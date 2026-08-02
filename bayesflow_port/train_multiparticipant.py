"""
train_multiparticipant.py — offline pilot + diagnostics for the milestone-2
group-template-pooling workflow (multiparticipant_workflow.py).

Recovers the shared 12-parameter group template from a SET of N participants'
confusion matrices via a SetTransformer summary network. Per-participant
attention nuisance parameters (k_A, k_B) are simulated but NOT inferred here —
see multiparticipant_workflow.py's module docstring for why, and for the
two-stage extension (this workflow, then bf_workflow.py conditioned on the
recovered group template) that would recover them.

Run from the repo root:

    conda activate grin_venv
    KERAS_BACKEND=torch python bayesflow_port/train_multiparticipant.py \
        --train 20000 --epochs 100 --n-participants 8

Sessions are cheap to simulate (pure numpy, same cost class as the
single-matrix simulator times n_participants), so this also uses the
fast-simulator pilot budget: 20,000 sessions, 100 epochs, offline.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

import multiparticipant_workflow as mp
from inspect_training import inspect_history
from check_diagnostics import check_diagnostics, suggest_next_steps

FIGURE_NAMES = {
    "losses": "loss.png",
    "recovery": "recovery.png",
    "calibration_ecdf": "calibration_ecdf.png",
    "coverage": "coverage.png",
    "z_score_contraction": "z_score_contraction.png",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="bayesflow_port/results/multiparticipant-base")
    ap.add_argument("--train", type=int, default=20_000, help="pilot session budget")
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-participants", type=int, default=8)
    ap.add_argument("--kind", default="flow_matching", choices=["flow_matching", "coupling"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    session_sim = mp.GrinSessionSimulator(n_participants=args.n_participants, seed=args.seed)

    workflow = mp.build_workflow(results_dir, simulator=session_sim,
                                  inference_kind=args.kind)

    # ---- pilot budget: pre-simulate once, split train/val ----
    all_sims = mp.simulate_sessions(args.train + args.val, args.n_participants, rng)
    train_data = {k: v[: args.train] for k, v in all_sims.items()}
    val_data = {k: v[args.train:] for k, v in all_sims.items()}

    history = workflow.fit_offline(
        data=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=val_data,
    )

    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

    training_report = inspect_history(history.history)
    print(json.dumps(training_report, indent=2))
    if not training_report["overall"]["ok"]:
        print("TRAINING ISSUES — address before continuing:")
        for issue in training_report["overall"]["issues"]:
            print(f"  - {issue}")

    # ---- in-silico diagnostics on a fresh held-out test set of sessions ----
    test_data = mp.simulate_sessions(args.test, args.n_participants, rng)

    figures = workflow.plot_default_diagnostics(test_data=test_data)
    for key, fig in figures.items():
        fig.savefig(os.path.join(results_dir, FIGURE_NAMES[key]), dpi=150, bbox_inches="tight")
        plt.close(fig)

    metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"))
    print(metrics)

    diag_report = check_diagnostics(metrics)
    next_steps = suggest_next_steps(training_report, diag_report)

    # ---- also compare against the single-matrix (unpooled) baseline, if it
    # exists, as the direct test of whether pooling actually helps ----
    baseline_note = _compare_to_single_matrix_baseline(results_dir)

    _write_report_md(results_dir, args, training_report, diag_report, next_steps,
                      n_train=args.train, baseline_note=baseline_note)

    print(f"\nWrote report to {os.path.join(results_dir, 'report.md')}")


def _compare_to_single_matrix_baseline(results_dir):
    """If bayesflow_port/single-matrix-base/metrics.csv exists (from
    train_and_diagnose.py), note it in the report so the pooled-vs-unpooled
    rho/NRMSE comparison — the actual point of milestone 2 — is one click
    away rather than requiring a manual diff."""
    baseline_path = os.path.join(os.path.dirname(results_dir), "single-matrix-base", "metrics.csv")
    if os.path.exists(baseline_path):
        return (f"Single-matrix (unpooled) baseline metrics available at "
                f"`{baseline_path}` — compare rho NRMSE/ECE directly to gauge "
                f"how much pooling across participants helps identify the group "
                f"template vs. inferring from one matrix alone.")
    return ("No single-matrix baseline found at ../single-matrix-base/metrics.csv "
            "(both live under bayesflow_port/results/) — run train_and_diagnose.py "
            "first to get a pooled-vs-unpooled comparison.")


def _write_report_md(results_dir, args, training_report, diag_report, next_steps,
                      n_train, baseline_note):
    net_label = {"flow_matching": "FlowMatching (Base)",
                 "coupling": "CouplingFlow, spline (Base)"}[args.kind]

    conv_issues = training_report["overall"]["issues"]
    conv_assessment = (
        "Training converged with no NaNs, overfitting, or under-training detected."
        if not conv_issues else " ".join(conv_issues)
    )

    param_lines = [f"**{param}** — {summary}"
                   for param, summary in diag_report.get("summary", {}).items()]
    steps_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(next_steps))

    report = f"""# Amortized Inference — Diagnostic Report (Milestone 2: Multi-Participant Pooling)

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | {net_label} |
| Summary network | SetTransformer (Base, summary_dim=36) |
| Participants per session | {args.n_participants} |
| Epochs | {args.epochs} |
| Batch size | {args.batch_size} |
| Validation data | {args.val} held-out sessions |
| Training mode | offline |
| Simulation budget | {n_train} sessions ({n_train * args.n_participants} participant-matrices) |

## What this model infers

Only the **shared group-level template** (zx, zy, rho — 12 parameters), pooled
from {args.n_participants} exchangeable participants' confusion matrices per
session via a SetTransformer. Per-participant dimensional-attention scalars
(k_A, k_B) are simulated as nuisance variation but not inferred by this model
— see `multiparticipant_workflow.py`'s module docstring for the two-stage
extension that would recover them.

## Convergence

![Training loss](loss.png)

**Assessment:** {conv_assessment}

## Parameter Recovery

![Parameter recovery](recovery.png)

**Assessment:** see per-parameter summary below.

## Calibration and Coverage

![Calibration ECDF](calibration_ecdf.png)

![Coverage](coverage.png)

**Assessment:** see per-parameter summary below.

## Posterior Z-Score and Contraction

![Z-score and contraction](z_score_contraction.png)

**Assessment:** see per-parameter summary below.

## Numerical Diagnostic Summary

Full numeric table in `metrics.csv`. Per-parameter qualitative ratings:

{chr(10).join(param_lines)}

## Pooled vs. unpooled comparison

{baseline_note}

The core hypothesis this milestone exists to test: pooling {args.n_participants}
participants sharing a group template should recover rho — the
information-limited parameter in the single-matrix case (see
`docs/DESIGN_RECORD.md` §7) — substantially better than one matrix alone,
since the group-shared correlation structure now has {args.n_participants}x
the effective evidence. If rho NRMSE/ECE here is NOT meaningfully better than
the single-matrix baseline, that is an important negative result: either the
attention-scaling nuisance model is absorbing the group signal, or the
SetTransformer needs more capacity/training before pooling shows its expected
benefit — check both before concluding pooling doesn't help.

## Suggested Next Steps

{steps_md}
"""
    with open(os.path.join(results_dir, "report.md"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
