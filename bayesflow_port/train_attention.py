"""
train_attention.py -- runnable offline-pilot pipeline for stage 2 of milestone 2:
recovering per-participant attention scalars (log k_A, log k_B) conditioned on
the group-level identified template. See attention_workflow.py's module
docstring for the full two-stage design and the plug-in/uncertainty-propagation
decisions locked in with the user before this was built.

This trains and diagnoses stage 2 IN ISOLATION, conditioned on the TRUE
simulated group template (the plug-in training design) -- it does NOT chain
onto a trained multiparticipant_workflow.py checkpoint. That composition
(stage-1 posterior draws -> stage-2 pooled samples) is
attention_workflow.sample_attention_propagated, exercised separately once both
stage-1 and stage-2 checkpoints exist; see that function's docstring for how to
call it on real per-participant data.

Run from the repo root:

    conda activate grin_venv
    KERAS_BACKEND=torch python bayesflow_port/train_attention.py \
        --train 20000 --epochs 100

Sessions with n_participants=1 are exactly as cheap to simulate as the
single-matrix port (same forward model, one matrix per row), so this uses the
same fast-simulator pilot budget: 20,000 rows, 100 epochs, offline.
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

import attention_workflow as aw
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
    ap.add_argument("--results-dir", default="bayesflow_port/results/attention-base")
    ap.add_argument("--train", type=int, default=20_000, help="pilot simulation budget")
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--attention-sd", type=float, default=0.25)
    ap.add_argument("--kind", default="flow_matching", choices=["flow_matching", "coupling"])
    ap.add_argument("--size", default="base", choices=["base", "large"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    participant_sim = aw.GrinParticipantSimulator(seed=args.seed, attention_sd=args.attention_sd)

    workflow = aw.build_workflow(results_dir, kind=args.kind, size=args.size,
                                  simulator=participant_sim)

    # ---- pilot budget: pre-simulate once, split train/val (fast simulator) ----
    all_sims = aw.simulate_participants(args.train + args.val, rng, attention_sd=args.attention_sd)
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

    # ---- in-silico diagnostics on a fresh held-out test set ----
    test_data = aw.simulate_participants(args.test, rng, attention_sd=args.attention_sd)

    figures = workflow.plot_default_diagnostics(test_data=test_data)
    for key, fig in figures.items():
        fig.savefig(os.path.join(results_dir, FIGURE_NAMES[key]), dpi=150, bbox_inches="tight")
        plt.close(fig)

    metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"))
    print(metrics)

    diag_report = check_diagnostics(metrics)
    next_steps = suggest_next_steps(training_report, diag_report)

    _write_report_md(results_dir, args, training_report, diag_report, next_steps,
                      n_train=args.train)

    print(f"\nWrote report to {os.path.join(results_dir, 'report.md')}")


def _write_report_md(results_dir, args, training_report, diag_report, next_steps, n_train):
    net_label = {
        ("flow_matching", "base"): "FlowMatching (Base)",
        ("flow_matching", "large"): "FlowMatching (Large)",
        ("coupling", "base"): "CouplingFlow, spline (Base)",
        ("coupling", "large"): "CouplingFlow, spline (Large)",
    }[(args.kind, args.size)]

    conv_issues = training_report["overall"]["issues"]
    conv_assessment = (
        "Training converged with no NaNs, overfitting, or under-training detected."
        if not conv_issues else " ".join(conv_issues)
    )

    param_lines = [f"**{param}** — {summary}"
                   for param, summary in diag_report.get("summary", {}).items()]
    steps_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(next_steps))

    report = f"""# Amortized Inference — Diagnostic Report (Milestone 2, Stage 2: Attention Scalars)

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | {net_label} |
| Summary network | none (fixed 32-d condition: 20-d participant matrix ++ 12-d group template) |
| Target | log(k_A), log(k_B) |
| Group template used in training | TRUE (plug-in), from the simulator — NOT a stage-1 posterior draw |
| Epochs | {args.epochs} |
| Batch size | {args.batch_size} |
| Validation data | {args.val} held-out rows |
| Training mode | offline |
| Simulation budget | {n_train} |
| attention_sd (prior) | {args.attention_sd} |

## Convergence

![Training loss](loss.png)

The training loss curve shows the optimization objective over epochs. A healthy curve decreases smoothly and plateaus. Key warning signs: (i) a growing gap between training and validation loss indicates overfitting; (ii) loss still visibly decreasing at the final epoch means the network could benefit from more epochs; (iii) NaN spikes indicate numerical instability, often caused by extreme simulator outputs or missing standardization.

**Assessment:** {conv_assessment}

## Parameter Recovery

![Parameter recovery](recovery.png)

Each panel plots the posterior median against the true log(k_A)/log(k_B) across held-out simulations, conditioned on the TRUE group template. This isolates stage 2's own performance from any stage-1 estimation error — see "Two-stage composition" below for how the two errors combine in application.

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

## Two-stage composition (important caveat)

This report diagnoses stage 2 **in isolation**, conditioned on the group template exactly as simulated — it does NOT reflect stage-1 (`multiparticipant_workflow.py`) estimation error. In application, the group template is itself a posterior, not a known value. Uncertainty is propagated by Monte Carlo mixture — `attention_workflow.sample_attention_propagated` runs stage 2 once per stage-1 posterior draw and pools the results — rather than a point-estimate plug-in (which would understate stage-2 uncertainty) or joint retraining. This is a **"cut" in the Bayesian-workflow sense**: individual-level fit cannot feed back and correct the group-level estimate. Accept this as a known limitation of the two-stage design, not a defect — a full joint hierarchical amortizer would be a substantially larger undertaking and was explicitly not the chosen design.

## Suggested Next Steps

{steps_md}
"""
    with open(os.path.join(results_dir, "report.md"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
