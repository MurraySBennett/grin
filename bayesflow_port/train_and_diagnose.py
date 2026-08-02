"""
train_and_diagnose.py — runnable offline-pilot pipeline for the single-matrix
BayesFlow port. Implements the amortized-workflow skill's mandatory steps that
the original quick-comparison script (compare_frameworks.py) was not designed
to cover on its own:

  - validation_data= passed to fit_offline
  - history.history saved as JSON, then run through scripts/inspect_training.py
  - workflow.compute_default_diagnostics / plot_default_diagnostics
  - a self-contained results_dir/ with the standard figure names, metrics.csv,
    and report.md, per references/reporting.md

Run from the repo root:

    conda activate grin_venv
    KERAS_BACKEND=torch python bayesflow_port/train_and_diagnose.py \
        --train 20000 --epochs 100 --kind flow_matching --size base

Fast simulator (a single simulate() call for 20k rows is well under a second),
so the pilot budget is 20,000 pre-simulated datasets trained offline for 100
epochs, per the skill's fast-simulator default — NOT the smaller 40k/20-epoch
smoke-test defaults in compare_frameworks.py, which are sized for a quick
cross-framework sanity check, not a reportable run.
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

import bf_simulator as sim
import bf_workflow as wf
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
    ap.add_argument("--results-dir", default="bayesflow_port/results/single-matrix-base")
    ap.add_argument("--train", type=int, default=20_000, help="pilot simulation budget")
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--kind", default="flow_matching", choices=["flow_matching", "coupling"])
    ap.add_argument("--size", default="base", choices=["base", "large"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    grin_sim = sim.GrinSimulator(seed=args.seed)

    workflow = wf.build_workflow(results_dir, kind=args.kind, size=args.size,
                                  simulator=grin_sim)

    # ---- pilot budget: pre-simulate once, split train/val (fast simulator) ----
    all_sims = sim.simulate(args.train + args.val, rng)
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
    test_data = sim.simulate(args.test, rng)

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
    """Fill the references/reporting.md template from the actual run."""
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

    param_lines = []
    for param, summary in diag_report.get("summary", {}).items():
        param_lines.append(f"**{param}** — {summary}")

    steps_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(next_steps))

    report = f"""# Amortized Inference — Diagnostic Report

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | {net_label} |
| Summary network | none (fixed 20-d featurised condition, single confusion matrix) |
| Epochs | {args.epochs} |
| Batch size | {args.batch_size} |
| Validation data | {args.val} held-out simulations |
| Training mode | offline |
| Simulation budget | {n_train} |

## Convergence

![Training loss](loss.png)

The training loss curve shows the optimization objective over epochs. A healthy curve decreases smoothly and plateaus. Key warning signs: (i) a growing gap between training and validation loss indicates overfitting; (ii) loss still visibly decreasing at the final epoch means the network could benefit from more epochs; (iii) NaN spikes indicate numerical instability, often caused by extreme simulator outputs or missing standardization.

**Assessment:** {conv_assessment}

## Parameter Recovery

![Parameter recovery](recovery.png)

Each panel plots the posterior median against the true parameter value across held-out simulations. Points on the diagonal indicate perfect recovery; vertical bars are 95% credible intervals.

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

## Suggested Next Steps

{steps_md}

## Notes specific to this port

- rho recovery is expected to lag z-score recovery — this is a property of the
  data (weak within-stimulus correlations are information-limited from a single
  confusion matrix), confirmed independently by the bespoke stack's MLE
  baseline in `docs/DESIGN_RECORD.md` section 7. Do not read a lower rho rating
  here as a BayesFlow-specific defect without checking that document first.
- This run has no summary network because the condition is a single fixed-length
  featurised vector (the "simple vector" case). The summary network only enters
  for the multi-participant extension — see `multiparticipant_workflow.py`.
"""
    with open(os.path.join(results_dir, "report.md"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
