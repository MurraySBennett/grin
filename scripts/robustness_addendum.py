"""
Robustness addendum for the manuscript.

This is deliberately an addendum, not a new method-comparison study. It asks what
GRIN reports when the count matrix is generated from near-GRT processes that violate
one assumption at a time:

  1. decisional-separability failure: the A decision bound tilts with perceived B;
  2. lapse/contamination: response probabilities are mixed with uniform responding;
  3. overdispersion: each row probability is drawn from a Dirichlet distribution
     before multinomial sampling.

The outputs are:
  results/validation/robustness_addendum.json
  results/figures/robustness_addendum.png

Run:
  python scripts/robustness_addendum.py --n-per-class 150 --seed 20260828
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import load_model, ENVELOPE_WARNING_THRESHOLD
from src.config import FIGURES_DIR, RESULTS_DIR, Z_MAX, R_MAX
from src.data.generator import GRTDataGenerator
from src.inference.model_posterior import amortized_compare
from src.inference.ood import envelope_deviance
from src.inference.predict import predict_point
from src.viz.labels import constructs_from_labels, labels_from_amortized
from src.viz.style import set_style, BLUE, BLUE_DEEP, RED, RED_DEEP, MUTE, INK
import src.grt_model as gm


OUT_JSON = os.path.join("results", "validation", "robustness_addendum.json")
OUT_FIG = os.path.join(FIGURES_DIR, "robustness_addendum.png")


def _draw_counts(probs, trials, rng):
    gen = GRTDataGenerator(n_per_class=1)
    return gen._multinomial_counts(np.asarray(probs), np.asarray(trials), rng).reshape(len(probs), 16)


def _sample_trials(n, rng, trial_range=(12, 200), imbalance=0.25):
    gen = GRTDataGenerator(n_per_class=1, trial_range=trial_range, imbalance=imbalance)
    return gen._sample_trial_counts(n, rng)


def _prior_batch(n_per_class, rng):
    params, labels = [], []
    for name in gm.MODEL_NAMES:
        zx, zy, rho = gm.sample_prior(name, n_per_class, rng, z_max=Z_MAX, r_max=R_MAX)
        params.append(gm.pack(zx, zy, rho))
        labels.extend([name] * n_per_class)
    return np.concatenate(params), np.asarray(labels, dtype=object)


def _tilted_a_bound_probs(zx, zy, rho, slope):
    """P(response) when the A bound is x = slope*y and the B bound remains y = 0.

    Define U = X - slope*Y and V = Y. The response rule is A2 iff U > 0 and
    B2 iff V > 0, so the probabilities are an ordinary bivariate normal quadrant
    calculation after standardising U.
    """
    var_u = 1.0 + slope * slope - 2.0 * slope * rho
    sd_u = np.sqrt(np.maximum(var_u, 1e-9))
    z_u = (zx - slope * zy) / sd_u
    r_uv = np.clip((rho - slope) / sd_u, -0.999, 0.999)
    return gm.forward_probabilities(z_u, zy, r_uv)


def _lapse_probs(probs, eps):
    return (1.0 - eps) * probs + eps * 0.25


def _overdispersed_counts(probs, trials, phi, rng):
    """Dirichlet-multinomial rows with intraclass correlation approximately phi."""
    if phi <= 0:
        return _draw_counts(probs, trials, rng)
    alpha0 = (1.0 / phi) - 1.0
    q = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        for s in range(4):
            q[i, s] = rng.dirichlet(np.clip(probs[i, s], 1e-8, 1.0) * alpha0)
    return _draw_counts(q, trials, rng)


def _metrics(model, counts, trials, truth, true_labels):
    ac = amortized_compare(model, counts, trials)
    pred_labels = labels_from_amortized(ac)
    pc, ps_a, ps_b = constructs_from_labels(pred_labels)
    tc, ts_a, ts_b = constructs_from_labels(true_labels)
    pred = predict_point(model, counts, trials).numpy()
    dev = envelope_deviance(model, counts, trials)

    p_pi = np.asarray(ac["p_PI"])
    p_a = np.asarray(ac["p_sep_A"])
    p_b = np.asarray(ac["p_sep_B"])
    evid_pi = np.abs(p_pi - 0.5) > 0.25
    evid_a = np.abs(p_a - 0.5) > 0.25
    evid_b = np.abs(p_b - 0.5) > 0.25

    any_false_evidence = (
        ((p_pi < 0.25) & (tc == 0)) |
        ((p_pi > 0.75) & (tc != 0)) |
        ((p_a < 0.25) & (ts_a == 1)) |
        ((p_a > 0.75) & (ts_a == 0)) |
        ((p_b < 0.25) & (ts_b == 1)) |
        ((p_b > 0.75) & (ts_b == 0))
    )

    return {
        "n": int(len(counts)),
        "parameter_mae": float(np.mean(np.abs(pred - truth))),
        "exact_class_accuracy": float(np.mean(pred_labels == true_labels)),
        "corr_structure_accuracy": float(np.mean(pc == tc)),
        "sep_a_accuracy": float(np.mean(ps_a == ts_a)),
        "sep_b_accuracy": float(np.mean(ps_b == ts_b)),
        "p_PI_mean": float(np.mean(p_pi)),
        "p_sep_A_mean": float(np.mean(p_a)),
        "p_sep_B_mean": float(np.mean(p_b)),
        "evidence_PI_rate": float(np.mean(evid_pi)),
        "evidence_sep_A_rate": float(np.mean(evid_a)),
        "evidence_sep_B_rate": float(np.mean(evid_b)),
        "any_false_evidence_rate": float(np.mean(any_false_evidence)),
        "envelope_deviance_median": float(np.median(dev)),
        "envelope_warning_rate": float(np.mean(dev > ENVELOPE_WARNING_THRESHOLD)),
    }


def _run_decisional(model, n, rng):
    zx, zy, rho = gm.sample_prior("pi_ps_ds", n, rng, z_max=Z_MAX, r_max=R_MAX)
    truth = gm.pack(zx, zy, rho)
    labels = np.asarray(["pi_ps_ds"] * n, dtype=object)
    trials = _sample_trials(n, rng)
    rows = []
    for slope in (0.0, 0.15, 0.30, 0.45, 0.60):
        probs = _tilted_a_bound_probs(zx, zy, rho, slope)
        counts = _draw_counts(probs, trials, rng)
        row = _metrics(model, counts, trials, truth, labels)
        row.update({"scenario": "decisional", "severity": float(slope),
                    "severity_label": f"slope={slope:.2f}"})
        rows.append(row)
        print("decisional", slope, row["any_false_evidence_rate"], row["envelope_warning_rate"])
    return rows


def _run_lapse(model, n_per_class, rng):
    truth, labels = _prior_batch(n_per_class, rng)
    zx, zy, rho = gm.unpack(truth)
    clean = gm.forward_probabilities(zx, zy, rho)
    trials = _sample_trials(len(truth), rng)
    rows = []
    for eps in (0.0, 0.02, 0.05, 0.10, 0.20):
        counts = _draw_counts(_lapse_probs(clean, eps), trials, rng)
        row = _metrics(model, counts, trials, truth, labels)
        row.update({"scenario": "lapse", "severity": float(eps),
                    "severity_label": f"eps={eps:.2f}"})
        rows.append(row)
        print("lapse", eps, row["exact_class_accuracy"], row["envelope_warning_rate"])
    return rows


def _run_overdispersion(model, n_per_class, rng):
    truth, labels = _prior_batch(n_per_class, rng)
    zx, zy, rho = gm.unpack(truth)
    clean = gm.forward_probabilities(zx, zy, rho)
    trials = _sample_trials(len(truth), rng)
    rows = []
    for phi in (0.0, 0.02, 0.05, 0.10, 0.20):
        counts = _overdispersed_counts(clean, trials, phi, rng)
        row = _metrics(model, counts, trials, truth, labels)
        row.update({"scenario": "overdispersion", "severity": float(phi),
                    "severity_label": f"phi={phi:.2f}"})
        rows.append(row)
        print("overdispersion", phi, row["exact_class_accuracy"], row["envelope_warning_rate"])
    return rows


def _plot(rows, path):
    import matplotlib.pyplot as plt

    set_style()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.8))
    ax = ax.ravel()

    dec = [r for r in rows if r["scenario"] == "decisional"]
    x = [r["severity"] for r in dec]
    ax[0].plot(x, [r["any_false_evidence_rate"] for r in dec], "-o", color=RED_DEEP,
               label="any false construct evidence")
    ax[0].plot(x, [r["envelope_warning_rate"] for r in dec], "-o", color=BLUE_DEEP,
               label="support warning")
    ax[0].set_xlabel("A-bound tilt")
    ax[0].set_ylabel("fraction of matrices")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_title("A   Decisional-separability violation")
    ax[0].legend(fontsize=8.5)

    for scenario, col, label in [("lapse", RED_DEEP, "lapse/contamination"),
                                 ("overdispersion", BLUE_DEEP, "overdispersion")]:
        rs = [r for r in rows if r["scenario"] == scenario]
        xs = [r["severity"] for r in rs]
        ax[1].plot(xs, [r["exact_class_accuracy"] for r in rs], "-o", color=col,
                   label=label)
        ax[2].plot(xs, [r["envelope_warning_rate"] for r in rs], "-o", color=col,
                   label=label)
        ax[3].plot(xs, [r["parameter_mae"] for r in rs], "-o", color=col,
                   label=label)

    ax[1].set_title("B   Model-class accuracy")
    ax[1].set_ylabel("12-way accuracy")
    ax[1].set_ylim(0, 1.03)
    ax[1].legend(fontsize=8.5)

    ax[2].set_title("C   Training-support warnings")
    ax[2].set_xlabel("departure severity")
    ax[2].set_ylabel("fraction flagged")
    ax[2].set_ylim(0, 1.03)

    ax[3].set_title("D   Parameter recovery")
    ax[3].set_xlabel("departure severity")
    ax[3].set_ylabel("mean absolute error")

    for a in ax[1:]:
        a.set_xlim(-0.005, 0.205)
    fig.suptitle("Robustness addendum: departures from the training simulator",
                 x=0.02, ha="left", fontweight="bold", fontsize=15, color=INK)
    fig.text(0.02, 0.945,
             "Departures are scored against the clean generating representation; "
             "support warnings use the calibrated training-support threshold of 29.3.",
             ha="left", va="top", fontsize=10, color=MUTE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = load_model()
    n_decisional = args.n_per_class * len(gm.MODEL_NAMES)
    rows = []
    rows.extend(_run_decisional(model, n_decisional, rng))
    rows.extend(_run_lapse(model, args.n_per_class, rng))
    rows.extend(_run_overdispersion(model, args.n_per_class, rng))

    out = {
        "meta": {
            "seed": args.seed,
            "n_per_class": args.n_per_class,
            "n_decisional": n_decisional,
            "envelope_threshold": ENVELOPE_WARNING_THRESHOLD,
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    _plot(rows, OUT_FIG)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
