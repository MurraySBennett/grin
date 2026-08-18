"""
checks_literature.py — the Week 1 "turn the reviewed empirical designs into
explicit validation scenarios" deliverable from the literature review
(docs/literature_review_findings.md, gitignored, ask the user for it if you need
the full write-up).

STATUS: staged, not yet run at grid scale. Every function here defaults to a
SMALL n_per_class so it can be smoke-tested for correctness on the laptop (a few
seconds, not a validation run) -- see `if __name__ == "__main__"` below. The real
Week 2 decision battery (large n_per_class, the full parameter grids) is
compute -- generation is cheap but the grids are large enough, and the current
production checkpoint is loaded for real inference throughout, so this stays
lab-computer work, not something to fire off here. Nothing in this file trains a
network; it only evaluates the ALREADY-shipped checkpoint against new simulated
regimes the standard validation/checks.py suite doesn't cover.

Predefined retraining trigger, exactly as the review specified: absolute coverage
error > COVERAGE_ERROR_TRIGGER_PP percentage points (at the 90% nominal level) in
any evaluated stratum. Decided ahead of seeing results, not tuned after.
"""
import os

import numpy as np

from src.config import RESULTS_DIR, TRIAL_RANGE, TRIAL_IMBALANCE, Z_MAX, R_MAX
from src.data.robustness_generator import ExemplarHeterogeneityGenerator, LearningMixtureGenerator
from src.inference.predict import predict_posterior
from src.inference.model_posterior import amortized_compare
from src.inference.ood import envelope_deviance

COVERAGE_ERROR_TRIGGER_PP = 3.0   # percentage points, at nominal 90% -- see module docstring
EXTERNAL_MATRIX_DIR = os.path.join(RESULTS_DIR, "..", "data", "literature", "external_matrices")


def _coverage90(samples, truth):
    """samples (S, n, 12), truth (n, 12) -> per-parameter-group coverage at 90%."""
    lo = np.quantile(samples, 0.05, axis=0)
    hi = np.quantile(samples, 0.95, axis=0)
    hit = (truth >= lo) & (truth <= hi)                      # (n, 12)
    return {
        "overall": float(hit.mean()),
        "z": float(hit[:, :8].mean()),
        "rho": float(hit[:, 8:].mean()),
    }


def _evaluate(model, X, y_params, X_trials, n_samples=400):
    post = predict_posterior(model, X, X_trials, n_samples=n_samples)
    mean = post["mean"].numpy()
    samples = post["samples"].numpy()
    mae = float(np.abs(mean - y_params).mean())
    cov = _coverage90(samples, y_params)
    return {"mae": mae, "coverage90": cov}


def lit_exemplar_heterogeneity_grid(model, n_per_class=30,
                                    n_exemplars_list=(1, 2, 4, 10, 35),
                                    heterogeneity_list=(0.0, 0.15, 0.3, 0.6),
                                    modes=("fixed_blocks", "iid_resample"),
                                    trial_range=TRIAL_RANGE, seed=0):
    """docs/literature_review_findings.md #3. n_per_class=30 here is a smoke-test
    default (12 classes x 30 = 360 matrices per grid cell); scale up for the real
    Week 2 run. Crosses exemplar count x heterogeneity x assignment mode."""
    out = {}
    for mode in modes:
        for K in n_exemplars_list:
            for het in heterogeneity_list:
                if K == 1 and het > 0:
                    continue   # heterogeneity is meaningless with a single exemplar
                g = ExemplarHeterogeneityGenerator(
                    n_per_class=n_per_class, trial_range=trial_range,
                    z_max=Z_MAX, r_max=R_MAX, imbalance=TRIAL_IMBALANCE,
                    n_exemplars=K, heterogeneity=het, exemplar_mode=mode, seed=seed)
                X, yp, Xt, _, _ = g.generate_all_model_cms()
                out[f"{mode}/K={K}/het={het}"] = _evaluate(model, X, yp, Xt)
    return out


def lit_learning_mixture_grid(model, n_per_class=30,
                              changepoint_fracs=(0.0, 0.2, 0.4, 0.6, 0.8),
                              drift_scales=(0.0, 0.3, 0.6, 1.0),
                              trial_range=TRIAL_RANGE, seed=1):
    """docs/literature_review_findings.md #4 (nonstationarity/learning). Ground
    truth is always the LATE representation -- this measures how badly a naive
    cumulative matrix misrepresents "current" state, at each changepoint
    location and drift magnitude, using the checkpoint exactly as deployed
    today (no rolling window, no change-point layer -- those don't exist yet)."""
    out = {}
    for cp in changepoint_fracs:
        for drift in drift_scales:
            if cp == 0.0 and drift > 0:
                continue   # no early phase to differ from
            g = LearningMixtureGenerator(
                n_per_class=n_per_class, trial_range=trial_range,
                z_max=Z_MAX, r_max=R_MAX, imbalance=TRIAL_IMBALANCE,
                changepoint_frac=cp, drift_scale=drift, seed=seed)
            X, yp, Xt, _, _ = g.generate_all_model_cms()
            out[f"cp={cp}/drift={drift}"] = _evaluate(model, X, yp, Xt)
    return out


def check_retraining_trigger(grid_results, nominal=0.90):
    """Apply the predefined trigger to a grid's results (as returned by either
    function above): any stratum whose 90%-nominal coverage error exceeds
    COVERAGE_ERROR_TRIGGER_PP fires it. Returns (triggered: bool, offending strata: list)."""
    bad = []
    for stratum, res in grid_results.items():
        err_pp = abs(res["coverage90"]["overall"] - nominal) * 100
        if err_pp > COVERAGE_ERROR_TRIGGER_PP:
            bad.append((stratum, err_pp))
    return bool(bad), bad


def load_one_matrix(name):
    """A single named external matrix. Checks two sources so neither duplicates
    the other: `external_matrices/<name>.csv` (one row per participant, e.g.
    `soto2015` -- transcriptions specific to this harness) first, then falls back
    to `data/real/real_matrices.csv` (one row per dataset, keyed by the `dataset`
    column -- the mdsdt-derived exports `scripts/R/fit_real_data.R` already
    produces: thomas01a/b, silbert09a/b, silbert12). Run
    `Rscript scripts/R/fit_real_data.R` first if the second file doesn't exist yet.
    Both use the same cm_00..cm_33 row-major canonical-order column convention."""
    import pandas as pd
    path = os.path.join(EXTERNAL_MATRIX_DIR, f"{name}.csv")
    cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df[cols].to_numpy().astype(float).reshape(-1, 4, 4)
    real_path = os.path.join(RESULTS_DIR, "..", "data", "real", "real_matrices.csv")
    if os.path.exists(real_path):
        df = pd.read_csv(real_path)
        row = df[df["dataset"] == name]
        if len(row):
            return row[cols].to_numpy().astype(float).reshape(-1, 4, 4)
    raise FileNotFoundError(
        f"'{name}' not found in {EXTERNAL_MATRIX_DIR} or {real_path}. If this is "
        f"one of thomas01a/thomas01b/silbert09a/silbert09b/silbert12, run "
        f"`Rscript scripts/R/fit_real_data.R` first (mdsdt-derived, no publisher "
        f"supplement needed). Otherwise it needs a provenance-preserving "
        f"transcription added under external_matrices/ (see soto2015.csv's "
        f"README there for the pattern).")


# Backward-compatible alias -- the name used in this module's earlier drafts and
# in any external caller that already picked it up.
load_external_matrix = load_one_matrix


def load_all_available_matrices():
    """Every external matrix currently on disk, from both sources, as a single
    {name: (4,4) array} dict -- the natural unit for the fixed-stimulus external
    validation the literature review's Week 2 battery calls for."""
    import pandas as pd
    out = {}
    if os.path.isdir(EXTERNAL_MATRIX_DIR):
        for fn in sorted(os.listdir(EXTERNAL_MATRIX_DIR)):
            if not fn.endswith(".csv"):
                continue
            stem = fn[:-4]
            df = pd.read_csv(os.path.join(EXTERNAL_MATRIX_DIR, fn))
            if "participant" in df.columns:
                for _, row in df.iterrows():
                    out[f"{stem}_p{int(row['participant'])}"] = load_one_matrix(stem)[
                        int(row["participant"]) - 1]
            else:
                mats = load_one_matrix(stem)
                for i in range(mats.shape[0]):
                    out[f"{stem}_{i}"] = mats[i]
    real_path = os.path.join(RESULTS_DIR, "..", "data", "real", "real_matrices.csv")
    if os.path.exists(real_path):
        df = pd.read_csv(real_path)
        cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
        for _, row in df.iterrows():
            out[row["dataset"]] = row[cols].to_numpy().astype(float).reshape(4, 4)
    return out


def lit_external_matrix_check(model):
    """The fixed-stimulus external-validation leg of the Week 2 battery: run the
    current checkpoint on every real matrix available from BOTH sources (Soto et
    al. 2015's 24 participants, transcribed from their supplement; the 5 mdsdt
    datasets), and report the envelope-check deviance and top construct read-out
    for each. Cheap -- a few dozen real matrices, one inference pass each -- and
    safe to run anywhere, unlike the simulated grids above. Not a coverage/bias
    check (no ground truth exists for real data); the point is the same one
    `scripts/compare_real_data.py` makes for the mdsdt 5, extended to the Soto
    sample: does the current checkpoint's envelope check ever fire on real,
    already-published human data, and if so, which observers."""
    mats = load_all_available_matrices()
    names = sorted(mats.keys())
    X = np.stack([mats[n] for n in names]).reshape(len(names), 16)
    Xt = X.reshape(-1, 4, 4).sum(2)
    ac = amortized_compare(model, X, Xt)
    dev = envelope_deviance(model, X, Xt)
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "name": name, "n_trials": int(Xt[i].sum()), "envelope_deviance": float(dev[i]),
            "p_PI": float(ac["p_PI"][i]), "p_sep_A": float(ac["p_sep_A"][i]),
            "p_sep_B": float(ac["p_sep_B"][i]),
        })
    return rows


if __name__ == "__main__":
    # Smoke test only -- tiny n_per_class, confirms the harness runs end to end
    # against the real shipped checkpoint. NOT the Week 2 battery; see module
    # docstring before scaling n_per_class up.
    from src.api import load_model
    print("loading production checkpoint...")
    m = load_model()
    print("\nexemplar-heterogeneity grid (smoke scale):")
    ex = lit_exemplar_heterogeneity_grid(m, n_per_class=5,
                                         n_exemplars_list=(1, 4), heterogeneity_list=(0.0, 0.5))
    for k, v in ex.items():
        print(f"  {k:30s} mae={v['mae']:.3f} coverage90={v['coverage90']['overall']:.3f}")
    triggered, bad = check_retraining_trigger(ex)
    print(f"  trigger: {triggered}  ({bad})")

    print("\nlearning-mixture grid (smoke scale):")
    lm = lit_learning_mixture_grid(m, n_per_class=5,
                                   changepoint_fracs=(0.0, 0.5), drift_scales=(0.0, 0.8))
    for k, v in lm.items():
        print(f"  {k:30s} mae={v['mae']:.3f} coverage90={v['coverage90']['overall']:.3f}")
    triggered, bad = check_retraining_trigger(lm)
    print(f"  trigger: {triggered}  ({bad})")
