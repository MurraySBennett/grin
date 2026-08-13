"""
build_benchmarks.py — computes web/assets/data/benchmarks.json, the single file
validate.html renders.

    python scripts/build_benchmarks.py

WHAT THIS DOES AND DOES NOT DO
-------------------------------
This mirrors the validation suite's own checks (see validation/checks.py, v03-v11,
and README_validation.md for the full v01-v16 taxonomy) so the numbers on the web
page are DEFINED THE SAME WAY as the numbers in the paper — same MAE definition,
same coverage definition, same speed protocol.

It runs in one of two modes, chosen automatically:

  MLE-only   (no trained model found)
             Every metric that can be computed without a network is computed —
             MLE recovery, MLE speed, MLE model-selection accuracy, MLE-vs-truth
             calibration is NOT computed (a plug-in Wald interval isn't the same
             claim as a trained posterior, and reporting it next to "GRIN
             calibration" would invite exactly the confusion this suite exists to
             prevent).

  GRIN+MLE   (a trained model is found at RESULTS/models/npe_model.pt)
             Everything above, PLUS the network's recovery, calibration (SBC-style
             interval coverage), speed, OOD detection, and amortized model
             comparison vs AIC/BIC.

Every metric in the output JSON carries a "status": "measured" | "pending_weights"
field. validate.html renders "pending_weights" panels as an explicit placeholder —
it NEVER fabricates or interpolates a GRIN number. Re-run this script once
npe_model.pt exists and the page comes alive with no code changes.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.grt_model as gm
from src.inference.mle import fit_class, fit_and_select, _expand

OUT = os.path.join(os.path.dirname(__file__), "..", "web", "assets", "data", "benchmarks.json")
SEED = 20260711


# --------------------------------------------------------------------------- #
# Try to find a trained model. Never raises — falls back to MLE-only mode.
# --------------------------------------------------------------------------- #
def try_load_model():
    try:
        from src.config import MODEL_FILE
        from src.api import load_model
        if not os.path.exists(MODEL_FILE):
            print(f"[build_benchmarks] no weights at {MODEL_FILE} -> MLE-only mode")
            return None
        m = load_model(MODEL_FILE, device="cpu")
        print(f"[build_benchmarks] loaded {MODEL_FILE} -> GRIN+MLE mode")
        return m
    except Exception as e:  # pragma: no cover — best-effort convenience path
        print(f"[build_benchmarks] could not load a trained model ({e}) -> MLE-only mode")
        return None


# --------------------------------------------------------------------------- #
# Shared dataset construction — same protocol as validation/checks.py
# --------------------------------------------------------------------------- #
def make_dataset(n_per_class, trial_range, seed, balanced=True):
    rng = np.random.default_rng(seed)
    Xs, ys, Xts, clss = [], [], [], []
    for name in gm.MODEL_NAMES:
        zx, zy, rho = gm.sample_prior(name, n_per_class, rng)
        params = gm.pack(zx, zy, rho)
        probs = gm.forward_probabilities(zx, zy, rho)  # (n,4,4)
        if balanced:
            t = np.full((n_per_class, 4), int(np.mean(trial_range)))
        else:
            lo, hi = trial_range
            t = np.exp(rng.uniform(np.log(lo), np.log(hi), (n_per_class, 4))).astype(int)
        counts = np.stack([[rng.multinomial(t[i, s], probs[i, s]) for s in range(4)]
                            for i in range(n_per_class)])
        Xs.append(counts.reshape(n_per_class, 16))
        ys.append(params)
        Xts.append(t)
        clss.extend([name] * n_per_class)
    X = np.concatenate(Xs); y = np.concatenate(ys); Xt = np.concatenate(Xts)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], Xt[idx], np.array(clss)[idx]


def cm_from_flat(x):
    return x.reshape(4, 4)


# --------------------------------------------------------------------------- #
# v03 — parameter recovery (MLE always; GRIN if available)
# --------------------------------------------------------------------------- #
def bench_recovery(model):
    X, y, Xt, _ = make_dataset(60, (200, 200), seed=42 + SEED)
    mle_pred = np.array([fit_class(cm_from_flat(X[i]), Xt[i], "ds")["params"] for i in range(len(X))])
    mle_z_mae = float(np.abs(mle_pred[:, :8] - y[:, :8]).mean())
    mle_r_mae = float(np.abs(mle_pred[:, 8:] - y[:, 8:]).mean())

    out = {
        "id": "v03", "claim": "parameter recovery",
        "n": len(X), "trials_per_stimulus": 200,
        "mle": {"zscore_MAE": mle_z_mae, "rho_MAE": mle_r_mae, "status": "measured"},
        "grin": {"status": "pending_weights"},
    }
    if model is not None:
        from src.inference.predict import predict_point
        p = predict_point(model, X, Xt).numpy()
        out["grin"] = {
            "zscore_MAE": float(np.abs(p[:, :8] - y[:, :8]).mean()),
            "rho_MAE": float(np.abs(p[:, 8:] - y[:, 8:]).mean()),
            "status": "measured",
        }
    return out


# --------------------------------------------------------------------------- #
# v04 — calibration. GRIN only: a Wald interval from MLE is a different claim
# from a trained posterior, and this suite's whole point is not to blur that.
# --------------------------------------------------------------------------- #
def bench_calibration(model):
    out = {"id": "v04", "claim": "posterior interval coverage",
           "note": ("Nominal X% intervals should contain the true parameter X% of "
                     "the time. This is the calibration claim, and it is specific to "
                     "a model that returns a posterior — an MLE point estimate has no "
                     "comparable number, so none is reported here."),
           "grin": {"status": "pending_weights"}}
    if model is None:
        return out
    from src.inference.predict import predict_posterior
    X, y, Xt, _ = make_dataset(50, (20, 400), seed=7 + SEED, balanced=False)
    s = predict_posterior(model, X, Xt, n_samples=400)["samples"].numpy()
    cov = {}
    for lvl in (0.5, 0.9, 0.95):
        lo = np.quantile(s, (1 - lvl) / 2, axis=0)
        hi = np.quantile(s, (1 + lvl) / 2, axis=0)
        cov[f"{int(lvl * 100)}%"] = float(((y >= lo) & (y <= hi)).mean())
    out["grin"] = {"coverage": cov, "n": len(X), "status": "measured"}
    return out


# --------------------------------------------------------------------------- #
# v05 — speed. MLE always measurable; GRIN if available.
# --------------------------------------------------------------------------- #
def bench_speed(model):
    X, y, Xt, _ = make_dataset(20, (200, 200), seed=5 + SEED)
    n = len(X)
    t0 = time.time()
    for i in range(n):
        fit_and_select(cm_from_flat(X[i]), Xt[i], "bic")
    mle_full_ms = 1e3 * (time.time() - t0) / n

    t0 = time.time()
    for i in range(n):
        fit_class(cm_from_flat(X[i]), Xt[i], "ds")
    mle_one_ms = 1e3 * (time.time() - t0) / n

    out = {"id": "v05", "claim": "amortized speedup vs MLE", "n": n,
           "mle_one_fit_ms": mle_one_ms, "mle_full_selection_ms": mle_full_ms,
           "grin": {"status": "pending_weights"}}
    if model is not None:
        from src.inference.predict import predict_point
        predict_point(model, X[:5], Xt[:5])  # warm up
        t0 = time.time()
        predict_point(model, X, Xt)
        grin_ms = 1e3 * (time.time() - t0) / n
        out["grin"] = {"ms_per_matrix": grin_ms,
                        "speedup_vs_one_mle_fit": mle_one_ms / grin_ms,
                        "speedup_vs_full_selection": mle_full_ms / grin_ms,
                        "status": "measured"}
    return out


# --------------------------------------------------------------------------- #
# v07 — reliability across trial counts. MLE recovery always measurable.
# --------------------------------------------------------------------------- #
def bench_trial_sweep(model):
    out = {"id": "v07", "claim": "reliability across trial counts", "by_trials": {}}
    for T in (10, 25, 50, 100, 200, 400, 1000):
        X, y, Xt, _ = make_dataset(30, (T, T), seed=100 + T + SEED)
        mle_pred = np.array([fit_class(cm_from_flat(X[i]), Xt[i], "ds")["params"] for i in range(len(X))])
        row = {"mle_MAE": float(np.abs(mle_pred - y).mean()), "grin": {"status": "pending_weights"}}
        if model is not None:
            from src.inference.predict import predict_posterior
            po = predict_posterior(model, X, Xt, n_samples=300)
            mae = float(np.abs(po["mean"].numpy() - y).mean())
            s = po["samples"].numpy()
            lo, hi = np.quantile(s, 0.05, 0), np.quantile(s, 0.95, 0)
            row["grin"] = {"MAE": mae, "coverage90": float(((y >= lo) & (y <= hi)).mean()),
                            "status": "measured"}
        out["by_trials"][str(T)] = row
    return out


# --------------------------------------------------------------------------- #
# v11 — model-class selection accuracy: MLE/AIC-BIC vs GRIN's comparison heads
# --------------------------------------------------------------------------- #
def bench_selection(model):
    X, y, Xt, true_cls = make_dataset(40, (400, 400), seed=11 + SEED)
    n = len(X)

    t0 = time.time()
    bic_pred = [fit_and_select(cm_from_flat(X[i]), Xt[i], "bic")[0]["model"] for i in range(n)]
    bic_ms = 1e3 * (time.time() - t0) / n
    bic_acc = float(np.mean(np.array(bic_pred) == true_cls))

    out = {"id": "v11", "claim": "model-class selection accuracy", "n": n,
           "mle_aic_bic": {"exact_class_accuracy": bic_acc, "ms_per_matrix": bic_ms,
                            "status": "measured"},
           "grin": {"status": "pending_weights"}}
    if model is not None:
        from src.inference.model_posterior import amortized_compare, construct_labels
        t0 = time.time()
        ac = amortized_compare(model, X, Xt)
        grin_ms = 1e3 * (time.time() - t0) / n
        tc, ta, tb = construct_labels(true_cls)
        pc = ac["p_corr"].argmax(1)
        psa = (ac["p_sep_A"] > 0.5).astype(int)
        psb = (ac["p_sep_B"] > 0.5).astype(int)
        out["grin"] = {
            "corr_accuracy": float(np.mean(pc == tc)),
            "sepA_accuracy": float(np.mean(psa == ta)),
            "sepB_accuracy": float(np.mean(psb == tb)),
            "ms_per_matrix": grin_ms, "status": "measured",
        }
    return out


# --------------------------------------------------------------------------- #
# v08 — the PI identifiability frontier: accuracy vs true |rho|, MLE-only proxy
# always measurable, GRIN heads if available
# --------------------------------------------------------------------------- #
def bench_pi_frontier(model):
    X, y, Xt, true_cls = make_dataset(80, (400, 400), seed=9 + SEED)
    mr = np.abs(y[:, 8:12]).max(1)
    bins = [(0, 0.001, "true_PI"), (0.001, 0.3, "weak"), (0.3, 0.6, "moderate"), (0.6, 0.95, "strong")]

    bic_pred = np.array([fit_and_select(cm_from_flat(X[i]), Xt[i], "bic")[0]["model"]
                          for i in range(len(X))])
    from src.grt_model import MODEL_SPECS
    bic_is_pi = np.array([MODEL_SPECS[m][0] == "pi" for m in bic_pred])
    true_is_pi = np.array([MODEL_SPECS[m][0] == "pi" for m in true_cls])

    mle_out = {}
    for lo, hi, lab in bins:
        m = (mr >= lo) & (mr < hi)
        if m.sum() > 5:
            mle_out[lab] = {"n": int(m.sum()), "accuracy": float(np.mean(bic_is_pi[m] == true_is_pi[m]))}

    out = {"id": "v08", "claim": "PI identifiability frontier (the honest limit)",
           "mle_aic_bic": mle_out, "grin": {"status": "pending_weights"}}
    if model is not None:
        from src.inference.model_posterior import amortized_compare, construct_labels
        ac = amortized_compare(model, X, Xt)
        pc = ac["p_corr"].argmax(1)  # 0=pi in construct_labels' convention
        tc, _, _ = construct_labels(true_cls)
        grin_out = {}
        for lo, hi, lab in bins:
            m = (mr >= lo) & (mr < hi)
            if m.sum() > 5:
                grin_out[lab] = {"n": int(m.sum()), "accuracy": float(np.mean((pc[m] == 0) == (tc[m] == 0)))}
        out["grin"] = {**grin_out, "status": "measured"}
    return out


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
def main():
    model = try_load_model()
    mode = "grin+mle" if model is not None else "mle_only"

    benchmarks = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "seed": SEED,
        "checks": {
            "v03_recovery": bench_recovery(model),
            "v04_calibration": bench_calibration(model),
            "v05_speed": bench_speed(model),
            "v07_trial_sweep": bench_trial_sweep(model),
            "v08_pi_frontier": bench_pi_frontier(model),
            "v11_selection": bench_selection(model),
        },
        # v12-v16 (RT-specific: collinearity probe, RT gain, architecture recovery,
        # speed-confound control, LBA recovery) need the RT model and RT dataset,
        # which is the same "pending_weights" story — wire in once npe_rt_model.pt
        # exists. See validation/checks_rt.py for the exact protocols to mirror.
        "rt_checks": {"status": "pending_weights",
                       "note": "See validation/checks_rt.py (v12-v16) for the protocols "
                               "to mirror once the RT model is trained."},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"[build_benchmarks] wrote {OUT}  (mode={mode})")


if __name__ == "__main__":
    main()
    