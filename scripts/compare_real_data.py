"""
Compare GRIN against mdsdt, grtools and a Python maximum-likelihood baseline on REAL
2x2 identification matrices, where there is no ground truth.

    Rscript scripts/R/fit_real_data.R      # 1. extract, fit, and thin the real matrices
    python scripts/compare_real_data.py    # 2. run GRIN on the same matrices  <- here

Three things are reported, because "does it agree" is only the first of them.

1. AGREEMENT ON CONCLUSIONS. Does GRIN reach the same verdict on perceptual
   independence and separability as the established packages' AIC model selection?
   This is the check that matters to a user deciding whether to trust the tool.
2. AGREEMENT ON PARAMETERS. Do the other methods' point estimates fall inside GRIN's
   credible intervals? Model-class agreement can hide substantial disagreement about
   the representation itself, and the perceptual-space figure shows it directly.
3. BEHAVIOUR AS DATA THIN. Each real matrix is resampled down to smaller trial counts
   (in fit_real_data.R) and every method refit. With no ground truth, the reference is
   each method's own full-data estimate: a method that is stable under thinning returns
   something close to what it returned with everything. This is the real-data analogue
   of the simulated sparse-data comparison.

Writes results/figures/real_data_*.png and results/real_data_comparison.json.
"""
import json, os, time
import numpy as np
import pandas as pd

from src.config import REAL_DATA_DIR, MLE_FITS_DIR, FIGURES_DIR, MODEL_FILE
from src.api import load_model
from src.inference.predict import predict_posterior
from src.inference.model_posterior import amortized_compare
from src.inference.mle import fit_selected
from src.inference.ood import envelope_deviance

MATS = os.path.join(REAL_DATA_DIR, "real_matrices.csv")
RFIT = os.path.join(MLE_FITS_DIR, "real_data_fits.csv")
RSUB = os.path.join(MLE_FITS_DIR, "real_subsample_fits.csv")
OUT_JSON = os.path.join("results", "real_data_comparison.json")
AIC_PARSIMONY = 1.0
CM_COLS = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
PNAMES = ([f"zx_{i}" for i in range(4)] + [f"zy_{i}" for i in range(4)]
          + [f"rho_{i}" for i in range(4)])
CORR_LABELS = ["independence", "one shared rho", "free rho"]


def _load_matrices():
    df = pd.read_csv(MATS)
    X = df[CM_COLS].to_numpy(float)
    return df["dataset"].tolist(), X, X.reshape(-1, 4, 4).sum(2)


def _grin(model, X, Xt, n_samples=4000):
    post = predict_posterior(model, X, Xt, n_samples=n_samples)
    ac = amortized_compare(model, X, Xt, parsimony=AIC_PARSIMONY)
    smp = post["samples"].numpy()
    return dict(mean=post["mean"].numpy(),
                lo=np.quantile(smp, 0.025, axis=0), hi=np.quantile(smp, 0.975, axis=0),
                p_corr=ac["p_corr"], p_PI=ac["p_PI"],
                p_sep_A=ac["p_sep_A"], p_sep_B=ac["p_sep_B"],
                deviance=envelope_deviance(model, X, Xt))


def _python_mle(X, Xt):
    out, secs = [], []
    for c, t in zip(X, Xt):
        t0 = time.time()
        try:
            # fit_selected returns {"model", "params", "nll", "loglik", "k"};
            # "params" is the packed 12-vector, in GRIN's canonical order.
            fit = fit_selected(c.reshape(4, 4), t, criterion="aic")
            th = np.asarray(fit["params"], float)
        except Exception as e:
            print(f"  (python MLE failed: {type(e).__name__}: {e})")
            th = np.full(12, np.nan)
        secs.append(1e3 * (time.time() - t0)); out.append(th)
    return np.asarray(out), np.asarray(secs)


def _r_params(rf, name, pkg):
    """Pull one method's 12-vector out of the R fit table; NaN if absent."""
    if rf is None or name not in rf.index:
        return np.full(12, np.nan)
    row = rf.loc[name]
    cols = ([f"{pkg}_zx_{i}" for i in range(4)] + [f"{pkg}_zy_{i}" for i in range(4)]
            + [f"{pkg}_rho_{i}" for i in range(4)])
    if not all(c in rf.columns for c in cols):
        return np.full(12, np.nan)
    return row[cols].to_numpy(float)


def _sep_from_params(z, dim, tol=1e-3):
    """Is this dimension's marginal sensitivity tied across levels of the OTHER one?

    Canonical stimulus order is A1B1, A1B2, A2B1, A2B2, so B varies fastest. The two
    dimensions therefore pair up differently, and this is easy to get wrong:

      separability on A: z_x must not depend on the level of B
                         -> zx[A1B1] == zx[A1B2]  and  zx[A2B1] == zx[A2B2]
                         -> indices (0,1) and (2,3)
      separability on B: z_y must not depend on the level of A
                         -> zy[A1B1] == zy[A2B1]  and  zy[A1B2] == zy[A2B2]
                         -> indices (0,2) and (1,3)

    Using the A pairing for B silently reports every separable-on-B fit as
    non-separable, which is exactly what it did before this was fixed -- caught because
    mdsdt's own model label said PS(B) where this function said otherwise.

    Read from the fitted values rather than each package's label, so all four methods
    are classified by one rule instead of three naming schemes.
    """
    z = np.asarray(z, float)
    if np.all(np.isnan(z)):
        return None
    pairs = ((0, 1), (2, 3)) if dim == "A" else ((0, 2), (1, 3))
    return bool(all(abs(z[i] - z[j]) < tol for i, j in pairs))


def _corr_class_from_params(rho, tol=1e-6):
    """0 = independence, 1 = one shared correlation, 2 = free. Derived from the fitted
    values rather than a package's own label, so the three methods are classified by one
    rule instead of three different naming schemes."""
    rho = np.asarray(rho, float)
    if np.all(np.isnan(rho)):
        return -1
    if np.allclose(rho, 0.0, atol=1e-4):
        return 0
    return 1 if np.nanstd(rho) < tol * 10 or np.ptp(rho) < 1e-3 else 2


def main():
    names, X, Xt = _load_matrices()
    model = load_model(MODEL_FILE)

    t0 = time.time()
    g = _grin(model, X, Xt)
    grin_ms = 1e3 * (time.time() - t0) / len(X)

    mle_theta, mle_ms = _python_mle(X, Xt)

    rf = None
    if os.path.exists(RFIT):
        rf = pd.read_csv(RFIT)
        rf = rf[rf["rep"] == 0].set_index("dataset") if "rep" in rf.columns \
            else rf.set_index("dataset")
    else:
        print(f"(!) {RFIT} not found — run scripts/R/fit_real_data.R first")

    methods = {}
    for pkg in ("mdsdt", "grtools"):
        methods[pkg] = np.vstack([_r_params(rf, n, pkg) for n in names])
    methods["python_mle"] = mle_theta
    methods["grin"] = g["mean"]

    # ---------------- report ------------------------------------------------
    rows = []
    print(f"\nGRIN: {len(X)} real observers, {grin_ms:.3f} ms each "
          f"(python MLE {np.nanmedian(mle_ms):.0f} ms each)\n")
    for i, name in enumerate(names):
        gc = int(g["p_corr"][i].argmax())
        r = dict(dataset=name, n_trials=int(Xt[i].sum()),
                 grin_corr_class=gc, grin_p_PI=float(g["p_PI"][i]),
                 grin_p_sep_A=float(g["p_sep_A"][i]), grin_p_sep_B=float(g["p_sep_B"][i]),
                 grin_deviance=float(g["deviance"][i]))
        for pkg in ("mdsdt", "grtools", "python_mle"):
            r[f"{pkg}_corr_class"] = _corr_class_from_params(methods[pkg][i, 8:12])
            r[f"{pkg}_sep_A"] = _sep_from_params(methods[pkg][i, 0:4], "A")
            r[f"{pkg}_sep_B"] = _sep_from_params(methods[pkg][i, 4:8], "B")
        r["grin_sep_A"] = bool(g["p_sep_A"][i] > 0.5)
        r["grin_sep_B"] = bool(g["p_sep_B"][i] > 0.5)
        if rf is not None and name in rf.index:
            r["mdsdt_label"] = str(rf.loc[name].get("best_model", ""))
            r["grtools_ok"] = bool(rf.loc[name].get("grtools_ok", False))
        rows.append(r)

        print(f"--- {name}  (n = {r['n_trials']}) ---")
        print(f"    GRIN      corr class {CORR_LABELS[gc]:16s} "
              f"P(PI) {r['grin_p_PI']:.2f}   sep A {r['grin_p_sep_A']:.2f} "
              f"B {r['grin_p_sep_B']:.2f}")
        def _sepstr(a, b):
            f = lambda v: "-" if v is None else ("yes" if v else "no")
            return f"sep A {f(a):>3s} B {f(b):>3s}"
        print(f"    {'GRIN':10s} {'':16s}   {_sepstr(r['grin_sep_A'], r['grin_sep_B'])}")
        for pkg in ("mdsdt", "grtools", "python_mle"):
            c = r[f"{pkg}_corr_class"]
            lab = CORR_LABELS[c] if c >= 0 else "no fit"
            mark = "=" if c == gc else ("x" if c >= 0 else " ")
            sa, sb = r[f"{pkg}_sep_A"], r[f"{pkg}_sep_B"]
            smark = "=" if (sa == r["grin_sep_A"] and sb == r["grin_sep_B"]) else "x"
            print(f"    {pkg:10s} corr class {lab:16s} [{mark}]   "
                  f"{_sepstr(sa, sb)} [{smark}]")
        print(f"    training-support deviance {r['grin_deviance']:.1f}\n")

    print()
    for pkg in ("mdsdt", "grtools", "python_mle"):
        cagree = sum(r[f"{pkg}_corr_class"] == r["grin_corr_class"] for r in rows)
        sagree = sum(r[f"{pkg}_sep_A"] == r["grin_sep_A"]
                     and r[f"{pkg}_sep_B"] == r["grin_sep_B"] for r in rows)
        print(f"GRIN vs {pkg:<11}: correlation {cagree}/{len(rows)}   "
              f"separability (both dims) {sagree}/{len(rows)}")
    mg_c = sum(r["mdsdt_corr_class"] == r["grtools_corr_class"] for r in rows)
    mg_s = sum(r["mdsdt_sep_A"] == r["grtools_sep_A"]
               and r["mdsdt_sep_B"] == r["grtools_sep_B"] for r in rows)
    print(f"mdsdt vs grtools    : correlation {mg_c}/{len(rows)}   "
          f"separability (both dims) {mg_s}/{len(rows)}")
    print()

    # how many of the other methods' point estimates land inside GRIN's 95% interval
    inside = {}
    for pkg in ("mdsdt", "grtools", "python_mle"):
        th = methods[pkg]
        ok = (th >= g["lo"]) & (th <= g["hi"]) & ~np.isnan(th)
        valid = ~np.isnan(th)
        inside[pkg] = dict(
            all=float(ok.sum() / max(valid.sum(), 1)),
            z=float(ok[:, :8].sum() / max(valid[:, :8].sum(), 1)),
            rho=float(ok[:, 8:].sum() / max(valid[:, 8:].sum(), 1)))
        print(f"{pkg:10s} point estimates inside GRIN's 95% CI: "
              f"{100*inside[pkg]['all']:.0f}% (z {100*inside[pkg]['z']:.0f}%, "
              f"rho {100*inside[pkg]['rho']:.0f}%)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(per_observer=rows, inside_grin_ci=inside,
                       grin_ms=grin_ms, python_mle_ms=float(np.nanmedian(mle_ms)),
                       methods={k: v.tolist() for k, v in methods.items()},
                       grin_lo=g["lo"].tolist(), grin_hi=g["hi"].tolist(),
                       names=names), f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    # ---- GRIN on the SAME thinned matrices the R baselines were refitted to -------
    # The thinning comparison asks how far each method drifts from its own full-data
    # estimate as trials are removed. That is only a like-for-like comparison if every
    # method sees the identical resampled matrices, so fit_real_data.R writes them out
    # and GRIN is run on those rather than on an independent draw of its own.
    sub = None
    if os.path.exists(RSUB):
        sub = pd.read_csv(RSUB)
        if all(c in sub.columns for c in CM_COLS):
            thin = sub[CM_COLS].to_numpy(float)
            keep = np.isfinite(thin).all(1) & (thin.sum(1) > 0)
            gp = np.full((len(sub), 12), np.nan)
            if keep.any():
                tt = thin[keep].reshape(-1, 4, 4).sum(2)
                gp[keep] = predict_posterior(model, thin[keep], tt,
                                             n_samples=400)["mean"].numpy()
            for j, nm_ in enumerate(PNAMES):
                sub[f"grin_{nm_}"] = gp[:, j]
            sub["grin_ok"] = np.isfinite(gp[:, 0])

            # fit_real_data.R covers the two R packages; the Python baseline has to be
            # fitted here, on the same matrices, or it would be absent from the
            # thinning comparison for no reason other than which language it lives in
            mp = np.full((len(sub), 12), np.nan)
            for i in np.where(keep)[0]:
                try:
                    mp[i] = np.asarray(
                        fit_selected(thin[i].reshape(4, 4), tt[list(np.where(keep)[0]).index(i)],
                                     criterion="aic")["params"], float)
                except Exception:
                    pass
            for j, nm_ in enumerate(PNAMES):
                sub[f"python_mle_{nm_}"] = mp[:, j]
            sub["python_mle_ok"] = np.isfinite(mp[:, 0])
            print(f"added Python MLE fits for {int(np.isfinite(mp[:,0]).sum())} thinned matrices")
            sub.to_csv(RSUB, index=False)
            print(f"added GRIN fits for {int(keep.sum())} thinned matrices -> {RSUB}")
        else:
            print("(!) subsample file has no matrix columns; re-run fit_real_data.R")

    from src.viz.real_data import real_data_figures
    real_data_figures(names, X, Xt, g, methods, FIGURES_DIR,
                      subsample_path=RSUB if os.path.exists(RSUB) else None,
                      model=model)


if __name__ == "__main__":
    main()
