"""
Compare GRIN against mdsdt on REAL confusion matrices (no ground truth).

    Rscript scripts/R/fit_real_data.R      # 1. extract + fit the real matrices
    python scripts/compare_real_data.py    # 2. run GRIN on the SAME matrices  <- here

Because there is no ground truth on real data, the check is **agreement with the published
gold standard**: does GRIN reach the same conclusions (perceptual independence, separability)
as mdsdt's AIC model selection — and does it do so in microseconds rather than seconds?

Also reports GRIN's OOD/goodness-of-fit score for each real observer: a high value means the
matrix has structure the GRT-Gaussian family cannot produce, so the estimates should be
treated with caution.

CONSISTENCY WITH compare_to_r.py (these used to differ, which made the two scripts'
"agreement" numbers non-comparable):
  * model selection uses argmax over the 3-way correlation head (the analogue of "AIC picks
    the best model"), NOT p_PI > 0.5 — with three classes those are different rules and can
    disagree.
  * the head is evaluated at AIC-matched parsimony, because mdsdt selects by AIC and the
    head is otherwise a pure-likelihood classifier. Both settings are printed.
  * R model labels are parsed into the (correlation class, sep-A, sep-B) triple, so this
    works whether fit_real_data.R emits short labels ('pi', 'ps', 'full') or the 12-node
    hierarchy labels used by fit_baselines.R.

Writes results/figures/real_data_comparison.png (built here rather than in src/viz/figures.py,
which has no real-data figure).
"""
import os
import time
import numpy as np
import pandas as pd

from src.config import REAL_DATA_DIR, MLE_FITS_DIR, FIGURES_DIR
from src.api import load_model
from src.inference.predict import predict_posterior
from src.inference.model_posterior import amortized_compare
from src.inference.ood import ood_deviance
import src.grt_model as gm

MATS = os.path.join(REAL_DATA_DIR, "real_matrices.csv")
RFIT = os.path.join(MLE_FITS_DIR, "real_data_fits.csv")

AIC_PARSIMONY = 1.0        # see compare_to_r.py for the derivation
OOD_THRESHOLD = 40.0


def _parse_r_model(label):
    """
    Parse either naming scheme into (corr_idx, sepA, sepB); None if unparseable.
      fit_baselines.R style : '{PI, PS(A), DS}', 'GRT-{1_RHO, PS, DS}'
      fit_real_data.R style : 'pi', 'ps', 'full', 'pi_ps', 'rho1'
    """
    if not isinstance(label, str):
        return None
    s = label.upper().replace("GRT-", "").strip()
    if not s or s.startswith("ERROR") or s == "NAN":
        return None
    if "{" in s or "," in s:
        toks = [t.strip() for t in s.strip("{}").split(",")]
        corr = 0 if "PI" in toks else (1 if "1_RHO" in toks else 2)
        if "PS" in toks:
            return corr, True, True
        return corr, "PS(A)" in toks, "PS(B)" in toks
    # short labels: underscore-separated flags
    toks = s.split("_")
    corr = 0 if "PI" in toks else (1 if ("RHO1" in toks or "1" in toks) else 2)
    sep = "PS" in toks
    return corr, sep, sep


def main():
    if not os.path.exists(MATS):
        raise SystemExit(f"{MATS} not found — run: Rscript scripts/R/fit_real_data.R")
    df = pd.read_csv(MATS)
    cm_cols = [f"cm_{s}{r}" for s in range(4) for r in range(4)]
    missing = [c for c in cm_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{MATS} is missing columns {missing[:4]}... — regenerate it")
    X = df[cm_cols].to_numpy().astype(float)
    Xt = X.reshape(-1, 4, 4).sum(2)                     # trials = row sums
    names = list(df["dataset"])

    model = load_model()
    t0 = time.time()
    post = predict_posterior(model, X, Xt, n_samples=2000)
    ac = amortized_compare(model, X, Xt, parsimony=AIC_PARSIMONY)
    ac0 = amortized_compare(model, X, Xt, parsimony=0.0)
    dev = ood_deviance(model, X, Xt)
    grin_ms = 1e3 * (time.time() - t0) / len(X)

    grin_corr = ac["p_corr"].argmax(1)                  # 0=PI, 1=RHO1, 2=free
    grin_pi = grin_corr == 0
    grin_sa, grin_sb = ac["p_sep_A"] > 0.5, ac["p_sep_B"] > 0.5

    rfits = None
    if os.path.exists(RFIT):
        rf = pd.read_csv(RFIT)
        if "dataset" in rf.columns:
            rfits = rf.set_index("dataset")
        else:
            print(f"(!) {RFIT} has no 'dataset' column — skipping baseline comparison")
    else:
        print(f"(no mdsdt fits at {RFIT} — run: Rscript scripts/R/fit_real_data.R)")

    model_col = None
    if rfits is not None:
        for c in ("best_model", "mdsdt_model", "model"):
            if c in rfits.columns:
                model_col = c
                break
        if model_col is None:
            print(f"(!) no model column in {RFIT} (looked for best_model/mdsdt_model/model)")

    print(f"\nGRIN: {len(X)} real observers in {grin_ms*len(X):.1f} ms "
          f"({grin_ms:.3f} ms each)\n")

    agree_flags = []          # True / False / None (None = no baseline fit available)
    for i, name in enumerate(names):
        m = post["mean"][i].numpy()
        sA = np.abs(m[0:4]).mean(); sB = np.abs(m[4:8]).mean()
        print(f"--- {name}  (n = {int(Xt[i].sum())} trials) ---")
        print(f"    sensitivity  A {sA:.2f}   B {sB:.2f}")
        print(f"    correlation  {m[8:12].mean():+.2f}  (mean across stimuli)")
        print(f"    corr class   {['PI','RHO1','free'][grin_corr[i]]}  "
              f"(P(PI) {ac['p_PI'][i]:.2f} AIC-matched / {ac0['p_PI'][i]:.2f} unpenalised)")
        print(f"    separability  A {ac['p_sep_A'][i]:.2f}  B {ac['p_sep_B'][i]:.2f}")
        flag = "ok" if dev[i] < OOD_THRESHOLD else "CHECK — may be outside the GRT-Gaussian family"
        print(f"    model fit (deviance) {dev[i]:.1f}  [{flag}]")

        agree = None
        if model_col is not None and name in rfits.index:
            raw = rfits.loc[name, model_col]
            parsed = _parse_r_model(raw)
            if parsed is not None:
                r_corr, r_sa, r_sb = parsed
                agree = bool(grin_pi[i] == (r_corr == 0))
                extra = ""
                # only report separability agreement if the R label actually encodes it
                if isinstance(raw, str) and ("{" in raw or "PS" in raw.upper()):
                    extra = (f" | sepA {'=' if grin_sa[i] == r_sa else 'x'}"
                             f" sepB {'=' if grin_sb[i] == r_sb else 'x'}")
                print(f"    mdsdt best model: {str(raw):20s} | "
                      f"GRIN {'PI' if grin_pi[i] else 'not PI'} -> "
                      f"{'AGREE' if agree else 'DISAGREE'}{extra}")
            else:
                print(f"    mdsdt best model: {str(raw)} (unparseable — no comparison)")
        agree_flags.append(agree)
        print()

    n_cmp = sum(a is not None for a in agree_flags)
    if n_cmp:
        n_ok = sum(bool(a) for a in agree_flags if a is not None)
        print(f"=== agreement with mdsdt on independence: {n_ok}/{n_cmp} observers ===")
    if (dev >= OOD_THRESHOLD).any():
        bad = [names[i] for i in np.where(dev >= OOD_THRESHOLD)[0]]
        print(f"=== {len(bad)} observer(s) flagged as possibly out-of-family: {', '.join(bad)} ===")

    # ---------------------------------------------------------------- figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz.style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, INK
    set_style()

    fig, ax = plt.subplots(1, 2, figsize=(13.5, max(4.2, 0.42 * len(names) + 1.6)))
    y = np.arange(len(names))
    # grey where there is no baseline to agree with, blue where GRIN and mdsdt agree,
    # rose where they disagree — so the eye goes straight to the disagreements.
    col = [MUTE if a is None else (BLUE_DEEP if a else RED_DEEP) for a in agree_flags]

    ax[0].barh(y, ac["p_PI"], color=col)
    ax[0].axvline(.5, color=INK, lw=1, ls=(0, (3, 3)))
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("P(perceptual independence), AIC-matched")
    ax[0].invert_yaxis()
    ax[0].set_title("Independence per observer\n(blue = agrees with mdsdt, rose = disagrees)",
                    fontsize=10)

    ax[1].barh(y, dev, color=[BLUE if d < OOD_THRESHOLD else RED_DEEP for d in dev])
    ax[1].axvline(OOD_THRESHOLD, color=INK, lw=1, ls=(0, (3, 3)))
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlabel("goodness-of-fit deviance")
    ax[1].set_title(f"Model fit (rose = above {OOD_THRESHOLD:.0f}, treat with caution)",
                    fontsize=10)

    fpath = os.path.join(FIGURES_DIR, "real_data_comparison.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout(); fig.savefig(fpath, dpi=150); plt.close(fig)
    print(f"figure -> {fpath}")


if __name__ == "__main__":
    main()