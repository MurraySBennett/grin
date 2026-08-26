"""
Recovery and construct classification as a function of OBSERVED per-dimension accuracy.

Everything else in this paper stratifies by trials per stimulus. Trial count is only half
of what determines whether a matrix is informative, and it is the half a researcher has
least control over: an experiment's budget is usually fixed, whereas stimulus separation
is chosen, piloted, and often staircased. Accuracy is the quantity practitioners actually
tune, so this script reports performance against it directly.

The frontier analysis argues from the Fisher information that the correlation is best
identified near chance and the marginal sensitivities near ceiling, leaving roughly
60-80% per dimension as the band serving both. That is an analytic claim about the design.
This measures whether the trained network's behaviour follows it.

Accuracy here is OBSERVED marginal accuracy per dimension, computed from the matrix
itself, not from the generating parameters -- because that is what a researcher can see
during a pilot block, and the whole point of reporting it this way is that it can be
acted on before the main experiment starts.

Writes results/validation/accuracy_stratified.json and
results/figures/accuracy_stratified.png.

    python scripts/accuracy_stratified.py [--n-per-class 2000] [--seed 999]
"""
import argparse, json, os
import numpy as np

from src.config import MODEL_FILE, FIGURES_DIR, Z_MAX, R_MAX
from src.data.generator import GRTDataGenerator
from src.inference.predict import predict_posterior
from src.inference.model_posterior import amortized_compare
from src.api import load_model
import src.grt_model as gm

OUT_JSON = os.path.join("results", "validation", "accuracy_stratified.json")
# Per-dimension accuracy has a FLOOR at 0.50, not 0.25: under the design's sign
# convention |z| >= 0, so an observer cannot be systematically below chance on a
# dimension without having the response mapping reversed. Overall 4AFC accuracy has
# chance at 0.25 and does reach the low values practitioners talk about; the two scales
# are reported side by side because the same design sits at very different-looking
# numbers on each (per-dimension 0.70 is roughly 0.58 overall).
ACC_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
OVERALL_EDGES = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
TPS_EDGES = [5, 20, 50, 200, 1000]
Z_SL, RHO_SL = slice(0, 8), slice(8, 12)


def marginal_accuracy(counts, trials):
    """Observed per-dimension accuracy, averaged over the two dimensions.

    Canonical stimulus/response order is A1B1, A1B2, A2B1, A2B2, so dimension A is
    correct when stimulus and response fall in the same half {0,1} or {2,3}, and
    dimension B when they share parity. This reads the matrix exactly as an
    experimenter would during a pilot block.
    """
    c = counts.reshape(-1, 4, 4)
    n = c.sum((1, 2))
    a_ok = np.zeros(len(c)); b_ok = np.zeros(len(c))
    for s in range(4):
        for r in range(4):
            if (s // 2) == (r // 2):
                a_ok += c[:, s, r]
            if (s % 2) == (r % 2):
                b_ok += c[:, s, r]
    return 0.5 * (a_ok / n + b_ok / n)


def true_accuracy(theta):
    """Per-dimension accuracy implied by the GENERATING parameters, free of sampling
    noise. Observed accuracy is what a practitioner can see, so it is the primary axis;
    but it is a noisy estimate of this at low trial counts, which pushes matrices into
    the tails of the observed distribution and leaves the extreme bins enriched with
    small-n observers. Stratifying on the true value separates the effect of accuracy
    from the effect of estimating it badly."""
    out = np.empty(len(theta))
    for i, th in enumerate(theta):
        P = gm.forward_probabilities(th[0:4], th[4:8], th[8:12])
        a = sum(P[s, r] for s in range(4) for r in range(4) if (s // 2) == (r // 2)) / 4
        b = sum(P[s, r] for s in range(4) for r in range(4) if (s % 2) == (r % 2)) / 4
        out[i] = 0.5 * (a + b)
    return out


def main(n_per_class=2000, seed=999, n_samples=400):
    model = load_model(MODEL_FILE)
    gen = GRTDataGenerator(n_per_class=n_per_class, z_max=Z_MAX, r_max=R_MAX, seed=seed)
    X, y, Xt, _, labels = gen.generate_all_model_cms()
    labels = np.asarray(labels)

    # The training prior draws z uniformly on (0, Z_MAX), which concentrates observed
    # accuracy well above the band the frontier analysis recommends -- so the recommended
    # window's LOWER half is barely populated by the standard test set. Augment with a
    # draw at reduced sensitivity purely to cover that region. These matrices are still
    # inside the trained envelope (smaller |z| is interior, not extrapolation); they are
    # simply drawn from a different region of it.
    for k, zc in enumerate((0.6, 1.0, 1.6, 2.2)):
        g2 = GRTDataGenerator(n_per_class=max(n_per_class // 2, 1), z_max=zc,
                              r_max=R_MAX, seed=seed + 1 + k)
        X2, y2, Xt2, _, l2 = g2.generate_all_model_cms()
        X = np.concatenate([X, X2]); y = np.concatenate([y, y2])
        Xt = np.concatenate([Xt, Xt2]); labels = np.concatenate([labels, l2])
    print(f"augmented to {len(X)} matrices across z_max in (0.6, 1.0, 1.6, 2.2, 3.0) "
          f"so every accuracy band is populated; the standard prior alone leaves the\n"
          f"low-accuracy region -- where most identification studies are run -- nearly empty")

    post = predict_posterior(model, X, Xt, n_samples=n_samples)
    mean = post["mean"].numpy()
    ac = amortized_compare(model, X, Xt, parsimony=0.0)

    acc = marginal_accuracy(X, Xt)
    acc_true = true_accuracy(y)
    cc = X.reshape(-1, 4, 4)
    overall = np.trace(cc, axis1=1, axis2=2) / cc.sum((1, 2))   # all-4AFC-correct
    tps = Xt.mean(1)
    specs = [gm.MODEL_SPECS[l] for l in labels]
    true_pi = np.array([s[0] == "pi" for s in specs])
    true_sa = np.array([bool(s[1]) for s in specs])
    true_sb = np.array([bool(s[2]) for s in specs])
    pred_pi = ac["p_PI"] > 0.5
    pred_sa = ac["p_sep_A"] > 0.5
    pred_sb = ac["p_sep_B"] > 0.5

    ae = np.abs(mean - y)
    mae_z = ae[:, Z_SL].mean(1)
    # The frontier analysis's claim about the sensitivities is about their COEFFICIENT OF
    # VARIATION -- error relative to the parameter's own size -- not absolute error. Near
    # ceiling the z-scores are large, so the same relative precision shows up as a larger
    # absolute error, and the two measures can point in opposite directions without
    # contradicting each other. Both are reported so the comparison is like for like.
    abs_z = np.abs(y[:, Z_SL]).mean(1)
    # only score rho where it is a free parameter; under PI the true value is exactly 0
    rho_free = ~true_pi
    mae_rho = np.where(rho_free, ae[:, RHO_SL].mean(1), np.nan)

    out = dict(meta=dict(n=int(len(X)), seed=seed, acc_edges=ACC_EDGES,
                         tps_edges=TPS_EDGES, n_posterior_samples=n_samples),
               by_accuracy=[], by_accuracy_x_trials=[])

    print(f"N = {len(X)}   observed accuracy {acc.min():.2f}-{acc.max():.2f}\n")
    print(f"{'per-dim band':>14} {'4AFC':>6} {'n':>6} {'MAE z':>7} {'rel z':>7} "
          f"{'MAE rho':>8} {'PI acc':>7} {'sepA acc':>9} {'sepB acc':>9}")
    for lo, hi in zip(ACC_EDGES[:-1], ACC_EDGES[1:]):
        m = (acc >= lo) & (acc < hi)
        if m.sum() < 100:
            continue
        mean_abs_z = float(np.nanmean(abs_z[m]))
        row = dict(lo=lo, hi=hi, n=int(m.sum()),
                   mean_overall_accuracy=float(np.nanmean(overall[m])),
                   median_tps=float(np.median(tps[m])),
                   mae_z=float(np.nanmean(mae_z[m])),
                   mean_abs_z=mean_abs_z,
                   rel_err_z=float(np.nanmean(mae_z[m]) / mean_abs_z),
                   mae_rho=float(np.nanmean(mae_rho[m])),
                   n_rho=int((m & rho_free).sum()),
                   acc_PI=float((pred_pi[m] == true_pi[m]).mean()),
                   acc_sepA=float((pred_sa[m] == true_sa[m]).mean()),
                   acc_sepB=float((pred_sb[m] == true_sb[m]).mean()))
        out["by_accuracy"].append(row)
        print(f"{lo:.2f}-{hi:.2f}".rjust(14),
              f"{row['mean_overall_accuracy']:>6.2f} "
              f"{row['n']:>6} {row['mae_z']:>7.3f} {row['rel_err_z']:>7.3f} "
              f"{row['mae_rho']:>8.3f} "
              f"{row['acc_PI']:>7.3f} {row['acc_sepA']:>9.3f} {row['acc_sepB']:>9.3f}")

    # Same stratification on TRUE accuracy. If the conclusions hold on both axes they
    # are about accuracy; if they hold only on the observed axis they are partly about
    # the noise in estimating it.
    out["by_true_accuracy"] = []
    print(f"\n{'TRUE band':>14} {'n':>6} {'med tps':>8} {'MAE rho':>8} "
          f"{'PI acc':>7} {'sepA acc':>9}")
    for lo, hi in zip(ACC_EDGES[:-1], ACC_EDGES[1:]):
        m = (acc_true >= lo) & (acc_true < hi)
        if m.sum() < 100:
            continue
        r = dict(lo=lo, hi=hi, n=int(m.sum()), median_tps=float(np.median(tps[m])),
                 mae_rho=float(np.nanmean(mae_rho[m])),
                 rel_err_z=float(np.nanmean(mae_z[m]) / np.nanmean(abs_z[m])),
                 acc_PI=float((pred_pi[m] == true_pi[m]).mean()),
                 acc_sepA=float((pred_sa[m] == true_sa[m]).mean()),
                 acc_sepB=float((pred_sb[m] == true_sb[m]).mean()))
        out["by_true_accuracy"].append(r)
        print(f"{lo:.2f}-{hi:.2f}".rjust(14),
              f"{r['n']:>6} {r['median_tps']:>8.1f} {r['mae_rho']:>8.3f} "
              f"{r['acc_PI']:>7.3f} {r['acc_sepA']:>9.3f}")

    # accuracy x trial count: does the informative accuracy band shift with data?
    for tlo, thi in zip(TPS_EDGES[:-1], TPS_EDGES[1:]):
        for lo, hi in zip(ACC_EDGES[:-1], ACC_EDGES[1:]):
            m = (acc >= lo) & (acc < hi) & (tps >= tlo) & (tps < thi)
            if m.sum() < 40:
                continue
            out["by_accuracy_x_trials"].append(dict(
                tps_lo=tlo, tps_hi=thi, acc_lo=lo, acc_hi=hi, n=int(m.sum()),
                mae_z=float(np.nanmean(mae_z[m])),
                mae_rho=float(np.nanmean(mae_rho[m])),
                acc_PI=float((pred_pi[m] == true_pi[m]).mean()),
                acc_sepA=float((pred_sa[m] == true_sa[m]).mean()),
                acc_sepB=float((pred_sb[m] == true_sb[m]).mean())))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_JSON}")

    best = min(out["by_accuracy"], key=lambda r: r["mae_rho"])
    print(f"lowest rho error in the {best['lo']:.2f}-{best['hi']:.2f} accuracy band "
          f"(MAE {best['mae_rho']:.3f})")
    bestpi = max(out["by_accuracy"], key=lambda r: r["acc_PI"])
    print(f"best independence classification in the {bestpi['lo']:.2f}-{bestpi['hi']:.2f} "
          f"band ({bestpi['acc_PI']:.3f})")

    from src.viz.accuracy_panel import accuracy_stratified_figure
    accuracy_stratified_figure(out, os.path.join(FIGURES_DIR, "accuracy_stratified.png"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--n-samples", type=int, default=400)
    main(**vars(ap.parse_args()))
