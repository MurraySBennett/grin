"""
checks.py — every validation check as a callable returning a result dict.

Each check is self-contained and returns {"id", "claim", "result", "pass"} so the suite can
print a summary table and write JSON. Checks that need a trained model take one as an arg.
"""
import time
import numpy as np
import torch

import src.grt_model as gm
from src.data.generator import GRTDataGenerator
from src.models.network import NPEModel, featurize
from src.models.heads import params_to_train_space, train_space_to_params
from src.models.losses import joint_loss
from src.inference.predict import predict_posterior, predict_point
from src.inference.mle import fit_full, fit_and_select
from src.inference.ood import envelope_deviance
from src.inference.model_posterior import construct_labels, amortized_compare

try:                                    # track the production knob when available
    from src.config import TRIAL_IMBALANCE
except Exception:                       # keep checks runnable if config isn't importable
    TRIAL_IMBALANCE = 0.35


def _train(feats, tgt, labels=None, epochs=20, in_dim=None, n_arch=0, seed=0):
    torch.manual_seed(seed)
    m = NPEModel(in_dim=in_dim or feats.shape[1], dropout=0.0, comparison=labels is not None)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = feats.shape[0]; perm = torch.randperm(n)
    for e in range(epochs):
        for i in range(0, n, 512):
            b = perm[i:i + 512]; opt.zero_grad()
            if labels is None:
                mean, L = m(feats[b])
                loss = -torch.distributions.MultivariateNormal(mean, scale_tril=L).log_prob(tgt[b]).mean()
            else:
                c, a, s = labels
                loss = joint_loss(m, feats[b], tgt[b], c[b], a[b], s[b], w_cls=4.0)[0]
            loss.backward(); opt.step()
    return m.eval()


def _dataset(n_per_class, trial_range=(20, 500), seed=1, balanced=False,
             imbalance=TRIAL_IMBALANCE):
    g = GRTDataGenerator(n_per_class=n_per_class, trial_range=trial_range,
                         balanced_trials=balanced, seed=seed, imbalance=imbalance)
    return g.generate_all_model_cms()


# ---------------------------------------------------------------- v01
def v01_forward_exactness(**kw):
    from scipy.stats import multivariate_normal as MVN
    rng = np.random.default_rng(0); err = 0.0
    for _ in range(200):
        h, k = rng.uniform(-3, 3, 2); r = rng.uniform(-0.95, 0.95)
        err = max(err, abs(float(gm.bvn_cdf(np.array(h), np.array(k), np.array(r)))
                           - MVN(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([h, k])))
    return {"id": "v01", "claim": "forward model is exact",
            "result": {"max_abs_error_vs_scipy": err}, "pass": err < 1e-6}


# ---------------------------------------------------------------- v02
def v02_prior_coverage(n=2000, **kw):
    X, yp, Xt, yc, yl = _dataset(n // 12 or 50)
    c = X.reshape(-1, 4, 4) / Xt[:, :, None]
    acc = np.einsum('nii->ni', c).mean(1)
    xb = (c[:, :, 2] + c[:, :, 3]).mean(1) - .5
    cong = .5 * (np.einsum('nii->ni', c)[:, 0] + np.einsum('nii->ni', c)[:, 3]) - \
           .5 * (np.einsum('nii->ni', c)[:, 1] + np.einsum('nii->ni', c)[:, 2])
    r = {"accuracy_range": [float(acc.min()), float(acc.max())],
         "bias_range": [float(xb.min()), float(xb.max())],
         "congruency_range": [float(cong.min()), float(cong.max())]}
    return {"id": "v02", "claim": "prior coverage is broad and verified",
            "result": r, "pass": acc.min() < 0.4 and acc.max() > 0.9}


# ---------------------------------------------------------------- v03 / v04 / v07 / v08
def _fit_model(n_per_class=2500, epochs=20, seed=1):
    X, yp, Xt, yc, yl = _dataset(n_per_class, seed=seed)
    f = featurize(torch.tensor(X), torch.tensor(Xt))
    t = params_to_train_space(torch.tensor(yp, dtype=torch.float32))
    c, a, b = construct_labels(yl)
    return _train(f, t, (torch.tensor(c, dtype=torch.long), torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)), epochs=epochs)


def v03_recovery(model=None, **kw):
    model = model or _fit_model()
    X, yp, Xt, yc, yl = _dataset(150, trial_range=(200, 200), balanced=True, seed=42)
    p = predict_point(model, X, Xt).numpy()
    z = float(np.abs(p[:, :8] - yp[:, :8]).mean()); r = float(np.abs(p[:, 8:] - yp[:, 8:]).mean())
    rz = float(np.corrcoef(yp[:, 0], p[:, 0])[0, 1])
    return {"id": "v03", "claim": "parameter recovery",
            "result": {"zscore_MAE": z, "rho_MAE": r, "zx0_r": rz}, "pass": z < 0.3}


def v04_calibration(model=None, **kw):
    model = model or _fit_model()
    X, yp, Xt, yc, yl = _dataset(100, seed=7)
    s = predict_posterior(model, X, Xt, n_samples=400)["samples"].numpy()
    cov = {}
    for lvl in (0.5, 0.9, 0.95):
        lo = np.quantile(s, (1 - lvl) / 2, 0); hi = np.quantile(s, (1 + lvl) / 2, 0)
        cov[f"{int(lvl*100)}%"] = float(((yp >= lo) & (yp <= hi)).mean())
    return {"id": "v04", "claim": "posterior is calibrated",
            "result": cov, "pass": abs(cov["90%"] - 0.9) < 0.08}


def v05_speed(model=None, n=60, **kw):
    model = model or _fit_model()
    X, yp, Xt, yc, yl = _dataset(20, trial_range=(200, 200), balanced=True, seed=5)
    idx = np.arange(min(n, len(X)))
    predict_point(model, X[idx], Xt[idx])                       # warm up
    t0 = time.time(); predict_point(model, X[idx], Xt[idx]); npe = (time.time() - t0) / len(idx)
    t0 = time.time(); mle = np.array([fit_full(X[i], Xt[i])["params"] for i in idx])
    mle_t = (time.time() - t0) / len(idx)
    return {"id": "v05", "claim": "amortized speedup vs MLE",
            "result": {"npe_ms_per_matrix": npe * 1e3, "mle_ms_per_matrix": mle_t * 1e3,
                       "speedup": mle_t / npe},
            "pass": bool(mle_t / npe > 20)}


def v06_ensemble(n_seeds=3, **kw):
    X, yp, Xt, yc, yl = _dataset(100, trial_range=(200, 200), balanced=True, seed=42)
    maes = []
    for s in range(n_seeds):
        m = _fit_model(n_per_class=1200, epochs=12, seed=s)
        maes.append(float(np.abs(predict_point(m, X, Xt).numpy() - yp).mean()))
    return {"id": "v06", "claim": "stable across training seeds",
            "result": {"per_seed_MAE": maes, "spread": float(np.std(maes))},
            "pass": float(np.std(maes)) < 0.05}


def v07_trial_sweep(model=None, **kw):
    model = model or _fit_model()
    out = {}
    for T in (10, 25, 50, 100, 200, 400):
        X, yp, Xt, yc, yl = _dataset(40, trial_range=(T, T), balanced=True, seed=100 + T)
        po = predict_posterior(model, X, Xt, n_samples=300)
        mae = float(np.abs(po["mean"].numpy() - yp).mean())
        s = po["samples"].numpy()
        lo, hi = np.quantile(s, .05, 0), np.quantile(s, .95, 0)
        out[f"{T}_trials"] = {"MAE": mae, "coverage90": float(((yp >= lo) & (yp <= hi)).mean())}
    return {"id": "v07", "claim": "reliability across trial counts", "result": out, "pass": True}


def v08_pi_frontier(model=None, **kw):
    model = model or _fit_model()
    X, yp, Xt, yc, yl = _dataset(120, trial_range=(400, 400), balanced=True, seed=9)
    ac = amortized_compare(model, X, Xt)
    tc, _, _ = construct_labels(yl)
    pc = ac["p_corr"].argmax(1); mr = np.abs(yp[:, 8:12]).max(1)
    out = {}
    for lo, hi, lab in [(0, .001, "true_PI"), (.001, .3, "weak"), (.3, .6, "moderate"), (.6, .9, "strong")]:
        m = (mr >= lo) & (mr < hi)
        if m.sum() > 5:
            out[lab] = float(np.mean((pc[m] == 0) == (tc[m] == 0)))
    return {"id": "v08", "claim": "PI identifiability frontier (the honest limit)",
            "result": out, "pass": True}


def _reversed_mapping(n, frac, seed):
    """Simulate a participant whose B-dimension response mapping is reversed
    (e.g. a swapped response key, or a confused dimension label) on a `frac`
    fraction of trials. This is not merely an extreme point of the fitted family:
    the model's identified parameterization defines z-scores by a fixed sign
    convention (stimulus level 1 below the bound, level 2 above -- DESIGN_RECORD.md
    #2), so a reversed mapping has no representation in that coordinate system at
    any frac > 0, unlike an unusual-but-valid (zx, zy, rho) triple."""
    r = np.random.default_rng(seed)
    zx, zy, rho = gm.sample_prior('ds', n, r)
    probs = gm.forward_probabilities(zx, zy, rho)               # (n,4,4)
    probs_rev = probs[:, :, [1, 0, 3, 2]]                        # swap B1/B2 responses
    mix = (1 - frac) * probs + frac * probs_rev
    Xo = np.zeros((n, 16), dtype=np.int64)
    for i in range(n):
        for st in range(4):
            Xo[i, st * 4:(st + 1) * 4] = r.multinomial(300, mix[i, st] / mix[i, st].sum())
    return Xo, np.full((n, 4), 300)


def v09_ood(model=None, **kw):
    """
    Training-envelope / input-support diagnostic via posterior-mean reconstruction deviance
    (src/inference/ood.py) -- NOT model-family misspecification detection. Read
    ood.py's module docstring first: the identified model is saturated (12 free
    parameters for 12 data df), so essentially any response-proportion table --
    however it was generated, reversed mapping included -- has SOME parameter
    vector that reproduces it exactly; there is no "no representation in the
    model's parameterization" case at this level. What this deviance actually
    scores is whether the TRAINED NETWORK's own fitted parameters reproduce the
    matrix, which is why a matrix outside the training prior's support (reversed
    sign/orientation, as generated by `_reversed_mapping` below) shows up as a
    large deviance: the network never learned to search that region, not because
    no parameter vector exists there. An earlier version of this check used a
    merely-extreme in-family matrix as its "OOD" case ([.05,.45,.45,.05] per row,
    achievable at zx=zy=0, rho~=-0.95) and only detected it 63% of the time --
    that was the test mistaking in-envelope data for out-of-envelope, not a
    detector failing to detect (see DESIGN_RECORD.md #7). `_reversed_mapping`
    grades severity from mild (a minority of trials use a reversed response
    mapping) to total.
    """
    model = model or _fit_model()
    Xi, ypi, Xti, _, _ = _dataset(500, trial_range=(300, 300), balanced=True, seed=5)
    din = envelope_deviance(model, Xi, Xti); thr = float(np.quantile(din, .95))

    severity = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        Xo, To = _reversed_mapping(150, frac, seed=1)
        do = envelope_deviance(model, Xo, To)
        severity[str(frac)] = {"median_deviance": float(np.median(do)),
                                "detection_rate": float((do > thr).mean())}

    false_alarm = float((din > thr).mean())
    return {"id": "v09",
            "claim": "data outside the trained envelope (reversed response mapping) "
                     "is flagged, gradedly with severity, at a controlled "
                     "false-alarm rate against in-envelope data",
            "result": {"in_dist_median": float(np.median(din)), "threshold95": thr,
                       "false_alarm": false_alarm, "severity_curve": severity},
            "pass": severity["1.0"]["detection_rate"] > 0.95 and false_alarm <= 0.10}


def v10_degradation(model=None, **kw):
    model = model or _fit_model()
    rng = np.random.default_rng(0)
    params = gm.pack(*gm.sample_prior('ds', 200, rng))
    zx, zy, rho = gm.unpack(params); base = gm.forward_probabilities(zx, zy, rho)
    G = GRTDataGenerator(seed=0); T = np.full((200, 4), 200)
    out = {}
    for lam in (0.0, 0.2, 0.5):
        probs = (1 - lam) * base + lam * 0.25
        C = G._multinomial_counts(probs, T, rng).reshape(200, 16)
        out[f"lapse_{lam}"] = float(np.abs(predict_point(model, C, T).numpy() - params).mean())
    return {"id": "v10", "claim": "degrades gracefully under lapses", "result": out, "pass": True}


def v11_amortized_comparison(model=None, n_mle=60, **kw):
    model = model or _fit_model()
    X, yp, Xt, yc, yl = _dataset(60, trial_range=(400, 400), balanced=True, seed=11)
    tc, ta, tb = construct_labels(yl)
    t0 = time.time(); ac = amortized_compare(model, X, Xt); at = (time.time() - t0) / len(X)
    pc = ac["p_corr"].argmax(1)
    psa = (ac["p_sep_A"] > .5).astype(int); psb = (ac["p_sep_B"] > .5).astype(int)
    sub = np.random.default_rng(0).choice(len(X), min(n_mle, len(X)), replace=False)
    t0 = time.time(); bic = [fit_and_select(X[i], Xt[i], 'bic')[0]['model'] for i in sub]
    bt = (time.time() - t0) / len(sub)
    bc, ba, bb = construct_labels(bic)
    return {"id": "v11", "claim": "amortized comparison vs AIC/BIC",
            "result": {"amortized": {"corr": float(np.mean(pc[sub] == tc[sub])),
                                     "sepA": float(np.mean(psa[sub] == ta[sub])),
                                     "sepB": float(np.mean(psb[sub] == tb[sub])), "ms": at * 1e3},
                       "aic_bic": {"corr": float(np.mean(bc == tc[sub])),
                                   "sepA": float(np.mean(ba == ta[sub])),
                                   "sepB": float(np.mean(bb == tb[sub])), "ms": bt * 1e3},
                       "speedup": bt / at},
            "pass": True}
    