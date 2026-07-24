"""
model_posterior.py — calibrated probabilities for the GRT structural questions,
reported honestly (including where the evidence is thin).

Instead of forcing a single 12-way label, we return, for each construct, the
posterior probability that it holds, plus an "insufficient evidence" flag when the
answer is genuinely undetermined by the data:

    perceptual independence (PI)   — is there NO within-stimulus correlation?
    separability of dimension A    — is x-sensitivity invariant across B?
    separability of dimension B     — is y-sensitivity invariant across A?

Honest caveat: PI / correlation questions are information-limited from a single
confusion matrix — weak correlations are genuinely unrecoverable at realistic trial
counts (see the recoverability analysis). These probabilities REPORT that limit
(they widen toward 0.5 and flag low evidence); they do not remove it. Stronger
identifiability needs more information: multiple decision-bound conditions, response
times, or group-level pooling.
"""
import numpy as np
import torch
from scipy.stats import chi2

try:
    from . import _ood_dev
except Exception:
    pass

from .predict import predict_posterior


def _deviance_1(counts4, zx, zy, rho):
    try:
        from .. import grt_model as _gm
    except ImportError:
        import grt_model as _gm
    p = np.clip(_gm.forward_probabilities(zx[None], zy[None], rho[None])[0], 1e-12, 1.0)
    T = counts4.sum(1, keepdims=True)
    phat = np.clip(counts4 / np.maximum(T, 1), 1e-12, 1.0)
    return 2.0 * np.sum(counts4 * (np.log(phat) - np.log(p)))


def _pi_probability(counts4, zx, zy, rho):
    """LR test: deviance with rho=0 minus deviance with rho free (>=0). Large => PI violated."""
    dev_free = _deviance_1(counts4, zx, zy, rho)
    dev_pi = _deviance_1(counts4, zx, zy, np.zeros(4))
    lr = max(0.0, dev_pi - dev_free)
    return float(chi2.sf(lr, df=4))          # P(data consistent with PI)


def _p_ci_excludes_zero(samples):
    """P(quantity != 0) summarised as how far the posterior mass sits from zero."""
    below = (samples < 0).mean(0)
    return 2 * np.maximum(below, 1 - below) - 1          # 0 (mass on 0) .. 1 (all one side)


def model_posterior(model, counts, trials, n_samples=1000, evidence_tol=0.5):
    """
    counts (N,16), trials (N,4). Returns a list of dicts, one per matrix:
      {p_PI, p_sep_A, p_sep_B, rho_est, rho_ci, evidence_PI, ...}
    where each p_* is the probability the construct HOLDS, and evidence_* is False
    when the posterior is too diffuse to decide (flag for "insufficient evidence").
    """
    counts = np.asarray(counts); trials = np.asarray(trials)
    post = predict_posterior(model, counts, trials, n_samples=n_samples)
    S = post["samples"].numpy()                          # (n_samples, N, 12)
    out = []
    for i in range(S.shape[1]):
        zx, zy, rho = S[:, i, 0:4], S[:, i, 4:8], S[:, i, 8:12]
        # PI via likelihood-ratio: does forcing all correlations to 0 hurt the fit?
        # (pools evidence across the whole matrix -> far more powerful than a marginal CI)
        mean_zx, mean_zy, mean_rho = zx.mean(0), zy.mean(0), rho.mean(0)
        p_pi = _pi_probability(counts[i].reshape(4, 4).astype(float), mean_zx, mean_zy, mean_rho)
        # separability: the tied differences are ~0
        dA = np.concatenate([(zx[:, 0] - zx[:, 1])[:, None], (zx[:, 2] - zx[:, 3])[:, None]], 1)
        dB = np.concatenate([(zy[:, 0] - zy[:, 2])[:, None], (zy[:, 1] - zy[:, 3])[:, None]], 1)
        p_sepA = 1 - _p_ci_excludes_zero(dA).max()
        p_sepB = 1 - _p_ci_excludes_zero(dB).max()
        rho_mean = rho.mean(0)
        k = int(np.argmax(np.abs(rho_mean)))
        out.append({
            "p_PI": float(p_pi), "p_sep_A": float(p_sepA), "p_sep_B": float(p_sepB),
            "rho_est": float(rho_mean[k]),
            "rho_ci": [float(np.quantile(rho[:, k], .05)), float(np.quantile(rho[:, k], .95))],
            # evidence is "sufficient" only when the probability is decisive either way
            "evidence_PI": bool(abs(p_pi - 0.5) > (0.5 - evidence_tol / 2)),
            "evidence_sep_A": bool(abs(p_sepA - 0.5) > (0.5 - evidence_tol / 2)),
            "evidence_sep_B": bool(abs(p_sepB - 0.5) > (0.5 - evidence_tol / 2)),
        })
    return out


_CORR_IDX = {"pi": 0, "rho1": 1, "free": 2}


def construct_labels(class_names):
    """Map GRT class names -> (corr_idx[0..2], sepA{0,1}, sepB{0,1}) for training/eval."""
    try:
        from .. import grt_model as _gm
    except ImportError:
        import grt_model as _gm
    corr, sa, sb = [], [], []
    for name in class_names:
        c, px, py = _gm.MODEL_SPECS[name]
        corr.append(_CORR_IDX[c]); sa.append(int(px)); sb.append(int(py))
    return np.array(corr), np.array(sa), np.array(sb)


# free correlation parameters per correlation class (PI, RHO1, free) -> parsimony penalty
_CORR_COMPLEXITY = np.array([0.0, 1.0, 4.0])


def amortized_compare(model, counts, trials, parsimony=0.0):
    """
    One forward pass -> {p_corr (N,3): PI/RHO1/free, p_sep_A (N,), p_sep_B (N,)}.

    `parsimony` (>=0) adds a complexity log-prior to the correlation logits before the
    softmax: -parsimony * (0, 1, 4) for (PI, RHO1, free). This is the Occam prior AIC/BIC
    have built in and the uniform-label head lacks, so the head correctly defaults toward
    independence unless the evidence overcomes the penalty. 0 = pure likelihood.
    """
    import torch
    from ..models.network import featurize
    counts = np.asarray(counts); trials = np.asarray(trials)
    device = next(model.parameters()).device
    with torch.no_grad():
        x = featurize(torch.as_tensor(counts), torch.as_tensor(trials)).to(device)
        cl, al, bl = model.compare_logits(x)
        cl = cl.cpu().numpy() - parsimony * _CORR_COMPLEXITY[None, :]
        cl = cl - cl.max(1, keepdims=True)
        p_corr = np.exp(cl); p_corr /= p_corr.sum(1, keepdims=True)
        p_sepA = torch.softmax(al, -1)[:, 1].cpu().numpy()
        p_sepB = torch.softmax(bl, -1)[:, 1].cpu().numpy()
    return {"p_corr": p_corr, "p_PI": p_corr[:, 0], "p_sep_A": p_sepA, "p_sep_B": p_sepB}
