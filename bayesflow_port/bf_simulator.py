"""
bf_simulator.py — GRIN generative model as a BayesFlow-compatible simulator.

This is the SHARED generative model. It reuses grin_core.grt_model exactly (same
identified 12-parameter forward model, same class-specific prior, same log-uniform
trial-count regime), so the BayesFlow branch and the bespoke NPE branch are trained
and evaluated on identical draws. Nothing about the science changes here — only the
inference machinery downstream differs between frameworks.

A BayesFlow v2 simulator returns a dict of numpy arrays with a leading batch axis.
We emit:
    theta : (B, 12)  canonical identified vector [zx_0..3, zy_0..3, rho_0..3]
    x     : (B, 20)  featurised confusion matrix: 16 row-proportions + 4 log10 trials
    trials: (B, 4)   per-stimulus trial counts (carried for diagnostics / stratifying)
    cls   : (B,)     integer class index into grt_model.MODEL_NAMES

`x` uses the SAME featurisation as the bespoke network's `featurize()`:
proportions are the signal, log10 trial counts tell the network the noise level.
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.grt_model as gm

TRIAL_RANGE = (1, 1000)   # matches src.config.TRIAL_RANGE
Z_MAX = 3.0
R_MAX = 0.9


def _sample_trial_counts(n, rng, trial_range=TRIAL_RANGE):
    """Per-stimulus log-uniform counts, independent per stimulus (unbalanced),
    covering both offline (~balanced) and online/adaptive (sparse) regimes."""
    lo, hi = trial_range
    counts = np.round(np.exp(rng.uniform(np.log(lo), np.log(hi), (n, 4)))).astype(int)
    return np.clip(counts, 1, None)


def _multinomial_counts(probs, trials, rng):
    """Vectorised multinomial via sequential conditional binomials (matches the
    bespoke generator's _multinomial_counts). probs (n,4,4), trials (n,4)."""
    n = probs.shape[0]
    counts = np.zeros((n, 4, 4), dtype=np.int64)
    for s in range(4):
        p = probs[:, s, :]
        remaining_T = trials[:, s].astype(np.int64).copy()
        remaining_p = np.ones(n)
        for r in range(3):
            cond_p = np.clip(p[:, r] / np.maximum(remaining_p, 1e-12), 0.0, 1.0)
            c = rng.binomial(remaining_T, cond_p)
            counts[:, s, r] = c
            remaining_T = remaining_T - c
            remaining_p = remaining_p - p[:, r]
        counts[:, s, 3] = remaining_T
    return counts


def featurize_np(counts16, trials4):
    """Numpy twin of network.featurize: (N,16) counts + (N,4) trials -> (N,20)
    row-proportions (16) ++ log10 trial counts (4)."""
    counts = counts16.reshape(-1, 4, 4).astype(float)
    trials = np.clip(trials4.astype(float), 1, None)
    props = counts / trials[:, :, None]
    log_trials = np.log10(trials)
    return np.concatenate([props.reshape(-1, 16), log_trials], axis=-1)


def simulate(batch_size, rng=None, z_max=Z_MAX, r_max=R_MAX, trial_range=TRIAL_RANGE):
    """Draw `batch_size` GRIN training examples across a uniform mix of the 12
    identified model classes. Returns a dict of numpy arrays for BayesFlow."""
    rng = np.random.default_rng() if rng is None else rng
    n_cls = len(gm.MODEL_NAMES)
    cls = rng.integers(0, n_cls, size=batch_size)

    theta = np.empty((batch_size, 12), dtype=np.float64)
    counts16 = np.empty((batch_size, 16), dtype=np.int64)
    trials4 = np.empty((batch_size, 4), dtype=np.int64)

    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        name = gm.MODEL_NAMES[c]
        zx, zy, rho = gm.sample_prior(name, len(idx), rng, z_max=z_max, r_max=r_max)
        probs = gm.forward_probabilities(zx, zy, rho)          # (m,4,4)
        trials = _sample_trial_counts(len(idx), rng, trial_range)
        counts = _multinomial_counts(probs, trials, rng)       # (m,4,4)
        theta[idx] = gm.pack(zx, zy, rho)
        counts16[idx] = counts.reshape(len(idx), 16)
        trials4[idx] = trials

    x = featurize_np(counts16, trials4)                        # (B,20)
    return {
        "theta": theta.astype(np.float32),         # full 12-vector (for eval)
        "z":     theta[:, :8].astype(np.float32),  # real-valued sensitivities
        "rho":   theta[:, 8:].astype(np.float32),  # correlations in (-1,1)
        "x":     x.astype(np.float32),             # 20-dim featurised condition
        "trials": trials4.astype(np.float32),
        "cls":    cls.astype(np.int32),
    }


# BayesFlow's simulator protocol wants a callable that accepts a batch shape.
class GrinSimulator:
    """Thin wrapper so this plugs into bf.simulators / approximator.fit(simulator=...)."""
    def __init__(self, seed=None, **kw):
        self.rng = np.random.default_rng(seed)
        self.kw = kw

    def sample(self, batch_shape):
        n = int(np.prod(batch_shape))
        return simulate(n, self.rng, **self.kw)


if __name__ == "__main__":
    out = simulate(2000, np.random.default_rng(0))
    for k, v in out.items():
        print(f"{k:7s} {v.shape} {v.dtype}")
    # quick sanity: proportions in [0,1], rho in (-1,1)
    props = out["x"][:, :16]
    print("props range:", props.min(), props.max())
    print("rho range:", out["theta"][:, 8:].min(), out["theta"][:, 8:].max())
    print("classes present:", np.bincount(out["cls"]))
