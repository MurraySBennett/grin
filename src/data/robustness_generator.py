"""
robustness_generator.py — the two simulators the literature review's Week 1/2
program asked for, staged ahead of actually running the battery (compute-heavy
evaluation stays deferred; these are pure data generators, cheap to write and to
smoke-test, not to run at full grid scale on the laptop).

Both wrap GRTDataGenerator's prior/trial-count machinery and only change the LAST
step -- how response counts are drawn from the underlying representation(s) --
so a generated batch is drop-in compatible with everything that already consumes
GRTDataGenerator output (X, X_trials, y_params, y_cls, y_label).

1. ExemplarHeterogeneityGenerator -- docs/literature_review_findings.md #3.
   GRIN's standard simulator draws one stationary multinomial per stimulus. Several
   reviewed studies (Silbert 2012: 4 tokens/category; Farris et al. 2010: 140
   targets; Richler et al. 2008: randomly recombined face parts) instead pool
   several physically-different exemplars into one factorial cell. If exemplars
   differ from each other, the resulting counts are overdispersed relative to a
   single multinomial with the cell's average probability -- a compound/mixture
   distribution, not a multinomial one. This generator produces exactly that
   mixture so the current checkpoint's coverage can be evaluated against it.

2. LearningMixtureGenerator -- docs/literature_review_findings.md #4 (nonstationarity).
   Both Soto papers trim an individually-estimated learning phase before forming a
   matrix specifically because early- and late-block trials come from different
   representations. This generator produces a matrix that blends an "early" and a
   "late" representation at a controllable mixing fraction and trial-level
   change-point, so the current checkpoint's behaviour when handed
   already-blended data (the naive cumulative-matrix approach) can be measured
   directly, and so a future rolling-window/change-point layer has a known-truth
   target to validate detection power against.
"""
import numpy as np

try:
    from src import grt_model as gm
    from src.data.generator import GRTDataGenerator
except ImportError:
    import grt_model as gm
    from data.generator import GRTDataGenerator


def _perturb_exemplars(zx, zy, rho, heterogeneity, n_exemplars, rng, z_max, r_max):
    """(n,) base zx/zy/rho -> (n, n_exemplars) per-exemplar variants, perturbed in
    an unconstrained space (rho via atanh) so every exemplar's rho stays valid.
    heterogeneity=0 collapses every exemplar back onto the base (the standard
    single-multinomial generator as a special case, not a separate code path)."""
    n = zx.shape[0]
    if heterogeneity <= 0 or n_exemplars <= 1:
        rep = lambda v: np.repeat(v[:, None], n_exemplars, axis=1)
        return rep(zx), rep(zy), rep(rho)
    ezx = zx[:, None] + rng.normal(0, heterogeneity, (n, n_exemplars))
    ezy = zy[:, None] + rng.normal(0, heterogeneity, (n, n_exemplars))
    rho_z = np.arctanh(np.clip(rho, -0.999, 0.999))
    erho = np.tanh(rho_z[:, None] + rng.normal(0, heterogeneity, (n, n_exemplars)))
    return (np.clip(ezx, -z_max * 1.5, z_max * 1.5),
            np.clip(ezy, -z_max * 1.5, z_max * 1.5),
            np.clip(erho, -r_max, r_max))


class ExemplarHeterogeneityGenerator(GRTDataGenerator):
    """
    n_exemplars: how many physically-different items populate each factorial cell
        (1 reproduces the standard generator exactly).
    heterogeneity: SD, in the same unconstrained space training uses (raw z-units
        for zx/zy, atanh-rho-units for rho), of the per-exemplar perturbation
        around the cell's nominal representation. 0 = no heterogeneity.
    exemplar_mode:
        "fixed_blocks"  -- a FIXED set of n_exemplars is drawn once per matrix and
            trials are split into n_exemplars equal-size blocks, one block per
            exemplar (repeated presentation of a fixed exemplar set, e.g. Silbert's
            four tokens/category). More between-block clustering.
        "iid_resample"  -- each trial independently draws which exemplar (of a
            fixed pool of n_exemplars) produced it. Still overdispersed relative to
            a single multinomial (mixture variance), but without block clustering.
    """
    def __init__(self, *args, n_exemplars=4, heterogeneity=0.3,
                exemplar_mode="fixed_blocks", **kwargs):
        super().__init__(*args, **kwargs)
        self.n_exemplars = int(n_exemplars)
        self.heterogeneity = float(heterogeneity)
        assert exemplar_mode in ("fixed_blocks", "iid_resample")
        self.exemplar_mode = exemplar_mode

    def _multinomial_counts_heterogeneous(self, zx, zy, rho, trials, rng):
        """zx/zy/rho (n,4); trials (n,4) -> counts (n,4,4), drawn from
        n_exemplars-many perturbed variants of each stimulus's representation
        instead of one stationary probability vector."""
        n = zx.shape[0]
        counts = np.zeros((n, 4, 4), dtype=np.int64)
        for s in range(4):
            ezx, ezy, erho = _perturb_exemplars(
                zx[:, s], zy[:, s], rho[:, s], self.heterogeneity, self.n_exemplars,
                rng, self.z_max, self.r_max)                          # (n, K)
            K = self.n_exemplars
            T = trials[:, s].astype(np.int64)
            if self.exemplar_mode == "fixed_blocks":
                # split T as evenly as possible across K blocks; each block is a
                # multinomial draw from ITS exemplar's own probability vector
                base_block = T // K
                remainder = T - base_block * K
                for k in range(K):
                    block_T = base_block + (remainder > k).astype(np.int64)
                    probs_k = gm.forward_probabilities(
                        ezx[:, k:k + 1], ezy[:, k:k + 1], erho[:, k:k + 1])[:, 0, :]  # (n,4)
                    counts[:, s, :] += self._draw_multinomial_row(probs_k, block_T, rng)
            else:  # iid_resample: each of T trials independently picks an exemplar
                exemplar_draw = rng.integers(0, K, (n, int(T.max()) if T.max() > 0 else 1))
                for k in range(K):
                    probs_k = gm.forward_probabilities(
                        ezx[:, k:k + 1], ezy[:, k:k + 1], erho[:, k:k + 1])[:, 0, :]  # (n,4)
                    # trials assigned to exemplar k, respecting each row's own T
                    valid = np.arange(exemplar_draw.shape[1])[None, :] < T[:, None]
                    block_T = ((exemplar_draw == k) & valid).sum(1)
                    counts[:, s, :] += self._draw_multinomial_row(probs_k, block_T, rng)
        return counts

    @staticmethod
    def _draw_multinomial_row(probs, T, rng):
        """probs (n,4), T (n,) -> counts (n,4) via sequential conditional binomials."""
        n = probs.shape[0]
        out = np.zeros((n, 4), dtype=np.int64)
        remaining_T = T.copy()
        remaining_p = np.ones(n)
        for r in range(3):
            cond_p = np.clip(probs[:, r] / np.maximum(remaining_p, 1e-12), 0.0, 1.0)
            c = rng.binomial(remaining_T, cond_p)
            out[:, r] = c
            remaining_T = remaining_T - c
            remaining_p = remaining_p - probs[:, r]
        out[:, 3] = remaining_T
        return out

    def generate_all_model_cms(self, seed=None):
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        X, y_params, X_trials, y_cls, y_label = [], [], [], [], []
        n = self.n_per_class
        for cls_idx, name in enumerate(self.model_names):
            zx, zy, rho = gm.sample_prior(name, n, rng, z_max=self.z_max, r_max=self.r_max)
            trials = self._sample_trial_counts(n, rng)
            counts = self._multinomial_counts_heterogeneous(zx, zy, rho, trials, rng)
            X.append(counts.reshape(n, 16)); X_trials.append(trials)
            y_params.append(gm.pack(zx, zy, rho))
            y_cls.append(np.full(n, cls_idx, dtype=int))
            y_label.append(np.array([name] * n))
        return (np.concatenate(X), np.concatenate(y_params), np.concatenate(X_trials),
                np.concatenate(y_cls), np.concatenate(y_label))


class LearningMixtureGenerator(GRTDataGenerator):
    """
    Simulates a matrix built the naive cumulative way across a learning
    transition: the first `changepoint_frac` of each stimulus's trials come from
    an "early" representation, the rest from an independently-sampled "late"
    representation of the SAME model class. y_params returned is the LATE
    representation (the one an online/adaptive user actually wants to know), so
    evaluating the current checkpoint against it directly measures how much a
    naive cumulative matrix misleads about "current" representation -- this is
    the known-truth target a future rolling-window/change-point detector should
    be validated against, not itself a change-point detector.

    changepoint_frac: fraction of each stimulus's trials from the early state
        (0 = no learning effect / recovers the standard generator; 0.5 = evenly
        split).
    drift_scale: SD, in the same unconstrained space as heterogeneity above, of
        how far the late representation moves from the early one.
    """
    def __init__(self, *args, changepoint_frac=0.4, drift_scale=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.changepoint_frac = float(changepoint_frac)
        self.drift_scale = float(drift_scale)

    def generate_all_model_cms(self, seed=None):
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        X, y_params, X_trials, y_cls, y_label = [], [], [], [], []
        n = self.n_per_class
        for cls_idx, name in enumerate(self.model_names):
            zx0, zy0, rho0 = gm.sample_prior(name, n, rng, z_max=self.z_max, r_max=self.r_max)
            if self.changepoint_frac >= 1.0 or self.drift_scale <= 0:
                zx1, zy1, rho1 = zx0, zy0, rho0
            else:
                ezx, ezy, erho = _perturb_exemplars(
                    zx0.ravel(), zy0.ravel(), rho0.ravel(), self.drift_scale, 1,
                    rng, self.z_max, self.r_max)
                zx1 = ezx[:, 0].reshape(zx0.shape); zy1 = ezy[:, 0].reshape(zy0.shape)
                rho1 = erho[:, 0].reshape(rho0.shape)
            trials = self._sample_trial_counts(n, rng)
            T_early = np.round(trials * self.changepoint_frac).astype(np.int64)
            T_late = trials - T_early
            probs0 = gm.forward_probabilities(zx0, zy0, rho0)   # (n,4,4)
            probs1 = gm.forward_probabilities(zx1, zy1, rho1)
            counts = (self._multinomial_counts(probs0, T_early, rng)
                     + self._multinomial_counts(probs1, T_late, rng))
            X.append(counts.reshape(n, 16)); X_trials.append(trials)
            y_params.append(gm.pack(zx1, zy1, rho1))    # ground truth = the LATE state
            y_cls.append(np.full(n, cls_idx, dtype=int))
            y_label.append(np.array([name] * n))
        return (np.concatenate(X), np.concatenate(y_params), np.concatenate(X_trials),
                np.concatenate(y_cls), np.concatenate(y_label))
