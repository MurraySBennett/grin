"""Simulation generator for the experimental two-dimensional 3x3 GRT model."""

import numpy as np

from src import grt_model_3x3 as gm


class GRT3x3DataGenerator:
    def __init__(self, n_per_class=5000, trial_range=(5, 500), z_max=3.0,
                 r_max=0.9, bound_range=(0.75, 3.0), seed=None,
                 balanced_trials=False, imbalance=0.35, model_module=gm,
                 prior_kwargs=None):
        self.n_per_class = int(n_per_class)
        self.trial_range = trial_range
        self.z_max = z_max
        self.r_max = r_max
        self.bound_range = bound_range
        self.seed = seed
        self.imbalance = 0.0 if balanced_trials else float(imbalance)
        self.gm = model_module
        self.prior_kwargs = dict(prior_kwargs or {})
        self.model_names = self.gm.MODEL_NAMES

    def _sample_trial_counts(self, n, rng):
        lo, hi = self.trial_range
        base = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        if self.imbalance <= 0:
            counts = np.repeat(base[:, None], 9, axis=1)
        else:
            factors = rng.uniform(1.0 - min(self.imbalance, 1.0), 1.0, (n, 9))
            counts = base[:, None] * factors
        return np.clip(np.round(counts).astype(np.int64), 1, None)

    @staticmethod
    def _multinomial_counts(probs, trials, rng):
        n, n_stim, n_resp = probs.shape
        counts = np.zeros((n, n_stim, n_resp), dtype=np.int64)
        for s in range(n_stim):
            remaining_t = trials[:, s].copy()
            remaining_p = np.ones(n)
            for r in range(n_resp - 1):
                conditional = np.clip(probs[:, s, r] / np.maximum(remaining_p, 1e-12), 0, 1)
                draw = rng.binomial(remaining_t, conditional)
                counts[:, s, r] = draw
                remaining_t -= draw
                remaining_p -= probs[:, s, r]
            counts[:, s, -1] = remaining_t
        return counts

    def generate_all_model_cms(self, seed=None):
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        xs, ys, ts, classes, labels = [], [], [], [], []
        n = self.n_per_class
        for class_index, name in enumerate(self.model_names):
            prior_kwargs = dict(self.prior_kwargs)
            if self.gm is gm:
                prior_kwargs.setdefault("bound_range", self.bound_range)
            theta = self.gm.sample_prior(name, n, rng, z_max=self.z_max,
                                         r_max=self.r_max, **prior_kwargs)
            probs = self.gm.forward_probabilities(*theta)
            trials = self._sample_trial_counts(n, rng)
            counts = self._multinomial_counts(probs, trials, rng)
            xs.append(counts.reshape(n, 81))
            ys.append(self.gm.pack(*theta))
            ts.append(trials)
            classes.append(np.full(n, class_index, dtype=int))
            labels.append(np.full(n, name))
        return tuple(map(np.concatenate, (xs, ys, ts, classes, labels)))


class GRT3x3HeteroDataGenerator(GRT3x3DataGenerator):
    """Convenience wrapper for the free-marginal-variance 3x3 model."""

    def __init__(self, *args, sd_range=(0.5, 2.0), **kwargs):
        from src import grt_model_3x3_hetero
        super().__init__(*args, model_module=grt_model_3x3_hetero,
                         prior_kwargs={"sd_range": sd_range}, **kwargs)
