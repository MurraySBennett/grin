"""
GRT_data_generator.py  —  Phase 1, rebuilt on grt_model.py.

The generative pipeline is now direct and exact, with no rejection/calibration
loop:

    1. sample identified parameters (z-scores + correlations) from the explicit
       per-class prior                                     (grt_model.sample_prior)
    2. map parameters -> exact confusion-matrix probabilities
                                                    (grt_model.forward_probabilities)
    3. draw per-stimulus trial counts, then multinomial-sample response counts

Everything is vectorised over the whole batch (the multinomial is drawn via a
sequential-binomial decomposition using numpy's vectorised rng.binomial), so a
large premium dataset generates in seconds and no multiprocessing is needed.

Output (the .npz written by the __main__ block):
    X            (N, 16)  flattened 4x4 confusion matrices of COUNTS, row-major
                          (stimulus-major): [s0r0, s0r1, s0r2, s0r3, s1r0, ...]
    X_trials     (N, 4)   per-stimulus trial counts (each CM row sums to these)
    y_params     (N, 12)  canonical identified vector [zx_0..3, zy_0..3, rho_0..3]
    y_model_cls  (N,)     integer class index into grt_model.MODEL_NAMES
    y_cls_label  (N,)     class-name string

Trial counts are drawn log-uniformly across a broad range so the network sees the
full noise spectrum (few trials -> noisy/uncertain, many -> precise). This is the
knob that teaches the network the trials -> uncertainty relationship the adaptive
loop relies on.
"""

import numpy as np

try:                                    # in-package (src/grt_model.py)
    from src import grt_model as gm
except ImportError:                     # standalone (tests, same directory)
    import grt_model as gm


class GRTDataGenerator:
    def __init__(self, n_per_class=5000, trial_range=(5, 500),
                 z_max=3.0, r_max=0.9, seed=None, balanced_trials=False,
                 imbalance=0.35):
        self.n_per_class = int(n_per_class)
        self.trial_range = trial_range          # per-participant BASE count, log-uniform
        self.z_max = z_max
        self.r_max = r_max
        self.seed = seed
        # Within-set trial imbalance, matched to RTLBAGenerator so the counts and RT
        # pipelines see the SAME trial-count regime. One base count is drawn per matrix;
        # each stimulus then keeps a fraction in [1 - imbalance, 1] of it. imbalance=0.0
        # is perfectly balanced. balanced_trials=True forces imbalance=0.0 and is kept as
        # a back-compatible alias for callers (e.g. the showcase set in make_figures.py)
        # that pass it explicitly.
        self.balanced_trials = balanced_trials
        self.imbalance = 0.0 if balanced_trials else float(imbalance)
        self.model_names = gm.MODEL_NAMES

    # ------------------------------------------------------------------ #
    def _sample_trial_counts(self, n, rng):
        """
        Per-stimulus trial counts with BOUNDED within-set imbalance, identical in
        spirit to RTLBAGenerator._sample_trial_counts so the counts and RT pipelines
        see the same regime. One per-matrix BASE count is drawn log-uniformly across
        trial_range (preserving the few->many magnitude spectrum the network needs to
        learn the trials->uncertainty relation); the 4 stimuli then differ only by
        realistic attrition, each keeping a fraction in [1 - imbalance, 1] of the base.
        This rules out the pathological independent-sampling sets (e.g. 1/648/5/732).
        balanced_trials=True (or imbalance=0.0) forces equal counts per matrix.
        """
        lo, hi = self.trial_range
        base = np.exp(rng.uniform(np.log(lo), np.log(hi), n))          # (n,) per matrix
        f = 0.0 if getattr(self, "balanced_trials", False) else min(self.imbalance, 1.0)
        if f <= 0.0:
            counts = np.repeat(base[:, None], 4, axis=1)               # perfectly balanced
        else:
            factors = rng.uniform(1.0 - f, 1.0, (n, 4))               # each stimulus keeps 1-f..1
            counts = base[:, None] * factors
        counts = np.round(counts).astype(np.int64)
        return np.clip(counts, 1, None)                               # (n, 4)

    def _multinomial_counts(self, probs, trials, rng):
        """
        Vectorised multinomial sampling via sequential conditional binomials.
        probs  (n, 4, 4)  response probabilities per matrix/stimulus
        trials (n, 4)     trial counts per matrix/stimulus
        returns (n, 4, 4) integer counts, each row summing to its trial count.
        """
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

    # ------------------------------------------------------------------ #
    def generate_all_model_cms(self, seed=None):
        """Generate the full dataset. Returns (X, y_params, X_trials, y_cls, y_label)."""
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)

        X, y_params, X_trials, y_cls, y_label = [], [], [], [], []
        n = self.n_per_class
        for cls_idx, name in enumerate(self.model_names):
            zx, zy, rho = gm.sample_prior(name, n, rng, z_max=self.z_max, r_max=self.r_max)
            probs = gm.forward_probabilities(zx, zy, rho)          # (n,4,4)
            trials = self._sample_trial_counts(n, rng)             # (n,4)
            counts = self._multinomial_counts(probs, trials, rng)  # (n,4,4)

            X.append(counts.reshape(n, 16))
            X_trials.append(trials)
            y_params.append(gm.pack(zx, zy, rho))                  # (n,12)
            y_cls.append(np.full(n, cls_idx, dtype=int))
            y_label.append(np.array([name] * n))

        return (np.concatenate(X), np.concatenate(y_params), np.concatenate(X_trials),
                np.concatenate(y_cls), np.concatenate(y_label))

    # ------------------------------------------------------------------ #
    def coverage_report(self, X, X_trials, y_params, y_cls, y_label,
                        figure_path=None):
        """
        Summarise (and optionally plot) the induced distributions over accuracy and
        structural features, so coverage is verified rather than assumed.
        Returns a dict of arrays keyed by feature.
        """
        counts = X.reshape(-1, 4, 4).astype(float)
        trials = X_trials.astype(float)
        props = counts / trials[:, :, None]                        # row proportions

        per_stim_acc = np.einsum('nii->ni', props)                 # (N,4)
        overall_acc = per_stim_acc.mean(1)
        # response bias: tendency to respond level-2 on each dimension
        # respond a2 on x = responses a2b1(r2)+a2b2(r3); respond b2 on y = a1b2(r1)+a2b2(r3)
        x_bias = (props[:, :, 2] + props[:, :, 3]).mean(1) - 0.5
        y_bias = (props[:, :, 1] + props[:, :, 3]).mean(1) - 0.5
        # congruency asymmetry: acc on same-level stimuli (s0=A1B1, s3=A2B2)
        #                       minus different-level stimuli (s1=A1B2, s2=A2B1)
        congruency = 0.5 * (per_stim_acc[:, 0] + per_stim_acc[:, 3]) \
            - 0.5 * (per_stim_acc[:, 1] + per_stim_acc[:, 2])
        zx, zy, rho = gm.unpack(y_params)

        stats = {
            "overall_accuracy": overall_acc,
            "per_stimulus_accuracy": per_stim_acc,
            "x_response_bias": x_bias,
            "y_response_bias": y_bias,
            "congruency_asymmetry": congruency,
            "abs_zscore": np.abs(np.concatenate([zx, zy], axis=1)).ravel(),
            "correlation": rho.ravel(),
            "trials_per_stimulus": trials[:, 0],
        }

        def q(a):
            return (np.min(a), np.quantile(a, .25), np.median(a),
                    np.quantile(a, .75), np.max(a))

        print(f"{'feature':22s} {'min':>7s} {'q25':>7s} {'med':>7s} {'q75':>7s} {'max':>7s}")
        for key in ["overall_accuracy", "x_response_bias", "y_response_bias",
                    "congruency_asymmetry", "abs_zscore", "correlation",
                    "trials_per_stimulus"]:
            lo, q1, md, q3, hi = q(stats[key])
            print(f"{key:22s} {lo:7.2f} {q1:7.2f} {md:7.2f} {q3:7.2f} {hi:7.2f}")

        if figure_path is not None:
            self._plot_coverage(stats, figure_path, self.z_max, self.r_max)
        return stats

    @staticmethod
    def _plot_coverage(stats, figure_path, z_max=3.0, r_max=0.9):
        """Delegate to src.viz.generation so the coverage report uses the house style.

        This used to draw six seaborn-default histograms inline, which made it the only
        figure in the project with top/right spines, an off-palette colour scheme and
        dpi=110. It now goes through set_style() like everything else, marks the prior
        bounds on the panels whose job is to show prior coverage, and also writes each
        panel individually to <figures>/generation/.
        """
        import os
        try:
            from ..viz.generation import coverage_figures
        except ImportError:
            from src.viz.generation import coverage_figures
        panel_dir = os.path.join(os.path.dirname(os.path.abspath(figure_path)), "generation")
        coverage_figures(stats, figure_path, panel_dir=panel_dir,
                         z_max=z_max, r_max=r_max)
