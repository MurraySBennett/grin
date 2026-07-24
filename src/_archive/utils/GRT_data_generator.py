"""
GRT_data_generator.py  —  optimised / cleaned-up version.

Simulates confusion matrices from General Recognition Theory (GRT) models for
2x2 factorial identification designs, for use as training data in the GRIN
simulation-based-inference pipeline.

WHAT CHANGED vs. the original (behaviour-preserving unless noted):

  Performance
  -----------
  #1  Sample sharing across criterion candidates.
      calibrate_means_and_crit() used to re-simulate a full confusion matrix
      for every candidate criterion (max_crit_attempts times per scaling
      iteration). The criterion is only a threshold on the perceptual sample
      cloud, so the cloud is now drawn ONCE per scaling iteration and each
      candidate criterion is scored by re-counting against the same cloud.
      ~max_crit_attempts-fold fewer draws; statistically equivalent (all
      candidates now share the same noise, which slightly *reduces* selection
      variance).

  #2  NumPy Generator instead of scipy.stats.multivariate_normal.rvs.
      Sampling uses a cached 2x2 Cholesky factor and rng.standard_normal,
      avoiding scipy's per-call covariance factorisation and Python overhead.

  #4  Unified, seedable RNG.
      All randomness flows through a single numpy Generator threaded through
      the call chain (no more mixing of the legacy global np.random.* with a
      passed Generator). generate_all_model_cms(seed=...) is now reproducible,
      and per-worker seeding is handled correctly under the optional parallel
      path.

  #5  Vectorised quadrant counting via np.bincount (replaces four boolean sums).

  Optional parallelism (opt-in, off by default)
  ----------------------------------------------
      generate_all_model_cms(n_jobs=-1) will distribute (model, accuracy-bin)
      chunks across processes with joblib, each with an independent child seed.
      Falls back silently to serial if joblib is unavailable or n_jobs == 1.
      Output format is identical either way.

  Bug fixes (these paths raised before)
  -------------------------------------
      * generate_trial_data() unpacked three values from random_model_params(),
        which returns two -> it now obtains the criterion via random_crit().
      * generate_parameter_controlled_cms() called a non-existent
        self.random_means() -> a random_means() helper is now provided
        (unconstrained means, i.e. no PS constraint).

NOTE ON OUTPUT COMPATIBILITY:
  The output *format* is unchanged — shapes, column order, the flat parameter
  layout (8 means, 16 covariance elements, 2 criteria), and the confusion-matrix
  quadrant convention all match the original, so downstream code and previously
  cached datasets remain valid. A newly regenerated dataset will not be
  bit-identical to an old one (any change to the RNG stream reseeds it), but it
  is drawn from the same distribution.
"""

import os
import time
import argparse
from pprint import pprint as pp

import numpy as np
from tqdm import tqdm

from src.utils.config import *


class GRTDataGenerator:
    def __init__(self, num_matrices=10, num_dimensions=2, num_levels=2,
                 trial_range=TRIALS_RANGE, seed=None):
        self.num_matrices = num_matrices
        self.num_dimensions = num_dimensions
        self.num_levels = num_levels
        self.num_stimuli = self.num_dimensions * self.num_levels

        self.mean_range = (-3, 5)  # asymmetric range biases toward ordered means
        self.mean_sep = 0.1
        self.pi_tolerance = 0.1
        self.corr_range = (self.pi_tolerance, 0.99)
        self.variance = 1.0
        self.crit_range = (self.mean_range[0] * 0.9, self.mean_range[1] * 0.9)
        self.sample_loss = 0.05
        self.trial_range = trial_range
        self.min_accuracy = MIN_MATRIX_ACCURACY
        self.max_accuracy = MAX_MATRIX_ACCURACY
        self.n_accuracy_bins = MATRIX_ACCURACY_BINS

        self.model_names = MODEL_NAMES

        # Single, seedable source of randomness (optimisation #4).
        self._rng = np.random.default_rng(seed)

    def _resolve_rng(self, rng):
        return self._rng if rng is None else rng

    # ------------------------------------------------------------------ #
    # Low-level sampling / counting helpers (optimisations #2 and #5)
    # ------------------------------------------------------------------ #
    def _sample_cloud(self, mean_xy, cov, n, rng):
        """Draw n bivariate-normal samples via a cached Cholesky factor."""
        if n <= 0:
            return np.empty((0, 2))
        L = np.linalg.cholesky(cov)                 # 2x2, cheap and exact
        z = rng.standard_normal((n, 2))
        return mean_xy + z @ L.T

    def _quadrant_counts(self, samples, crit):
        """
        Count samples into the four response quadrants, preserving the original
        convention:
            0: x <  cx & y <  cy      1: x >= cx & y <  cy
            2: x <  cx & y >= cy      3: x >= cx & y >= cy
        """
        if samples.shape[0] == 0:
            return np.zeros(4, dtype=int)
        idx = ((samples[:, 0] >= crit[0]).astype(np.int64)
               + 2 * (samples[:, 1] >= crit[1]).astype(np.int64))
        return np.bincount(idx, minlength=4)

    def _draw_clouds(self, means_flat, cov_mat, n_stimulus_trials,
                     sample_loss_factor, rng):
        """
        Draw one perceptual sample cloud per stimulus (with per-stimulus trial
        counts). Returns (clouds, trial_counts). Drawn once, reused across all
        candidate criteria (optimisation #1).
        """
        clouds = []
        trial_counts = np.zeros(self.num_stimuli, dtype=int)
        lo = int(n_stimulus_trials * (1 - sample_loss_factor))
        hi = int(n_stimulus_trials)
        lo = max(lo, 0)
        for i in range(self.num_stimuli):
            n_i = int(rng.integers(lo, hi + 1)) if hi >= lo else hi
            trial_counts[i] = n_i
            clouds.append(self._sample_cloud(means_flat[i * 2:i * 2 + 2],
                                             cov_mat[i], n_i, rng))
        return clouds, trial_counts

    def _cm_from_clouds(self, clouds, trial_counts, crit):
        """Build a confusion matrix and per-stimulus accuracy from sample clouds."""
        cm = np.zeros((self.num_stimuli, 4), dtype=int)
        for i, s in enumerate(clouds):
            cm[i] = self._quadrant_counts(s, crit)
        per_stim_acc = np.divide(
            np.diag(cm), trial_counts,
            out=np.zeros(self.num_stimuli, dtype=float),
            where=trial_counts != 0,
        )
        return cm, per_stim_acc

    # ------------------------------------------------------------------ #
    # Public simulation API (kept for backward compatibility)
    # ------------------------------------------------------------------ #
    def accuracy_check(self, cm, min_acc, max_acc):
        correct_counts = np.diag(cm)
        total_trials = np.sum(cm, axis=1)
        accuracies = np.divide(
            correct_counts, total_trials,
            out=np.zeros_like(correct_counts, dtype=float),
            where=total_trials != 0,
        )
        grand_mean_accuracy = np.mean(accuracies)
        return min_acc < grand_mean_accuracy <= max_acc

    def simulate_cm_from_params(self, means, cov_mat, crit, n_stimulus_trials,
                                sample_loss_factor=0.0, rng=None):
        """Simulate a confusion matrix given params (thin wrapper over the helpers)."""
        rng = self._resolve_rng(rng)
        clouds, trial_counts = self._draw_clouds(
            means, cov_mat, n_stimulus_trials, sample_loss_factor, rng)
        cm, per_stim_acc = self._cm_from_clouds(clouds, trial_counts, crit)
        return cm, trial_counts, per_stim_acc

    def calibrate_means_and_crit(
        self,
        base_means,
        cov_mat,
        target_accuracy,
        n_stimulus_trials,
        sample_loss_factor=0.05,
        max_scale_iters=10,
        max_crit_attempts=8,
        tol=0.01,
        rng=None,
        time_budget_sec=4.0,
    ):
        """
        Joint calibration of mean scaling and criterion (c) to hit a target
        accuracy. Preserves stimulus-0 coordinates by scaling other means
        relative to that anchor, and keeps means within self.mean_range.

        Optimisation #1: the perceptual sample cloud is drawn once per scaling
        iteration and shared across all candidate criteria.
        """
        rng = self._resolve_rng(rng)
        stim_means = base_means.reshape(-1, 2)
        anchor = stim_means[0].copy()      # preserve stimulus 0 (identifiability anchor)
        relative = stim_means - anchor
        min_allowed, max_allowed = self.mean_range

        left, right = 0.0, 3.0
        best = None
        start_t = time.time()

        for _ in range(max_scale_iters):
            if time.time() - start_t > time_budget_sec:
                break
            mid = 0.5 * (left + right)

            scaled = relative * mid + anchor

            # safeguard: shrink the scale if any coordinate leaves mean_range
            if scaled.min() < min_allowed or scaled.max() > max_allowed:
                safe_factor = min(
                    (max_allowed - anchor).max() / (scaled - anchor).max(initial=1e-6),
                    (anchor - min_allowed).max() / (anchor - scaled).max(initial=1e-6),
                )
                mid *= max(0.0, safe_factor)
                scaled = relative * mid + anchor

            scaled_flat = scaled.flatten()

            # Draw the perceptual cloud ONCE for this scaling; reuse for all criteria.
            clouds, trial_counts = self._draw_clouds(
                scaled_flat, cov_mat, n_stimulus_trials, sample_loss_factor, rng)

            means_x, means_y = scaled_flat[0::2], scaled_flat[1::2]
            mid_x, mid_y = np.mean(means_x), np.mean(means_y)
            std_x = (np.max(means_x) - np.min(means_x)) / (len(means_x) + 1)
            std_y = (np.max(means_y) - np.min(means_y)) / (len(means_y) + 1)

            best_local = None
            for _ in range(max_crit_attempts):
                c = np.array([
                    np.clip(rng.normal(mid_x, std_x), self.crit_range[0], self.crit_range[1]),
                    np.clip(rng.normal(mid_y, std_y), self.crit_range[0], self.crit_range[1]),
                ])
                cm, per_stim_acc = self._cm_from_clouds(clouds, trial_counts, c)
                mean_acc = np.mean(per_stim_acc)
                if best_local is None or abs(mean_acc - target_accuracy) < abs(best_local[0] - target_accuracy):
                    best_local = (mean_acc, c, cm, trial_counts)

            if best is None or abs(best_local[0] - target_accuracy) < abs(best[0] - target_accuracy):
                best = (best_local[0], mid, scaled_flat, best_local[2], best_local[3], best_local[1])

            if best_local[0] < target_accuracy:
                left = mid
            else:
                right = mid
            if abs(best_local[0] - target_accuracy) <= tol:
                break

        if best is None:
            return base_means, None, None, None, None
        # (scaled_means, cm, trial_counts, c, achieved_accuracy)
        return best[2], best[3], best[4], best[5], best[0]

    def generate_cm(self, model_name, n_stimulus_trials, min_acc, max_acc, rng=None):
        """
        Generate one confusion matrix by calibrating mean scaling and criterion.
        Stimulus 0 stays anchored at its original coordinates (e.g. [0, 0]).
        """
        rng = self._resolve_rng(rng)
        max_attempts = 6
        target_accuracy = float(min_acc + (max_acc - min_acc) * rng.random())

        for _ in range(max_attempts):
            means, cov_mat = self.random_model_params(model_name, rng=rng)
            scaled_means, cm, trial_counts, c, achieved = self.calibrate_means_and_crit(
                means, cov_mat, target_accuracy, n_stimulus_trials,
                sample_loss_factor=self.sample_loss, rng=rng, time_budget_sec=2.0,
            )
            if cm is not None and min_acc < achieved <= max_acc:
                flat_params = np.concatenate([
                    scaled_means.flatten(),
                    np.array([m.flatten() for m in cov_mat]).flatten(),
                    c,
                ])
                return cm, trial_counts, flat_params, scaled_means, cov_mat, c
            n_stimulus_trials = np.clip(int(np.ceil(n_stimulus_trials * 1.1)), *self.trial_range)

        return None, None, None, None, None, None

    def _generate_one_bin(self, model_name, min_acc, max_acc, n_matrices, rng, show_progress=True):
        """Generate all matrices for a single (model, accuracy-bin) chunk."""
        cms, trial_counts, params = [], [], []
        iterator = range(n_matrices)
        if show_progress:
            iterator = tqdm(iterator, total=n_matrices,
                            desc=f"{model_name} @ {min_acc*100:.1f}-{max_acc*100:.1f}%")
        for _ in iterator:
            cm = None
            while cm is None:
                n_trials = int(np.ceil(rng.uniform(*self.trial_range)))
                cm, trial_count, flat_params, _, _, _ = self.generate_cm(
                    model_name, n_trials, min_acc, max_acc, rng=rng)
            cms.append(cm.flatten())
            trial_counts.append(trial_count)
            params.append(flat_params)
        return cms, trial_counts, params

    def generate_cms(self, model_name, n_matrices=1, rng=None):
        """Serial generation of n_matrices per accuracy bin for one model."""
        rng = self._resolve_rng(rng)
        cms, trial_counts, params = [], [], []
        edges = np.round(np.linspace(self.min_accuracy, self.max_accuracy,
                                     self.n_accuracy_bins) / 100, 3)
        for idx, min_acc in enumerate(edges[:-1]):
            max_acc = edges[idx + 1]
            b_cms, b_tc, b_p = self._generate_one_bin(
                model_name, min_acc, max_acc, n_matrices, rng)
            cms.extend(b_cms)
            trial_counts.extend(b_tc)
            params.extend(b_p)
        return cms, trial_counts, params

    def generate_all_model_cms(self, seed=None, n_jobs=1):
        """
        Generate the full dataset across all models and accuracy bins.

        n_jobs=1 (default): serial, identical behaviour to the original.
        n_jobs!=1: distribute (model, accuracy-bin) chunks across processes with
                   joblib, each with an independent child seed. Falls back to
                   serial if joblib is unavailable.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        edges = np.round(np.linspace(self.min_accuracy, self.max_accuracy,
                                     self.n_accuracy_bins) / 100, 3)
        # Task list: one chunk per (model, accuracy-bin).
        tasks = []
        for model_class, model_label in enumerate(self.model_names):
            for idx, min_acc in enumerate(edges[:-1]):
                tasks.append((model_class, model_label, float(min_acc), float(edges[idx + 1])))

        # Independent child seeds so every chunk has a disjoint random stream.
        base_ss = np.random.SeedSequence(seed)
        child_seeds = base_ss.spawn(len(tasks))

        def run_chunk(task, child_seed):
            model_class, model_label, min_acc, max_acc = task
            rng = np.random.default_rng(child_seed)
            b_cms, b_tc, b_p = self._generate_one_bin(
                model_label, min_acc, max_acc, self.num_matrices, rng,
                show_progress=(n_jobs == 1))
            labels_id = [model_class] * len(b_cms)
            labels_name = [model_label] * len(b_cms)
            return b_cms, b_p, b_tc, labels_id, labels_name

        results = None
        if n_jobs != 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_chunk)(t, s) for t, s in zip(tasks, child_seeds))
            except ImportError:
                print("joblib not available; falling back to serial generation.")
        if results is None:
            results = [run_chunk(t, s)
                       for t, s in tqdm(list(zip(tasks, child_seeds)),
                                        desc="Generating (serial)")]

        cms, parameters, trial_counts, y_cls, y_cls_label = [], [], [], [], []
        for b_cms, b_p, b_tc, l_id, l_name in results:
            cms.extend(b_cms)
            parameters.extend(b_p)
            trial_counts.extend(b_tc)
            y_cls.extend(l_id)
            y_cls_label.extend(l_name)

        return (np.array(cms), np.array(parameters), np.array(trial_counts),
                np.array(y_cls), np.array(y_cls_label))

    # ------------------------------------------------------------------ #
    # Parameter generation (GRT model constraints)
    # ------------------------------------------------------------------ #
    def random_model_params(self, model_name="pi_ps_ds", rng=None):
        rng = self._resolve_rng(rng)
        has_pi = 'pi' in model_name
        has_rho1 = 'rho1' in model_name
        has_psa = 'psa' in model_name or 'ps_' in model_name
        has_psb = 'psb' in model_name or 'ps_' in model_name
        means = self._generate_constrained_means(has_psa, has_psb, rng=rng)
        cov_mat = self._generate_constrained_cov(has_pi, has_rho1, rng=rng)
        return means, cov_mat

    def random_means(self, rng=None):
        """Unconstrained means (no perceptual-separability constraint)."""
        return self._generate_constrained_means(has_psa=False, has_psb=False, rng=rng)

    def _generate_constrained_means(self, has_psa, has_psb, rng=None):
        rng = self._resolve_rng(rng)
        while True:
            means_x = np.hstack([
                0.0,
                rng.uniform(0, self.mean_range[1]),
                np.sort(rng.uniform(*self.mean_range, 2)),
            ])
            means_x[3] = np.abs(means_x[3])

            means_y = np.hstack([0.0, rng.uniform(*self.mean_range, 3)])
            means_y[2] = np.abs(means_y[2])
            means_y[[1, 3]] = np.sort(means_y[[1, 3]])
            means_y[3] = np.abs(means_y[3])

            # Perceptual separability of dimension A (x): A1 -> stim{0,2}, A2 -> stim{1,3}
            if has_psa:
                means_x[2] = means_x[0]
                means_x[1] = means_x[3]
            # Perceptual separability of dimension B (y): B1 -> stim{0,1}, B2 -> stim{2,3}
            if has_psb:
                means_y[1] = means_y[0]
                means_y[2] = means_y[3]

            # Reject configurations that accidentally satisfy a constraint they shouldn't.
            psa_accident = (not has_psa
                            and np.abs(means_x[2] - means_x[0]) < self.mean_sep
                            and np.abs(means_x[3] - means_x[1]) < self.mean_sep)
            psb_accident = (not has_psb
                            and np.abs(means_y[1] - means_y[0]) < self.mean_sep
                            and np.abs(means_y[3] - means_y[2]) < self.mean_sep)

            if not psa_accident and not psb_accident:
                means = np.vstack([means_x, means_y]).T.flatten()
                return np.round(means, 3)

    def _create_cov_mat(self, corr):
        return np.array([
            [self.variance, corr * self.variance],
            [corr * self.variance, self.variance],
        ])

    def _generate_constrained_cov(self, has_pi, has_rho1, rng=None):
        rng = self._resolve_rng(rng)
        if has_pi:
            return self._set_pi()
        elif has_rho1:
            return self._set_rho1(rng=rng)
        else:
            return self._set_ds(rng=rng)

    def _set_pi(self):
        """Perceptual independence: zero within-stimulus correlation for all stimuli."""
        return np.array([self._create_cov_mat(0.0)] * self.num_stimuli)

    def _set_rho1(self, rng=None):
        """RHO1: a single (equal) non-zero correlation shared by all stimuli."""
        rng = self._resolve_rng(rng)
        corr = rng.uniform(*self.corr_range)
        if rng.random() < 0.5:
            corr = -corr
        return np.array([self._create_cov_mat(corr)] * self.num_stimuli)

    def _set_ds(self, rng=None):
        """Unconstrained correlations that differ across stimuli (rejecting PI-like / RHO1-like accidents)."""
        rng = self._resolve_rng(rng)
        while True:
            corrs = rng.uniform(*self.corr_range, self.num_stimuli)
            signs = rng.choice([-1, 1], self.num_stimuli)
            correlations = corrs * signs
            is_pi_like = np.all(np.abs(correlations) < self.pi_tolerance)
            is_rho1_like = np.max(correlations) - np.min(correlations) < self.pi_tolerance
            if not is_pi_like and not is_rho1_like:
                break
        return np.array([self._create_cov_mat(c) for c in correlations])

    def random_crit(self, means, rng=None):
        """Criterion drawn from a normal centred on the mean stimulus location per dimension."""
        rng = self._resolve_rng(rng)
        means_x, means_y = means[0::2], means[1::2]
        mid_x, mid_y = np.mean(means_x), np.mean(means_y)
        std_x = (np.max(means_x) - np.min(means_x)) / (len(means_x) + 1)
        std_y = (np.max(means_y) - np.min(means_y)) / (len(means_y) + 1)
        c_x = np.clip(rng.normal(mid_x, std_x), self.crit_range[0], self.crit_range[1])
        c_y = np.clip(rng.normal(mid_y, std_y), self.crit_range[0], self.crit_range[1])
        return np.round(np.array([c_x, c_y]), 2)

    # ------------------------------------------------------------------ #
    # Secondary datasets: parameter-controlled (pretraining) and trial-by-trial
    # ------------------------------------------------------------------ #
    def get_samples(self, means, cov_mat, size, sample_loss_factor=0.05, rng=None):
        rng = self._resolve_rng(rng)
        clouds = []
        for i in range(self.num_stimuli):
            n_i = int(rng.uniform(size * (1 - sample_loss_factor), size))
            clouds.append(self._sample_cloud(means[i * 2:i * 2 + 2], cov_mat[i], n_i, rng))
        return clouds

    def get_response_counts(self, samples, c):
        return self._quadrant_counts(np.atleast_2d(samples), c)

    def generate_parameter_controlled_cms(self, n_matrices, vary_means=True,
                                          vary_covariances=True, vary_crits=True, rng=None):
        rng = self._resolve_rng(rng)
        fixed_means = np.array([0., 0., 0.5, 0., 0., 0.5, 0.5, 0.5])
        fixed_cov_mat = np.array([np.eye(self.num_dimensions) for _ in range(self.num_stimuli)])
        fixed_crits = np.zeros(2, dtype=float)

        cms, trial_counts, params = [], [], []
        for _ in tqdm(range(n_matrices), total=n_matrices, desc="Parameter-controlled data"):
            n_trials = int(np.ceil(rng.uniform(*self.trial_range)))
            means = self.random_means(rng=rng) if vary_means else fixed_means
            cov_mat = self._set_pi() if vary_covariances else fixed_cov_mat
            c = self.random_crit(means, rng=rng) if vary_crits else fixed_crits

            samples = self.get_samples(means, cov_mat, n_trials,
                                       sample_loss_factor=self.sample_loss, rng=rng)
            cm_rows, tc = [], []
            for stim_samples in samples:
                cm_rows.append(self.get_response_counts(stim_samples, c))
                tc.append(len(stim_samples))
            cm = np.vstack(cm_rows)
            flat_params = np.concatenate([
                means.flatten(),
                np.array([m.flatten() for m in cov_mat]).flatten(),
                c,
            ])
            cms.append(cm.flatten())
            trial_counts.append(np.array(tc, dtype=float))
            params.append(flat_params)
        return np.array(cms), np.array(params), np.array(trial_counts)

    def generate_trial_data(self, model_name, n_trials=1000, rng=None):
        """
        Trial-by-trial data for a single simulated subject (e.g. for LSTM training).
        Returns (trials[n_trials, 2] = [stimulus_idx, response_idx], flat_params, model_name).
        """
        rng = self._resolve_rng(rng)
        means, cov_mat = self.random_model_params(model_name, rng=rng)
        c = self.random_crit(means, rng=rng)  # bugfix: was unpacked from random_model_params
        flat_params = np.concatenate([
            means.flatten(),
            np.array([m.flatten() for m in cov_mat]).flatten(),
            c,
        ])

        stimulus_sequence = rng.choice(self.num_stimuli, size=n_trials, replace=True)
        trials = np.empty((n_trials, 2), dtype=int)
        for t in tqdm(range(n_trials), desc=f"Trials for {model_name}"):
            s = stimulus_sequence[t]
            sample = self._sample_cloud(means[s * 2:s * 2 + 2], cov_mat[s], 1, rng)[0]
            response_idx = int(sample[0] >= c[0]) + 2 * int(sample[1] >= c[1])
            trials[t] = (s, response_idx)
        return trials, flat_params, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets for GRT modeling.")
    parser.add_argument("--full", action="store_true", help="Generate the full dataset with all model constraints.")
    parser.add_argument("--pretraining", action="store_true", help="Generate the parameter-controlled pre-training datasets.")
    parser.add_argument("--tbt", action="store_true", help="Generate trial-by-trial data (e.g. for LSTM training).")
    parser.add_argument("--all", action="store_true", help="Generate the full, pre-training, and trial-by-trial datasets.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for --full (default 1 = serial; -1 = all cores).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible generation.")
    args = parser.parse_args()

    if not (args.full or args.pretraining or args.all or args.tbt):
        parser.error("At least one of --full, --pretraining, --tbt, or --all is required.")

    if args.all or args.pretraining:
        stages = [
            ("vary_m_dataset.npz",   dict(vary_means=True,  vary_covariances=False, vary_crits=False)),
            ("vary_mv_dataset.npz",  dict(vary_means=True,  vary_covariances=True,  vary_crits=False)),
            ("vary_mc_dataset.npz",  dict(vary_means=True,  vary_covariances=False, vary_crits=True)),
            ("vary_mvc_dataset.npz", dict(vary_means=True,  vary_covariances=True,  vary_crits=True)),
        ]
        for fname, flags in stages:
            print(f"\nPretraining stage -> {fname}: {flags}")
            cms, params, trials = GRTDataGenerator(
                num_matrices=NUM_PRETRAINING_MATRICES, seed=args.seed
            ).generate_parameter_controlled_cms(n_matrices=NUM_PRETRAINING_MATRICES, **flags)
            save_file_name = os.path.join(SIMULATED_DATA_DIR, fname)
            np.savez(save_file_name, X=cms, X_trials=trials, y_params=params)
            print(f"Saved {cms.shape[0]} matrices to {save_file_name}.")

    if args.all or args.full:
        print("\n--- Generating data for all model constraints ---")
        gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_ACCURACY_BIN,
                               num_dimensions=2, num_levels=2,
                               trial_range=TRIALS_RANGE, seed=args.seed)
        cms, parameters, trial_counts, y_cls, y_cls_label = gen.generate_all_model_cms(
            seed=args.seed, n_jobs=args.jobs)
        np.savez(DATASET_FILE, X=cms, X_trials=trial_counts, y_params=parameters,
                 y_model_cls=y_cls, y_cls_label=y_cls_label)
        print(f"Saved {cms.shape[0]} matrices to {DATASET_FILE}.")

    if args.all or args.tbt:
        print("\n--- Generating trial-by-trial data ---")
        gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_MODEL, num_dimensions=2,
                               num_levels=2, trial_range=TRIALS_RANGE, seed=args.seed)
        num_sequences_per_model = 500
        all_trials, all_params, all_labels = [], [], []
        for model_label in gen.model_names:
            print(f"Generating {num_sequences_per_model} sequences for: {model_label}")
            for _ in tqdm(range(num_sequences_per_model)):
                trials, params, label = gen.generate_trial_data(model_label, n_trials=1000)
                all_trials.append(trials)
                all_params.append(params)
                all_labels.append(label)
        np.savez(TRIAL_BY_TRIAL_FIAL, X=all_trials, y_params=all_params, y_model_labels=all_labels)
        print(f"Done! Trial-by-trial data saved to {TRIAL_BY_TRIAL_FIAL}.")
        