"""Unit tests for src/data/generator.py's GRTDataGenerator."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import grt_model as gm
from src.data.generator import GRTDataGenerator


def test_generate_all_model_cms_shapes():
    gen = GRTDataGenerator(n_per_class=10, trial_range=(20, 200), seed=0)
    X, y_params, X_trials, y_cls, y_label = gen.generate_all_model_cms()
    n = 10 * len(gm.MODEL_NAMES)
    assert X.shape == (n, 16)
    assert y_params.shape == (n, 12)
    assert X_trials.shape == (n, 4)
    assert y_cls.shape == (n,)
    assert y_label.shape == (n,)


def test_counts_sum_to_trials():
    gen = GRTDataGenerator(n_per_class=20, trial_range=(10, 300), seed=1)
    X, _, X_trials, _, _ = gen.generate_all_model_cms()
    row_sums = X.reshape(-1, 4, 4).sum(axis=-1)
    assert np.array_equal(row_sums, X_trials)


def test_counts_are_nonnegative_integers():
    gen = GRTDataGenerator(n_per_class=20, trial_range=(5, 500), seed=2)
    X, _, _, _, _ = gen.generate_all_model_cms()
    assert X.dtype.kind in "iu"
    assert np.all(X >= 0)


def test_labels_match_model_names_in_order():
    gen = GRTDataGenerator(n_per_class=5, seed=3)
    _, _, _, y_cls, y_label = gen.generate_all_model_cms()
    for cls_idx, name in enumerate(gm.MODEL_NAMES):
        mask = y_cls == cls_idx
        assert mask.sum() == 5
        assert np.all(y_label[mask] == name)


def test_balanced_trials_gives_equal_per_stimulus_counts():
    gen = GRTDataGenerator(n_per_class=10, trial_range=(50, 500), seed=4, balanced_trials=True)
    _, _, X_trials, _, _ = gen.generate_all_model_cms()
    assert np.all(X_trials == X_trials[:, :1])  # every row's 4 stimuli agree


def test_imbalance_zero_is_equivalent_to_balanced_trials():
    a = GRTDataGenerator(n_per_class=10, trial_range=(50, 500), seed=5, imbalance=0.0)
    b = GRTDataGenerator(n_per_class=10, trial_range=(50, 500), seed=5, balanced_trials=True)
    _, _, Xt_a, _, _ = a.generate_all_model_cms()
    _, _, Xt_b, _, _ = b.generate_all_model_cms()
    assert np.array_equal(Xt_a, Xt_b)


def test_imbalance_bounds_per_stimulus_attrition():
    """Each stimulus keeps a fraction in [1 - imbalance, 1] of the per-matrix base
    count -- so no stimulus should ever fall below that floor (up to rounding)."""
    imbalance = 0.35
    gen = GRTDataGenerator(n_per_class=200, trial_range=(100, 100), seed=6, imbalance=imbalance)
    _, _, X_trials, _, _ = gen.generate_all_model_cms()
    floor = 100 * (1 - imbalance) - 1  # -1 slack for rounding
    assert np.all(X_trials >= floor)
    assert np.all(X_trials <= 100)


def test_same_seed_is_reproducible():
    a = GRTDataGenerator(n_per_class=15, trial_range=(20, 200), seed=42)
    b = GRTDataGenerator(n_per_class=15, trial_range=(20, 200), seed=42)
    Xa, ya, Xta, _, _ = a.generate_all_model_cms()
    Xb, yb, Xtb, _, _ = b.generate_all_model_cms()
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(ya, yb)
    assert np.array_equal(Xta, Xtb)


def test_different_seeds_differ():
    a = GRTDataGenerator(n_per_class=15, trial_range=(20, 200), seed=1)
    b = GRTDataGenerator(n_per_class=15, trial_range=(20, 200), seed=2)
    Xa, *_ = a.generate_all_model_cms()
    Xb, *_ = b.generate_all_model_cms()
    assert not np.array_equal(Xa, Xb)


def test_trial_range_is_respected():
    lo, hi = 30, 60
    gen = GRTDataGenerator(n_per_class=50, trial_range=(lo, hi), seed=7, imbalance=0.0)
    _, _, X_trials, _, _ = gen.generate_all_model_cms()
    # log-uniform base draw should stay within [lo, hi] (balanced, so exact)
    assert X_trials.min() >= lo
    assert X_trials.max() <= hi


@pytest.mark.parametrize("name", gm.MODEL_NAMES)
def test_expected_proportions_match_forward_model(name):
    """Statistical check: averaged over many draws, the empirical response
    proportions should track forward_probabilities (the generator's own claimed
    generative process), well within multinomial sampling noise."""
    rng = np.random.default_rng(8)
    zx, zy, rho = gm.sample_prior(name, 1, rng)
    probs = gm.forward_probabilities(zx, zy, rho)[0]  # (4,4)

    gen = GRTDataGenerator(n_per_class=1, trial_range=(2000, 2000), seed=9, imbalance=0.0)
    counts = gen._multinomial_counts(probs[None], np.full((1, 4), 2000), np.random.default_rng(9))[0]
    empirical = counts / counts.sum(axis=-1, keepdims=True)
    assert empirical == pytest.approx(probs, abs=0.03)  # generous: n=2000/cell, 4-way multinomial
