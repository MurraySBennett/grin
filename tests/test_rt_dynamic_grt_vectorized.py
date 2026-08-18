"""Correctness check for the vectorised/GPU dynamic-GRT generator: it must reproduce
the validated scalar reference simulator's distribution (response proportions AND RT
quantiles), not just run fast. This is this module's own "gate 1" -- the fast
implementation is only trustworthy insofar as it matches the slow, already-tested one.
"""
import numpy as np
import pytest
import torch

from src.data.rt_dynamic_grt import DynamicRTParameters, simulate_dynamic_grt_trials
from src.data.rt_dynamic_grt_vectorized import QUANTILES, DynamicGRTVectorizedGenerator


def _run_vectorized_single_condition(zx, zy, rho, t0, boundary, rate, arch_id, n, seed, device="cpu"):
    gen = DynamicGRTVectorizedGenerator(dt=0.0025, max_internal_time=25.0, device=device)
    zx_arr = np.full((1, 4), zx)
    zy_arr = np.full((1, 4), zy)
    rho_arr = np.full((1, 4), rho)
    dynamic_params = np.array([[t0, boundary, rate]])
    arch_arr = np.array([arch_id])
    n_per = np.full((1, 4), n, dtype=np.int64)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)
    counts, rtq, censor_rate = gen._build_examples(
        zx_arr, zy_arr, rho_arr, dynamic_params, arch_arr, n_per, torch_gen)
    return counts[0, 0], rtq[0, 0], censor_rate[0, 0]        # stimulus 0 (all 4 identical here)


@pytest.mark.parametrize("architecture,arch_id", [("serial_exhaustive", 0), ("parallel_exhaustive", 1)])
def test_vectorized_matches_scalar_response_and_rt_distribution(architecture, arch_id):
    zx, zy, rho = 0.6, -0.4, 0.3
    params = DynamicRTParameters(t0=0.2, boundary=1.0, rate=2.0)
    n = 300_000

    scalar = simulate_dynamic_grt_trials(zx, zy, rho, n, architecture, params, np.random.default_rng(11))
    counts, rtq, censor_rate = _run_vectorized_single_condition(
        zx, zy, rho, params.t0, params.boundary, params.rate, arch_id, n, seed=11)

    scalar_complete = ~scalar.censored
    scalar_probs = np.bincount(scalar.response[scalar_complete], minlength=4) / scalar_complete.sum()
    vec_probs = counts / counts.sum()

    assert vec_probs == pytest.approx(scalar_probs, abs=0.01)
    assert censor_rate < 0.01

    modal = int(np.argmax(counts))
    scalar_rt_modal = scalar.rt[scalar_complete & (scalar.response == modal)]
    scalar_quantiles = np.quantile(scalar_rt_modal, QUANTILES)
    assert rtq[modal] == pytest.approx(scalar_quantiles, abs=0.02)


def test_architectures_share_identical_response_counts_when_identically_seeded():
    """Direct vectorised analogue of
    test_serial_and_parallel_share_responses_but_combine_time_differently: same channel
    draws (same seed) must give the same response counts under both architectures, and
    the serial RT quantiles must be >= the parallel ones."""
    zx, zy, rho = -0.3, 0.8, -0.5
    params = (0.25, 0.9, 2.5)
    n = 50_000

    counts_serial, rtq_serial, _ = _run_vectorized_single_condition(zx, zy, rho, *params, 0, n, seed=99)
    counts_parallel, rtq_parallel, _ = _run_vectorized_single_condition(zx, zy, rho, *params, 1, n, seed=99)

    assert np.array_equal(counts_serial, counts_parallel)
    has_both = (counts_serial > 0)
    assert np.all(rtq_serial[has_both] >= rtq_parallel[has_both] - 1e-9)
    assert np.any(rtq_serial[has_both] > rtq_parallel[has_both] + 1e-6)


def test_censoring_reduces_effective_count_without_being_clipped():
    """A boundary far larger than the simulation horizon can reach forces near-total
    censoring; the response-cell counts must drop accordingly (never backfilled with a
    placeholder response or an RT pinned to the horizon)."""
    gen = DynamicGRTVectorizedGenerator(dt=0.01, max_internal_time=0.02, device="cpu")
    zx_arr = np.zeros((1, 4))
    zy_arr = np.zeros((1, 4))
    rho_arr = np.zeros((1, 4))
    dynamic_params = np.array([[0.2, 10.0, 1.0]])
    arch_arr = np.array([0])
    n_per = np.full((1, 4), 500, dtype=np.int64)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(7)
    counts, rtq, censor_rate = gen._build_examples(
        zx_arr, zy_arr, rho_arr, dynamic_params, arch_arr, n_per, torch_gen)

    assert np.all(censor_rate > 0.99)
    assert counts.sum() < 5           # almost nothing answered
    assert np.all(np.isfinite(rtq))   # no inf/nan leaked into the quantile array


def test_generate_end_to_end_smoke():
    """Full generate() path (all 12 model classes, chunking) at pilot scale runs and
    returns internally-consistent shapes with X_trials matching administered trials."""
    gen = DynamicGRTVectorizedGenerator(n_per_class=25, trial_range=(20, 200), chunk=7,
                                         dt=0.005, max_internal_time=15.0, device="cpu")
    X, RTQ, X_trials, censor_rate, y_params, y_dynamic, y_cls, y_label, y_arch = gen.generate(
        seed=123, verbose=False)

    n = 25 * 12
    assert X.shape == (n, 16)
    assert RTQ.shape == (n, 80)
    assert X_trials.shape == (n, 4)
    assert censor_rate.shape == (n, 4)
    assert y_params.shape == (n, 12)
    assert y_dynamic.shape == (n, 3)
    assert set(np.unique(y_arch)).issubset({0, 1})
    assert np.all((y_dynamic[:, 1] > 0) & (y_dynamic[:, 2] > 0))          # boundary, rate positive
    # every stimulus's answered counts + implied censored trials <= administered trials
    answered = X.reshape(n, 4, 4).sum(-1)
    assert np.all(answered <= X_trials + 1)   # +1 for rounding slack in censor_rate*n_per
