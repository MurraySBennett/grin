import numpy as np
import pytest

from src.data.rt_dynamic_grt import (
    ARCHITECTURES,
    DynamicRTParameters,
    _first_passage_1d,
    sample_dynamic_rt_parameters,
    sample_latent_drifts,
    simulate_dynamic_grt_trials,
)


def test_latent_drift_sampler_recovers_requested_moments():
    draws = sample_latent_drifts(0.4, -0.7, 0.55, 80_000, np.random.default_rng(1))
    assert draws[:, 0].mean() == pytest.approx(0.4, abs=0.015)
    assert draws[:, 1].mean() == pytest.approx(-0.7, abs=0.015)
    assert draws[:, 0].std() == pytest.approx(1.0, abs=0.015)
    assert draws[:, 1].std() == pytest.approx(1.0, abs=0.015)
    assert np.corrcoef(draws.T)[0, 1] == pytest.approx(0.55, abs=0.015)


def test_deterministic_first_passage_matches_boundary_over_drift():
    response, time, censored = _first_passage_1d(
        np.array([2.0, -4.0]),
        boundary=1.0,
        dt=0.001,
        max_internal_time=2.0,
        rng=np.random.default_rng(2),
        diffusion_sd=0.0,
    )
    assert np.array_equal(response, np.array([1, 0]))
    assert time == pytest.approx(np.array([0.5, 0.25]), abs=0.001)
    assert not np.any(censored)


def test_serial_and_parallel_share_responses_but_combine_time_differently():
    parameters = DynamicRTParameters(t0=0.25, boundary=0.9, rate=2.5)
    serial = simulate_dynamic_grt_trials(
        0.5,
        -0.25,
        0.4,
        500,
        "serial_exhaustive",
        parameters,
        np.random.default_rng(3),
    )
    parallel = simulate_dynamic_grt_trials(
        0.5,
        -0.25,
        0.4,
        500,
        "parallel_exhaustive",
        parameters,
        np.random.default_rng(3),
    )

    assert np.array_equal(serial.latent_drift, parallel.latent_drift)
    assert np.array_equal(serial.channel_response, parallel.channel_response)
    assert np.array_equal(serial.response, parallel.response)
    assert np.array_equal(serial.censored, parallel.censored)
    assert serial.channel_time_internal == pytest.approx(parallel.channel_time_internal)
    assert np.all(serial.rt[~serial.censored] >= parallel.rt[~parallel.censored])
    assert np.any(serial.rt[~serial.censored] > parallel.rt[~parallel.censored])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_response_codes_and_rts_are_valid_without_clipping(architecture):
    result = simulate_dynamic_grt_trials(
        0.0,
        0.0,
        -0.3,
        1_000,
        architecture,
        DynamicRTParameters(t0=0.2, boundary=1.0, rate=2.0),
        np.random.default_rng(4),
    )
    complete = ~result.censored
    assert complete.mean() > 0.999
    assert set(np.unique(result.response[complete])).issubset({0, 1, 2, 3})
    assert np.all(np.isfinite(result.rt[complete]))
    assert np.all(result.rt[complete] > 0.2)
    assert not np.any(result.rt[complete] == 10.0)


def test_censoring_is_explicit_not_a_horizon_point_mass():
    result = simulate_dynamic_grt_trials(
        0.0,
        0.0,
        0.0,
        50,
        "parallel_exhaustive",
        DynamicRTParameters(t0=0.2, boundary=10.0, rate=1.0),
        np.random.default_rng(5),
        max_internal_time=0.01,
    )
    assert np.all(result.censored)
    assert np.all(result.response == -1)
    assert np.all(np.isnan(result.rt))


def test_more_positive_drift_increases_positive_responses_and_speeds_decisions():
    parameters = DynamicRTParameters(t0=0.2, boundary=1.0, rate=2.0)
    low = simulate_dynamic_grt_trials(
        -1.5,
        0.5,
        0.0,
        4_000,
        "parallel_exhaustive",
        parameters,
        np.random.default_rng(6),
    )
    high = simulate_dynamic_grt_trials(
        1.5,
        0.5,
        0.0,
        4_000,
        "parallel_exhaustive",
        parameters,
        np.random.default_rng(7),
    )
    low_complete = ~low.censored
    high_complete = ~high.censored
    low_positive_x = low.channel_response[low_complete, 0].mean()
    high_positive_x = high.channel_response[high_complete, 0].mean()
    assert high_positive_x > low_positive_x + 0.5
    assert np.median(high.channel_time_internal[high_complete, 0]) < np.median(
        low.channel_time_internal[low_complete, 0]
    )


def test_pilot_parameter_sampler_is_reproducible_and_in_range():
    first = sample_dynamic_rt_parameters(np.random.default_rng(8), 100)
    second = sample_dynamic_rt_parameters(np.random.default_rng(8), 100)
    assert np.array_equal(first, second)
    assert np.all((first[:, 0] >= 0.15) & (first[:, 0] <= 0.45))
    assert np.all((first[:, 1] >= 0.75) & (first[:, 1] <= 1.50))
    assert np.all((first[:, 2] >= 1.0) & (first[:, 2] <= 6.0))


@pytest.mark.parametrize(
    "parameters",
    [
        DynamicRTParameters(t0=-0.1, boundary=1.0, rate=1.0),
        DynamicRTParameters(t0=0.1, boundary=0.0, rate=1.0),
        DynamicRTParameters(t0=0.1, boundary=1.0, rate=0.0),
    ],
)
def test_invalid_dynamic_parameters_fail_loudly(parameters):
    with pytest.raises(ValueError):
        parameters.validate()

