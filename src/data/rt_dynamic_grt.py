"""Reference simulator for the dynamic-GRT response-time model.

This module is deliberately a small, readable scientific reference rather than
the high-volume training generator.  It implements the model recorded in
``docs/dynamic_grt_rt_design.md`` and is the implementation against which a
later vectorised/GPU generator must be checked.

The latent GRT distribution is a bivariate-normal distribution of trial-level
drift rates.  Conditional on those drifts, two independent Wiener processes
make the dimensional decisions.  Serial- and parallel-exhaustive architectures
use the same dimensional outcomes and differ only in whether their first-passage
times are summed or maximised.

This is not an LBA.  It has within-trial stochastic accumulation, symmetric
absorbing boundaries, and no race between response-specific ballistic
accumulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Architecture = Literal["serial_exhaustive", "parallel_exhaustive"]
ARCHITECTURES: tuple[Architecture, ...] = (
    "serial_exhaustive",
    "parallel_exhaustive",
)
DYNAMIC_PARAMETER_NAMES = ("t0", "boundary", "rate")


@dataclass(frozen=True)
class DynamicRTParameters:
    """Participant-level dynamic decision parameters.

    ``boundary`` is the symmetric absorbing boundary in internal evidence units.
    ``rate`` converts internal diffusion time to seconds.  The within-trial
    diffusion coefficient is fixed to one to identify the scale.
    """

    t0: float
    boundary: float
    rate: float

    def validate(self) -> None:
        values = np.asarray((self.t0, self.boundary, self.rate), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("dynamic RT parameters must be finite")
        if self.t0 < 0:
            raise ValueError("t0 must be nonnegative")
        if self.boundary <= 0:
            raise ValueError("boundary must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")


@dataclass(frozen=True)
class DynamicGRTTrials:
    """Raw joint outcomes from one stimulus condition.

    Censored trials have ``response == -1`` and ``rt == NaN``.  Their channel
    fields retain any channel that did finish, making the numerical failure
    explicit rather than silently clipping it into the RT distribution.
    """

    response: np.ndarray
    rt: np.ndarray
    channel_response: np.ndarray
    channel_time_internal: np.ndarray
    latent_drift: np.ndarray
    censored: np.ndarray
    architecture: Architecture


def sample_dynamic_rt_parameters(
    rng: np.random.Generator,
    n: int = 1,
    *,
    t0_range: tuple[float, float] = (0.15, 0.45),
    boundary_range: tuple[float, float] = (0.75, 1.50),
    rate_range: tuple[float, float] = (1.0, 6.0),
) -> np.ndarray:
    """Draw provisional nuisance parameters for prior-predictive auditing.

    The rate prior is log-uniform because it is a positive scale parameter.
    These ranges are *pilot priors*, not production defaults; the prior-
    predictive audit is expected to change them.
    """

    n = int(n)
    if n < 1:
        raise ValueError("n must be at least one")
    for label, pair in (
        ("t0", t0_range),
        ("boundary", boundary_range),
        ("rate", rate_range),
    ):
        lo, hi = (float(v) for v in pair)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"invalid {label} range: {pair}")
        if label != "t0" and lo <= 0:
            raise ValueError(f"{label} range must be positive")
        if label == "t0" and lo < 0:
            raise ValueError("t0 range must be nonnegative")

    t0 = rng.uniform(*t0_range, size=n)
    boundary = rng.uniform(*boundary_range, size=n)
    rate = np.exp(rng.uniform(np.log(rate_range[0]), np.log(rate_range[1]), size=n))
    return np.column_stack((t0, boundary, rate))


def sample_latent_drifts(
    zx: float,
    zy: float,
    rho: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw trial-level bivariate-normal drifts with unit marginal variance."""

    values = np.asarray((zx, zy, rho), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("zx, zy, and rho must be finite")
    if abs(rho) >= 1:
        raise ValueError("rho must lie strictly between -1 and 1")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least one")

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    return np.column_stack(
        (
            zx + z1,
            zy + rho * z1 + np.sqrt(1.0 - rho**2) * z2,
        )
    )


def _first_passage_1d(
    drift: np.ndarray,
    *,
    boundary: float,
    dt: float,
    max_internal_time: float,
    rng: np.random.Generator,
    diffusion_sd: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euler-Maruyama first passage to symmetric boundaries for many drifts.

    Returns ``(response, time, censored)``.  Response is zero for the negative
    boundary, one for the positive boundary, and -1 for a censored path.
    ``diffusion_sd`` is fixed to one in the scientific model; exposing it here
    permits deterministic limiting-case and convergence tests.
    """

    drift = np.asarray(drift, dtype=float)
    if drift.ndim != 1 or not np.all(np.isfinite(drift)):
        raise ValueError("drift must be a finite one-dimensional array")
    if boundary <= 0 or dt <= 0 or max_internal_time <= 0:
        raise ValueError("boundary, dt, and max_internal_time must be positive")
    if diffusion_sd < 0 or not np.isfinite(diffusion_sd):
        raise ValueError("diffusion_sd must be finite and nonnegative")

    n = drift.size
    evidence = np.zeros(n, dtype=float)
    response = np.full(n, -1, dtype=np.int8)
    time = np.full(n, np.nan, dtype=float)
    active = np.ones(n, dtype=bool)
    max_steps = int(np.ceil(max_internal_time / dt))
    noise_scale = diffusion_sd * np.sqrt(dt)

    for step in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        previous = evidence[idx]
        increment = drift[idx] * dt
        if diffusion_sd:
            increment = increment + noise_scale * rng.standard_normal(idx.size)
        updated = previous + increment
        evidence[idx] = updated

        crossed = np.abs(updated) >= boundary
        if not np.any(crossed):
            continue

        crossed_idx = idx[crossed]
        previous_crossed = previous[crossed]
        updated_crossed = updated[crossed]
        positive = updated_crossed >= 0
        target = np.where(positive, boundary, -boundary)
        denominator = updated_crossed - previous_crossed
        fraction = np.divide(
            target - previous_crossed,
            denominator,
            out=np.ones_like(target),
            where=denominator != 0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)

        response[crossed_idx] = positive.astype(np.int8)
        time[crossed_idx] = (step + fraction) * dt
        active[crossed_idx] = False

    return response, time, active


def simulate_dynamic_grt_trials(
    zx: float,
    zy: float,
    rho: float,
    n: int,
    architecture: Architecture,
    parameters: DynamicRTParameters,
    rng: np.random.Generator,
    *,
    dt: float = 0.0025,
    max_internal_time: float = 25.0,
    diffusion_sd: float = 1.0,
) -> DynamicGRTTrials:
    """Simulate one stimulus condition under the dynamic-GRT model.

    The same latent representation and channel processes are used for both
    architectures.  Architecture is applied only after both dimensional first
    passages have been generated.
    """

    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    parameters.validate()

    latent = sample_latent_drifts(zx, zy, rho, n, rng)
    channel_response = np.full((int(n), 2), -1, dtype=np.int8)
    channel_time = np.full((int(n), 2), np.nan, dtype=float)
    channel_censored = np.ones((int(n), 2), dtype=bool)

    for dimension in range(2):
        response_d, time_d, censored_d = _first_passage_1d(
            latent[:, dimension],
            boundary=parameters.boundary,
            dt=dt,
            max_internal_time=max_internal_time,
            rng=rng,
            diffusion_sd=diffusion_sd,
        )
        channel_response[:, dimension] = response_d
        channel_time[:, dimension] = time_d
        channel_censored[:, dimension] = censored_d

    censored = np.any(channel_censored, axis=1)
    response = np.full(int(n), -1, dtype=np.int8)
    completed = ~censored
    response[completed] = (
        2 * channel_response[completed, 0] + channel_response[completed, 1]
    ).astype(np.int8)

    rt = np.full(int(n), np.nan, dtype=float)
    if architecture == "serial_exhaustive":
        decision_time = np.sum(channel_time[completed], axis=1)
    else:
        decision_time = np.max(channel_time[completed], axis=1)
    rt[completed] = parameters.t0 + decision_time / parameters.rate

    return DynamicGRTTrials(
        response=response,
        rt=rt,
        channel_response=channel_response,
        channel_time_internal=channel_time,
        latent_drift=latent,
        censored=censored,
        architecture=architecture,
    )

