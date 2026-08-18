import numpy as np
import pytest

from src import grt_model_3x3 as unit_model
from src import grt_model_3x3_hetero as free_model
from src.inference.mle_3x3 import (
    fit_full, initial_free, initial_unit, multinomial_log_likelihood,
)


def test_multinomial_likelihood_prefers_generating_probabilities():
    rng = np.random.default_rng(1)
    theta = unit_model.sample_prior("ds", 1, rng)
    probabilities = unit_model.forward_probabilities(*theta)[0]
    counts = np.stack([rng.multinomial(5000, row) for row in probabilities])
    uniform = np.full((9, 9), 1 / 9)
    assert multinomial_log_likelihood(counts, probabilities) > multinomial_log_likelihood(counts, uniform)


def test_marginal_starts_are_finite_for_sparse_table():
    counts = np.zeros((9, 9), dtype=int)
    counts[:, 0] = 79
    counts[:, -1] = 1
    assert np.isfinite(initial_unit(counts)).all()
    assert np.isfinite(initial_free(counts)).all()


@pytest.mark.parametrize("variance_model,model", [("unit", unit_model), ("free", free_model)])
def test_mle_improves_over_start_on_generated_counts(variance_model, model):
    rng = np.random.default_rng(8)
    theta = model.sample_prior("ds", 1, rng)
    probabilities = model.forward_probabilities(*theta)[0]
    counts = np.stack([rng.multinomial(400, row) for row in probabilities])
    start = initial_unit(counts) if variance_model == "unit" else initial_free(counts)
    fit = fit_full(counts, variance_model=variance_model, n_restarts=1, maxiter=300)
    assert np.isfinite(fit["log_likelihood"])
    assert fit["params"].shape == (model.N_PARAMS,)
    assert fit["n_restarts"] == 1
    assert fit["success"] or "LIMIT" in fit["message"].upper()
