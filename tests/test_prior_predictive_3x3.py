import numpy as np

from scripts.prior_predictive_3x3 import information_regime, matrix_metrics, sd_sparsity_table
from src import grt_model_3x3_hetero as gm
from src.data.generator_3x3 import GRT3x3HeteroDataGenerator


def test_matrix_metrics_on_perfect_identification():
    counts = np.zeros((1, 9, 9), dtype=int)
    counts[0, np.arange(9), np.arange(9)] = 80
    metrics = matrix_metrics(counts).iloc[0]
    assert metrics.accuracy == 1
    assert metrics.mean_row_entropy == 0
    assert metrics.empty_cell_fraction == 8 / 9


def test_sd_sparsity_table_accounts_for_every_dimension_and_stimulus():
    generator = GRT3x3HeteroDataGenerator(
        n_per_class=2, trial_range=(80, 80), balanced_trials=True, seed=9
    )
    X, y, *_ = generator.generate_all_model_cms()
    table = sd_sparsity_table(X, y)
    assert table.n.sum() == len(X) * 9 * 2
    assert table.sparse_rate.between(0, 1).all()
    assert y.shape[1] == gm.N_PARAMS


def test_information_strata_keep_the_two_thomas_regimes_distinct():
    assert information_regime(0.403, 1.550) == "moderate_information"
    assert information_regime(0.272, 1.827) == "low_information"
    assert information_regime(0.70, 0.90) == "high_information"
