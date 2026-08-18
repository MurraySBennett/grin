import numpy as np

from src import grt_model_3x3 as gm
from src.data.generator_3x3 import GRT3x3DataGenerator, GRT3x3HeteroDataGenerator


def test_generate_shapes_and_row_totals():
    generator = GRT3x3DataGenerator(n_per_class=3, trial_range=(20, 50), seed=3)
    X, y, trials, classes, labels = generator.generate_all_model_cms()
    n = 3 * len(gm.MODEL_NAMES)
    assert X.shape == (n, 81)
    assert y.shape == (n, 29)
    assert trials.shape == (n, 9)
    assert classes.shape == labels.shape == (n,)
    assert np.array_equal(X.reshape(n, 9, 9).sum(axis=-1), trials)


def test_generation_is_reproducible():
    kwargs = dict(n_per_class=2, trial_range=(10, 30), seed=12)
    first = GRT3x3DataGenerator(**kwargs).generate_all_model_cms()
    second = GRT3x3DataGenerator(**kwargs).generate_all_model_cms()
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_heteroscedastic_generator_shapes_and_row_totals():
    generator = GRT3x3HeteroDataGenerator(n_per_class=2, trial_range=(20, 50), seed=4)
    X, y, trials, classes, labels = generator.generate_all_model_cms()
    n = 2 * len(gm.MODEL_NAMES)
    assert X.shape == (n, 81)
    assert y.shape == (n, 45)
    assert trials.shape == (n, 9)
    assert classes.shape == labels.shape == (n,)
    assert np.array_equal(X.reshape(n, 9, 9).sum(axis=-1), trials)


def test_square_featurizer_supports_3x3_without_changing_2x2_contract():
    import pytest
    torch = pytest.importorskip("torch")
    from src.models.network import featurize_square
    counts_3 = torch.ones((2, 81))
    trials_3 = torch.full((2, 9), 9)
    assert featurize_square(counts_3, trials_3, 9).shape == (2, 90)

    counts_2 = torch.ones((2, 16))
    trials_2 = torch.full((2, 4), 4)
    assert featurize_square(counts_2, trials_2, 4).shape == (2, 20)
