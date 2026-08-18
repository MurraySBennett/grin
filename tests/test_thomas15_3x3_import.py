from pathlib import Path

import pandas as pd


PATH = Path(__file__).parents[1] / "data" / "real" / "thomas15_3x3.csv"


def test_thomas_export_has_canonical_stimulus_order_and_provenance():
    data = pd.read_csv(PATH)
    assert list(data.columns) == [
        "dataset", "stimulus", "source_position", "source_label",
        "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33",
    ]
    expected_positions = [1, 4, 7, 2, 5, 8, 3, 6, 9]
    for dataset in ("thomas15a", "thomas15b"):
        observer = data[data.dataset == dataset]
        assert observer.stimulus.tolist() == list(range(1, 10))
        assert observer.source_label.tolist() == list(range(1, 10))
        assert observer.source_position.tolist() == expected_positions


def test_thomas_export_is_balanced_and_has_expected_total():
    data = pd.read_csv(PATH)
    response_columns = [f"r{a}{b}" for a in range(1, 4) for b in range(1, 4)]
    assert (data[response_columns].sum(axis=1) == 80).all()
    assert data.groupby("dataset")[response_columns].sum().sum(axis=1).to_dict() == {
        "thomas15a": 720,
        "thomas15b": 720,
    }


def test_known_cells_guard_against_response_or_stimulus_transpose():
    data = pd.read_csv(PATH).set_index(["dataset", "stimulus"])
    # These sentinels are taken directly from the labeled R xtabs. Together with
    # source_position they fail if either the stimulus grid or response grid is
    # flattened in column-major order by accident.
    assert data.loc[("thomas15a", 1), ["r11", "r12", "r21"]].tolist() == [23, 23, 9]
    assert data.loc[("thomas15a", 2), ["r11", "r12", "r21"]].tolist() == [3, 22, 0]
    assert data.loc[("thomas15b", 1), ["r11", "r12", "r21"]].tolist() == [15, 7, 17]
