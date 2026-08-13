"""Tests for grintools.plot. Requires the [plot] extra: pip install grintools[plot]."""
import matplotlib
matplotlib.use("Agg")  # headless, no display needed for tests
import matplotlib.pyplot as plt
import pytest

import grintools as gt

plot = pytest.importorskip("grintools.plot")

M1 = [[71, 17, 9, 5], [20, 67, 5, 9], [13, 6, 63, 20], [5, 10, 15, 71]]
M2 = [[50, 10, 15, 25], [12, 55, 20, 13], [18, 22, 48, 12], [8, 14, 18, 60]]


@pytest.fixture
def out1():
    return gt.infer(M1)


@pytest.fixture
def out2():
    return gt.infer(M2)


@pytest.fixture
def many(out1, out2):
    return [out1, out2]


def test_tidy_single_and_many(out1, many):
    df1 = plot.tidy(out1)
    assert len(df1) == 1
    assert df1["id"].iloc[0] == "p1"

    df = plot.tidy(many)
    assert len(df) == 2
    assert list(df["id"]) == ["p1", "p2"]
    for col in ("model_class", "zx_0", "rho_3", "p_PI", "evidence_sep_A"):
        assert col in df.columns


def test_tidy_custom_ids(many):
    df = plot.tidy(many, ids=["alice", "bob"])
    assert list(df["id"]) == ["alice", "bob"]


def test_tidy_rejects_mismatched_ids(many):
    with pytest.raises(ValueError):
        plot.tidy(many, ids=["only_one"])


def test_individual_plots_build_and_render(out1):
    for fig_or_ax in [
        plot.plot_space(out1[0]),
        plot.plot_params(out1[0]),
        plot.plot_constructs(out1[0], out1[1]),
    ]:
        fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
        fig.canvas.draw()  # forces a real render pass, not just object construction
        plt.close(fig)


def test_group_plots_build_and_render(many):
    outputs = [
        plot.plot_space_group(many, facet=True),
        plot.plot_space_group(many, facet=False),
        plot.plot_params_group(many),
        plot.plot_model_classes(many),
        plot.plot_precision_group(many),
    ]
    for fig_or_ax in outputs:
        fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
        fig.canvas.draw()
        plt.close(fig)


def test_plot_constructs_flags_insufficient_evidence(out1):
    stub_constructs = {
        "p_PI": 0.52, "p_sep_A": 0.97, "p_sep_B": 0.10,
        "p_corr": [0.52, 0.30, 0.18],
        "evidence_PI": False, "evidence_sep_A": True, "evidence_sep_B": True,
    }
    ax = plot.plot_constructs(out1[0], stub_constructs)
    texts = [t.get_text() for t in ax.texts]
    assert any("insufficient" in t for t in texts)
    plt.close(ax.figure)


def test_long_params_reshapes_wide_to_long(many):
    df = plot.tidy(many)
    long = plot._long_params(df)
    assert len(long) == 12 * 2  # 12 params x 2 participants
    for col in ("id", "group", "param", "estimate", "sd"):
        assert col in long.columns
    row = long[(long["id"] == "p1") & (long["param"] == "zx_0")]
    assert row["estimate"].iloc[0] == pytest.approx(many[0][0].params[0])
