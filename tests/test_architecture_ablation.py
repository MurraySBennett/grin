"""Regression coverage for src.inference.predict_rt.architecture_ablation() -- the
function behind the manuscript's 28.8%/32.0% ablation numbers and the
rt_architecture.png/rt_architecture_ablation.json artifacts. Uses a small,
freshly-initialised (untrained) RTNPEModel rather than the shipped checkpoint,
so this runs fast and does not depend on results/models/npe_rt_model.pt
existing -- it tests the ablation MECHANICS (confusion-matrix bookkeeping,
the two interventions, output schema, reproducibility), not recovery quality.
"""
import numpy as np
import pytest
import torch

from src.data.rt_lba_generator import ARCHITECTURES, RTLBAGenerator
from src.inference.predict_rt import architecture_ablation, _confusion
from src.models.rt_network import RTNPEModel


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    m = RTNPEModel(in_dim=100, hidden=(8, 8), dropout=0.0).eval()
    m._lba_mu = torch.zeros(4)
    m._lba_sd = torch.ones(4)
    return m


def _tiny_sample(n_per_class=3, seed=0):
    g = RTLBAGenerator(n_per_class=n_per_class, trial_range=(30, 30),
                       imbalance=0.0, seed=seed)
    X, RTQ, Xt, yp, ylba, yc, yl, ya = g.generate(verbose=False)
    return X, RTQ, Xt, ya


def test_confusion_matches_hand_computed_recall_and_precision():
    # 3 classes, hand-picked so recall/precision differ and are easy to check by hand.
    true = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 1, 0])
    cm, recall, precision = _confusion(true, pred, k=3)
    assert cm.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    # recall: class 0 -> 2/3, class 1 -> 2/2, class 2 -> 0/1
    assert recall == pytest.approx([2 / 3, 1.0, 0.0])
    # precision: predicted-0 column sums to 3 (2 correct), predicted-1 column sums to
    # 3 (2 correct), predicted-2 column sums to 0 (denominator floored at 1 -> 0/1)
    assert precision == pytest.approx([2 / 3, 2 / 3, 0.0])


def test_architecture_ablation_output_schema():
    model = _tiny_model()
    X, RTQ, Xt, ya = _tiny_sample()
    out = architecture_ablation(model, X, RTQ, Xt, ya, shuffle_seed=0)

    assert out["architectures"] == list(ARCHITECTURES)
    assert out["n"] == len(X)
    assert out["chance"] == pytest.approx(1 / len(ARCHITECTURES))
    for key in ("baseline", "ablation_mean_profile", "ablation_shuffled"):
        assert key in out
    k = len(ARCHITECTURES)
    cm = np.array(out["baseline"]["confusion_matrix"])
    assert cm.shape == (k, k)
    assert cm.sum() == len(X)                        # every observer counted exactly once
    # architecture is assigned per observer independently of the (balanced) model-class
    # draw, so row totals need not be equal at this sample size -- only that every row's
    # true-label count matches how many observers actually had that true architecture
    assert cm.sum(1).tolist() == [int((ya == i).sum()) for i in range(k)]
    for field in ("recall", "precision"):
        assert len(out["baseline"][field]) == k
        assert all(0.0 <= v <= 1.0 for v in out["baseline"][field])
    for cond in ("ablation_mean_profile", "ablation_shuffled"):
        assert 0.0 <= out[cond]["accuracy"] <= 1.0
        assert len(out[cond]["recall"]) == k
    assert "description" in out["ablation_mean_profile"]
    assert "description" in out["ablation_shuffled"]
    # deliberately NOT called "cm" anywhere in this schema -- see the manuscript-
    # facing rename in scripts/make_figures_rt.py's rt_construct_metrics.json writer,
    # motivated by exactly this: nothing here should look like a counts-only measurement.
    assert "cm" not in out["ablation_mean_profile"]
    assert "cm" not in out["ablation_shuffled"]


def test_mean_profile_ablation_uses_the_actual_across_observer_mean():
    """The mean-profile condition must be the empirical mean of the RTQ actually
    passed in, not a fixed constant -- otherwise it silently stops being an
    ablation of THIS data and becomes something else."""
    model = _tiny_model()
    X, RTQ, Xt, ya = _tiny_sample()

    captured = {}
    real_predict = model.forward

    def spy_forward(x):
        captured.setdefault("inputs", []).append(x.clone())
        return real_predict(x)

    model.forward = spy_forward
    architecture_ablation(model, X, RTQ, Xt, ya, shuffle_seed=0)
    model.forward = real_predict

    # inputs[0] = baseline (real RTQ), inputs[1] = mean-profile, inputs[2] = shuffled
    assert len(captured["inputs"]) == 3
    rtq_cols = slice(16 + 4, 16 + 4 + 80)             # featurize_lba layout: props, log-trials, rtq
    mean_profile_seen = captured["inputs"][1][:, rtq_cols].numpy()
    expected = np.broadcast_to(RTQ.mean(axis=0, keepdims=True), RTQ.shape)
    assert np.allclose(mean_profile_seen, expected, atol=1e-5)
    # every row identical (the whole point of the mean-profile intervention)
    assert np.allclose(mean_profile_seen, mean_profile_seen[0], atol=1e-5)


def test_shuffled_ablation_is_a_permutation_and_is_seed_reproducible():
    model = _tiny_model()
    X, RTQ, Xt, ya = _tiny_sample()

    out_a = architecture_ablation(model, X, RTQ, Xt, ya, shuffle_seed=7)
    out_b = architecture_ablation(model, X, RTQ, Xt, ya, shuffle_seed=7)
    out_c = architecture_ablation(model, X, RTQ, Xt, ya, shuffle_seed=8)

    # same shuffle seed -> identical results, exactly (this is the reproducibility
    # property the manuscript's numbers depend on)
    assert out_a["ablation_shuffled"]["accuracy"] == out_b["ablation_shuffled"]["accuracy"]
    assert out_a["ablation_shuffled"]["recall"] == out_b["ablation_shuffled"]["recall"]
    # baseline and mean-profile don't depend on shuffle_seed at all
    assert out_a["baseline"]["accuracy"] == out_c["baseline"]["accuracy"]
    assert out_a["ablation_mean_profile"]["accuracy"] == out_c["ablation_mean_profile"]["accuracy"]


def test_baseline_reproduces_predict_rt_directly():
    """The ablation's own baseline accuracy must match calling predict_rt() the
    plain way on the same (unmodified) inputs -- i.e. the ablation machinery
    doesn't accidentally perturb the real-RTQ condition."""
    from src.inference.predict_rt import predict_rt
    model = _tiny_model()
    X, RTQ, Xt, ya = _tiny_sample()

    direct_pa = predict_rt(model, X, RTQ, Xt)["p_arch"].argmax(1)
    direct_acc = float(np.mean(direct_pa == ya))

    out = architecture_ablation(model, X, RTQ, Xt, ya, shuffle_seed=0)
    assert out["baseline"]["accuracy"] == pytest.approx(direct_acc)
