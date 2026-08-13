"""
Smoke test against the INSTALLED grintools wheel, run from a directory containing
none of the source (see the CI workflow / porting_status.md's manual publish
checklist, which this automates). Exercises exactly the public API surface a user
following the README would touch: infer(), the ordering/counts guards, and the
Criterion/Target/Decision stopping API.

Run against an installed grintools:
    pip install .
    pytest grintools/tests/test_packaged.py
"""
import numpy as np
import pytest

import grintools as gt

M = [[71, 17, 9, 5],
     [20, 67, 5, 9],
     [13, 6, 63, 20],
     [5, 10, 15, 71]]


def test_version_and_bundled_model_path():
    assert gt.__version__
    import os
    assert os.path.exists(gt.default_model_path())


def test_infer_returns_a_sane_posterior_and_constructs():
    result, constructs = gt.infer(M)
    assert len(result.params) == 12
    assert len(result.std) == 12
    assert np.all(np.asarray(result.std) > 0)
    assert 0.0 <= constructs["p_PI"] <= 1.0
    assert 0.0 <= constructs["p_sep_A"] <= 1.0
    assert 0.0 <= constructs["p_sep_B"] <= 1.0
    assert set(constructs) >= {"p_PI", "p_sep_A", "p_sep_B", "p_corr",
                               "evidence_PI", "evidence_sep_A", "evidence_sep_B"}


def test_infer_is_deterministic_and_session_cached():
    r1, c1 = gt.infer(M)
    r2, c2 = gt.infer(M)
    assert list(r1.params) == list(r2.params)
    assert c1["p_PI"] == c2["p_PI"]


def test_to_confusion_refuses_a_bare_unlabelled_matrix():
    with pytest.raises(ValueError, match="refus"):
        gt.to_confusion(M)


def test_to_confusion_accepts_the_canonical_assertion():
    ci = gt.to_confusion(M, order="canonical")
    assert np.array_equal(ci.counts, M)
    assert list(ci.trials) == [102, 101, 102, 101]


def test_criterion_stopping_decision_end_to_end():
    result, constructs = gt.infer(M)
    crit = gt.Criterion([
        gt.Target.precision(params=["zx", "zy"], sd_max=0.10),
        gt.Target.probability("PS_A", at_least=0.90),
    ], combine="any")
    decision = crit.evaluate(result, constructs)
    assert isinstance(decision.stop, bool)
    assert isinstance(decision.blocked_by, list)
