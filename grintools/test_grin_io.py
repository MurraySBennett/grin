"""Tests for grin_io against the real test_cm matrix. Plain asserts, no torch."""
import numpy as np
import pandas as pd
import grin_io as gio

M = np.array([[71, 17,  9,  5],
              [20, 67,  5,  9],
              [13,  6, 63, 20],
              [ 5, 10, 15, 71]], dtype=int)
FA = ("Old", "Young")     # dimension A: A1=Old, A2=Young
FB = ("Neg", "Pos")       # dimension B: B1=Neg, B2=Pos
LABELS = ["Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"]

passed = 0
def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1; print(f"  ok  {name}")

print("grin_io tests\n" + "=" * 52)

# --- normalisation ---
ci = gio.to_confusion(M, order="canonical")
check("canonical assert unchanged", np.array_equal(ci.counts, M))
check("trials from row sums", list(ci.trials) == [102, 101, 102, 101])

perm = [2, 0, 3, 1]
Ms = M[np.ix_(perm, perm)]
labs = [LABELS[p] for p in perm]
ci2 = gio.to_confusion(Ms, stim_labels=labs, resp_labels=labs, factor_a=FA, factor_b=FB)
check("ordering guard: scramble+labels -> repaired", np.array_equal(ci2.counts, M))
check("placement recorded", ci2.placement["A1B1"] == "Old/Neg")

try:
    gio.to_confusion(Ms); check("bare unlabelled refused", False)
except ValueError:
    check("bare unlabelled refused", True)

df = pd.DataFrame(M, columns=LABELS)
ci4 = gio.to_confusion(df, factor_a=FA, factor_b=FB)
check("labelled DataFrame -> canonical", np.array_equal(ci4.counts, M))

long_rows = [(LABELS[perm[i]], LABELS[perm[j]], int(M[perm[i], perm[j]]))
             for i in range(4) for j in range(4)]
ci5 = gio.to_confusion(long_rows, long=True, factor_a=FA, factor_b=FB)
check("aggregated long -> canonical", np.array_equal(ci5.counts, M))

trial_rows = []
for i in range(4):
    for j in range(4):
        trial_rows += [(LABELS[i], LABELS[j])] * int(M[i, j])
ci5b = gio.to_confusion(trial_rows, long=True, factor_a=FA, factor_b=FB)
check("trial-level long -> canonical", np.array_equal(ci5b.counts, M))

props = M / M.sum(1, keepdims=True)
try:
    gio.to_confusion(props, order="canonical"); check("proportions refused", False)
except ValueError:
    check("proportions refused", True)
ci6 = gio.to_confusion(props, order="canonical", trials=[102, 101, 102, 101])
check("proportions+trials -> counts", np.array_equal(ci6.counts, M))

M_thin = np.array([[15, 0, 0, 0], [0, 14, 0, 1], [1, 0, 13, 0], [0, 0, 2, 12]], int)
ci7 = gio.to_confusion(M_thin, order="canonical")
check("sparse warning", any("only" in w and "trials" in w for w in ci7.warnings))
check("empty-cell warning", any("empty cell" in w for w in ci7.warnings))

# --- stopping decisions (stubbed posterior + constructs; no network needed) ---
class Res:
    names = gio.PARAM_NAMES
    def __init__(self, std):
        self.std = np.asarray(std, float)
        self.ci_low = -self.std * 1.64
        self.ci_high = self.std * 1.64

tight = Res([0.05] * 12)
loose = Res([0.05] * 8 + [0.40] * 4)          # rho block still wide

# precision target
d = gio.Criterion([gio.Target.precision(sd_max=0.10)]).evaluate(tight)
check("precision all-tight -> stop", bool(d) is True)
d = gio.Criterion([gio.Target.precision(sd_max=0.10)]).evaluate(loose)
check("precision wide-rho -> no stop", bool(d) is False)
d = gio.Criterion([gio.Target.precision(params=["zx", "zy"], sd_max=0.10)]).evaluate(loose)
check("precision on zx/zy only -> stop", bool(d) is True)
check("convenience stop_on_precision", bool(gio.stop_on_precision(tight, sd_max=0.10)) is True)

# probability target on constructs (as returned by model_posterior)
decisive = {"p_PI": 0.02, "p_sep_A": 0.97, "p_sep_B": 0.10,
            "evidence_PI": True, "evidence_sep_A": True, "evidence_sep_B": True}
undecided_pi = {"p_PI": 0.52, "p_sep_A": 0.97, "p_sep_B": 0.10,
                "evidence_PI": False, "evidence_sep_A": True, "evidence_sep_B": True}

d = gio.Criterion([gio.Target.probability("PS_A", at_least=0.9)]).evaluate(None, decisive)
check("prob PS_A holds -> stop", bool(d) is True)
d = gio.Criterion([gio.Target.probability("PS_B", at_least=0.9)]).evaluate(None, decisive)
check("prob PS_B (0.10) -> no stop", bool(d) is False)
d = gio.Criterion([gio.Target.probability("PI_violated", at_least=0.9)]).evaluate(None, decisive)
check("prob PI_violated (1-0.02=0.98) -> stop", bool(d) is True)

# PI identifiability: unreachable target is flagged, never silently stops
d = gio.Criterion([gio.Target.probability("PI", at_least=0.9)]).evaluate(None, undecided_pi)
check("undecidable PI -> no stop", bool(d) is False)
check("undecidable PI -> blocked_by set", d.blocked_by == ["PI"])

# combine: verdict AND precision (his scenario 3, both wanted)
crit = gio.Criterion([gio.Target.probability("PS_A", at_least=0.9),
                      gio.Target.precision(params=["zx", "zy"], sd_max=0.10)], combine="all")
check("combine all: verdict+precision both met -> stop", bool(crit.evaluate(loose, decisive)) is True)
crit_any = gio.Criterion([gio.Target.probability("PS_B", at_least=0.9),
                          gio.Target.precision(sd_max=0.10)], combine="any")
check("combine any: neither met -> no stop", bool(crit_any.evaluate(loose, decisive)) is False)
check("needs_constructs flag", crit.needs_constructs is True)

# --- describe ---
print("-" * 52)
rep = gio.describe(Ms, stim_labels=labs, resp_labels=labs, factor_a=FA, factor_b=FB)
check("describe ready", rep["ready"] is True)
print("-" * 52)
rep2 = gio.describe(Ms)
check("describe reports error, no raise", rep2["ready"] is False)

print("=" * 52 + f"\nALL {passed} CHECKS PASSED")