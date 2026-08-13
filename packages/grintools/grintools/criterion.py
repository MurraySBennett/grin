"""
criterion.py: stopping decisions for adaptive designs.

Split out of io.py so `grintools.criterion` can be imported (and reasoned about)
independently of the input-normalisation layer; `Target`/`Criterion` only need
PARAM_NAMES/PARAM_GROUPS from io.py, not any of its parsing machinery.
"""
from __future__ import annotations
import numpy as np

from .io import PARAM_NAMES, PARAM_GROUPS

# =========================================================================== #
# Stopping decisions. The EXPERIMENTER declares a Criterion from Targets; GRIN
# evaluates it. Four ways to use it:
#
#   1. passthrough  : build no Criterion; read result.* and constructs yourself.
#   2. precision    : Target.precision(params=..., sd_max=... | ci_width_max=...)
#                     stop when the parameter posterior is tight enough. This is
#                     the "I want the space measured, verdict aside" target.
#   3. probability  : Target.probability("PI"|"PS_A"|"PS_B"|"*_violated", at_least=0.9)
#                     stop when a construct probability (from model_posterior)
#                     crosses a threshold. This is the "I want the verdict" target.
#   4. combine      : Criterion([...targets...], combine="all"|"any")
#
# The construct targets read model_posterior's per-matrix dict directly, including
# its evidence_* flags. When an evidence flag is False the target is reported as
# unreachable (Decision.blocked_by), which is how the PI identifiability limit
# surfaces: a threshold on a construct the data cannot decide will never be met,
# and we say so rather than letting the loop run forever.
# =========================================================================== #
_CONSTRUCT_MAP = {
    "PI":            ("p_PI",    "evidence_PI",    False),
    "PS_A":          ("p_sep_A", "evidence_sep_A", False),
    "PS_B":          ("p_sep_B", "evidence_sep_B", False),
    "PI_violated":   ("p_PI",    "evidence_PI",    True),
    "PS_A_violated": ("p_sep_A", "evidence_sep_A", True),
    "PS_B_violated": ("p_sep_B", "evidence_sep_B", True),
}


def _select_indices(params):
    if params is None or params == "all":
        return list(range(len(PARAM_NAMES)))
    idx = []
    for p in ([params] if isinstance(params, str) else list(params)):
        if p in PARAM_GROUPS:
            idx.extend(PARAM_GROUPS[p])
        elif p in PARAM_NAMES:
            idx.append(PARAM_NAMES.index(p))
        else:
            raise ValueError(f"unknown parameter selector {p!r}; use a name in "
                             f"{PARAM_NAMES} or a group in {list(PARAM_GROUPS)}")
    return sorted(set(idx))


class Target:
    def __init__(self, kind, cfg):
        self.kind = kind
        self.cfg = cfg

    @classmethod
    def precision(cls, *, params=None, sd_max=None, ci_width_max=None):
        if (sd_max is None) == (ci_width_max is None):
            raise ValueError("precision target needs exactly one of sd_max or ci_width_max")
        return cls("precision", dict(params=params, sd_max=sd_max, ci_width_max=ci_width_max))

    @classmethod
    def probability(cls, construct, *, at_least):
        if construct not in _CONSTRUCT_MAP:
            raise ValueError(f"unknown construct {construct!r}; choose from {list(_CONSTRUCT_MAP)}")
        return cls("probability", dict(construct=construct, at_least=float(at_least)))

    def check(self, result, constructs):
        if self.kind == "precision":
            idx = _select_indices(self.cfg["params"])
            names = getattr(result, "names", PARAM_NAMES)
            if self.cfg["sd_max"] is not None:
                vals = np.asarray(result.std, float)[idx]; thr = self.cfg["sd_max"]; q = "sd"
            else:
                vals = (np.asarray(result.ci_high, float) - np.asarray(result.ci_low, float))[idx]
                thr = self.cfg["ci_width_max"]; q = "ci_width"
            worst = int(np.argmax(vals))
            return dict(met=bool(np.all(vals <= thr)), value=float(vals[worst]),
                        name=f"{q}:{names[idx[worst]]}", threshold=thr, reachable=True, note="")
        # probability
        c = self.cfg["construct"]; thr = self.cfg["at_least"]
        if constructs is None:
            return dict(met=False, value=float("nan"), name=c, threshold=thr,
                        reachable=None, note="no constructs supplied; cannot evaluate")
        key, evkey, violated = _CONSTRUCT_MAP[c]
        p = float(constructs[key])
        if violated:
            p = 1.0 - p
        ev = bool(constructs.get(evkey, True))
        note = "" if ev else ("evidence flag is False: the data may not be able to decide "
                              "this construct in the current regime (a property of the data, "
                              "not of GRIN)")
        return dict(met=bool(p >= thr and ev), value=p, name=c, threshold=thr,
                    reachable=ev, note=note)


class Decision:
    def __init__(self, stop, checks, combine, blocked):
        self.stop = bool(stop)
        self.checks = checks
        self.combine = combine
        self.blocked_by = blocked

    def __bool__(self):
        return self.stop

    def summary(self):
        lines = [f"stop = {self.stop}  (combine='{self.combine}')"]
        for c in self.checks:
            mark = "met" if c["met"] else "not met"
            val = "nan" if c["value"] != c["value"] else f"{c['value']:.3f}"
            lines.append(f"    {c['name']:16s} {val} vs {c['threshold']:.3f}  [{mark}]")
            if c["note"]:
                lines.append(f"        note: {c['note']}")
        if self.blocked_by:
            lines.append(f"    unreachable target(s): {self.blocked_by} "
                         f"(threshold may never be met with current data)")
        return "\n".join(lines)


class Criterion:
    def __init__(self, targets, combine="all"):
        self.targets = list(targets)
        if combine not in ("all", "any"):
            raise ValueError("combine must be 'all' or 'any'")
        self.combine = combine

    @property
    def needs_constructs(self):
        return any(t.kind == "probability" for t in self.targets)

    def evaluate(self, result=None, constructs=None):
        checks = [t.check(result, constructs) for t in self.targets]
        mets = [c["met"] for c in checks]
        stop = all(mets) if self.combine == "all" else any(mets)
        blocked = [c["name"] for c in checks if c["reachable"] is False]
        return Decision(stop, checks, self.combine, blocked)


def stop_on_precision(result, *, sd_max=None, ci_width_max=None, params=None):
    """Convenience for the common single-target precision case."""
    return Criterion([Target.precision(params=params, sd_max=sd_max,
                                       ci_width_max=ci_width_max)]).evaluate(result)
