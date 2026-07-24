"""
labels.py — translate the R baselines' model-class strings into GRIN's class names.

mdsdt and grtools both report a winning model from the same 12-node hierarchy GRIN
uses, but they print it in their own notation ("{PI, PS(A), DS}"), and grtools prefixes
its string ("GRT-{PI, PS, DS}"). This module normalises both into `grt_model.MODEL_NAMES`
so that per-construct and 12-way correctness are computed identically for every method.

DELIBERATELY STRICT. An unrecognised label RAISES rather than falling back to "wrong".
A silent parse failure would mark every baseline point incorrect and make GRIN look
perfect, which is the single most dangerous failure mode in the whole comparison.
"""
import re

import numpy as np

try:
    from .. import grt_model as gm
except ImportError:  # flat layout / direct execution
    import grt_model as gm


# Canonical R notation -> GRIN class name. All 12 nodes, one-to-one with MODEL_NAMES.
_R_TO_GRIN = {
    "PI,PS,DS":      "pi_ps_ds",
    "PI,PS(A),DS":   "pi_psa_ds",
    "PI,PS(B),DS":   "pi_psb_ds",
    "1_RHO,PS,DS":   "rho1_ps_ds",
    "1_RHO,PS(A),DS": "rho1_psa_ds",
    "1_RHO,PS(B),DS": "rho1_psb_ds",
    "PI,DS":         "pi_ds",
    "PS,DS":         "ps_ds",
    "1_RHO,DS":      "rho1_ds",
    "PS(A),DS":      "psa_ds",
    "PS(B),DS":      "psb_ds",
    "DS":            "ds",
}

assert set(_R_TO_GRIN.values()) == set(gm.MODEL_NAMES), \
    "label table has drifted from grt_model.MODEL_NAMES"


def _canon(raw):
    """'GRT-{1_RHO, PS(A), DS}' -> '1_RHO,PS(A),DS'. Whitespace/case/prefix tolerant."""
    s = str(raw).strip()
    s = re.sub(r"^\s*GRT\s*-\s*", "", s, flags=re.IGNORECASE)   # grtools prefix
    s = s.strip().strip("{}").strip()
    s = re.sub(r"\s+", "", s)                                    # kill all whitespace
    s = s.upper()
    s = s.replace("RHO_1", "1_RHO").replace("ONE_RHO", "1_RHO")  # defensive aliases
    return s


def to_grin_label(raw, strict=True):
    """One R label string -> a GRIN class name.

    Returns None for a genuine non-fit (NA / empty / an ERROR: ... string written by
    fit_baselines.R). Raises ValueError on a string that looks like a label but does
    not parse — that is a bug, not a failed fit, and must not be silently scored wrong.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"na", "nan", "none"} or s.startswith("ERROR:"):
        return None
    key = _canon(s)
    if key in _R_TO_GRIN:
        return _R_TO_GRIN[key]
    if key in gm.MODEL_NAMES:          # already a GRIN name
        return key
    lower = key.lower()
    if lower in gm.MODEL_NAMES:
        return lower
    if strict:
        raise ValueError(
            f"unrecognised model label {raw!r} (canonical form {key!r}). "
            "Add it to labels._R_TO_GRIN — do NOT let it fall through as 'incorrect'."
        )
    return None


def to_grin_labels(raws, strict=True):
    """Vectorised `to_grin_label`. Returns an object array with None for non-fits."""
    return np.array([to_grin_label(r, strict=strict) for r in np.asarray(raws, dtype=object)],
                    dtype=object)


def constructs_from_labels(names):
    """GRIN class names (with None allowed) -> (corr, ps_x, ps_y) int arrays.

    corr is 0/1/2 for pi/rho1/free; ps_x and ps_y are 0/1. Entries whose label is None
    (a failed fit) are filled with -1 so they can never accidentally compare equal to a
    ground-truth construct.
    """
    corr_idx = {"pi": 0, "rho1": 1, "free": 2}
    corr, sx, sy = [], [], []
    for n in names:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            corr.append(-1); sx.append(-1); sy.append(-1)
            continue
        c, px, py = gm.MODEL_SPECS[n]
        corr.append(corr_idx[c]); sx.append(int(px)); sy.append(int(py))
    return np.array(corr), np.array(sx), np.array(sy)


def grin_labels_from_constructs(corr, ps_x, ps_y):
    """(corr, ps_x, ps_y) -> GRIN class names. Inverse of constructs_from_labels.

    This is how GRIN gets a 12-way label: argmax of the comparison head's three
    constructs, composed. MODEL_SPECS is exactly the 3 x 2 x 2 product, so the
    composition is total — every combination is a real class.
    """
    inv = {(c, px, py): name for name, (c, px, py) in gm.MODEL_SPECS.items()}
    corr_name = {0: "pi", 1: "rho1", 2: "free"}
    out = []
    for c, px, py in zip(np.asarray(corr), np.asarray(ps_x), np.asarray(ps_y)):
        out.append(inv[(corr_name[int(c)], bool(px), bool(py))])
    return np.array(out, dtype=object)


def labels_from_amortized(ac, threshold=0.5):
    """`amortized_compare(...)` output -> (N,) GRIN class names.

    This is GRIN's 12-way class: argmax of the correlation head, plus the two separability
    heads thresholded. Kept here, pure-numpy, so every caller composes it identically --
    a figure whose shapes disagree with its own confusion matrix is worse than no figure.
    """
    corr = np.asarray(ac["p_corr"]).argmax(1)
    ps_a = (np.asarray(ac["p_sep_A"]) > threshold).astype(int)
    ps_b = (np.asarray(ac["p_sep_B"]) > threshold).astype(int)
    return grin_labels_from_constructs(corr, ps_a, ps_b)
