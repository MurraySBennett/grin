"""
grin_io.py: the input contract and stopping decisions for GRIN.

This is the thin layer between an experiment and the trained network. It does
three jobs and nothing else:

    NORMALISE   whatever the experiment has (labelled 4x4, bare 4x4, long trial
                log, aggregated long) -> the canonical-order 4x4 count matrix plus
                per-stimulus trial totals that api.infer expects.
    DESCRIBE    echo back exactly what was parsed, WITHOUT running the network, so
                an experimenter can confirm their wiring before trusting a result.
    DECIDE      turn a posterior (and, optionally, the construct probabilities from
                model_posterior) into a stop / keep-collecting decision, using a
                criterion the EXPERIMENTER declares. GRIN does not choose the rule.

Principle: liberal about shape, strict about meaning. Any container/layout is
coerced, but two things are never guessed because guessing returns a confident
wrong answer: (1) stimulus/response ORDER for a bare unlabelled matrix, and (2)
counts vs proportions (the network reads trial totals as a second input, so
proportions silently wreck the posterior's width). Both are refused unless the
caller resolves them explicitly.

Canonical layout (from grt_model.py): rows = stimuli, cols = responses, both in
order A1B1, A1B2, A2B1, A2B2 (dimension A = x, B = y, B fastest). Canonical index
of a cell = 2*a_level + b_level.

Australian English. numpy required; pandas optional (duck-typed).
"""

from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- #
# Canonical layout (mirror grt_model.py; fall back so this runs standalone).
# --------------------------------------------------------------------------- #
try:
    from grt_model import STIMULUS_ORDER as CANON_STIM, RESPONSE_ORDER as CANON_RESP
    from grt_model import PARAM_NAMES
except Exception:
    CANON_STIM = ["A1B1", "A1B2", "A2B1", "A2B2"]
    CANON_RESP = ["a1b1", "a1b2", "a2b1", "a2b2"]
    PARAM_NAMES = ([f"zx_{i}" for i in range(4)]
                   + [f"zy_{i}" for i in range(4)]
                   + [f"rho_{i}" for i in range(4)])

PARAM_GROUPS = {"zx": list(range(0, 4)), "zy": list(range(4, 8)), "rho": list(range(8, 12))}

_SPARSE_TRIALS = 20
_PROP_TOL = 1e-6
_INT_TOL = 1e-8


# =========================================================================== #
# Containers
# =========================================================================== #
class ConfusionInput:
    """A normalised, network-ready input plus a record of how it was parsed."""
    def __init__(self, counts, trials, placement, warnings, asserted_order):
        self.counts = counts                  # (4,4) int, canonical order
        self.trials = trials                  # (4,)  int, per-stimulus totals
        self.placement = placement            # canonical cell -> source label
        self.warnings = warnings              # list[str]
        self.asserted_order = asserted_order  # did the caller assert canonical order?

    @property
    def ready(self):
        return self.counts is not None

    def __repr__(self):
        return (f"ConfusionInput(trials={list(map(int, self.trials))}, "
                f"warnings={len(self.warnings)})")


# =========================================================================== #
# Label resolution (strict): map 4 labels onto canonical order, or fail loudly.
# =========================================================================== #
def _canonical_index(a_level, b_level):
    return 2 * int(a_level) + int(b_level)


def _parse_cell_label(label, factor_a, factor_b, sep):
    if isinstance(label, (tuple, list)):
        if len(label) != 2:
            raise ValueError(f"label {label!r} is not a 2-tuple (a_name, b_name)")
        a_name, b_name = label
    else:
        parts = str(label).split(sep)
        if len(parts) != 2:
            raise ValueError(
                f"cannot split label {label!r} on sep {sep!r} into two factor "
                f"levels; pass a (a_name, b_name) tuple or use a matching sep")
        a_name, b_name = parts[0].strip(), parts[1].strip()
    if a_name not in factor_a:
        raise ValueError(f"'{a_name}' from {label!r} is not a level of factor_a={factor_a}")
    if b_name not in factor_b:
        raise ValueError(f"'{b_name}' from {label!r} is not a level of factor_b={factor_b}")
    return factor_a.index(a_name), factor_b.index(b_name)


def _permutation_to_canonical(labels, factor_a, factor_b, sep):
    """Return (perm, placement) so axis[perm[c]] is canonical cell c. Raises unless
    the 4 labels form a complete 2x2 factorial."""
    canon_tokens = [t.lower() for t in CANON_STIM]
    low = [str(x).lower() for x in labels]
    if low == canon_tokens or low == [t.lower() for t in CANON_RESP]:
        return list(range(4)), {CANON_STIM[i]: labels[i] for i in range(4)}
    if factor_a is None or factor_b is None:
        raise ValueError(
            "labels are not canonical A1B1.. tokens, so I need factor_a=(A1,A2) and "
            "factor_b=(B1,B2) to place them. Refusing to assume an order.")
    factor_a, factor_b = tuple(factor_a), tuple(factor_b)
    source_of_canon, seen = {}, set()
    for pos, lab in enumerate(labels):
        a, b = _parse_cell_label(lab, factor_a, factor_b, sep)
        c = _canonical_index(a, b)
        if c in seen:
            raise ValueError(f"label {lab!r} duplicates canonical cell {CANON_STIM[c]}; "
                             f"the 4 labels must be a complete 2x2 factorial")
        seen.add(c); source_of_canon[c] = pos
    missing = [CANON_STIM[c] for c in range(4) if c not in seen]
    if missing:
        raise ValueError(f"labels do not cover the full factorial; missing: {missing}")
    perm = [source_of_canon[c] for c in range(4)]
    placement = {CANON_STIM[c]: labels[source_of_canon[c]] for c in range(4)}
    return perm, placement


# =========================================================================== #
# Normaliser
# =========================================================================== #
def _is_dataframe(x):
    return hasattr(x, "columns") and hasattr(x, "itertuples") and hasattr(x, "index")


def _counts_guard(counts, trials, warnings):
    counts = np.asarray(counts, dtype=float)
    row_sums = counts.sum(axis=1)
    looks_like_props = np.allclose(row_sums, 1.0, atol=1e-3) and np.all(counts <= 1.0 + _PROP_TOL)
    non_integer = np.any(np.abs(counts - np.round(counts)) > _INT_TOL)
    if looks_like_props or non_integer:
        if trials is None:
            raise ValueError(
                "input looks like PROPORTIONS, not counts (rows sum to ~1 or contain "
                "non-integers). The network reads per-stimulus trial totals as a "
                "separate input, so proportions would silently wreck the posterior "
                "uncertainty. Pass counts, or pass trials=[...] to rescale.")
        trials = np.asarray(trials, dtype=float).reshape(4)
        counts = np.round(counts * trials[:, None])
        warnings.append("input treated as proportions and rescaled to counts using the "
                        "supplied trials; verify this is what you intended")
    return np.round(counts).astype(int)


def _row_warnings(counts, trials, warnings):
    for i, n in enumerate(trials):
        if n < _SPARSE_TRIALS:
            warnings.append(f"stimulus {CANON_STIM[i]} has only {int(n)} trials "
                            f"(< {_SPARSE_TRIALS}); posterior will be wide here")
    if np.any(counts == 0):
        z = [(CANON_STIM[i], CANON_RESP[j]) for i in range(4) for j in range(4)
             if counts[i, j] == 0]
        warnings.append(f"{len(z)} empty cell(s) {z}; fine for GRIN, but this is the "
                        f"cell-separation regime where MLE baselines diverge")


def _from_long(data, factor_a, factor_b, sep, warnings):
    stim, resp, cnt = [], [], []
    if _is_dataframe(data):
        cols = [str(c).lower() for c in data.columns]
        def col(name): return data.columns[cols.index(name)]
        if "stimulus" not in cols or "response" not in cols:
            raise ValueError("long DataFrame needs 'stimulus' and 'response' columns "
                             "(a 'count' column is optional)")
        has_count = "count" in cols
        for row in data.itertuples(index=False):
            d = row._asdict()
            stim.append(d[col("stimulus")]); resp.append(d[col("response")])
            cnt.append(d[col("count")] if has_count else 1)
    else:
        for row in data:
            row = tuple(row)
            if len(row) == 2:
                stim.append(row[0]); resp.append(row[1]); cnt.append(1)
            elif len(row) == 3:
                stim.append(row[0]); resp.append(row[1]); cnt.append(row[2])
            else:
                raise ValueError(f"long row {row!r} must be (stimulus, response) or "
                                 f"(stimulus, response, count)")
    stim_labels = list(dict.fromkeys(stim))
    resp_labels = list(dict.fromkeys(resp))
    if len(stim_labels) != 4 or len(resp_labels) != 4:
        raise ValueError(f"expected 4 distinct stimuli and 4 responses; got "
                         f"{len(stim_labels)} stimuli, {len(resp_labels)} responses")
    s_perm, s_place = _permutation_to_canonical(stim_labels, factor_a, factor_b, sep)
    r_perm, _ = _permutation_to_canonical(resp_labels, factor_a, factor_b, sep)
    s_to_c = {stim_labels[s_perm[c]]: c for c in range(4)}
    r_to_c = {resp_labels[r_perm[c]]: c for c in range(4)}
    counts = np.zeros((4, 4), dtype=float)
    for s, r, c in zip(stim, resp, cnt):
        counts[s_to_c[s], r_to_c[r]] += c
    return counts, s_place


def to_confusion(data, *, stim_labels=None, resp_labels=None, factor_a=None,
                 factor_b=None, order=None, trials=None, sep="/", long=False):
    """Normalise any supported input into a canonical-order ConfusionInput.
    Raises ValueError on any ambiguity of meaning."""
    warnings = []
    factor_a = tuple(factor_a) if factor_a is not None else None
    factor_b = tuple(factor_b) if factor_b is not None else None

    is_long = long or (_is_dataframe(data)
                       and "stimulus" in [str(c).lower() for c in data.columns])
    if is_long:
        counts, placement = _from_long(data, factor_a, factor_b, sep, warnings)
        asserted = False
    else:
        if _is_dataframe(data):
            if resp_labels is None:
                resp_labels = list(data.columns)
            idx = list(data.index)
            if stim_labels is None and not all(isinstance(i, (int, np.integer)) for i in idx):
                stim_labels = idx
            mat = np.asarray(data.values, dtype=float)
        else:
            mat = np.asarray(data, dtype=float)
        if mat.shape != (4, 4):
            mat = mat.reshape(4, 4)

        if order == "canonical":
            counts = mat
            placement = {c: c for c in CANON_STIM}
            asserted = True
        elif stim_labels is not None or resp_labels is not None:
            if resp_labels is not None and stim_labels is None:
                stim_labels = list(resp_labels)
                warnings.append("rows were unlabelled; assumed to follow the same "
                                "category order as the column labels")
            if stim_labels is None or resp_labels is None:
                raise ValueError("need both stim_labels and resp_labels (or "
                                 "order='canonical') to place a 4x4 matrix")
            r_perm, placement = _permutation_to_canonical(stim_labels, factor_a, factor_b, sep)
            c_perm, _ = _permutation_to_canonical(resp_labels, factor_a, factor_b, sep)
            counts = mat[np.ix_(r_perm, c_perm)]
            asserted = False
        else:
            raise ValueError(
                "a bare 4x4 with no labels and no order assertion is refused: I will "
                "not guess the stimulus/response order and hand back a confident wrong "
                "posterior. Either pass order='canonical' to assert your matrix is "
                "already A1B1,A1B2,A2B1,A2B2, or pass stim_labels/resp_labels with "
                "factor_a and factor_b.")

    counts = _counts_guard(counts, trials, warnings)
    resolved_trials = (np.asarray(trials, dtype=float).round().astype(int).reshape(4)
                       if trials is not None else counts.sum(axis=1))
    if trials is not None and not np.array_equal(resolved_trials, counts.sum(axis=1)):
        warnings.append("supplied trials disagree with row sums of the counts; using "
                        "the supplied trials")
    _row_warnings(counts, resolved_trials, warnings)
    return ConfusionInput(counts, resolved_trials, placement, warnings, asserted)


# =========================================================================== #
# Describe: setup guide / dev mode / exception surface in one. Never raises.
# =========================================================================== #
def describe(data, *, printout=True, **kwargs):
    report = {"ready": False, "errors": [], "warnings": [], "counts": None,
              "trials": None, "placement": None, "asserted_order": None}
    try:
        ci = to_confusion(data, **kwargs)
        report.update(ready=True, counts=ci.counts, trials=ci.trials,
                      placement=ci.placement, warnings=ci.warnings,
                      asserted_order=ci.asserted_order)
    except Exception as e:
        report["errors"].append(f"{type(e).__name__}: {e}")
    if printout:
        lines = ["GRIN input check", "-" * 52]
        if report["ready"]:
            lines.append("parsed OK, ready for inference"
                         + ("  (order asserted by caller)" if report["asserted_order"]
                            else "  (order resolved from labels)"))
            lines.append("canonical placement (canonical cell <- your label):")
            for c in CANON_STIM:
                lines.append(f"    {c} <- {report['placement'][c]!r}")
            lines.append(f"trials per stimulus: {list(map(int, report['trials']))}")
            lines.append("counts (canonical order):")
            for i, rowc in enumerate(report["counts"]):
                lines.append(f"    {CANON_STIM[i]:5s} | " + " ".join(f"{v:4d}" for v in rowc))
        else:
            lines.append("NOT ready, could not parse:")
        for w in report["warnings"]:
            lines.append(f"  warning: {w}")
        for e in report["errors"]:
            lines.append(f"  ERROR:   {e}")
        print("\n".join(lines))
    return report


# =========================================================================== #
# Response bias: computed directly from a confusion matrix, no model fit
# required. GRT's separability/independence machinery lives entirely in
# infer()'s 12 identified parameters; the raw tendency to over- or
# under-report one level of a dimension, independent of how well the observer
# discriminates it, is a property of the data alone and worth a name.
# =========================================================================== #
def response_bias(counts, trials=None):
    """Response bias from a raw confusion matrix.

    The signed tendency to report level 2 of a dimension more or less often
    than level 1, averaged across the four stimuli: 0 is unbiased, positive
    means the observer favours the "2" response on that dimension more than a
    fair coin would, negative means they favour "1". This describes the data,
    independent of infer()'s model fit -- it needs no trained network and
    works even on a matrix GRIN can't otherwise fit.

    `counts`: a canonical-order 4x4 matrix, or a length-16 vector read
    row-major. `trials`: optional per-stimulus trial totals; defaults to row
    sums.

    Returns a dict: `x_bias`, `y_bias` (each in [-0.5, 0.5]), and `p_resp2`,
    a (4, 2) array giving P(respond level 2) on each dimension for each of
    the four stimuli (so a systematic bias can be told apart from one driven
    by a single stimulus).
    """
    cm = np.asarray(counts, dtype=float).reshape(4, 4)
    if trials is None:
        trials = cm.sum(axis=1)
    trials = np.asarray(trials, dtype=float).reshape(4)
    props = cm / trials[:, None]
    p_x2 = props[:, 2] + props[:, 3]   # respond "A2" (canonical cols a2b1, a2b2)
    p_y2 = props[:, 1] + props[:, 3]   # respond "B2" (canonical cols a1b2, a2b2)
    return {
        "x_bias": float(p_x2.mean() - 0.5),
        "y_bias": float(p_y2.mean() - 0.5),
        "p_resp2": np.stack([p_x2, p_y2], axis=1),
    }


# =========================================================================== #
# Convenience: normalise -> infer (-> constructs -> decide) in one call. Network
# imports are lazy so this module and its I/O are usable without torch present.
# =========================================================================== #
def _import_infer():
    try:
        from api import infer, load_model
    except Exception:
        from .api import infer, load_model
    return infer, load_model


def _import_model_posterior():
    try:
        from model_posterior import model_posterior
    except Exception:
        from .model_posterior import model_posterior
    return model_posterior


def fit(data, *, stim_labels=None, resp_labels=None, factor_a=None, factor_b=None,
        order=None, trials=None, sep="/", long=False, criterion=None,
        constructs=False, model=None, n_samples=1000):
    """Normalise `data`, run GRIN, optionally compute construct probabilities and a
    stopping decision. Returns the InferenceResult with `.input_` attached, plus
    `.constructs` and `.decision` when requested."""
    ci = to_confusion(data, stim_labels=stim_labels, resp_labels=resp_labels,
                      factor_a=factor_a, factor_b=factor_b, order=order,
                      trials=trials, sep=sep, long=long)
    infer, load_model = _import_infer()
    want_constructs = constructs or (criterion is not None and criterion.needs_constructs)
    if model is None and want_constructs:
        model = load_model()                      # load once, share with model_posterior
    result = infer(ci.counts, trials=ci.trials, model=model, n_samples=n_samples)
    result.input_ = ci
    if want_constructs:
        model_posterior = _import_model_posterior()
        result.constructs = model_posterior(
            model, ci.counts.reshape(1, 16).astype(float),
            ci.trials.reshape(1, 4).astype(float), n_samples=n_samples)[0]
    if criterion is not None:
        result.decision = criterion.evaluate(result, getattr(result, "constructs", None))
    return result
