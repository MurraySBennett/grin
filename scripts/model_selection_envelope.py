"""
Regenerate the `model_selection` block of results/model_selection_envelope.json:
GRIN's construct classification accuracy against AIC selection over the Python
maximum-likelihood hierarchy, on the stratified software-comparison export.

    python scripts/model_selection_envelope.py

Why this script exists. Until 2026-09-29 that JSON had no generator at all. Nothing
in the repository read or wrote it -- it entered git inside a figures commit -- yet
it is the source of four values the manuscript reports (separability 71.9% vs 63.5%,
correlation structure 48.8% vs 42.8%, and the per-dimension 83.3%/84.5% against
77.6%/79.8%), and the manuscript claims every numerical summary it reports is
produced by a named script. The values themselves were verified correct on
2026-09-29: re-derived from the released checkpoint they reproduce to four decimal
places. This script is that derivation, committed so the claim holds and so the
numbers can be regenerated when the weights change.

Reads (all tier-3 bulk, restore from the archive if absent):
    data/simulated/test_set_for_R.csv                     -- matrices + generating class
    results/mle_fits/compare_to_r_python_mle_cache.csv    -- AIC-selected class per matrix

Existing blocks in the JSON (envelope_v09_production, timings) are preserved; only
`model_selection` is rewritten.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "packages", "grintools")))

import grintools as gt  # noqa: E402  (needs the path inserts above)
from src import grt_model as gm  # noqa: E402
from src.config import SIMULATED_DATA_DIR, MLE_FITS_DIR, RESULTS_DIR  # noqa: E402

CSV = os.path.join(SIMULATED_DATA_DIR, "test_set_for_R.csv")
MLE_CACHE = os.path.join(MLE_FITS_DIR, "compare_to_r_python_mle_cache.csv")
OUT = os.path.join(RESULTS_DIR, "model_selection_envelope.json")

_CORR = ["PI", "RHO1", "free"]


def components(name):
    """'rho1_psa_ds' -> ('RHO1', sep_A=True, sep_B=False).

    Decisional separability is assumed throughout, so the trailing `ds` carries no
    information and is dropped. A name with neither `pi` nor `rho1` has a free
    correlation per stimulus.
    """
    toks = [t for t in str(name).split("_") if t != "ds"]
    corr = "PI" if "pi" in toks else ("RHO1" if "rho1" in toks else "free")
    return corr, ("ps" in toks or "psa" in toks), ("ps" in toks or "psb" in toks)


def main():
    # A parser that collapsed two classes would silently inflate every accuracy
    # below, so prove it separates all twelve before using it on data.
    assert len({components(n) for n in gm.MODEL_NAMES}) == 12, \
        "components() does not separate the 12 model classes"

    for path in (CSV, MLE_CACHE):
        if not os.path.isfile(path):
            raise SystemExit(f"missing input: {path}\n"
                             "(tier-3 bulk -- restore it from the archive)")

    df = pd.read_csv(CSV)
    mle = pd.read_csv(MLE_CACHE)
    if len(df) != len(mle):
        raise SystemExit(f"row mismatch: {len(df)} matrices vs {len(mle)} MLE fits")

    cm_cols = [c for c in df.columns if c.startswith("cm_")]
    trial_cols = [f"trials_{i}" for i in range(4)]

    truth = [components(x) for x in df["model_label"]]
    aic = [components(x) for x in mle["model"]]

    grin = []
    for _, row in df.iterrows():
        _, con = gt.infer([row[c] for c in cm_cols], [row[c] for c in trial_cols])
        grin.append((_CORR[int(np.argmax(con["p_corr"]))],
                     con["p_sep_A"] > 0.5, con["p_sep_B"] > 0.5))

    def acc(pred, idx):
        return float(np.mean([p[idx] == t[idx] for p, t in zip(pred, truth)]))

    def joint_sep(pred):
        return float(np.mean([p[1] == t[1] and p[2] == t[2]
                              for p, t in zip(pred, truth)]))

    block = {
        "n": len(df),
        "aicbic_parsed_ok": len(mle),
        "separability": {"grin": joint_sep(grin), "aicbic": joint_sep(aic)},
        "corr_structure": {"grin": acc(grin, 0), "aicbic": acc(aic, 0)},
        "sepA": {"grin": acc(grin, 1), "aicbic": acc(aic, 1)},
        "sepB": {"grin": acc(grin, 2), "aicbic": acc(aic, 2)},
        "generated_by": "scripts/model_selection_envelope.py",
        "model_sha256": json.load(open(os.path.join(
            os.path.dirname(gt.default_model_path()),
            "model_provenance.json"), encoding="utf-8"))["sha256"],
    }

    existing = {}
    if os.path.isfile(OUT):
        with open(OUT, encoding="utf-8") as fh:
            existing = json.load(fh)
        old = existing.get("model_selection", {})
        for key in ("separability", "corr_structure", "sepA", "sepB"):
            if key in old:
                for who in ("grin", "aicbic"):
                    was, now = old[key].get(who), block[key][who]
                    flag = "" if was is not None and abs(was - now) < 5e-4 else "  <-- CHANGED"
                    print(f"  {key:15} {who:7} {was!s:<20} -> {now:.6f}{flag}")
        # Timing lives in the same block but is measured elsewhere; carry it over
        # rather than silently dropping it.
        for key in ("ms_per_matrix", "speedup"):
            if key in old:
                block[key] = old[key]

    existing["model_selection"] = block
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=1)
        fh.write("\n")
    print(f"wrote {os.path.relpath(OUT, os.path.dirname(os.path.dirname(__file__)))}")


if __name__ == "__main__":
    main()
