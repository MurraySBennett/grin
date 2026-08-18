"""
run_all.py — run the GRIN validation suite.

    python validation/run_all.py               # everything
    python validation/run_all.py --quick       # fast smoke version
    python validation/run_all.py --only v14    # one check
    python validation/run_all.py --list        # show the manifest

Writes results/validation/<id>.json plus a summary table (results/validation/SUMMARY.md).
Every claim in DESIGN_RECORD.md maps to a check here.

SCOPE, encoded here so it can't drift out of sync with the generated summary (see
validation/README_validation.md for the long version): every check below trains its
own small, fresh network for that run -- none load the released production checkpoint
(results/models/npe_model.pt). This is a development/CI regression suite, not
production-checkpoint validation; the manuscript's headline numbers come from
separate, larger evaluations (results/validation/manuscript_recovery.json,
results/rt_metrics.json, results/mle_fits/baseline_fits.csv), not from this table.
GATE_IDS below are checks with a real threshold in their own "pass" condition;
everything else runs unconditionally and its "pass" records only that it produced
output, not that anything was met -- SUMMARY.md must keep that distinction visible
rather than render every row as a uniform PASS.
"""
import argparse, json, os, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR
from validation import checks as C
from validation import checks_rt as R

REGISTRY = [
    ("v01", C.v01_forward_exactness), ("v02", C.v02_prior_coverage),
    ("v03", C.v03_recovery), ("v04", C.v04_calibration), ("v05", C.v05_speed),
    ("v06", C.v06_ensemble), ("v07", C.v07_trial_sweep), ("v08", C.v08_pi_frontier),
    ("v09", C.v09_ood), ("v10", C.v10_degradation), ("v11", C.v11_amortized_comparison),
    ("v12", R.v12_rt_collinearity), ("v13", R.v13_rt_gain),
    ("v14", R.v14_architecture), ("v15", R.v15_speed_confound), ("v16", R.v16_lba_confound),
]

# Checks whose own "pass" condition is a stated quantitative threshold (see the
# literal `"pass": ...` expression in checks.py/checks_rt.py for each). Everything
# NOT listed here runs unconditionally ("pass": True hardcoded) and has no
# pass/fail criterion at all -- update this set if a check's own threshold changes,
# not just the markdown this script writes.
GATE_IDS = {"v01", "v02", "v03", "v04", "v05", "v06", "v09", "v14", "v15", "v16"}

SCOPE_HEADER = """# GRIN validation summary

**This is the development/CI smoke-check ledger, not production-checkpoint
validation.** Every row below trains its own small, fresh network for that run
(2,500 examples/class, 20 epochs by default; v06 uses 1,200/12) -- none load the
released production checkpoint (`results/models/npe_model.pt`). See
`validation/README_validation.md` for the full scope note. The manuscript's
headline numbers come from separate, larger, production-checkpoint evaluations
(`results/validation/manuscript_recovery.json`, `results/rt_metrics.json`,
`results/mle_fits/baseline_fits.csv`), not from this table.

`GATE` below means the check has a stated quantitative pass/fail threshold and the
network was required to clear it. `REPORT` means there is no threshold -- the check
runs unconditionally and its "PASS" records only that it produced output, not that
anything was met.

"""


def _json(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, _ in REGISTRY: print(i)
        return

    out_dir = os.path.join(RESULTS_DIR, "validation")
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(n_per_class=250, epochs=6) if a.quick else {}
    rows = []
    for cid, fn in REGISTRY:
        if a.only and cid != a.only:
            continue
        t0 = time.time()
        try:
            res = fn(**kw)
            ok = res.get("pass", True)
        except Exception as e:
            traceback.print_exc()
            res = {"id": cid, "claim": "ERROR", "result": {"error": str(e)}, "pass": False}
            ok = False
        dt = time.time() - t0
        res["seconds"] = round(dt, 1)
        with open(os.path.join(out_dir, f"{cid}.json"), "w") as f:
            json.dump(_json(res), f, indent=2)
        kind = "GATE" if cid in GATE_IDS else "REPORT"
        flag = "PASS" if ok else "FAIL"
        status = flag if kind == "GATE" else ("ran, reported" if ok else "FAIL")
        print(f"[{flag}] {cid}  {res['claim']}  ({dt:.0f}s)  [{kind}]")
        print(f"        {json.dumps(_json(res['result']))[:150]}")
        rows.append((cid, res["claim"], status, kind, res["seconds"]))

    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as f:
        f.write(SCOPE_HEADER)
        f.write("| id | claim | status | type | sec |\n|---|---|---|---|---|\n")
        for cid, claim, status, kind, sec in rows:
            f.write(f"| {cid} | {claim} | {status} | {kind} | {sec} |\n")
    print(f"\nwrote {out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
