"""
run_all.py — run the GRIN validation suite.

    python validation/run_all.py               # everything
    python validation/run_all.py --quick       # fast smoke version
    python validation/run_all.py --only v14    # one check
    python validation/run_all.py --list        # show the manifest

Writes results/validation/<id>.json plus a summary table (results/validation/SUMMARY.md).
Every claim in DESIGN_RECORD.md maps to a check here.
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
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {cid}  {res['claim']}  ({dt:.0f}s)")
        print(f"        {json.dumps(_json(res['result']))[:150]}")
        rows.append((cid, res["claim"], flag, res["seconds"]))

    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as f:
        f.write("# GRIN validation summary\n\n| id | claim | status | sec |\n|---|---|---|---|\n")
        for cid, claim, flag, sec in rows:
            f.write(f"| {cid} | {claim} | {flag} | {sec} |\n")
    print(f"\nwrote {out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
"""
run_all.py — run the GRIN validation suite.

    python validation/run_all.py               # everything
    python validation/run_all.py --quick       # fast smoke version
    python validation/run_all.py --only v14    # one check
    python validation/run_all.py --list        # show the manifest

Writes results/validation/<id>.json plus a summary table (results/validation/SUMMARY.md).
Every claim in DESIGN_RECORD.md maps to a check here.
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
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {cid}  {res['claim']}  ({dt:.0f}s)")
        print(f"        {json.dumps(_json(res['result']))[:150]}")
        rows.append((cid, res["claim"], flag, res["seconds"]))

    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as f:
        f.write("# GRIN validation summary\n\n| id | claim | status | sec |\n|---|---|---|---|\n")
        for cid, claim, flag, sec in rows:
            f.write(f"| {cid} | {claim} | {flag} | {sec} |\n")
    print(f"\nwrote {out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
