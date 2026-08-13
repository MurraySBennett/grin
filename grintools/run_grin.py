#!/usr/bin/env python3
"""
run_grin.py: end-to-end GRIN on a confusion matrix, with a stopping decision.

Torch-free: runs the exported ONNX model. Put this file, grin_io.py and grin_onnx.py
together in ANY directory (e.g. a `grin_run/` subfolder of your repo); paths are
resolved relative to this file and the discovered repo root, so it runs from anywhere.

    python run_grin.py
    python run_grin.py --csv mydata.csv --model path/to/npe_model.onnx
    python run_grin.py --construct PS_A --at-least 0.9 --sd 0.15 --combine any
    python run_grin.py --offline
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                     # find grin_io / grin_onnx next to this file
import grin_io as gio


def find_repo_root(start):
    """Walk up from `start` looking for the grin repo root (has setup.py or src/)."""
    d = start
    for _ in range(6):
        if os.path.exists(os.path.join(d, "setup.py")) or os.path.isdir(os.path.join(d, "src")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return start


REPO_ROOT = find_repo_root(HERE)
sys.path.insert(0, REPO_ROOT)                # let `src.*` imports resolve if wanted

MODEL_CANDIDATES = ["web/assets/models/cm/npe_model.onnx",
                    "results/models/npe_model.onnx", "npe_model.onnx"]


def read_matrix(path):
    with open(path) as f:
        first = f.readline().strip()
    def is_num(t):
        try:
            float(t); return True
        except ValueError:
            return False
    if all(is_num(t) for t in first.split(",")):
        return np.loadtxt(path, delimiter=",", dtype=int), None
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    labels = [c.strip() for c in rows[0]]
    mat = np.array([[int(float(x)) for x in r] for r in rows[1:]], dtype=int)
    return mat, labels


def find_model(explicit):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for base in (REPO_ROOT, os.getcwd()):    # anchored to repo root, then cwd
        for c in MODEL_CANDIDATES:
            p = os.path.join(base, c)
            if os.path.exists(p):
                return p
    return None


def resolve_csv(arg):
    if os.path.isabs(arg) or os.path.exists(arg):
        return arg
    beside = os.path.join(HERE, arg)         # default example sits next to this file
    return beside if os.path.exists(beside) else arg


def placeholder():
    class R:
        names = gio.PARAM_NAMES
    r = R()
    r.params = np.array([-1.1, -1.1, 0.9, 1.0, -0.7, 0.7, -0.7, 0.8,
                         0.15, -0.02, 0.01, 0.15], float)
    r.std = np.array([0.17] * 8 + [0.18] * 4, float)
    r.ci_low = r.params - 1.645 * r.std
    r.ci_high = r.params + 1.645 * r.std
    r.model_class = "(placeholder)"
    constructs = {"p_PI": 0.81, "p_sep_A": 0.95, "p_sep_B": 0.93,
                  "evidence_PI": True, "evidence_sep_A": True, "evidence_sep_B": True}
    return r, constructs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="example_cm.csv")
    ap.add_argument("--model", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sd", type=float, default=0.15)
    ap.add_argument("--construct", default="PS_A")
    ap.add_argument("--at-least", type=float, default=0.90)
    ap.add_argument("--combine", default="any", choices=["all", "any"])
    args = ap.parse_args()

    mat, labels = read_matrix(resolve_csv(args.csv))
    if labels is None:
        ci = gio.to_confusion(mat, order="canonical")
    else:
        ci = gio.to_confusion(mat, stim_labels=labels, resp_labels=labels,
                              factor_a=("Old", "Young"), factor_b=("Neg", "Pos"))

    print("=" * 60)
    gio.describe(ci.counts, order="canonical")
    print("=" * 60)

    criterion = gio.Criterion([
        gio.Target.precision(params=["zx", "zy"], sd_max=args.sd),
        gio.Target.probability(args.construct, at_least=args.at_least),
    ], combine=args.combine)

    if args.offline:
        print("OFFLINE: placeholder posterior (NOT inferred from your data)\n")
        result, constructs = placeholder()
    else:
        model_path = find_model(args.model)
        if model_path is None:
            raise SystemExit(
                "No ONNX model found. Pass --model path/to/npe_model.onnx (searched under\n"
                f"repo root {REPO_ROOT!r}), or try:  python run_grin.py --offline")
        from grin_onnx import GrinOnnx
        result, constructs = GrinOnnx(model_path)(ci.counts, ci.trials)
        print(f"model: {model_path}\n")

    print("posterior (mean +/- SD):")
    for i, n in enumerate(result.names):
        print(f"    {n:7s} {result.params[i]:+.2f} +/- {result.std[i]:.2f}")
    print(f"\nconstructs (P holds):  PI={constructs['p_PI']:.2f}"
          f"  PS_A={constructs['p_sep_A']:.2f}  PS_B={constructs['p_sep_B']:.2f}")
    print(f"evidence sufficient:   PI={constructs['evidence_PI']}"
          f"  PS_A={constructs['evidence_sep_A']}  PS_B={constructs['evidence_sep_B']}")

    decision = criterion.evaluate(result, constructs)
    print("\nstopping decision\n" + "-" * 60)
    print(decision.summary())
    print("-" * 60)
    print(f"\n>>> {'STOP collecting' if decision.stop else 'KEEP collecting'} <<<")


if __name__ == "__main__":
    main()
    