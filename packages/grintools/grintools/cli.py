"""Command-line entry point: `grin-fit`. Installed via the console_scripts hook."""
import argparse
import os
import numpy as np

from . import io as gio
from .criterion import Criterion, Target
from . import default_model_path


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
    return np.array([[int(float(x)) for x in r] for r in rows[1:]], dtype=int), labels


def main(argv=None):
    ap = argparse.ArgumentParser(prog="grin-fit",
                                 description="Run GRIN on a confusion matrix with a stopping decision.")
    ap.add_argument("--csv", default=None, help="4x4 CSV; defaults to the bundled example")
    ap.add_argument("--model", default=None, help="path to npe_model.onnx (bundled model if omitted)")
    ap.add_argument("--sd", type=float, default=0.15, help="precision target: zx/zy SD threshold")
    ap.add_argument("--construct", default="PS_A", help="PI/PS_A/PS_B or a *_violated variant")
    ap.add_argument("--at-least", type=float, default=0.90, help="probability threshold")
    ap.add_argument("--combine", default="any", choices=["all", "any"])
    args = ap.parse_args(argv)

    csv_path = args.csv or os.path.join(os.path.dirname(__file__), "data", "example_cm.csv")
    mat, labels = read_matrix(csv_path)
    if labels is None:
        ci = gio.to_confusion(mat, order="canonical")
    else:
        ci = gio.to_confusion(mat, stim_labels=labels, resp_labels=labels,
                              factor_a=("Old", "Young"), factor_b=("Neg", "Pos"))

    print("=" * 60)
    gio.describe(ci.counts, order="canonical")
    print("=" * 60)

    from .onnx import GrinOnnx
    result, constructs = GrinOnnx(args.model or default_model_path())(ci.counts, ci.trials)

    print("posterior (mean +/- SD):")
    for i, n in enumerate(result.names):
        print(f"    {n:7s} {result.params[i]:+.2f} +/- {result.std[i]:.2f}")
    print(f"\nconstructs (P holds):  PI={constructs['p_PI']:.2f}"
          f"  PS_A={constructs['p_sep_A']:.2f}  PS_B={constructs['p_sep_B']:.2f}")

    crit = Criterion([Target.precision(params=["zx", "zy"], sd_max=args.sd),
                      Target.probability(args.construct, at_least=args.at_least)],
                     combine=args.combine)
    decision = crit.evaluate(result, constructs)
    print("\nstopping decision\n" + "-" * 60)
    print(decision.summary())
    print("-" * 60)
    print(f"\n>>> {'STOP collecting' if decision.stop else 'KEEP collecting'} <<<")


if __name__ == "__main__":
    main()
