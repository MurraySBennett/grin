"""
Generate the reference constants pinned by packages/grin/tests/testthat/test-parity.R.
Run from the repo root: `python tests/gen_parity_reference.py`, then paste the
printed block over the one in that test.

Why this exists: those constants used to be produced by hand from "the Python
package", with no record of which weights were inside it. Between 2026-08-25 and
2026-09-29 the weights inside it were the PRE-release checkpoint, so the constants
and the bundled model agreed with each other and the test passed while both
packages shipped the wrong network. This script reads the bundled model through
grintools' own loader -- which verifies the model against its provenance sidecar --
and prints the sidecar's hashes into the comment, so the next reader can tell which
weights a given set of constants describes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "packages", "grintools")))

import grintools as gt  # noqa: E402  (needs the path inserts above)

# The same matrix the R test uses. Keep the two in step.
M = [[71, 17, 9, 5],
     [20, 67, 5, 9],
     [13, 6, 63, 20],
     [5, 10, 15, 71]]


def _c_vector(values, per_line=4, indent=16):
    cells = [f"{v:.4f}" for v in values]
    rows = [", ".join(cells[i:i + per_line]) for i in range(0, len(cells), per_line)]
    return (",\n" + " " * indent).join(rows)


def main():
    path = gt.default_model_path()
    result, constructs = gt.infer(M)

    sidecar_path = os.path.join(os.path.dirname(path), "model_provenance.json")
    with open(sidecar_path, encoding="utf-8") as fh:
        sidecar = json.load(fh)

    print(f"  # Generated from grintools with the v{sidecar['version']} ONNX,")
    print(f"  # sha256 {sidecar['sha256']},")
    print(f"  # itself exported from checkpoint {sidecar.get('checkpoint_file')} "
          f"sha256 {sidecar.get('checkpoint_sha256')}.")
    print(f"  ref_mean <- c({_c_vector(result.params)})")
    print(f"  ref_std  <- c({_c_vector(result.std)})")
    print("  ref_p_corr <- c("
          + ", ".join(f"{v:.4f}" for v in constructs["p_corr"]) + ")")
    print("  ref_p_sep  <- c("
          + f"{constructs['p_sep_A']:.4f}, {constructs['p_sep_B']:.4f}" + ")")


if __name__ == "__main__":
    main()
