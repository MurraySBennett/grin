"""
Cross-runtime parity: the ONNX graph vendored by grintools and the TorchScript
graph vendored by the R package must be the same network.

The manuscript claims the two packages "wrap the same trained weights" and are
"verified to agree within floating-point tolerance". Until 2026-09-29 nothing
checked that. The R-side test compared hard-coded constants against the R package
alone, so it could only detect the R export drifting from a snapshot -- not the
two packages drifting from each other, and not either drifting from the released
checkpoint. Both packages shipped the 2026-08-12 pre-release network for six weeks
while the web app served v1.0.0, and every test passed.

This test runs both graphs on the same inputs and compares all four outputs. It
needs torch only to *read* the TorchScript file; it is skipped, not failed, where
torch is unavailable -- but the hash checks below run regardless, because a wrong
bundled model is a packaging fault rather than an environment one.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY_MODEL = os.path.join(REPO, "packages", "grintools", "grintools", "models",
                        "npe_model.onnx")
R_MODEL = os.path.join(REPO, "packages", "grin", "inst", "models", "npe_model_ts.pt")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(model_path):
    with open(os.path.join(os.path.dirname(model_path), "model_provenance.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("model_path", [PY_MODEL, R_MODEL])
def test_bundled_model_matches_its_sidecar(model_path):
    rec = _sidecar(model_path)
    assert rec["model_file"] == os.path.basename(model_path)
    assert _sha256(model_path) == rec["sha256"]


def test_both_packages_were_built_from_one_checkpoint():
    """The comparison whose absence let the drift ship."""
    py, r = _sidecar(PY_MODEL), _sidecar(R_MODEL)
    assert py["checkpoint_sha256"] == r["checkpoint_sha256"]
    assert py["version"] == r["version"]


def test_onnx_and_torchscript_agree():
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")

    ts = torch.jit.load(R_MODEL).eval()
    sess = ort.InferenceSession(PY_MODEL, providers=["CPUExecutionProvider"])

    # One real observer (thomas01a, the worked example in the paper) plus random
    # matrices spanning sparse and dense counts.
    cases = [torch.tensor([[83, 112, 47, 11, 38, 154, 28, 33,
                            15, 27, 117, 94, 6, 36, 75, 136]], dtype=torch.float32)]
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        cases.append(torch.randint(0, 60, (1, 16), generator=g).float())

    worst = 0.0
    for counts in cases:
        trials = counts.reshape(1, 4, 4).sum(-1)
        with torch.no_grad():
            a = ts(counts, trials)
        b = sess.run(None, {"counts": counts.numpy(), "trials": trials.numpy()})
        for x, y in zip(a, b):
            worst = max(worst, float(np.abs(x.numpy() - y).max()))

    assert worst < 1e-5, f"ONNX and TorchScript disagree by {worst:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
