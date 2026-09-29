"""
grin_onnx.py: torch-free GRIN inference from the exported ONNX model.

This is the distributable wrapper. It depends only on numpy + onnxruntime. Retrain
your pipeline, re-export the .onnx, and this file is unchanged. The graph takes raw
counts (B,16) and trials (B,4) and returns parameter-space mean/std plus the trained
construct heads, so there is no featurisation, no link functions, and no sampling to
reproduce here.

    from grin_onnx import GrinOnnx
    grin = GrinOnnx("web/assets/models/cm/npe_model.onnx")
    result, constructs = grin(counts_4x4)          # trials default to row sums

`result` is compatible with grin_io.Criterion (.params/.std/.ci_low/.ci_high/.names).
`constructs` matches the keys grin_io's probability targets expect.
"""
import hashlib
import json
import os
from importlib import resources

import numpy as np
import onnxruntime as ort

try:
    from grt_model import PARAM_NAMES
except Exception:
    try:
        from src.grt_model import PARAM_NAMES
    except Exception:
        PARAM_NAMES = ([f"zx_{i}" for i in range(4)] + [f"zy_{i}" for i in range(4)]
                       + [f"rho_{i}" for i in range(4)])

# argmax(p_corr) index -> correlation-structure label
_CORR_LABEL = {0: "PI", 1: "RHO1", 2: "free"}


# Per-family posterior scale factors, fitted on held-out simulations by
# scripts/fit_recalibration.py in the research repository and validated on a further
# held-out set. Applied ONLY when the caller asks for it: see infer(calibrated=...).
_RECAL_CACHE = {}


def _recalibration():
    """Load the shipped scale factors, or None if this build has none."""
    if "spec" not in _RECAL_CACHE:
        import json
        try:
            path = resources.files("grintools").joinpath("models", "recalibration.json")
            _RECAL_CACHE["spec"] = json.loads(path.read_text())
        except Exception:
            _RECAL_CACHE["spec"] = None
    return _RECAL_CACHE["spec"]


def _recal_scales(spec, n=12):
    s = np.ones(n)
    if spec:
        s[0:8] = spec["global_scale"]["z"]
        s[8:12] = spec["global_scale"]["rho"]
    return s


class OnnxResult:
    """InferenceResult-compatible posterior from the ONNX marginal-Gaussian head.

    `calibrated=True` widens the intervals by the per-family factors described above.
    Point estimates are never affected, so model selection and the fitted perceptual
    space are identical either way -- only the stated uncertainty changes.
    """
    def __init__(self, mean, std, model_class, calibrated=False):
        self.params = np.asarray(mean, float)
        self.std_raw = np.asarray(std, float)
        self.calibrated = bool(calibrated)
        self.scale = _recal_scales(_recalibration() if calibrated else None)
        self.std = self.std_raw * self.scale
        self.ci_low = self.params - 1.645 * self.std      # 90% marginal, Gaussian
        self.ci_high = self.params + 1.645 * self.std
        self.names = PARAM_NAMES
        self.model_class = model_class
        self.samples = None                                # ONNX head is analytic

    def summary(self):
        tag = " [calibrated]" if self.calibrated else ""
        lines = [f"GRIN inference (onnx){tag}", "-" * 46]
        for i, n in enumerate(self.names):
            lines.append(f"  {n:7s} = {self.params[i]:+.2f}  +/- {self.std[i]:.2f}"
                         f"   [90% {self.ci_low[i]:+.2f}, {self.ci_high[i]:+.2f}]")
        lines.append("-" * 46)
        lines.append(f"  componentwise modal structure : {self.model_class}")
        if not self.calibrated:
            lines.append("  intervals are the network's own; pass calibrated=True for")
            lines.append("  width-corrected intervals (see the package documentation)")
        return "\n".join(lines)


def _class_label(p_corr, p_sep_a, p_sep_b):
    corr = _CORR_LABEL[int(np.argmax(p_corr))]
    parts = [corr]
    parts.append("PS(A)" if p_sep_a >= 0.5 else "!PS(A)")
    parts.append("PS(B)" if p_sep_b >= 0.5 else "!PS(B)")
    return " + ".join(parts)


def _decision(probability, evidence_tol):
    """Direction of evidence: 'for', 'against', or 'undecided'."""
    lower = 0.5 - evidence_tol / 2.0
    if probability < lower:
        return "against"
    if probability > 1.0 - lower:
        return "for"
    return "undecided"


def verify_bundled_model(path):
    """Check a bundled model against the `model_provenance.json` written beside it.

    Returns the sidecar dict, or None when there is no sidecar -- a user-supplied
    `model_path` is not covered by one and is not our business to police.

    This exists because of a real failure: between 2026-08-25 and 2026-09-29 the
    packages shipped the PRE-release checkpoint while the web app served v1.0.0,
    with the release-era recalibration factors applied on top. Nothing anywhere
    compared the two, so the only symptom was a worked example in the paper whose
    numbers no user could reproduce. A mismatch is now loud and immediate.
    """
    sidecar_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                "model_provenance.json")
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path, encoding="utf-8") as fh:
        sidecar = json.load(fh)
    if sidecar.get("model_file") != os.path.basename(path):
        return None                                   # sidecar describes a different file

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != sidecar.get("sha256"):
        raise RuntimeError(
            f"bundled GRIN model does not match its provenance record.\n"
            f"  file     {path}\n"
            f"  expected {sidecar.get('sha256')}  (version {sidecar.get('version')})\n"
            f"  actual   {actual}\n"
            "Reinstall grintools, or re-run scripts/export_onnx.py --install.")
    return sidecar


class GrinOnnx:
    def __init__(self, path, verify=True):
        if verify:
            self.provenance = verify_bundled_model(path)
        else:
            self.provenance = None
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]

    def __call__(self, counts, trials=None, evidence_tol=0.5, calibrated=False):
        """counts: (4,4) or length-16 canonical-order counts. Returns (OnnxResult, constructs)."""
        counts = np.asarray(counts, dtype=np.float32).reshape(1, 16)
        if trials is None:
            trials = counts.reshape(1, 4, 4).sum(2)
        trials = np.asarray(trials, dtype=np.float32).reshape(1, 4)
        mean, std, p_corr, p_sep = self.session.run(
            None, {"counts": counts, "trials": trials})
        p_pi = float(p_corr[0, 0]); p_a = float(p_sep[0, 0]); p_b = float(p_sep[0, 1])
        band = 0.5 - evidence_tol / 2.0                    # matches model_posterior's flag
        constructs = {
            "p_PI": p_pi, "p_sep_A": p_a, "p_sep_B": p_b,
            "p_corr": [float(x) for x in p_corr[0]],       # [PI, RHO1, free]
            "decision_PI": _decision(p_pi, evidence_tol),
            "decision_sep_A": _decision(p_a, evidence_tol),
            "decision_sep_B": _decision(p_b, evidence_tol),
            # Back-compatible decisiveness flags. These do not encode direction;
            # prefer decision_* in reporting code.
            "evidence_PI": bool(abs(p_pi - 0.5) > band),
            "evidence_sep_A": bool(abs(p_a - 0.5) > band),
            "evidence_sep_B": bool(abs(p_b - 0.5) > band),
        }
        result = OnnxResult(mean[0], std[0], _class_label(p_corr[0], p_a, p_b),
                            calibrated=calibrated)
        return result, constructs

