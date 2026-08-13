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


class OnnxResult:
    """InferenceResult-compatible posterior from the ONNX marginal-Gaussian head."""
    def __init__(self, mean, std, model_class):
        self.params = np.asarray(mean, float)
        self.std = np.asarray(std, float)
        self.ci_low = self.params - 1.645 * self.std      # 90% marginal, Gaussian
        self.ci_high = self.params + 1.645 * self.std
        self.names = PARAM_NAMES
        self.model_class = model_class
        self.samples = None                                # ONNX head is analytic

    def summary(self):
        lines = ["GRIN inference (onnx)", "-" * 46]
        for i, n in enumerate(self.names):
            lines.append(f"  {n:7s} = {self.params[i]:+.2f}  +/- {self.std[i]:.2f}"
                         f"   [90% {self.ci_low[i]:+.2f}, {self.ci_high[i]:+.2f}]")
        lines.append("-" * 46)
        lines.append(f"  most likely structure : {self.model_class}")
        return "\n".join(lines)


def _class_label(p_corr, p_sep_a, p_sep_b):
    corr = _CORR_LABEL[int(np.argmax(p_corr))]
    parts = [corr]
    parts.append("PS(A)" if p_sep_a >= 0.5 else "!PS(A)")
    parts.append("PS(B)" if p_sep_b >= 0.5 else "!PS(B)")
    return " + ".join(parts)


class GrinOnnx:
    def __init__(self, path):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]

    def __call__(self, counts, trials=None, evidence_tol=0.5):
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
            "evidence_PI": bool(abs(p_pi - 0.5) > band),
            "evidence_sep_A": bool(abs(p_a - 0.5) > band),
            "evidence_sep_B": bool(abs(p_b - 0.5) > band),
        }
        result = OnnxResult(mean[0], std[0], _class_label(p_corr[0], p_a, p_b))
        return result, constructs
        