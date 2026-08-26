"""
grintools: amortised Bayesian inference for General Recognition Theory (GRT).

Feed a 2x2 identification confusion matrix, get a calibrated posterior over the 12
GRT parameters plus construct probabilities (perceptual independence, separability),
and an optional stopping decision for adaptive designs. Torch-free at run time: the
trained network ships as an ONNX graph and runs under onnxruntime.

    import grintools as gt
    result, constructs = gt.infer([[71,17,9,5],[20,67,5,9],[13,6,63,20],[5,10,15,71]])
    print(result.summary())

The confusion matrix must be in canonical order (rows and columns A1B1, A1B2, A2B1,
A2B2). If yours is not, normalise it first with gt.to_confusion(...) using labels and
factor levels, or assert gt.to_confusion(M, order="canonical").
"""
from importlib import resources

from .io import (to_confusion, describe, empirical_bias, response_bias,
                 ConfusionInput, PARAM_NAMES, PARAM_GROUPS)
from .criterion import Criterion, Target, Decision, stop_on_precision
from .onnx import GrinOnnx, OnnxResult

__version__ = "0.1.0"

__all__ = ["infer", "to_confusion", "describe", "empirical_bias", "response_bias",
           "ConfusionInput", "Criterion", "Target", "Decision", "stop_on_precision",
           "GrinOnnx", "OnnxResult", "default_model_path", "PARAM_NAMES",
           "PARAM_GROUPS", "__version__"]

_SESSION_CACHE = {}


def default_model_path():
    """Filesystem path to the ONNX model bundled with this release."""
    return str(resources.files("grintools").joinpath("models", "npe_model.onnx"))


def infer(counts, trials=None, model_path=None, calibrated=False):
    """Run GRIN on a canonical-order 4x4 (or length-16) count matrix.

    Returns (OnnxResult, constructs). `trials` defaults to row sums. Pass model_path
    to use a specific .onnx; otherwise the bundled model is used and cached.

    calibrated: False (default) returns the network's own posterior, which is what
    every result in the GRIN paper is based on. True additionally rescales the
    interval widths by per-family factors fitted on held-out simulations, correcting
    a known asymmetry -- the sensitivity intervals run wider than nominal and the
    correlation intervals narrower. Point estimates are unchanged either way. The
    correction is estimated under the training prior and may not transfer to observers
    far outside it, which is why it is opt-in.
    """
    path = model_path or default_model_path()
    grin = _SESSION_CACHE.get(path)
    if grin is None:
        grin = _SESSION_CACHE[path] = GrinOnnx(path)
    return grin(counts, trials, calibrated=calibrated)
