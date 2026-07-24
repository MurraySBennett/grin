"""
Public inference API for GRIN — the three-line interface.

    from grin.api import infer          # (or: from grin import infer)
    result = infer(confusion_matrix, trials)
    print(result.summary())

`load_model` is robust to how the checkpoint was saved: it accepts either a rich
checkpoint (state_dict + architecture) or a bare weights-only file, and always
rebuilds the network with the matching architecture (dropout included), so the
"missing/unexpected key" mismatch can't happen.
"""
import numpy as np
import torch

from .config import MODEL_FILE, HIDDEN_LAYERS, ACTIVATION, DROPOUT
from .models.network import NPEModel
from .inference.predict import predict_posterior
from .inference.model_selection import infer_class
from .inference.ood import ood_deviance

try:
    from . import grt_model as gm
except ImportError:
    import grt_model as gm


class InferenceResult:
    """The recovered representation for one confusion matrix, with uncertainty."""
    def __init__(self, params, std, ci_low, ci_high, model_class, fit_deviance, samples):
        self.params = params            # (12,) posterior mean, param space
        self.std = std                  # (12,) marginal posterior SD
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.model_class = model_class  # inferred GRT model label
        self.fit_deviance = float(fit_deviance)
        self.samples = samples          # (S,12) posterior draws
        self.names = gm.PARAM_NAMES

    def as_dict(self):
        return {n: {"estimate": float(self.params[i]), "sd": float(self.std[i]),
                    "ci90": [float(self.ci_low[i]), float(self.ci_high[i])]}
                for i, n in enumerate(self.names)}

    def summary(self):
        lines = ["GRIN inference", "-" * 46]
        for i, n in enumerate(self.names):
            lines.append(f"  {n:7s} = {self.params[i]:+.2f}  ± {self.std[i]:.2f}"
                         f"   [90% {self.ci_low[i]:+.2f}, {self.ci_high[i]:+.2f}]")
        lines.append("-" * 46)
        lines.append(f"  inferred model : {self.model_class}")
        flag = "ok" if self.fit_deviance < 40 else "CHECK — may be out of model family"
        lines.append(f"  fit / OOD      : deviance {self.fit_deviance:.1f}  ({flag})")
        return "\n".join(lines)


def load_model(path=MODEL_FILE, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:              # rich checkpoint
        hidden = tuple(ckpt.get("hidden", HIDDEN_LAYERS))
        dropout = ckpt.get("dropout", DROPOUT)
        activation = ckpt.get("activation", ACTIVATION)
        comparison = ckpt.get("comparison", False)
        state = ckpt["state_dict"]
    else:                                                            # bare weights
        hidden, dropout, activation, comparison, state = HIDDEN_LAYERS, DROPOUT, ACTIVATION, False, ckpt
    model = NPEModel(hidden=hidden, activation=activation, dropout=dropout, comparison=comparison)
    model.load_state_dict(state)                                     # architecture now matches
    return model.to(device).eval()


def infer(confusion_matrix, trials=None, model=None, n_samples=1000):
    """
    confusion_matrix : (4,4) or length-16 array of response COUNTS (stimulus-major).
    trials           : length-4 per-stimulus totals; if None, taken from row sums.
    Returns an InferenceResult.
    """
    if model is None:
        model = load_model()
    counts = np.asarray(confusion_matrix, float).reshape(1, 16)
    if trials is None:
        trials = counts.reshape(1, 4, 4).sum(2)
    else:
        trials = np.asarray(trials, float).reshape(1, 4)
    post = predict_posterior(model, counts, trials, n_samples=n_samples)
    samples = post["samples"].numpy()[:, 0, :]
    cls = infer_class(torch.tensor(samples))
    dev = ood_deviance(model, counts, trials)[0]
    return InferenceResult(post["mean"][0].numpy(), post["std"][0].numpy(),
                           post["ci_low"][0].numpy(), post["ci_high"][0].numpy(),
                           cls, dev, samples)
    