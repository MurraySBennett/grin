"""
recalibrate.py — optional post-hoc rescaling of GRIN's posterior width.

GRIN's interval calibration differs by parameter family: the marginal sensitivities'
intervals are conservative, the within-stimulus correlations' are modestly
overconfident. scripts/fit_recalibration.py estimates a per-family scale factor from
held-out simulations and validates it on a further held-out set; this module applies it.

OPT-IN BY DESIGN. Nothing here runs unless a caller asks for it. `predict_posterior`
and the packages' `infer()` return the network's own posterior by default, and the
rescaled version is a distinct, explicitly requested object. Three reasons:

  * The correction is estimated under the training prior and inherits its coverage.
    For observers in regions the prior samples thinly it may not transfer, and a user
    cannot tell from their own data whether it has.
  * A rescaled interval is a calibrated interval DERIVED FROM the posterior, not the
    posterior itself. Returning it silently would change what `infer()` means.
  * The uncorrected behaviour is documented and reproducible; a default correction
    would make published GRIN results depend on which package version produced them.

The scale multiplies the posterior standard deviation, which for the Gaussian head is
equivalent to widening every credible interval by the same factor. Point estimates are
untouched, so model selection, recovery error, and the perceptual space are unaffected.
"""
import json
import os

import numpy as np

DEFAULT_PATH = os.path.join("results", "models", "recalibration.json")
FAM_SLICES = {"z": slice(0, 8), "rho": slice(8, 12)}


class Recalibration:
    """Per-family posterior scale factors, with optional trial-count dependence."""

    def __init__(self, spec):
        self.spec = spec
        self.mode = spec.get("recommended", "global")
        self.global_scale = spec["global_scale"]
        self.by_trials = spec.get("by_trials", {})

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no recalibration file at {path}; run scripts/fit_recalibration.py")
        with open(path) as f:
            return cls(json.load(f))

    def scales(self, n_params=12, trials_per_stimulus=None):
        """(n_params,) or (N, n_params) multipliers for the posterior SD."""
        if trials_per_stimulus is None or self.mode == "global":
            s = np.ones(n_params)
            for fam, sl in FAM_SLICES.items():
                s[sl] = self.global_scale[fam]
            return s
        tps = np.atleast_1d(np.asarray(trials_per_stimulus, float))
        s = np.ones((len(tps), n_params))
        for fam, sl in FAM_SLICES.items():
            col = np.full(len(tps), self.global_scale[fam])
            for k, v in self.by_trials.get(fam, {}).items():
                lo, hi = (float(x) for x in k.split("-"))
                col[(tps >= lo) & (tps < hi)] = v
            s[:, sl] = col[:, None]
        return s

    def apply(self, mean, std, trials_per_stimulus=None, levels=(0.9,)):
        """Rescale std and rebuild central intervals around the unchanged mean."""
        from scipy.stats import norm
        mean = np.asarray(mean, float); std = np.asarray(std, float)
        s = self.scales(mean.shape[-1], trials_per_stimulus)
        std_c = std * s
        out = {"mean": mean, "std": std_c, "scale": s}
        for lv in levels:
            h = norm.ppf(0.5 + lv / 2) * std_c
            out[f"ci_low_{lv:g}"] = mean - h
            out[f"ci_high_{lv:g}"] = mean + h
        return out

    def __repr__(self):
        g = ", ".join(f"{k}={v:.3f}" for k, v in self.global_scale.items())
        return f"Recalibration(mode={self.mode}, {g})"
