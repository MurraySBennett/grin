"""
model_selection.py — first-cut model-class inference from the posterior.

Chooses the most parsimonious class whose constraints the posterior does NOT rule
out (credible interval of the relevant quantity contains 0). This is a fast
heuristic; Phase 3 will validate/replace it with AIC/BIC model selection computed
through grt_model's exact forward likelihood, cross-checked against grtools/mdsdt.
"""
try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm


def _ci_contains_zero(x, alpha=0.1):
    return bool((x.quantile(alpha / 2) <= 0) and (x.quantile(1 - alpha / 2) >= 0))


def infer_class(posterior_samples, alpha=0.1):
    """posterior_samples: (S,12) for ONE matrix, in parameter space -> class name."""
    s = posterior_samples
    zx, zy, rho = s[:, 0:4], s[:, 4:8], s[:, 8:12]
    pi = all(_ci_contains_zero(rho[:, i], alpha) for i in range(4))
    rho1 = all(_ci_contains_zero(rho[:, i] - rho[:, j], alpha)
               for i in range(4) for j in range(i + 1, 4))
    psa = (_ci_contains_zero(zx[:, 0] - zx[:, 1], alpha)
           and _ci_contains_zero(zx[:, 2] - zx[:, 3], alpha))
    psb = (_ci_contains_zero(zy[:, 0] - zy[:, 2], alpha)
           and _ci_contains_zero(zy[:, 1] - zy[:, 3], alpha))
    corr = "pi" if pi else ("rho1" if rho1 else "free")
    for name, (c, a, b) in gm.MODEL_SPECS.items():
        if c == corr and a == psa and b == psb:
            return name
    return "ds"