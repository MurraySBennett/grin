"""
heads.py — NPE distributional head + link transforms.

The head maps encoder features to a full-covariance Gaussian over the 12 identified
parameters, expressed in an UNCONSTRAINED "train space": the 8 z-scores stay as-is
(real-valued), and the 4 correlations are modelled via Fisher-z = atanh(rho), so a
Gaussian there always maps back to valid correlations in (-1, 1) via tanh. The
full covariance (not diagonal) captures posterior correlations between parameters
(e.g. the z-score / correlation trade-off), which diagonal heads miss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PARAMS = 12
Z_SLICE = slice(0, 8)       # zx_0..3, zy_0..3  (real-valued)
RHO_SLICE = slice(8, 12)    # rho_0..3          (Fisher-z in train space)


def params_to_train_space(params):
    z = params[..., Z_SLICE]
    rho = params[..., RHO_SLICE].clamp(-0.999, 0.999)
    return torch.cat([z, torch.atanh(rho)], dim=-1)


def train_space_to_params(t):
    z = t[..., Z_SLICE]
    # clamp Fisher-z before tanh so reported correlations stay strictly in (-1,1)
    # (a rho of exactly +/-1 would make a reconstructed covariance singular)
    rho = torch.tanh(t[..., RHO_SLICE].clamp(-7.0, 7.0))
    return torch.cat([z, rho], dim=-1)


def build_scale_tril(diag_raw, lower_raw, dim=N_PARAMS, eps=1e-5):
    """Assemble a valid lower-triangular Cholesky factor (positive diagonal)."""
    batch = diag_raw.shape[:-1]
    L = torch.zeros(*batch, dim, dim, device=diag_raw.device, dtype=diag_raw.dtype)
    diag = F.softplus(diag_raw) + eps
    idx = torch.arange(dim, device=diag_raw.device)
    L[..., idx, idx] = diag
    ti, tj = torch.tril_indices(dim, dim, offset=-1, device=diag_raw.device)
    L[..., ti, tj] = lower_raw
    return L


class GaussianHead(nn.Module):
    def __init__(self, in_features, dim=N_PARAMS):
        super().__init__()
        self.dim = dim
        self.mean = nn.Linear(in_features, dim)
        self.diag = nn.Linear(in_features, dim)
        self.lower = nn.Linear(in_features, dim * (dim - 1) // 2)

    def forward(self, h):
        return self.mean(h), build_scale_tril(self.diag(h), self.lower(h), self.dim)

    def distribution(self, h):
        mean, L = self.forward(h)
        return torch.distributions.MultivariateNormal(mean, scale_tril=L)
    