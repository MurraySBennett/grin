"""Parameter transforms for the experimental 29-parameter 3x3 model."""

import torch


N_PARAMS = 29


def params_to_train_space(params):
    means = params[..., :18]
    rho = params[..., 18:27].clamp(-0.999, 0.999)
    bounds = params[..., 27:29].clamp_min(1e-6)
    return torch.cat([means, torch.atanh(rho), torch.log(bounds)], dim=-1)


def train_space_to_params(values):
    means = values[..., :18]
    rho = torch.tanh(values[..., 18:27].clamp(-7.0, 7.0))
    bounds = torch.exp(values[..., 27:29].clamp(-10.0, 10.0))
    return torch.cat([means, rho, bounds], dim=-1)
