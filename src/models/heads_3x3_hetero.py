"""Parameter transforms for the 45-parameter heteroscedastic 3x3 model."""

import torch


N_PARAMS = 45


def params_to_train_space(params):
    means = params[..., :18]
    standard_deviations = params[..., 18:36].clamp_min(1e-6)
    rho = params[..., 36:45].clamp(-0.999, 0.999)
    return torch.cat([means, torch.log(standard_deviations), torch.atanh(rho)], dim=-1)


def train_space_to_params(values):
    means = values[..., :18]
    standard_deviations = torch.exp(values[..., 18:36].clamp(-10.0, 10.0))
    rho = torch.tanh(values[..., 36:45].clamp(-7.0, 7.0))
    return torch.cat([means, standard_deviations, rho], dim=-1)
