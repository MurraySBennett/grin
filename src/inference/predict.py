"""predict.py — extract a posterior from a trained NPE model, in parameter space.

Runs on whatever device the model is on (inferred from its parameters), so it
follows the model to GPU automatically — no device argument to remember. Batched
over matrices so it scales to large test sets without allocating a giant
(n_samples x N x P x P) tensor inside the multivariate-normal sampler.
"""
import torch
from ..models.network import featurize
from ..models.heads import train_space_to_params


def _model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def predict_posterior(model, counts, trials, n_samples=2000, batch_size=1024):
    model.eval()
    device = _model_device(model)
    counts = torch.as_tensor(counts)
    trials = torch.as_tensor(trials)
    N = counts.shape[0]
    means, stds, los, his, samps = [], [], [], [], []
    for i in range(0, N, batch_size):
        x = featurize(counts[i:i + batch_size], trials[i:i + batch_size]).to(device)
        s = train_space_to_params(model.distribution(x).sample((n_samples,)))   # (S,b,12)
        means.append(s.mean(0)); stds.append(s.std(0))
        los.append(s.quantile(0.05, dim=0)); his.append(s.quantile(0.95, dim=0))
        samps.append(s)
    return {
        "mean":    torch.cat(means, 0).cpu(),
        "std":     torch.cat(stds, 0).cpu(),
        "ci_low":  torch.cat(los, 0).cpu(),
        "ci_high": torch.cat(his, 0).cpu(),
        "samples": torch.cat(samps, 1).cpu(),      # (S, N, 12)
    }


@torch.no_grad()
def predict_point(model, counts, trials):
    """Fast point estimate (posterior mean, no sampling) -> (N,12), on CPU."""
    model.eval()
    device = _model_device(model)
    x = featurize(torch.as_tensor(counts), torch.as_tensor(trials)).to(device)
    mean_train, _ = model(x)
    return train_space_to_params(mean_train).cpu()
