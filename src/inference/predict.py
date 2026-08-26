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
def predict_posterior(model, counts, trials, n_samples=2000, batch_size=1024,
                      recalibrate=None):
    """Posterior over the twelve identified parameters.

    recalibrate: None (default) returns the network's own posterior. Pass a
    Recalibration, or True to load the shipped one, to additionally return
    width-corrected intervals under the keys `std_calibrated`, `ci_low_calibrated`
    and `ci_high_calibrated`. The uncorrected keys are always present and unchanged,
    so a caller never loses the raw posterior by asking for the corrected one.
    See src/inference/recalibrate.py for why this is opt-in.
    """
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
    out = {
        "mean":    torch.cat(means, 0).cpu(),
        "std":     torch.cat(stds, 0).cpu(),
        "ci_low":  torch.cat(los, 0).cpu(),
        "ci_high": torch.cat(his, 0).cpu(),
        "samples": torch.cat(samps, 1).cpu(),      # (S, N, 12)
    }
    if recalibrate is not None and recalibrate is not False:
        import numpy as _np
        from .recalibrate import Recalibration
        rc = Recalibration.load() if recalibrate is True else recalibrate
        tps = _np.asarray(trials, float).mean(-1)
        c = rc.apply(out["mean"].numpy(), out["std"].numpy(),
                     trials_per_stimulus=tps, levels=(0.9,))
        out["std_calibrated"] = torch.as_tensor(c["std"])
        out["ci_low_calibrated"] = torch.as_tensor(c["ci_low_0.9"])
        out["ci_high_calibrated"] = torch.as_tensor(c["ci_high_0.9"])
        out["recalibration"] = rc
    return out


@torch.no_grad()
def predict_point(model, counts, trials):
    """Fast point estimate (posterior mean, no sampling) -> (N,12), on CPU."""
    model.eval()
    device = _model_device(model)
    x = featurize(torch.as_tensor(counts), torch.as_tensor(trials)).to(device)
    mean_train, _ = model(x)
    return train_space_to_params(mean_train).cpu()
