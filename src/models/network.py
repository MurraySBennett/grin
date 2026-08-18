"""network.py — input featurisation, encoder, and the full NPE model."""
import torch
import torch.nn as nn
from .heads import GaussianHead, N_PARAMS


def featurize_square(counts, trials, n_stimuli):
    """Featurize a square identification matrix without assuming a 2x2 design."""
    counts = counts.reshape(-1, n_stimuli, n_stimuli).float()
    trials = trials.reshape(-1, n_stimuli).float().clamp(min=1)
    props = counts / trials.unsqueeze(-1)
    return torch.cat([props.reshape(-1, n_stimuli ** 2), torch.log10(trials)], dim=-1)


def featurize(counts, trials):
    """
    counts (N,16) int, trials (N,4) -> (N,20): row-proportions (16) + log10 trial
    counts (4). Proportions are the signal; log-trials tell the net the noise level
    so the posterior can widen when data are scarce.
    """
    return featurize_square(counts, trials, n_stimuli=4)


def _mlp(in_dim, hidden, activation, dropout):
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        if dropout:
            layers += [nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


class NPEModel(nn.Module):
    def __init__(self, in_dim=20, param_dim=N_PARAMS, hidden=(128, 128, 128),
                 activation="tanh", dropout=0.1, comparison=False, n_arch=0):
        super().__init__()
        self.encoder, feat = _mlp(in_dim, hidden, activation, dropout)
        self.head = GaussianHead(feat, param_dim)
        self.comparison = comparison
        self.n_arch = n_arch
        if n_arch:
            self.arch_head = nn.Sequential(nn.Linear(feat, 64), nn.GELU(), nn.Linear(64, n_arch))
        if comparison:
            def _clshead(out):
                return nn.Sequential(nn.Linear(feat, 64), nn.GELU(), nn.Linear(64, out))
            self.corr_head = _clshead(3)     # PI / RHO1 / free
            self.sepA_head = _clshead(2)     # separable-A: no / yes
            self.sepB_head = _clshead(2)     # separable-B: no / yes

    def forward(self, x):
        return self.head(self.encoder(x))

    def distribution(self, x):
        return self.head.distribution(self.encoder(x))

    def compare_logits(self, x):
        """Factorized model-comparison logits (needs comparison=True)."""
        h = self.encoder(x)
        return self.corr_head(h), self.sepA_head(h), self.sepB_head(h)

    def arch_logits(self, x):
        """Processing-architecture logits (needs n_arch>0)."""
        return self.arch_head(self.encoder(x))

    def forward_arch(self, x):
        h = self.encoder(x)
        mean, L = self.head(h)
        return (mean, L, self.corr_head(h), self.sepA_head(h), self.sepB_head(h),
                self.arch_head(h))

    def forward_all(self, x):
        """Single encoder pass -> (mean, L, corr_logits, sepA_logits, sepB_logits)."""
        h = self.encoder(x)
        mean, L = self.head(h)
        return mean, L, self.corr_head(h), self.sepA_head(h), self.sepB_head(h)
