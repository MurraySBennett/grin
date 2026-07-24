"""
rt_network.py — the RT-augmented model: GRT parameters + constructs + processing
architecture + LBA parameters, from one shared encoder.

Used only when your data include response times. Input is the confusion-matrix
proportions, log trial counts, and the RT quantile vector (see
`src/data/rt_lba_generator.featurize_lba`).
"""
import torch
import torch.nn as nn

from .network import _mlp
from .heads import GaussianHead, N_PARAMS
from ..data.rt_lba_generator import ARCHITECTURES, LBA_NAMES


class RTNPEModel(nn.Module):
    def __init__(self, in_dim=100, hidden=None, activation="tanh", dropout=None):
        from ..config import RT_HIDDEN_LAYERS, RT_DROPOUT, ACTIVATION
        hidden = RT_HIDDEN_LAYERS if hidden is None else hidden
        dropout = RT_DROPOUT if dropout is None else dropout
        activation = activation or ACTIVATION
        super().__init__()
        self.encoder, f = _mlp(in_dim, tuple(hidden), activation, dropout)
        self.head = GaussianHead(f, N_PARAMS)
        mk = lambda o: nn.Sequential(nn.Linear(f, 64), nn.GELU(), nn.Linear(64, o))
        self.corr_head = mk(3)                    # PI / RHO1 / free
        self.sepA_head = mk(2)
        self.sepB_head = mk(2)
        self.arch_head = mk(len(ARCHITECTURES))   # 5-way SFT architecture
        self.lba_head = mk(len(LBA_NAMES))        # t0, threshold, drift_k_A, drift_k_B

    def forward(self, x):
        h = self.encoder(x)
        mean, L = self.head(h)
        return (mean, L, self.corr_head(h), self.sepA_head(h), self.sepB_head(h),
                self.arch_head(h), self.lba_head(h))

    def distribution(self, x):
        return self.head.distribution(self.encoder(x))
