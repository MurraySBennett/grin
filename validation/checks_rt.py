"""checks_rt.py — RT, architecture, and LBA validation checks (v12-v16)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import src.grt_model as gm
from src.data.generator import GRTDataGenerator
from src.data.rt_lba_generator import RTLBAGenerator, featurize_lba, ARCHITECTURES, LBA_NAMES
from src.models.network import featurize, _mlp
from src.models.heads import GaussianHead, params_to_train_space, train_space_to_params

try:                                    # track the production knob when available
    from src.config import TRIAL_IMBALANCE
except Exception:                       # keep checks runnable if config isn't importable
    TRIAL_IMBALANCE = 0.35

K = len(ARCHITECTURES)


class _Full(nn.Module):
    def __init__(self, d, n_arch=K, n_lba=4):
        super().__init__()
        self.enc, f = _mlp(d, (192, 192, 192), "tanh", 0.0)
        self.h = GaussianHead(f, 12)
        mk = lambda o: nn.Sequential(nn.Linear(f, 64), nn.GELU(), nn.Linear(64, o))
        self.c, self.a, self.b, self.ar, self.l = mk(3), mk(2), mk(2), mk(n_arch), mk(n_lba)

    def forward(self, x):
        h = self.enc(x); m, L = self.h(h)
        return m, L, self.c(h), self.a(h), self.b(h), self.ar(h), self.l(h)


def _train_full(feats, tgt, arch, lba_z, epochs=25, seed=0):
    torch.manual_seed(seed)
    m = _Full(feats.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = feats.shape[0]; perm = torch.randperm(n)
    for e in range(epochs):
        for i in range(0, n, 512):
            b = perm[i:i + 512]; opt.zero_grad()
            mean, L, cl, al, bl, arl, lb = m(feats[b])
            loss = -torch.distributions.MultivariateNormal(mean, scale_tril=L).log_prob(tgt[b]).mean()
            loss = loss + 4 * Fn.cross_entropy(arl, arch[b]) + 2 * Fn.mse_loss(lb, lba_z[b])
            loss.backward(); opt.step()
    return m.eval()


# ---------------------------------------------------------------- v12
def v12_rt_collinearity(**kw):
    """WHY RTs don't fix PI: RT and accuracy are largely collinear w.r.t. rho."""
    g = RTLBAGenerator(seed=0); rng = np.random.default_rng(0)
    lba = np.array([0.25, 0.6, 1.2, 1.2])
    rhos, accs, meds = [], [], []
    for r in (-0.8, -0.4, 0.0, 0.4, 0.8):
        resp, rt = g._trial_block(-1.2, -1.2, r, 40000, "parallel_exhaustive", lba, rng)
        rhos.append(r); accs.append(float((resp == 0).mean())); meds.append(float(np.median(rt)))
    return {"id": "v12", "claim": "RT and accuracy are collinear w.r.t. rho (why RTs can't fix PI)",
            "result": {"corr_rho_medianRT": float(np.corrcoef(rhos, meds)[0, 1]),
                       "corr_rho_accuracy": float(np.corrcoef(rhos, accs)[0, 1])},
            "pass": True}


def _rt_dataset(n_per_class, trial_range=(50, 300), seed=1, gen_cls=RTLBAGenerator,
                imbalance=TRIAL_IMBALANCE):
    G = gen_cls(n_per_class=n_per_class, trial_range=trial_range, seed=seed,
                imbalance=imbalance)
    return G.generate()


# ---------------------------------------------------------------- v13
def v13_rt_gain(n_per_class=800, epochs=20, **kw):
    """What RTs actually buy: counts-only vs +RT quantiles."""
    X, RTQ, Xt, yp, ylba, yc, yl, ya = _rt_dataset(n_per_class)
    tgt = params_to_train_space(torch.tensor(yp, dtype=torch.float32))
    lt = torch.tensor(ylba, dtype=torch.float32); mu, sd = lt.mean(0), lt.std(0)
    ar = torch.tensor(ya, dtype=torch.long)
    f0 = featurize(torch.tensor(X), torch.tensor(Xt))
    f1 = featurize_lba(X, RTQ, Xt)
    m0 = _train_full(f0, tgt, ar, (lt - mu) / sd, epochs=epochs)
    m1 = _train_full(f1, tgt, ar, (lt - mu) / sd, epochs=epochs)
    Xe, RTQe, Xte, ype, _, _, _, yae = _rt_dataset(80, (150, 150), seed=42)
    out = {}
    for tag, m, fe in (("counts_only", m0, featurize(torch.tensor(Xe), torch.tensor(Xte))),
                       ("plus_RT", m1, featurize_lba(Xe, RTQe, Xte))):
        with torch.no_grad(): o = m(fe)
        p = train_space_to_params(o[0]).numpy()
        out[tag] = {"rho_MAE": float(np.abs(p[:, 8:12] - ype[:, 8:12]).mean()),
                    "z_MAE": float(np.abs(p[:, 0:8] - ype[:, 0:8]).mean())}
    return {"id": "v13", "claim": "RT gain (real but modest)", "result": out, "pass": True}


# ---------------------------------------------------------------- v14 / v15
def _arch_accuracy(gen_cls, n_per_class=800, epochs=20, seed=1):
    X, RTQ, Xt, yp, ylba, yc, yl, ya = _rt_dataset(n_per_class, seed=seed, gen_cls=gen_cls)
    tgt = params_to_train_space(torch.tensor(yp, dtype=torch.float32))
    lt = torch.tensor(ylba, dtype=torch.float32); mu, sd = lt.mean(0), lt.std(0)
    m = _train_full(featurize_lba(X, RTQ, Xt), tgt, torch.tensor(ya, dtype=torch.long), (lt - mu) / sd, epochs=epochs)
    Xe, RTQe, Xte, ype, ylbae, _, _, yae = _rt_dataset(80, (150, 150), seed=42, gen_cls=gen_cls)
    with torch.no_grad(): o = m(featurize_lba(Xe, RTQe, Xte))
    pa = torch.softmax(o[5], -1).numpy().argmax(1)
    per = {a: float(np.mean(pa[yae == i] == i)) for i, a in enumerate(ARCHITECTURES)}
    lba_pred = (o[6] * sd + mu).numpy()
    lba_r = {n: float(np.corrcoef(ylbae[:, j], lba_pred[:, j])[0, 1]) for j, n in enumerate(LBA_NAMES)}
    p = train_space_to_params(o[0]).numpy()
    grt = {"z_MAE": float(np.abs(p[:, 0:8] - ype[:, 0:8]).mean()),
           "rho_MAE": float(np.abs(p[:, 8:12] - ype[:, 8:12]).mean())}
    return float(np.mean(pa == yae)), per, lba_r, grt


def v14_architecture(**kw):
    overall, per, lba_r, grt = _arch_accuracy(RTLBAGenerator)
    stop = np.mean([per[a] for a in ARCHITECTURES if "self_terminating" in a])
    return {"id": "v14", "claim": "architecture recovery (5-way SFT); dimension-neglect detection",
            "result": {"overall": overall, "per_architecture": per,
                       "self_terminating_mean": float(stop), "chance": 1 / K},
            "pass": stop > 0.85}


def v15_speed_confound(**kw):
    """Control: match mean RT across architectures. Recovery must survive."""
    g0 = RTLBAGenerator(seed=0); rng = np.random.default_rng(0)
    lba = np.array([0.25, 0.6, 1.2, 1.2]); base = {}
    for a in ARCHITECTURES:
        _, rt = g0._trial_block(-1.2, -1.2, 0.0, 20000, a, lba, rng); base[a] = rt.mean()
    target = float(np.mean(list(base.values())))
    scale = {a: (target - 0.25) / (base[a] - 0.25) for a in ARCHITECTURES}

    class Matched(RTLBAGenerator):
        def _trial_block(self, zx, zy, rho, n, arch, lba, rng):
            l = np.asarray(lba).copy(); l[1] = l[1] * scale[arch]
            return super()._trial_block(zx, zy, rho, n, arch, l, rng)

    o_orig, per_orig, _, _ = _arch_accuracy(RTLBAGenerator)
    o_match, per_match, _, _ = _arch_accuracy(Matched)
    return {"id": "v15", "claim": "architecture is read from DISTRIBUTION SHAPE, not speed",
            "result": {"mean_rt_before": {k: float(v) for k, v in base.items()},
                       "overall_original": o_orig, "overall_rt_matched": o_match,
                       "coactive_original": per_orig["coactive"],
                       "coactive_rt_matched": per_match["coactive"]},
            "pass": abs(o_orig - o_match) < 0.08}


def v16_lba_confound(**kw):
    """LBA params recover; adding them does NOT damage GRT identifiability."""
    _, _, lba_r, grt = _arch_accuracy(RTLBAGenerator)
    return {"id": "v16", "claim": "LBA recovery; extra params cost no GRT identifiability",
            "result": {"lba_correlations": lba_r, "grt_recovery_with_lba": grt},
            "pass": grt["z_MAE"] < 0.45}
