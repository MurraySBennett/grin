"""
rt_arch_generator.py — RT-augmented GRT with ONE RT per trial and an explicit,
inferable processing ARCHITECTURE.

Fixes the fatal flaw of the two-RT prototype: real identification tasks yield a
SINGLE RT per trial, so any signal must survive into that one number. How the two
dimensional decisions combine into one response+RT is the processing architecture,
and it is a *parameter*, not an assumption.

RT model — accumulator (LBA-style), per dimension:
    drift_i  ~ N(v0 * |z_i_sample| , s)      evidence rate grows with distance from bound
    t_i      = A / max(drift_i, eps)         time for that dimension to resolve
Both dimensions accumulate from the SAME perceptual sample (x, y), so their times are
coupled through the perceptual correlation — this is the channel that carries rho.

Architectures (how the two dimensional times -> one observed RT, and one response):
    'parallel_exhaustive' : both accumulate at once; RT = t0 + max(t_x, t_y).
                            Response uses BOTH dimensions. (Identification needs both.)
    'serial'              : one then the other; RT = t0 + t_x + t_y. Response uses both.
    'coactive'            : evidence pools into a single decision; RT = t0 + A/(v_x+v_y),
                            response uses both. (Integral dimensions, e.g. hue/saturation.)
    'self_terminating'    : responds when the FIRST dimension resolves and GUESSES the
                            other; RT = t0 + min(t_x, t_y). This is the "participant who
                            can't/won't use a dimension" mode — behaviourally real, and we
                            want to know whether GRIN can detect it.

Honest notes:
  * The drift-from-distance link replaces the earlier ad-hoc hyperbolic RT link.
  * RTs sharpen PI/PS but do NOT dissociate decisional separability (the RT theorems of
    Townsend, Houpt & Silbert 2012 assume stochastic DS).
"""
import numpy as np

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm

ARCHITECTURES = ["parallel_exhaustive", "serial", "coactive", "self_terminating"]


class RTArchGenerator:
    def __init__(self, n_per_class=2000, trial_range=(50, 300), z_max=3.0, r_max=0.9,
                 t0=0.25, A=0.6, v0=1.2, drift_sd=0.35, seed=None):
        self.n_per_class = int(n_per_class)
        self.trial_range = trial_range
        self.z_max, self.r_max = z_max, r_max
        self.t0, self.A, self.v0, self.drift_sd = t0, A, v0, drift_sd
        self.seed = seed
        self.model_names = gm.MODEL_NAMES

    def _trial_block(self, zx_s, zy_s, rho_s, n, arch, rng):
        """Simulate n trials of one stimulus. Returns responses (n,), rts (n,)."""
        L = np.linalg.cholesky([[1.0, rho_s], [rho_s, 1.0]])
        pts = np.array([zx_s, zy_s]) + rng.standard_normal((n, 2)) @ L.T
        # dimensional decisions from the sample (bounds at 0)
        rx = (pts[:, 0] >= 0).astype(int)
        ry = (pts[:, 1] >= 0).astype(int)
        # drift rates grow with |distance from bound|; both from the SAME sample
        vx = np.maximum(self.v0 * np.abs(pts[:, 0]) + rng.normal(0, self.drift_sd, n), 0.05)
        vy = np.maximum(self.v0 * np.abs(pts[:, 1]) + rng.normal(0, self.drift_sd, n), 0.05)
        tx, ty = self.A / vx, self.A / vy

        if arch == "parallel_exhaustive":
            rt = self.t0 + np.maximum(tx, ty)
            resp = 2 * rx + ry
        elif arch == "serial":
            rt = self.t0 + tx + ty
            resp = 2 * rx + ry
        elif arch == "coactive":
            rt = self.t0 + self.A / np.maximum(vx + vy, 0.05)
            resp = 2 * rx + ry
        elif arch == "self_terminating":
            rt = self.t0 + np.minimum(tx, ty)
            # responds on whichever dimension finished first; GUESSES the other
            first_x = tx <= ty
            gx = np.where(first_x, rx, rng.integers(0, 2, n))
            gy = np.where(first_x, rng.integers(0, 2, n), ry)
            resp = 2 * gx + gy
        else:
            raise ValueError(arch)
        return resp, np.clip(rt, 0.12, 8.0)

    def _summarize(self, resp, rt, s, counts, rtfeat):
        """Per-stimulus RT summaries, conditioned on the response cell (keeps joint info)."""
        counts[s] = np.bincount(resp, minlength=4)
        # overall RT moments for this stimulus
        rtfeat[s, 0] = rt.mean()
        rtfeat[s, 1] = rt.std()
        rtfeat[s, 2] = np.quantile(rt, 0.1)
        rtfeat[s, 3] = np.quantile(rt, 0.9)
        rtfeat[s, 4] = ((rt - rt.mean()) ** 3).mean() / (rt.std() ** 3 + 1e-9)   # skew
        # RT split by CORRECT vs error (correct = response == stimulus)
        m = resp == s
        rtfeat[s, 5] = rt[m].mean() if m.sum() else 0.0
        rtfeat[s, 6] = rt[~m].mean() if (~m).sum() else 0.0
        # mean RT per response cell (4) — the cell-wise joint structure
        for r in range(4):
            k = resp == r
            rtfeat[s, 7 + r] = rt[k].mean() if k.sum() else 0.0

    def generate(self, seed=None, architectures=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        archs = architectures or ARCHITECTURES
        X, RT, Xt, yp, ycls, ylab, yarch = [], [], [], [], [], [], []
        lo, hi = self.trial_range
        for ci, name in enumerate(self.model_names):
            zx, zy, rho = gm.sample_prior(name, self.n_per_class, rng,
                                          z_max=self.z_max, r_max=self.r_max)
            for i in range(self.n_per_class):
                arch = archs[int(rng.integers(len(archs)))]
                n_per = np.round(np.exp(rng.uniform(np.log(lo), np.log(hi), 4))).astype(int)
                counts = np.zeros((4, 4), dtype=np.int64)
                rtfeat = np.zeros((4, 11))
                for s in range(4):
                    resp, rt = self._trial_block(zx[i, s], zy[i, s], rho[i, s],
                                                 max(int(n_per[s]), 2), arch, rng)
                    self._summarize(resp, rt, s, counts, rtfeat)
                X.append(counts.reshape(16)); RT.append(rtfeat.reshape(44))
                Xt.append(counts.sum(1))
                yp.append(gm.pack(zx[i:i+1], zy[i:i+1], rho[i:i+1])[0])
                ycls.append(ci); ylab.append(name); yarch.append(ARCHITECTURES.index(arch))
        return (np.array(X), np.array(RT), np.array(Xt), np.array(yp),
                np.array(ycls), np.array(ylab), np.array(yarch))


def featurize_rt_arch(counts, rt, trials):
    """counts (N,16) + rt (N,44) + trials (N,4) -> (N,64)."""
    import torch
    c = torch.as_tensor(counts, dtype=torch.float32).reshape(-1, 4, 4)
    t = torch.as_tensor(trials, dtype=torch.float32).clamp(min=1)
    props = (c / t.unsqueeze(-1)).reshape(-1, 16)
    return torch.cat([props, torch.log10(t),
                      torch.as_tensor(rt, dtype=torch.float32)], dim=-1)
    