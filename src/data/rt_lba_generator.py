"""
rt_lba_generator.py — GRT + LBA + processing architecture, VECTORISED.

Produces exactly the same thing as the counts-only generator (`src/data/generator.py`)
— confusion matrices from the same GRT prior — and ADDITIONALLY the response times that
those same trials produced. Counts and RTs are MATCHED BY CONSTRUCTION: each trial draws
one perceptual sample, and that single sample determines both the response (which quadrant
it fell in) and the RT (its distance from each bound drives the LBA drift rate).

Everything else — the prior, z_max, r_max, trial_range, the model classes — is identical
to the counts-only generator. The ONLY difference is that RTs come out too.

Implementation notes (vectorisation)
------------------------------------
The naive version looped participants x stimuli in Python, simulating a few hundred trials
each. This version:
  * groups participants by architecture (so each group is one branch, no per-trial `if`),
  * pads each group's trials to a common length and masks the excess,
  * uses the closed-form 2x2 Cholesky instead of np.linalg.cholesky per participant,
  * computes per-cell RT quantiles by sorting with +inf padding and gathering by rank.
This is ~50-100x faster and lets you generate the same N as the counts-only pipeline.

SFT taxonomy: architecture (serial / parallel / coactive) x stopping rule (exhaustive /
self-terminating). Coactive has no stopping-rule crossing (evidence pools into a single
accumulator), hence 5 models, not 6.

NOTE on the self-terminating models: in an IDENTIFICATION task the response must name BOTH
levels, so stopping early necessarily means guessing the un-processed dimension. These
therefore represent the participant who is NOT using a dimension (incapacity, inattention,
or strategy) — a pathology to DETECT, not a normal processing mode.
"""
import numpy as np

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm

ARCHITECTURES = [
    "serial_exhaustive",          # RT = t0 + t_x + t_y
    "serial_self_terminating",    # RT = t0 + t_first          (processes ONE dim, guesses other)
    "parallel_exhaustive",        # RT = t0 + max(t_x, t_y)
    "parallel_self_terminating",  # RT = t0 + min(t_x, t_y)    (stops at first, guesses other)
    "coactive",                   # RT = t0 + A / (v_x + v_y)  (channel summation)
]
LBA_NAMES = ["t0", "threshold_A", "drift_k_A", "drift_k_B"]
QUANTILES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
N_Q = len(QUANTILES)


class RTLBAGenerator:
    def __init__(self, n_per_class=20000, trial_range=(5, 1000), z_max=3.0, r_max=0.9,
                 drift_sd=0.35, seed=None, chunk=2000, imbalance=0.35):
        self.n_per_class = int(n_per_class)
        self.trial_range = trial_range   # per-participant BASE count, log-uniform
        self.z_max, self.r_max, self.drift_sd = z_max, r_max, drift_sd
        self.seed = seed
        self.chunk = int(chunk)          # participants simulated per vectorised block
        # max fractional attrition a single stimulus may suffer relative to the
        # participant's base count (0.35 => smallest stimulus keeps >=65% of the
        # largest). Bounds within-set imbalance; 0.0 == perfectly balanced.
        self.imbalance = float(imbalance)
        self.model_names = gm.MODEL_NAMES

    # ------------------------------------------------------------------ #
    def _sample_trial_counts(self, n, rng, floor=3):
        """
        Per-stimulus trial counts with BOUNDED within-set imbalance — the RT twin of
        GRTDataGenerator._sample_trial_counts (kept identical in spirit so the counts
        and RT pipelines see the same trial-count regime).

        One per-participant BASE count is drawn log-uniformly across trial_range (this
        keeps the full few->many magnitude spectrum the network needs to learn the
        trials -> uncertainty relation). The 4 stimuli then differ only by realistic
        attrition — each keeps a fraction in [1 - imbalance, 1] of the base — so
        imbalance stays bounded and proportional to set size. This rules out the
        pathological sets independent log-uniform sampling produced (e.g. 1/648/5/732).

        `floor` is higher here than in the counts generator (default 3) because each
        (stimulus, response) cell needs a few trials for its RT quantiles to be defined.
        int64 dtype is pinned so downstream torch tensors are Long on every platform.
        """
        lo, hi = self.trial_range
        base = np.exp(rng.uniform(np.log(lo), np.log(hi), n))          # (n,) per participant
        if self.imbalance <= 0.0:
            counts = np.repeat(base[:, None], 4, axis=1)
        else:
            f = min(self.imbalance, 1.0)
            factors = rng.uniform(1.0 - f, 1.0, (n, 4))               # each stimulus keeps 1-f..1
            counts = base[:, None] * factors
        counts = np.round(counts).astype(np.int64)
        return np.clip(counts, floor, None)                          # (n, 4)

    # ------------------------------------------------------------------ #
    def sample_lba(self, rng, n=1):
        """Per-participant accumulator parameters: (n, 4) = [t0, threshold, k_A, k_B]."""
        return np.stack([rng.uniform(0.15, 0.45, n),      # non-decision time (s)
                         rng.uniform(0.35, 1.10, n),      # threshold / caution
                         rng.uniform(0.60, 2.00, n),      # drift scaling, dimension A
                         rng.uniform(0.60, 2.00, n)], 1)  # drift scaling, dimension B

    # ------------------------------------------------------------------ #
    def _trial_block(self, zx, zy, rho, n, arch, lba, rng):
        """
        Single-condition, single-architecture, unbatched simulator.
        zx, zy, rho : scalars (one GRT condition)      lba : (4,) = [t0, A, k_A, k_B]
        Returns resp (n,) int8 in {0,1,2,3} and rt (n,) float, one row per trial.

        This is the scalar counterpart to `_simulate_group` (which is batched over
        participants x stimulus quadrants and returns aggregated counts/quantiles).
        Kept for checks that need raw per-trial (resp, rt) at one fixed condition,
        e.g. RT/accuracy collinearity (v12) and speed-confound matching (v15).
        """
        n = int(n)
        t0, A, kA, kB = (float(v) for v in lba)

        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        x = zx + z1
        y = zy + rho * z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z2

        rx = (x >= 0).astype(np.int8)
        ry = (y >= 0).astype(np.int8)
        vx = np.maximum(kA * np.abs(x) + rng.normal(0, self.drift_sd, n), 0.05)
        vy = np.maximum(kB * np.abs(y) + rng.normal(0, self.drift_sd, n), 0.05)
        tx, ty = A / vx, A / vy

        coin = rng.integers(0, 2, n).astype(np.int8)   # guesses for ST models

        if arch == "serial_exhaustive":
            rt = t0 + tx + ty
            gx, gy = rx, ry
        elif arch == "serial_self_terminating":
            do_x = rng.random(n) < 0.5
            rt = t0 + np.where(do_x, tx, ty)
            gx = np.where(do_x, rx, coin)
            gy = np.where(do_x, coin, ry)
        elif arch == "parallel_exhaustive":
            rt = t0 + np.maximum(tx, ty)
            gx, gy = rx, ry
        elif arch == "parallel_self_terminating":
            rt = t0 + np.minimum(tx, ty)
            first_x = tx <= ty
            gx = np.where(first_x, rx, coin)
            gy = np.where(first_x, coin, ry)
        elif arch == "coactive":
            rt = t0 + A / np.maximum(vx + vy, 0.05)
            gx, gy = rx, ry
        else:
            raise ValueError(arch)

        resp = (2 * gx + gy).astype(np.int8)
        rt = np.clip(rt, 0.1, 10.0)
        return resp, rt

    # ------------------------------------------------------------------ #
    def _simulate_group(self, zx, zy, rho, lba, n_per, arch, rng):
        """
        Fully vectorised simulation for a group of participants sharing ONE architecture.
        zx, zy, rho : (B, 4)     n_per : (B, 4) trial counts     lba : (B, 4)
        Returns counts (B,4,4) and RT quantiles (B,4,4,N_Q).
        """
        B = zx.shape[0]
        n_max = int(n_per.max())
        t0 = lba[:, 0][:, None, None]                 # (B,1,1)
        A = lba[:, 1][:, None, None]
        kA = lba[:, 2][:, None, None]
        kB = lba[:, 3][:, None, None]

        # --- perceptual samples: closed-form 2x2 Cholesky, no per-participant loop ---
        z1 = rng.standard_normal((B, 4, n_max))
        z2 = rng.standard_normal((B, 4, n_max))
        r = rho[:, :, None]
        x = zx[:, :, None] + z1
        y = zy[:, :, None] + r * z1 + np.sqrt(np.maximum(1.0 - r ** 2, 0.0)) * z2

        # --- dimensional decisions and LBA times ---
        rx = (x >= 0).astype(np.int8)
        ry = (y >= 0).astype(np.int8)
        vx = np.maximum(kA * np.abs(x) + rng.normal(0, self.drift_sd, (B, 4, n_max)), 0.05)
        vy = np.maximum(kB * np.abs(y) + rng.normal(0, self.drift_sd, (B, 4, n_max)), 0.05)
        tx, ty = A / vx, A / vy

        coin = rng.integers(0, 2, (B, 4, n_max)).astype(np.int8)   # guesses for ST models

        if arch == "serial_exhaustive":
            rt = t0 + tx + ty
            gx, gy = rx, ry
        elif arch == "serial_self_terminating":
            do_x = rng.random((B, 4, n_max)) < 0.5
            rt = t0 + np.where(do_x, tx, ty)
            gx = np.where(do_x, rx, coin)
            gy = np.where(do_x, coin, ry)
        elif arch == "parallel_exhaustive":
            rt = t0 + np.maximum(tx, ty)
            gx, gy = rx, ry
        elif arch == "parallel_self_terminating":
            rt = t0 + np.minimum(tx, ty)
            first_x = tx <= ty
            gx = np.where(first_x, rx, coin)
            gy = np.where(first_x, coin, ry)
        elif arch == "coactive":
            rt = t0 + A / np.maximum(vx + vy, 0.05)
            gx, gy = rx, ry
        else:
            raise ValueError(arch)

        resp = (2 * gx + gy).astype(np.int8)                      # (B,4,n_max)
        rt = np.clip(rt, 0.1, 10.0)

        # --- mask the padded trials ---
        valid = np.arange(n_max)[None, None, :] < n_per[:, :, None]   # (B,4,n_max)

        # --- counts: one-hot over responses, masked ---
        counts = np.zeros((B, 4, 4), dtype=np.int64)
        for rr in range(4):
            counts[:, :, rr] = ((resp == rr) & valid).sum(-1)

        # --- RT quantiles per (participant, stimulus, response cell) ---
        # sort with +inf padding so invalid/other-cell trials fall to the end, then gather
        # by nearest-rank index. Cells with 0 trials stay 0 (the net sees count == 0).
        Q = np.zeros((B, 4, 4, N_Q))
        for rr in range(4):
            m = (resp == rr) & valid
            filled = np.where(m, rt, np.inf)
            srt = np.sort(filled, axis=-1)                          # (B,4,n_max)
            k = counts[:, :, rr]                                    # (B,4) trials in this cell
            has = k > 0
            kk = np.maximum(k - 1, 0)
            for qi, q in enumerate(QUANTILES):
                idx = np.rint(q * kk).astype(np.int64)              # nearest-rank
                idx = np.clip(idx, 0, max(n_max - 1, 0))
                vals = np.take_along_axis(srt, idx[:, :, None], axis=-1)[:, :, 0]
                Q[:, :, rr, qi] = np.where(has, vals, 0.0)
        return counts, Q

    # ------------------------------------------------------------------ #
    def generate(self, seed=None, verbose=True):
        """Returns X(N,16), RTQ(N,80), X_trials(N,4), y_params(N,12), y_lba(N,4),
        y_model_cls(N,), y_cls_label(N,), y_arch(N,)."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        Xs, Qs, Ts, Ps, Ls, Cs, Ns, As = [], [], [], [], [], [], [], []

        for ci, name in enumerate(self.model_names):
            n = int(self.n_per_class)
            if verbose:
                print(f"   [{ci+1:2d}/12] {name:12s} n={n}", flush=True)
            zx, zy, rho = gm.sample_prior(name, n, rng, z_max=self.z_max, r_max=self.r_max)
            lba = self.sample_lba(rng, n)
            arch_id = rng.integers(0, len(ARCHITECTURES), n)
            n_per = self._sample_trial_counts(n, rng, floor=3)
            counts = np.zeros((n, 4, 4), dtype=np.int64)
            Q = np.zeros((n, 4, 4, N_Q))
            # group by architecture (one vectorised branch each), chunked to bound memory
            for ai, aname in enumerate(ARCHITECTURES):
                sel = np.flatnonzero(arch_id == ai)
                for st in range(0, len(sel), self.chunk):
                    idx = sel[st:st + self.chunk]
                    c, q = self._simulate_group(zx[idx], zy[idx], rho[idx], lba[idx],
                                                n_per[idx], aname, rng)
                    counts[idx] = c; Q[idx] = q

            Xs.append(counts.reshape(n, 16))
            Qs.append(Q.reshape(n, 4 * 4 * N_Q))
            Ts.append(counts.sum(2))
            Ps.append(gm.pack(zx, zy, rho))
            Ls.append(lba)
            Cs.append(np.full(n, ci, dtype=np.int64)); Ns.append(np.array([name] * n))
            As.append(arch_id.astype(np.int64))

        return (np.concatenate(Xs), np.concatenate(Qs), np.concatenate(Ts),
                np.concatenate(Ps), np.concatenate(Ls), np.concatenate(Cs),
                np.concatenate(Ns), np.concatenate(As))


def featurize_lba(counts, rtq, trials):
    """counts(N,16) + rt quantiles(N,80) + trials(N,4) -> (N,100) network input."""
    import torch
    c = torch.as_tensor(counts, dtype=torch.float32).reshape(-1, 4, 4)
    t = torch.as_tensor(trials, dtype=torch.float32).clamp(min=1)
    props = (c / t.unsqueeze(-1)).reshape(-1, 16)
    q = torch.as_tensor(rtq, dtype=torch.float32)
    return torch.cat([props, torch.log10(t), q], dim=-1)
