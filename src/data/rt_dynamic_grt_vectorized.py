"""
rt_dynamic_grt_vectorized.py -- GPU/torch-batched generator for the dynamic-GRT RT
model (docs/dynamic_grt_rt_design.md).

Vectorised counterpart to the scalar reference simulator (src/data/rt_dynamic_grt.py),
which validated the model at gates 1-3 and, via the closed-form first-passage
probability (scripts/check_dynamic_grt_gate4_bridge.py), gate 4. This module exists
ONLY to generate training-scale data efficiently -- it must reproduce the scalar
simulator's distribution, never redefine the model. See
tests/test_rt_dynamic_grt_vectorized.py for the correctness check against the scalar
reference (this module's own "gate 1").

Per the design doc S5, this generator is infrastructure for gates 5-8 (identifiability,
architecture evidence, misspecification, empirical check) -- NOT yet a production/
manuscript training run. That waits until all 8 gates pass.

Shapes follow the count-only/RTLBA generator convention: each participant has 4
stimulus conditions (their own zx/zy/rho, matching src/data/generator.py) and ONE set
of nuisance decision parameters (t0, boundary, rate) shared across stimuli. Both
channels (x, y) step together in a single Euler-Maruyama loop; already-finished
(participant, stimulus, trial, channel) cells are FROZEN in place rather than removed,
so every step is a plain elementwise GPU op with no dynamic re-indexing -- the standard
GPU-friendly pattern for batched first-passage simulation (compute for everyone, mask
the ones that are already done, cheaper than compacting on most hardware).

Architecture is applied AFTER the channel simulation, not during it: serial-exhaustive
and parallel-exhaustive read the IDENTICAL channel outcomes (drift draw, response,
per-channel time) and differ only in whether the two channel times are summed or
maxed -- by construction, matching simulate_dynamic_grt_trials and
test_serial_and_parallel_share_responses_but_combine_time_differently.

Censoring is explicit and not clipped into a point mass (design doc S4): a censored
trial contributes to that stimulus's administered trial count but NOT to any response
cell's count or RT quantiles. The resulting shortfall (trials - sum of response-cell
counts) is exposed as a per-stimulus censor_rate feature rather than silently dropped.
"""
import numpy as np
import torch

try:                                        # in-package
    from src import grt_model as gm
    from src.data.rt_dynamic_grt import ARCHITECTURES, sample_dynamic_rt_parameters
except ImportError:                         # standalone (tests, same directory)
    import grt_model as gm
    from rt_dynamic_grt import ARCHITECTURES, sample_dynamic_rt_parameters

QUANTILES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
N_Q = len(QUANTILES)
DYNAMIC_PARAM_NAMES = ("t0", "boundary", "rate")


class DynamicGRTVectorizedGenerator:
    def __init__(self, n_per_class=20000, trial_range=(5, 1000), z_max=3.0, r_max=0.9,
                 seed=None, chunk=2000, imbalance=0.35, dt=0.0025, max_internal_time=25.0,
                 device=None):
        self.n_per_class = int(n_per_class)
        self.trial_range = trial_range          # per-participant BASE count, log-uniform
        self.z_max, self.r_max = z_max, r_max
        self.seed = seed
        self.chunk = int(chunk)                 # participants simulated per GPU block
        self.imbalance = float(imbalance)
        self.dt = float(dt)
        self.max_internal_time = float(max_internal_time)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_names = gm.MODEL_NAMES

    # ------------------------------------------------------------------ #
    def _sample_trial_counts(self, n, rng, floor=3):
        """Identical in spirit to RTLBAGenerator._sample_trial_counts: one log-uniform
        BASE count per participant, each of the 4 stimuli keeping a bounded fraction of
        it (attrition), never independently sampled per stimulus."""
        lo, hi = self.trial_range
        base = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        if self.imbalance <= 0.0:
            counts = np.repeat(base[:, None], 4, axis=1)
        else:
            f = min(self.imbalance, 1.0)
            factors = rng.uniform(1.0 - f, 1.0, (n, 4))
            counts = base[:, None] * factors
        counts = np.round(counts).astype(np.int64)
        return np.clip(counts, floor, None)                              # (n, 4)

    # ------------------------------------------------------------------ #
    def _simulate_channels(self, zx, zy, rho, boundary, n_max, torch_gen):
        """Batched Euler-Maruyama first passage for BOTH channels of a participant
        chunk, sharing one time-step loop.

        zx, zy, rho : (B, 4) numpy       boundary : (B,) numpy       n_max : int
        Returns numpy arrays channel_response (B,4,n_max,2) in {-1,0,1} (-1 = censored),
        channel_time (B,4,n_max,2) internal-time float (NaN where censored), each channel
        indexed [..., 0]=x, [..., 1]=y.
        """
        dev = self.device
        zx_t = torch.as_tensor(zx, dtype=torch.float32, device=dev)[:, :, None]
        zy_t = torch.as_tensor(zy, dtype=torch.float32, device=dev)[:, :, None]
        rho_t = torch.as_tensor(rho, dtype=torch.float32, device=dev)[:, :, None]
        boundary_t = torch.as_tensor(boundary, dtype=torch.float32, device=dev)[:, None, None, None]

        z1 = torch.randn((zx.shape[0], 4, n_max), generator=torch_gen, device=dev)
        z2 = torch.randn((zx.shape[0], 4, n_max), generator=torch_gen, device=dev)
        vx = zx_t + z1
        vy = zy_t + rho_t * z1 + torch.sqrt(torch.clamp(1.0 - rho_t ** 2, min=0.0)) * z2
        drift = torch.stack([vx, vy], dim=-1)                             # (B,4,n_max,2)

        evidence = torch.zeros_like(drift)
        done = torch.zeros_like(drift, dtype=torch.bool)
        response = torch.full_like(drift, -1.0)
        time = torch.full_like(drift, float("nan"))

        dt = self.dt
        noise_scale = dt ** 0.5
        max_steps = int(np.ceil(self.max_internal_time / dt))
        boundary_full = boundary_t.expand_as(evidence)
        # `.any()`/`.all()` on a CUDA tensor forces a host-device synchronisation --
        # checking every step (as an earlier version did) serialises the whole loop on
        # sync latency rather than compute, since each step's actual arithmetic is a
        # handful of cheap elementwise ops on an already-resident tensor. So: do the
        # torch.where-based update unconditionally every step (never gate it behind
        # `crossed.any()`, unconditional GPU compute is far cheaper than a sync), and
        # only pay for the early-exit sync every `check_every` steps.
        check_every = 20
        for step in range(max_steps):
            noise = torch.randn(drift.shape, generator=torch_gen, device=dev) * noise_scale
            updated = evidence + drift * dt + noise
            crossed = (~done) & (updated.abs() >= boundary_full)
            positive = updated >= 0
            target = torch.where(positive, boundary_full, -boundary_full)
            denom = updated - evidence
            frac = torch.where(denom != 0, (target - evidence) / denom, torch.ones_like(denom))
            frac = frac.clamp(0.0, 1.0)
            response = torch.where(crossed, positive.float(), response)
            time = torch.where(crossed, (step + frac) * dt, time)
            evidence = torch.where(done, evidence, updated)
            done = done | crossed
            if step % check_every == 0 and bool(done.all()):
                break

        censored = (~done).cpu().numpy()
        response_np = response.cpu().numpy()
        response_np = np.where(censored, -1, response_np).astype(np.int8)
        time_np = time.cpu().numpy()
        return response_np, time_np, censored

    # ------------------------------------------------------------------ #
    def _build_examples(self, zx, zy, rho, dynamic_params, arch_id, n_per, torch_gen):
        """One participant chunk, all 4 stimuli, both channels, architecture applied
        after the shared channel simulation.

        zx, zy, rho : (B,4)     dynamic_params : (B,3) = [t0, boundary, rate]
        arch_id : (B,) int in {0,1}      n_per : (B,4) trial counts
        Returns counts (B,4,4), rtq (B,4,4,N_Q), censor_rate (B,4).
        """
        B = zx.shape[0]
        n_max = int(n_per.max())
        t0 = dynamic_params[:, 0]
        boundary = dynamic_params[:, 1]
        rate = dynamic_params[:, 2]

        resp, time, chan_censored = self._simulate_channels(zx, zy, rho, boundary, n_max, torch_gen)
        # resp/time/chan_censored: (B,4,n_max,2), channel 0 = x, channel 1 = y
        resp_x, resp_y = resp[..., 0], resp[..., 1]
        time_x, time_y = time[..., 0], time[..., 1]
        trial_censored = chan_censored[..., 0] | chan_censored[..., 1]           # (B,4,n_max)

        decision_time_serial = time_x + time_y
        decision_time_parallel = np.maximum(time_x, time_y)
        is_serial = (arch_id == 0)[:, None, None]
        decision_time = np.where(is_serial, decision_time_serial, decision_time_parallel)

        rt = t0[:, None, None] + decision_time / rate[:, None, None]
        response = (2 * np.clip(resp_x, 0, 1) + np.clip(resp_y, 0, 1)).astype(np.int8)
        response = np.where(trial_censored, -1, response)

        valid = np.arange(n_max)[None, None, :] < n_per[:, :, None]              # (B,4,n_max)
        answered = valid & (~trial_censored)

        counts = np.zeros((B, 4, 4), dtype=np.int64)
        for rr in range(4):
            counts[:, :, rr] = ((response == rr) & answered).sum(-1)
        censor_rate = (valid & trial_censored).sum(-1) / np.maximum(valid.sum(-1), 1)   # (B,4)

        rtq = np.zeros((B, 4, 4, N_Q))
        for rr in range(4):
            m = (response == rr) & answered
            filled = np.where(m, rt, np.inf)
            srt = np.sort(filled, axis=-1)
            k = counts[:, :, rr]
            has = k > 0
            kk = np.maximum(k - 1, 0)
            for qi, q in enumerate(QUANTILES):
                idx = np.rint(q * kk).astype(np.int64)
                idx = np.clip(idx, 0, max(n_max - 1, 0))
                vals = np.take_along_axis(srt, idx[:, :, None], axis=-1)[:, :, 0]
                rtq[:, :, rr, qi] = np.where(has, vals, 0.0)

        return counts, rtq, censor_rate

    # ------------------------------------------------------------------ #
    def generate(self, seed=None, verbose=True):
        """Returns X(N,16), RTQ(N,80), X_trials(N,4), censor_rate(N,4), y_params(N,12),
        y_dynamic(N,3), y_model_cls(N,), y_cls_label(N,), y_arch(N,)."""
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator(device=self.device)
        torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))

        Xs, Qs, Ts, Cs_rate, Ps, Dy, Cls, Lbl, Arch = [], [], [], [], [], [], [], [], []

        for ci, name in enumerate(self.model_names):
            n = int(self.n_per_class)
            if verbose:
                print(f"   [{ci+1:2d}/12] {name:12s} n={n}", flush=True)
            zx, zy, rho = gm.sample_prior(name, n, rng, z_max=self.z_max, r_max=self.r_max)
            dynamic_params = sample_dynamic_rt_parameters(rng, n)          # (n,3)
            arch_id = rng.integers(0, len(ARCHITECTURES), n)
            n_per = self._sample_trial_counts(n, rng, floor=3)

            counts = np.zeros((n, 4, 4), dtype=np.int64)
            rtq = np.zeros((n, 4, 4, N_Q))
            censor_rate = np.zeros((n, 4))
            for st in range(0, n, self.chunk):
                sl = slice(st, min(st + self.chunk, n))
                c, q, cr = self._build_examples(
                    zx[sl], zy[sl], rho[sl], dynamic_params[sl], arch_id[sl], n_per[sl], torch_gen)
                counts[sl] = c; rtq[sl] = q; censor_rate[sl] = cr

            Xs.append(counts.reshape(n, 16))
            Qs.append(rtq.reshape(n, 4 * 4 * N_Q))
            Ts.append(counts.sum(2) + np.round(censor_rate * n_per).astype(np.int64))
            Cs_rate.append(censor_rate)
            Ps.append(gm.pack(zx, zy, rho))
            Dy.append(dynamic_params)
            Cls.append(np.full(n, ci, dtype=np.int64))
            Lbl.append(np.array([name] * n))
            Arch.append(arch_id.astype(np.int64))

        return (np.concatenate(Xs), np.concatenate(Qs), np.concatenate(Ts),
                np.concatenate(Cs_rate), np.concatenate(Ps), np.concatenate(Dy),
                np.concatenate(Cls), np.concatenate(Lbl), np.concatenate(Arch))


def featurize_dynamic(counts, rtq, trials, censor_rate):
    """counts(N,16) + rt quantiles(N,80) + trials(N,4) + censor_rate(N,4) -> (N,104)
    network input. censor_rate is exposed explicitly per S4 of the design doc rather
    than folded silently into the RT quantiles or trial counts."""
    import torch as _torch
    c = _torch.as_tensor(counts, dtype=_torch.float32).reshape(-1, 4, 4)
    t = _torch.as_tensor(trials, dtype=_torch.float32).clamp(min=1)
    props = (c / t.unsqueeze(-1)).reshape(-1, 16)
    q = _torch.as_tensor(rtq, dtype=_torch.float32)
    cr = _torch.as_tensor(censor_rate, dtype=_torch.float32)
    return _torch.cat([props, _torch.log10(t), cr, q], dim=-1)
