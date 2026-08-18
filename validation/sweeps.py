"""
sweeps.py — parameter sweeps establishing the SCOPE of GRIN's results.

Design: ONE-FACTOR-AT-A-TIME from a documented baseline (a full combinatorial grid is
thousands of runs and mostly redundant), with the trial-count sweep additionally run at
several prior settings to spot-check interactions.

    python validation/sweeps.py                 # all sweeps
    python validation/sweeps.py --only trials   # one axis
    python validation/sweeps.py --quick         # small/fast

Axes
----
  trials    : per-stimulus trial count, 5 -> 2000 (log-spaced). The regime axis.
              Also run at 3 prior settings to check for interactions.
  prior     : z_max in {2,3,4} x r_max in {0.7,0.9,0.95}. Shows the network is not
              tuned to one arbitrary parameter envelope.
  rt_speed  : RT regime spanning mean RT ~0.4s to ~5s (plus a slow/noisy condition).
              Shows the RT/architecture results are not an artifact of one speed range.
  capacity  : hidden layers {128x3, 256x3} x seeds {0,1,2}. Converts "we got a number"
              into "the result is stable".

NOT swept: batch size, learning rate, dropout, epochs — these affect convergence, not
conclusions.

Outputs: results/validation/sweeps/<axis>.json + sweeps_summary.csv + a figure.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR
import src.grt_model as gm
from src.data.generator import GRTDataGenerator
from src.data.rt_lba_generator import RTLBAGenerator, featurize_lba, ARCHITECTURES
from src.models.network import NPEModel, featurize, _mlp
from src.models.heads import (GaussianHead, params_to_train_space, train_space_to_params)
from src.models.losses import joint_loss
from src.inference.predict import predict_posterior, predict_point
from src.inference.model_posterior import construct_labels, amortized_compare

BASELINE = dict(n_per_class=1500, epochs=18, z_max=3.0, r_max=0.9,
                trial_range=(20, 500), hidden=(128, 128, 128), seed=0)
K = len(ARCHITECTURES)


# ------------------------------------------------------------------ counts-only model
def _fit(cfg):
    g = GRTDataGenerator(n_per_class=cfg["n_per_class"], trial_range=cfg["trial_range"],
                         z_max=cfg["z_max"], r_max=cfg["r_max"], seed=cfg["seed"] + 1)
    X, yp, Xt, yc, yl = g.generate_all_model_cms()
    f = featurize(torch.tensor(X), torch.tensor(Xt))
    t = params_to_train_space(torch.tensor(yp, dtype=torch.float32))
    c, a, b = construct_labels(yl)
    torch.manual_seed(cfg["seed"])
    m = NPEModel(in_dim=f.shape[1], hidden=cfg["hidden"], dropout=0.0, comparison=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = f.shape[0]; perm = torch.randperm(n)
    ct, at, bt = torch.tensor(c), torch.tensor(a), torch.tensor(b)
    for e in range(cfg["epochs"]):
        for i in range(0, n, 512):
            bb = perm[i:i + 512]; opt.zero_grad()
            joint_loss(m, f[bb], t[bb], ct[bb], at[bb], bt[bb], w_cls=4.0)[0].backward()
            opt.step()
    return m.eval()


def _eval(model, cfg, T, n=60, seed=42):
    g = GRTDataGenerator(n_per_class=n, trial_range=(T, T), balanced_trials=True,
                         z_max=cfg["z_max"], r_max=cfg["r_max"], seed=seed)
    X, yp, Xt, yc, yl = g.generate_all_model_cms()
    po = predict_posterior(model, X, Xt, n_samples=300)
    mean = po["mean"].numpy(); s = po["samples"].numpy()
    lo, hi = np.quantile(s, .05, 0), np.quantile(s, .95, 0)
    ac = amortized_compare(model, X, Xt); tc, _, _ = construct_labels(yl)
    pc = ac["p_corr"].argmax(1)
    return {
        "z_MAE": float(np.abs(mean[:, :8] - yp[:, :8]).mean()),
        "rho_MAE": float(np.abs(mean[:, 8:] - yp[:, 8:]).mean()),
        "coverage90": float(((yp >= lo) & (yp <= hi)).mean()),
        "PI_acc": float(np.mean((pc == 0) == (tc == 0))),
        "sepA_acc": float(np.mean((ac["p_sep_A"] > .5).astype(int) == construct_labels(yl)[1])),
    }


# ------------------------------------------------------------------ AXIS 1: trials
def sweep_trials(quick=False):
    cfg = dict(BASELINE)
    if quick: cfg.update(n_per_class=400, epochs=6)
    Ts = [5, 10, 25, 50, 100, 200, 500, 1000, 2000]
    if quick: Ts = [10, 50, 200, 1000]
    # train across the full trial range so the net sees every regime
    cfg["trial_range"] = (5, 2000)
    priors = [(3.0, 0.9)] if quick else [(2.0, 0.9), (3.0, 0.9), (4.0, 0.95)]
    out = []
    for zmax, rmax in priors:
        c = dict(cfg, z_max=zmax, r_max=rmax)
        m = _fit(c)
        for T in Ts:
            r = _eval(m, c, T)
            r.update(trials=T, z_max=zmax, r_max=rmax)
            out.append(r)
            print(f"   trials={T:5d} z_max={zmax} r_max={rmax} | z_MAE {r['z_MAE']:.3f} "
                  f"cov90 {r['coverage90']:.2f} PI {r['PI_acc']:.2f}")
    return out


# ------------------------------------------------------------------ AXIS 2: prior
def sweep_prior(quick=False):
    cfg = dict(BASELINE)
    if quick: cfg.update(n_per_class=400, epochs=6)
    grid = [(2.0, 0.7), (2.0, 0.9), (3.0, 0.7), (3.0, 0.9), (3.0, 0.95), (4.0, 0.9), (4.0, 0.95)]
    if quick: grid = [(2.0, 0.9), (3.0, 0.9), (4.0, 0.95)]
    out = []
    for zmax, rmax in grid:
        c = dict(cfg, z_max=zmax, r_max=rmax)
        m = _fit(c); r = _eval(m, c, 200)
        r.update(z_max=zmax, r_max=rmax)
        out.append(r)
        print(f"   z_max={zmax} r_max={rmax} | z_MAE {r['z_MAE']:.3f} rho_MAE {r['rho_MAE']:.3f} "
              f"cov90 {r['coverage90']:.2f}")
    return out


# ------------------------------------------------------------------ AXIS 3: RT speed regime
class _Full(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc, f = _mlp(d, (192, 192, 192), "tanh", 0.0)
        self.h = GaussianHead(f, 12)
        mk = lambda o: nn.Sequential(nn.Linear(f, 64), nn.GELU(), nn.Linear(64, o))
        self.ar = mk(K)

    def forward(self, x):
        h = self.enc(x); m, L = self.h(h); return m, L, self.ar(h)


def sweep_rt_speed(quick=False):
    """Regimes spanning mean RT ~0.4s to ~5s, plus a slow/noisy condition."""
    regimes = {
        "very_fast": dict(t0=(0.10, 0.20), A=(0.15, 0.35), drift_sd=0.35),
        "fast":      dict(t0=(0.15, 0.30), A=(0.25, 0.55), drift_sd=0.35),
        "typical":   dict(t0=(0.15, 0.45), A=(0.35, 1.10), drift_sd=0.35),
        "slow":      dict(t0=(0.30, 0.60), A=(1.00, 2.20), drift_sd=0.35),
        "slow_noisy": dict(t0=(0.30, 0.60), A=(1.00, 2.20), drift_sd=0.70),
    }
    if quick: regimes = {k: regimes[k] for k in ("fast", "typical", "slow")}
    npc, ep = (300, 6) if quick else (900, 18)
    out = []
    for name, spec in regimes.items():
        class G(RTLBAGenerator):
            def sample_lba(self, rng):
                return np.array([rng.uniform(*spec["t0"]), rng.uniform(*spec["A"]),
                                 rng.uniform(0.6, 2.0), rng.uniform(0.6, 2.0)])
        g = G(n_per_class=npc, trial_range=(50, 300), drift_sd=spec["drift_sd"], seed=1)
        X, RTQ, Xt, yp, ylba, yc, yl, ya = g.generate()
        mean_rt = float(np.mean(RTQ.reshape(len(X), 4, 4, 5)[:, :, :, 2][RTQ.reshape(len(X), 4, 4, 5)[:, :, :, 2] > 0]))
        f = featurize_lba(X, RTQ, Xt)
        t = params_to_train_space(torch.tensor(yp, dtype=torch.float32))
        ar = torch.tensor(ya)
        torch.manual_seed(0)
        m = _Full(f.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        n = f.shape[0]; perm = torch.randperm(n)
        for e in range(ep):
            for i in range(0, n, 512):
                b = perm[i:i + 512]; opt.zero_grad()
                mm, L, arl = m(f[b])
                loss = -torch.distributions.MultivariateNormal(mm, scale_tril=L).log_prob(t[b]).mean()
                loss = loss + 4 * Fn.cross_entropy(arl, ar[b]); loss.backward(); opt.step()
        ge = G(n_per_class=60, trial_range=(150, 150), drift_sd=spec["drift_sd"], seed=42)
        Xe, RTQe, Xte, ype, _, _, _, yae = ge.generate()
        with torch.no_grad(): o = m(featurize_lba(Xe, RTQe, Xte))
        p = train_space_to_params(o[0]).numpy()
        pa = torch.softmax(o[2], -1).numpy().argmax(1)
        st = [i for i, a in enumerate(ARCHITECTURES) if "self_terminating" in a]
        msk = np.isin(yae, st)
        r = {"regime": name, "median_RT_s": round(mean_rt, 2),
             "z_MAE": float(np.abs(p[:, :8] - ype[:, :8]).mean()),
             "rho_MAE": float(np.abs(p[:, 8:] - ype[:, 8:]).mean()),
             "arch_acc": float(np.mean(pa == yae)),
             "self_terminating_recall": float(np.mean(np.isin(pa[msk], st))) if msk.sum() else None}
        out.append(r)
        print(f"   {name:11s} medRT~{mean_rt:4.2f}s | z_MAE {r['z_MAE']:.3f} | arch {r['arch_acc']:.2f} "
              f"| self-terminating recall {r['self_terminating_recall']:.2f}")
    return out


# ------------------------------------------------------------------ AXIS 4: capacity/seed
def sweep_capacity(quick=False):
    cfg = dict(BASELINE)
    if quick: cfg.update(n_per_class=400, epochs=6)
    hiddens = [(128, 128, 128)] if quick else [(128, 128, 128), (256, 256, 256)]
    seeds = [0, 1] if quick else [0, 1, 2]
    out = []
    for h in hiddens:
        for s in seeds:
            c = dict(cfg, hidden=h, seed=s)
            m = _fit(c); r = _eval(m, c, 200)
            r.update(hidden=str(h), seed=s); out.append(r)
            print(f"   hidden={h} seed={s} | z_MAE {r['z_MAE']:.3f} cov90 {r['coverage90']:.2f}")
    return out


AXES = {"trials": sweep_trials, "prior": sweep_prior,
        "rt_speed": sweep_rt_speed, "capacity": sweep_capacity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=list(AXES))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(RESULTS_DIR, "validation", "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    all_rows = {}
    for name, fn in AXES.items():
        if a.only and name != a.only: continue
        print(f"\n=== SWEEP: {name} ===")
        t0 = time.time(); rows = fn(quick=a.quick)
        all_rows[name] = rows
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"   ({time.time()-t0:.0f}s) -> {out_dir}/{name}.json")
    # csv summary
    import csv
    with open(os.path.join(out_dir, "sweeps_summary.csv"), "w", newline="") as f:
        w = None
        for axis, rows in all_rows.items():
            for r in rows:
                row = dict(axis=axis, **r)
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
                w.writerow({k: row.get(k) for k in w.fieldnames})
    print(f"\nwrote {out_dir}/sweeps_summary.csv")
    try:
        make_figure(out_dir)
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()


def make_figure(out_dir=None):
    """Render the sweep summary figure (call after running sweeps).

    Changes from the first version: all four sweep axes are plotted, not two. `prior` and
    `capacity` were being written to JSON and never drawn, so the figure silently claimed
    less robustness than the sweeps actually establish. The capacity panel shows the SEED
    SPREAD rather than a single line, since the whole point of that axis is "the result is
    stable, not a lucky draw". Missing JSON files leave a labelled placeholder instead of
    an empty axis, so a partial run is legible as a partial run. The dpi override is gone;
    savefig.dpi from style.py now applies, matching every other figure in the project.
    """
    import json, os
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    from src.viz.style import set_style, BLUE, BLUE_DEEP, RED_DEEP, MUTE, INK
    out_dir = out_dir or os.path.join(RESULTS_DIR, "validation", "sweeps")
    set_style()
    cols = [BLUE, BLUE_DEEP, RED_DEEP, MUTE]

    def load(name):
        fp = os.path.join(out_dir, f"{name}.json")
        return json.load(open(fp)) if os.path.exists(fp) else None

    def missing(ax, name):
        ax.text(0.5, 0.5, f"{name}.json not found\n(run: python scripts/sweeps.py --only {name})",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color=MUTE)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes.ravel()

    # --- 1 & 2: recovery and calibration vs trial count, one line per prior envelope ---
    rows = load("trials")
    if rows:
        priors = sorted({(r["z_max"], r["r_max"]) for r in rows})
        for (z, rr), col in zip(priors, cols):
            sel = sorted([r for r in rows if (r["z_max"], r["r_max"]) == (z, rr)],
                         key=lambda r: r["trials"])
            t = [r["trials"] for r in sel]
            ax[0].plot(t, [r["z_MAE"] for r in sel], "o-", color=col, lw=2,
                       label=f"$Z_{{max}}$={z}, $R_{{max}}$={rr}")
            ax[1].plot(t, [r["coverage90"] for r in sel], "o-", color=col, lw=2,
                       label=f"$Z_{{max}}$={z}, $R_{{max}}$={rr}")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("trials per stimulus (log)"); ax[0].set_ylabel("z-score MAE (log)")
        ax[0].set_title("Recovery vs trial count"); ax[0].legend(fontsize=8.5)
        ax[1].axhline(0.9, color=INK, ls=(0, (4, 3)), lw=1.4, label="nominal 90%")
        ax[1].set_xscale("log"); ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("trials per stimulus (log)"); ax[1].set_ylabel("90% interval coverage")
        ax[1].set_title("Calibration holds across regimes"); ax[1].legend(fontsize=8.5)
    else:
        missing(ax[0], "trials"); missing(ax[1], "trials")

    # --- 3: prior envelope ---
    rows = load("prior")
    if rows:
        labs = [f"{r['z_max']}/{r['r_max']}" for r in rows]
        x = _np.arange(len(rows)); w = 0.38
        ax[2].bar(x - w / 2, [r["z_MAE"] for r in rows], w, color=BLUE, label="z MAE")
        ax[2].bar(x + w / 2, [r.get("rho_MAE", _np.nan) for r in rows], w, color=RED_DEEP,
                  label=r"$\rho$ MAE")
        ax[2].set_xticks(x); ax[2].set_xticklabels(labs, rotation=30, ha="right")
        ax[2].set_xlabel("$Z_{max}$ / $R_{max}$"); ax[2].set_ylabel("MAE")
        ax[2].set_title("Not tuned to one prior envelope"); ax[2].legend(fontsize=8.5)
    else:
        missing(ax[2], "prior")

    # --- 4: capacity, shown as seed spread ---
    rows = load("capacity")
    if rows:
        hs = sorted({str(r["hidden"]) for r in rows})
        for i, h in enumerate(hs):
            v = [r["z_MAE"] for r in rows if str(r["hidden"]) == h]
            ax[3].scatter([i] * len(v), v, s=42, color=BLUE, zorder=3,
                          label="individual seeds" if i == 0 else None)
            ax[3].hlines(_np.mean(v), i - 0.22, i + 0.22, color=BLUE_DEEP, lw=2.4,
                         label="mean" if i == 0 else None)
        ax[3].set_xticks(range(len(hs))); ax[3].set_xticklabels(hs, fontsize=8.5)
        ax[3].set_xlabel("hidden layers"); ax[3].set_ylabel("z-score MAE")
        ax[3].set_title("Stable across capacity and seed"); ax[3].legend(fontsize=8.5)
    else:
        missing(ax[3], "capacity")

    # --- 5: RT speed regime ---
    rows = load("rt_speed")
    if rows:
        x = [r["median_RT_s"] for r in rows]
        ax[4].plot(x, [r["arch_acc"] for r in rows], "o-", color=BLUE_DEEP, lw=2,
                   label="architecture")
        ax[4].plot(x, [r["self_terminating_recall"] for r in rows], "s-", color=RED_DEEP, lw=2,
                   label="self-terminating recall")
        ax[4].set_ylim(0, 1); ax[4].set_xlabel("median RT (s)"); ax[4].set_ylabel("accuracy")
        ax[4].set_title("Invariant to RT speed regime"); ax[4].legend(fontsize=8.5)
    else:
        missing(ax[4], "rt_speed")

    ax[5].set_visible(False)
    fig.suptitle("Robustness sweeps", x=0.02, ha="left", fontweight="bold",
                 fontsize=15, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(RESULTS_DIR, "figures", "sweeps.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p); plt.close(fig)
    print(f"figure -> {p}")
    return p
