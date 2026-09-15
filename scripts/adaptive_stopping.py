"""Reproduce the manuscript's exploratory precision-stopping simulation.

The simulation draws 33 parameter vectors from each of the twelve training-prior
classes. For every observer and stimulus, one categorical sequence is generated up
to 640 trials; every interim matrix is a prefix of that same sequence. This avoids
the incoherent alternative of drawing an independent data set at each look.

The rule uses the largest uncorrected parameter-space posterior SD. It is a
proof-of-concept efficiency calculation, not a sequentially calibrated decision rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import grt_model as gm
from src.api import load_model
from src.config import RESULTS_DIR, Z_MAX, R_MAX
from src.models.network import featurize


LEVELS = np.array([5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640])
THRESHOLDS = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25)
N_PER_CLASS = 33
SEED = 20260825


def _sample_parameters(rng: np.random.Generator):
    params, labels = [], []
    for name in gm.MODEL_NAMES:
        zx, zy, rho = gm.sample_prior(name, N_PER_CLASS, rng, z_max=Z_MAX, r_max=R_MAX)
        params.append(gm.pack(zx, zy, rho))
        labels.extend([name] * N_PER_CLASS)
    return np.concatenate(params), np.asarray(labels)


def _cumulative_counts(params: np.ndarray, rng: np.random.Generator):
    zx, zy, rho = gm.unpack(params)
    probs = gm.forward_probabilities(zx, zy, rho)
    m = len(params)
    responses = np.empty((m, 4, LEVELS[-1]), dtype=np.int8)
    for i in range(m):
        for s in range(4):
            responses[i, s] = rng.choice(4, size=LEVELS[-1], p=probs[i, s])
    out = []
    for level in LEVELS:
        cm = np.zeros((m, 4, 4), dtype=np.int64)
        prefix = responses[:, :, :level]
        for r in range(4):
            cm[:, :, r] = (prefix == r).sum(axis=2)
        out.append(cm)
    return out


@torch.no_grad()
def _parameter_std(model, counts: np.ndarray, level: int):
    device = next(model.parameters()).device
    trials = np.full((len(counts), 4), level, dtype=np.int64)
    x = featurize(torch.as_tensor(counts.reshape(len(counts), 16)),
                  torch.as_tensor(trials)).to(device)
    mean_train, scale_tril = model(x)
    std_train = (scale_tril.square().sum(-1)).clamp_min(1e-12).sqrt()
    rho = torch.tanh(mean_train[:, 8:12].clamp(-7.0, 7.0))
    std = torch.cat([std_train[:, :8], (1.0 - rho.square()) * std_train[:, 8:12]], dim=1)
    return std.cpu().numpy()


def run(seed: int = SEED):
    rng = np.random.default_rng(seed)
    params, labels = _sample_parameters(rng)
    matrices = _cumulative_counts(params, rng)
    model = load_model(device="cpu")
    max_sd = np.stack([_parameter_std(model, cm, int(level)).max(axis=1)
                       for cm, level in zip(matrices, LEVELS)], axis=1)

    by_threshold = {}
    for threshold in THRESHOLDS:
        met = max_sd <= threshold
        reached = met.any(axis=1)
        first = np.argmax(met, axis=1)
        stop_trials = np.where(reached, LEVELS[first], LEVELS[-1])
        coverage = float(reached.mean())
        fixed_candidates = np.flatnonzero(met.mean(axis=0) >= coverage - 1e-12)
        fixed = int(LEVELS[fixed_candidates[0]]) if len(fixed_candidates) else int(LEVELS[-1])
        adaptive_mean = float(stop_trials.mean())
        by_threshold[f"{threshold:g}"] = {
            "adaptive_mean_trials": adaptive_mean,
            "fixed_matched_coverage": fixed,
            "saving_matched": float(1.0 - adaptive_mean / fixed),
            "never_reached_frac": float(1.0 - coverage),
            "coverage": coverage,
        }

    return {
        "meta": {
            "seed": seed,
            "n_per_class": N_PER_CLASS,
            "n_model_classes": len(gm.MODEL_NAMES),
            "sampling": "cumulative categorical prefixes; balanced trials across stimuli",
            "criterion": "largest uncorrected parameter-space posterior SD",
            "calibrated_sequential_rule": False,
        },
        "M": int(len(params)),
        "levels": LEVELS.tolist(),
        "model_class_counts": {name: int((labels == name).sum()) for name in gm.MODEL_NAMES},
        "cohort_mean_sd": max_sd.mean(axis=0).tolist(),
        "by_sd_max": by_threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", type=Path,
                    default=Path(RESULTS_DIR) / "adaptive_stopping.json")
    args = ap.parse_args()
    result = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["by_sd_max"], indent=2))


if __name__ == "__main__":
    main()
