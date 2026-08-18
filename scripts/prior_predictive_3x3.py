"""Compare 3x3 prior-predictive simulations with the two Thomas (2015) matrices."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import grt_model_3x3 as gm
from src import grt_model_3x3_hetero as gm_free
from src.data.generator_3x3 import GRT3x3DataGenerator, GRT3x3HeteroDataGenerator


RESPONSE_COLUMNS = [f"r{a}{b}" for a in range(1, 4) for b in range(1, 4)]


def matrix_metrics(counts):
    counts = np.asarray(counts, dtype=float).reshape(-1, 9, 9)
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = counts / np.maximum(totals, 1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-300, 1))).sum(axis=-1)
    response_a = counts.reshape(-1, 9, 3, 3).sum(axis=3)
    response_b = counts.reshape(-1, 9, 3, 3).sum(axis=2)
    grand_total = counts.sum(axis=(1, 2))
    return pd.DataFrame({
        "accuracy": np.diagonal(counts, axis1=1, axis2=2).sum(axis=1) / grand_total,
        "mean_row_entropy": entropy.mean(axis=1),
        "empty_cell_fraction": (counts == 0).mean(axis=(1, 2)),
        "near_empty_cell_fraction": (counts <= 1).mean(axis=(1, 2)),
        "middle_response_a": response_a[:, :, 1].sum(axis=1) / grand_total,
        "middle_response_b": response_b[:, :, 1].sum(axis=1) / grand_total,
        "sparse_marginal_fraction": np.concatenate(
            [(response_a <= 1), (response_b <= 1)], axis=1
        ).mean(axis=(1, 2)),
    })


def load_thomas(path):
    data = pd.read_csv(path)
    matrices, names = [], []
    for dataset, observer in data.groupby("dataset", sort=False):
        observer = observer.sort_values("stimulus")
        if observer.source_label.tolist() != list(range(1, 10)):
            raise ValueError(f"{dataset} is not in canonical stimulus-label order")
        matrices.append(observer[RESPONSE_COLUMNS].to_numpy())
        names.append(dataset)
    return np.stack(matrices), names


def summarize(frame):
    return {
        column: {
            "min": float(frame[column].min()),
            "median": float(frame[column].median()),
            "max": float(frame[column].max()),
        }
        for column in frame.columns if column not in ("source", "dataset")
    }


def information_regime(accuracy, mean_row_entropy):
    """Descriptive audit strata, not inferential classifications.

    Thomas B anchors the low-information regime and Thomas A the moderate regime.
    Both accuracy and entropy are required so a biased but concentrated response
    pattern cannot be mislabeled high-information from entropy alone.
    """
    if accuracy <= 0.33 and mean_row_entropy >= 1.70:
        return "low_information"
    if accuracy > 0.50 and mean_row_entropy < 1.35:
        return "high_information"
    return "moderate_information"


def sd_sparsity_table(X, y_params):
    counts = X.reshape(-1, 9, 3, 3)
    response_a = counts.sum(axis=3)
    response_b = counts.sum(axis=2)
    _, _, sd_x, sd_y, _ = gm_free.unpack(y_params)
    sd = np.concatenate([sd_x.ravel(), sd_y.ravel()])
    sparse = np.concatenate([
        (response_a <= 1).any(axis=2).ravel(),
        (response_b <= 1).any(axis=2).ravel(),
    ])
    bins = [0.5, 0.75, 1.0, 1.5, 2.000001]
    labels = ["0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50-2.00"]
    groups = pd.cut(sd, bins=bins, labels=labels, include_lowest=True, right=False)
    return pd.DataFrame({"sd_band": groups, "has_response_margin_le_1": sparse}).groupby(
        "sd_band", observed=True
    ).agg(n=("has_response_margin_le_1", "size"),
          sparse_rate=("has_response_margin_le_1", "mean")).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument("--trial-min", type=int, default=80)
    parser.add_argument("--trial-max", type=int, default=80)
    parser.add_argument("--seed", type=int, default=315)
    parser.add_argument("--thomas", default="data/real/thomas15_3x3.csv")
    parser.add_argument("--output", default="results/validation/3x3_prior_predictive.json")
    parser.add_argument("--sd-output", default="results/validation/3x3_sd_sparsity.csv")
    args = parser.parse_args()

    common = dict(n_per_class=args.n_per_class,
                  trial_range=(args.trial_min, args.trial_max), seed=args.seed,
                  balanced_trials=True)
    unit = GRT3x3DataGenerator(**common).generate_all_model_cms()
    free = GRT3x3HeteroDataGenerator(**common).generate_all_model_cms()
    thomas_counts, thomas_names = load_thomas(args.thomas)

    unit_metrics = matrix_metrics(unit[0]); unit_metrics["source"] = "unit_simulation"
    free_metrics = matrix_metrics(free[0]); free_metrics["source"] = "free_simulation"
    thomas_metrics = matrix_metrics(thomas_counts)
    thomas_metrics["source"] = "Thomas_2015"
    thomas_metrics["dataset"] = thomas_names
    thomas_metrics["information_regime"] = [
        information_regime(row.accuracy, row.mean_row_entropy)
        for row in thomas_metrics.itertuples()
    ]

    report = {
        "regime": {"trials_per_stimulus": [args.trial_min, args.trial_max],
                   "n_per_class": args.n_per_class, "seed": args.seed},
        "unit_simulation": summarize(unit_metrics),
        "free_simulation": summarize(free_metrics),
        "thomas": thomas_metrics.to_dict(orient="records"),
        "notes": [
            "Thomas stimuli were reordered by labels from source positions 1,4,7,2,5,8,3,6,9.",
            "Sparsity conditional on SD is reported separately for the free-variance simulator.",
            "This is a prior-predictive scope audit, not a fitted-model result.",
            "Thomas A and B remain separate information strata; no pooled recovery headline is valid.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.sd_output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    sd_sparsity_table(free[0], free[1]).to_csv(args.sd_output, index=False)
    print(f"wrote {args.output}")
    print(f"wrote {args.sd_output}")


if __name__ == "__main__":
    main()
