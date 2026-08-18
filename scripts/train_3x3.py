"""Train an experimental 3x3 NPE checkpoint without touching production files."""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.losses import joint_loss
from src.models.network import NPEModel, featurize_square


def _construct_labels(names, model_module):
    specs = [model_module.MODEL_SPECS[str(name)] for name in names]
    corr = np.array([{"pi": 0, "rho1": 1, "free": 2}[s[0]] for s in specs])
    sep_a = np.array([int(s[1]) for s in specs])
    sep_b = np.array([int(s[2]) for s in specs])
    return corr, sep_a, sep_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/simulated/grt_3x3_dataset.npz")
    parser.add_argument("--output", default="results/models/npe_3x3_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.data, allow_pickle=True)
    if str(data.get("design", "")) != "3x3":
        raise ValueError("dataset is not marked as a 3x3 corpus")
    variance_model = str(data.get("variance_model", "unit"))
    if variance_model == "unit":
        from src import grt_model_3x3 as gm
        from src.models.heads_3x3 import params_to_train_space
    elif variance_model == "free":
        from src import grt_model_3x3_hetero as gm
        from src.models.heads_3x3_hetero import params_to_train_space
    else:
        raise ValueError(f"unknown 3x3 variance model: {variance_model}")
    features = featurize_square(torch.tensor(data["X"]), torch.tensor(data["X_trials"]), 9)
    targets = params_to_train_space(torch.tensor(data["y_params"], dtype=torch.float32))
    corr, sep_a, sep_b = _construct_labels(data["y_cls_label"], gm)
    labels = [torch.tensor(x, dtype=torch.long) for x in (corr, sep_a, sep_b)]

    permutation = torch.randperm(len(features), generator=torch.Generator().manual_seed(args.seed))
    n_validation = max(1, int(len(features) * args.validation_fraction))
    validation, training = permutation[:n_validation], permutation[n_validation:]
    train_loader = DataLoader(
        TensorDataset(features[training], targets[training], *(x[training] for x in labels)),
        batch_size=args.batch_size, shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(features[validation], targets[validation], *(x[validation] for x in labels)),
        batch_size=2048,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NPEModel(in_dim=90, param_dim=gm.N_PARAMS, hidden=(256, 256, 256),
                     activation="gelu", dropout=0.1, comparison=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for batch in train_loader:
            xb, yb, cb, ab, bb = (x.to(device) for x in batch)
            optimizer.zero_grad()
            loss, _, _ = joint_loss(model, xb, yb, cb, ab, bb, w_cls=4.0)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                joint_loss(model, *(x.to(device) for x in batch), w_cls=4.0)[0].item()
                for batch in validation_loader
            ])
        if val_loss < best:
            best = float(val_loss)
            torch.save({
                "state_dict": model.state_dict(),
                "design": "3x3",
                "variance_model": variance_model,
                "n_stimuli": 9,
                "input_dim": 90,
                "param_dim": gm.N_PARAMS,
                "param_names": gm.PARAM_NAMES,
                "hidden": [256, 256, 256],
                "activation": "gelu",
                "dropout": 0.1,
                "comparison": True,
                "training_data": os.path.abspath(args.data),
                "seed": args.seed,
                "best_epoch": epoch,
                "best_val_loss": best,
            }, args.output)
        print(f"epoch {epoch:3d} train={np.mean(train_loss):.4f} val={val_loss:.4f} best={best:.4f}")


if __name__ == "__main__":
    main()
