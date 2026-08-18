"""Train the NPE model on the simulated dataset. Run from the project root:
    python scripts/train.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.config import (
    DATASET_FILE, MODEL_FILE, DEVICE, TRAINING_HISTORY_DIR, FIGURES_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VAL_SPLIT, HIDDEN_LAYERS, ACTIVATION,
    DROPOUT, PATIENCE, MIN_DELTA, TRAIN_SEED, RLRP_FACTOR, RLRP_PATIENCE, RLRP_MIN_LR
)
from src.models.network import NPEModel, featurize
from src.models.heads import params_to_train_space
from src.models.losses import joint_loss
from src.inference.model_posterior import construct_labels
from src.provenance import build_manifest

TRAINING_CURVE_FIG = os.path.join(FIGURES_DIR, "training_curve.png")
TRAINING_HISTORY_CSV = os.path.join(TRAINING_HISTORY_DIR, "training_history.csv")

def main():
    torch.manual_seed(TRAIN_SEED)
    device = DEVICE if torch.cuda.is_available() else "cpu"
    d = np.load(DATASET_FILE, allow_pickle=True)
    feats = featurize(torch.tensor(d["X"]), torch.tensor(d["X_trials"]))
    targets = params_to_train_space(torch.tensor(d["y_params"], dtype=torch.float32))
    _corr, _sa, _sb = construct_labels(d["y_cls_label"])
    corr = torch.tensor(_corr, dtype=torch.long); sepA = torch.tensor(_sa, dtype=torch.long); sepB = torch.tensor(_sb, dtype=torch.long)

    # Computed once (hashes the ~350MB dataset file, so not worth redoing per checkpoint
    # save); the dynamic best_epoch/best_val_nll fields are merged in at save time below.
    manifest = build_manifest()

    n = feats.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(TRAIN_SEED))
    n_val = int(n * VAL_SPLIT)
    va, tr = idx[:n_val], idx[n_val:]
    train_dl = DataLoader(TensorDataset(feats[tr], targets[tr], corr[tr], sepA[tr], sepB[tr]),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(feats[va], targets[va], corr[va], sepA[va], sepB[va]), batch_size=2048)

    model = NPEModel(in_dim=feats.shape[1], hidden=HIDDEN_LAYERS,
                     activation=ACTIVATION, dropout=DROPOUT, comparison=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=RLRP_FACTOR, patience=RLRP_PATIENCE, min_lr=RLRP_MIN_LR)

    train_losses, val_losses, lrs = [], [], []
    best, wait = float("inf"), 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_losses = []
        for xb, yb, cb, ab, bb in train_dl:
            opt.zero_grad()
            loss, _, _ = joint_loss(model, xb.to(device), yb.to(device),
                                    cb.to(device), ab.to(device), bb.to(device), w_cls=4.0)
            loss.backward()
            opt.step()
            epoch_train_losses.append(loss.item())
        train_loss = float(np.mean(epoch_train_losses))
        model.eval()
        with torch.no_grad():
            vloss = np.mean([joint_loss(model, xb.to(device), yb.to(device),
                                        cb.to(device), ab.to(device), bb.to(device), w_cls=4.0)[0].item()
                             for xb, yb, cb, ab, bb in val_dl])
        scheduler.step(vloss)
        if vloss < best - MIN_DELTA:
            best, wait = vloss, 0
            torch.save({"state_dict": model.state_dict(), "hidden": list(HIDDEN_LAYERS),
                        "dropout": DROPOUT, "activation": ACTIVATION, "comparison": True,
                        "provenance": {**manifest, "best_epoch": epoch, "best_val_nll": best}},
                       MODEL_FILE)
        else:
            wait += 1
        lr = opt.param_groups[0]["lr"]
        train_losses.append(train_loss)
        val_losses.append(vloss)
        lrs.append(opt.param_groups[0]["lr"])
        print(f"epoch {epoch:3d}  train {train_loss:.4f} val_nll {vloss:.4f}  (best {best:.4f})  lr {lr:.2e}")
        if wait >= PATIENCE:
            print("early stopping."); break
    print(f"done. best val_nll = {best:.4f}  ->  {MODEL_FILE}")

    history = pd.DataFrame({
        "epoch": np.arange(len(train_losses)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "lr": lrs,
    })
    history.to_csv(TRAINING_HISTORY_CSV, index = False)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"],
            history["train_loss"],
            label="Training loss")

    plt.plot(history["epoch"],
            history["val_loss"],
            label="Validation loss")
    best_epoch = int(np.argmin(val_losses))
    plt.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best epoch ({best_epoch})"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NPE Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_FIG, dpi=300)
    plt.close()
    
    
if __name__ == "__main__":
    main()
