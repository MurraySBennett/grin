"""Train the RT-augmented model (GRT + constructs + architecture + LBA).

    python scripts/generate_data.py --rt      # first, make the RT dataset
    python scripts/train_rt.py                # then train

Only needed if your data include response times. The counts-only pipeline
(`scripts/train.py`) is independent and remains the default.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Fn
from torch.utils.data import TensorDataset, DataLoader

from src.config import (
    RT_DATASET_FILE, RT_MODEL_FILE, FIGURES_DIR, TRAINING_HISTORY_DIR,
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, VAL_SPLIT, PATIENCE, MIN_DELTA, TRAIN_SEED,
    RLRP_FACTOR, RLRP_PATIENCE, RLRP_MIN_LR, RT_HIDDEN_LAYERS, RT_DROPOUT
)

RT_TRAINING_CURVE_FIG = os.path.join(FIGURES_DIR, "training_curve_rt.png")
RT_TRAINING_HISTORY_CSV = os.path.join(TRAINING_HISTORY_DIR, "training_history_rt.csv")

from src.models.rt_network import RTNPEModel
from src.models.heads import params_to_train_space
from src.data.rt_lba_generator import featurize_lba
from src.inference.model_posterior import construct_labels


def main():
    torch.manual_seed(TRAIN_SEED)
    device = DEVICE if torch.cuda.is_available() else "cpu"
    d = np.load(RT_DATASET_FILE, allow_pickle=True)

    feats = featurize_lba(d["X"], d["RTQ"], d["X_trials"])
    tgt = params_to_train_space(torch.tensor(d["y_params"], dtype=torch.float32))
    corr, sepA, sepB = construct_labels(d["y_cls_label"])
    corr = torch.tensor(corr, dtype=torch.long)
    sepA = torch.tensor(sepA, dtype=torch.long)
    sepB = torch.tensor(sepB, dtype=torch.long)
    arch = torch.tensor(d["y_arch"], dtype=torch.long)
    lba = torch.tensor(d["y_lba"], dtype=torch.float32)
    lba_mu, lba_sd = lba.mean(0), lba.std(0)
    lba_z = (lba - lba_mu) / lba_sd                     # standardized LBA targets

    n = feats.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(TRAIN_SEED))
    n_val = int(n * VAL_SPLIT); va, tr = idx[:n_val], idx[n_val:]
    pack = lambda i: TensorDataset(feats[i], tgt[i], corr[i], sepA[i], sepB[i], arch[i], lba_z[i])
    train_dl = DataLoader(pack(tr), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(pack(va), batch_size=2048)

    model = RTNPEModel(in_dim=feats.shape[1], hidden=RT_HIDDEN_LAYERS,
                       dropout=RT_DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=RLRP_FACTOR, patience=RLRP_PATIENCE, min_lr=RLRP_MIN_LR)

    def loss_fn(batch):
        x, y, c, a, b, ar, lb = [t.to(device) for t in batch]
        mean, L, cl, al, bl, arl, lbl = model(x)
        nll = -torch.distributions.MultivariateNormal(mean, scale_tril=L).log_prob(y).mean()
        ce = (Fn.cross_entropy(cl, c) + Fn.cross_entropy(al, a) + Fn.cross_entropy(bl, b))
        return nll + 4 * ce + 4 * Fn.cross_entropy(arl, ar) + 2 * Fn.mse_loss(lbl, lb)

    train_losses = []
    val_losses = []
    lrs = []
    best, wait = float("inf"), 0
    for epoch in range(EPOCHS):
        epoch_train_losses = []
        model.train()
        for batch in train_dl:
            opt.zero_grad();
            loss = loss_fn(batch)
            loss.backward()
            opt.step()
            # loss_fn(batch).backward(); opt.step()
            epoch_train_losses.append(loss.item())
        train_loss = float(np.mean(epoch_train_losses))
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([loss_fn(b).item() for b in val_dl]))
        train_losses.append(train_loss)
        val_losses.append(vloss)
        lrs.append(opt.param_groups[0]["lr"])
        sched.step(vloss)
        if vloss < best - MIN_DELTA:
            best, wait = vloss, 0
            torch.save({"state_dict": model.state_dict(), "in_dim": feats.shape[1],
                        "hidden": list(RT_HIDDEN_LAYERS), "dropout": RT_DROPOUT,
                        "lba_mu": lba_mu, "lba_sd": lba_sd}, RT_MODEL_FILE)
        else:
            wait += 1
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {vloss:.4f}  (best {best:.4f})  "
              f"lr {opt.param_groups[0]['lr']:.2e}")
        if wait >= PATIENCE:
            print("early stopping."); break
    print(f"done. best {best:.4f} -> {RT_MODEL_FILE}")
    history = pd.DataFrame({
        "epoch": np.arange(len(train_losses)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "lr": lrs,
    })

    history.to_csv(RT_TRAINING_HISTORY_CSV, index=False)
       
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
    plt.title("RT-NPE Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RT_TRAINING_CURVE_FIG, dpi=300)
    plt.close() 

if __name__ == "__main__":
    main()
