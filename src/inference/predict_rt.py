"""rt_predict.py — load and run the trained RT-augmented model."""
import numpy as np
import torch

from ..config import RT_MODEL_FILE, RT_HIDDEN_LAYERS, RT_DROPOUT, DEVICE
from ..models.rt_network import RTNPEModel
from ..models.heads import train_space_to_params
from ..data.rt_lba_generator import featurize_lba, ARCHITECTURES, LBA_NAMES


def load_rt_model(path=RT_MODEL_FILE, device=None):
    """Rebuilds the matching architecture from the checkpoint (no key mismatches)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    hidden = tuple(ckpt.get("hidden", RT_HIDDEN_LAYERS))
    dropout = ckpt.get("dropout", RT_DROPOUT)
    in_dim = ckpt.get("in_dim", 100)
    m = RTNPEModel(in_dim=in_dim, hidden=hidden, dropout=dropout)
    m.load_state_dict(ckpt["state_dict"])
    m = m.to(device).eval()
    m._lba_mu = ckpt["lba_mu"].to(device)
    m._lba_sd = ckpt["lba_sd"].to(device)
    return m


@torch.no_grad()
def predict_rt(model, counts, rtq, trials):
    """One forward pass -> everything the RT model infers."""
    device = next(model.parameters()).device
    x = featurize_lba(counts, rtq, trials).to(device)
    mean, L, cl, al, bl, arl, lbl = model(x)
    params = train_space_to_params(mean).cpu().numpy()
    var = (L ** 2).sum(-1)
    return {
        "params": params,                                    # (N,12) GRT
        "params_sd": var.clamp_min(1e-12).sqrt().cpu().numpy(),
        "p_corr": torch.softmax(cl, -1).cpu().numpy(),       # PI / RHO1 / free
        "p_sep_A": torch.softmax(al, -1)[:, 1].cpu().numpy(),
        "p_sep_B": torch.softmax(bl, -1)[:, 1].cpu().numpy(),
        "p_arch": torch.softmax(arl, -1).cpu().numpy(),      # (N,5) SFT architecture
        "arch": np.array(ARCHITECTURES)[torch.softmax(arl, -1).argmax(-1).cpu().numpy()],
        "lba": (lbl * model._lba_sd + model._lba_mu).cpu().numpy(),   # (N,4)
        "lba_names": LBA_NAMES,
    }


def dimension_neglect(pred):
    """P(the participant is NOT using a dimension) — the self-terminating models."""
    st = [i for i, a in enumerate(ARCHITECTURES) if "self_terminating" in a]
    return pred["p_arch"][:, st].sum(1)
