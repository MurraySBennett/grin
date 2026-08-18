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
    # weights_only=False: see the matching comment in src/api.py's load_model() --
    # same reason (own-generated checkpoint, torch_version provenance field).
    ckpt = torch.load(path, map_location=device, weights_only=False)
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


def self_terminating_probability(pred):
    """Total probability assigned to the two self-terminating architectures.

    The simulator chooses which dimension is processed on each trial and guesses
    the other. This quantity is therefore not evidence of stable neglect of a
    particular dimension.
    """
    st = [i for i, a in enumerate(ARCHITECTURES) if "self_terminating" in a]
    return pred["p_arch"][:, st].sum(1)


def dimension_neglect(pred):
    """Deprecated alias for :func:`self_terminating_probability`."""
    return self_terminating_probability(pred)


def _confusion(true_idx, pred_idx, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(true_idx, pred_idx):
        cm[t, p] += 1
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    return cm, recall, precision


def architecture_ablation(model, counts, rtq, trials, arch_idx, shuffle_seed=0):
    """How much of the joint network's architecture recovery depends on genuinely
    paired response-time information, versus response-time information at all.

    IMPORTANT SCOPE, stated here once so every caller inherits it rather than
    re-deriving it: this ablates the ALREADY-TRAINED joint network's input --
    it does not train a fresh counts-only classifier from scratch, and its
    "mean profile" condition does not leave the input in the training
    distribution (replacing every observer's own 80-dim RT-quantile vector with
    the across-observer mean breaks the learned joint structure among counts,
    cell occupancy, RT quantiles, and parameters that training saw, it doesn't
    just remove information). What this measures is exactly: how much this
    trained network's architecture output depends on receiving a correctly
    paired RT summary. It does not measure the information content of RT in
    the abstract, and does not bound what a from-scratch counts-only
    classifier could achieve -- see the accompanying manuscript supplement for
    the exact wording this licenses.

    counts (N,16), rtq (N,80), trials (N,4), arch_idx (N,) true architecture
    index. Returns a dict with the baseline confusion matrix/recall/precision
    and both ablations' accuracy and per-architecture recall, ready to
    json.dump directly (plain Python types / nested lists, no numpy objects).
    """
    k = len(ARCHITECTURES)
    pred_real = predict_rt(model, counts, rtq, trials)
    pa_real = pred_real["p_arch"].argmax(1)
    cm, recall, precision = _confusion(arch_idx, pa_real, k)
    acc_real = float(np.mean(pa_real == arch_idx))

    rtq_mean = np.broadcast_to(rtq.mean(axis=0, keepdims=True), rtq.shape).copy()
    pa_mean = predict_rt(model, counts, rtq_mean, trials)["p_arch"].argmax(1)
    acc_mean = float(np.mean(pa_mean == arch_idx))
    recall_mean = _confusion(arch_idx, pa_mean, k)[1]

    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(len(rtq))
    pa_shuf = predict_rt(model, counts, rtq[perm], trials)["p_arch"].argmax(1)
    acc_shuf = float(np.mean(pa_shuf == arch_idx))
    recall_shuf = _confusion(arch_idx, pa_shuf, k)[1]

    return {
        "architectures": list(ARCHITECTURES),
        "n": int(len(counts)),
        "chance": 1 / k,
        "baseline": {
            "accuracy": acc_real,
            "confusion_matrix": cm.tolist(),
            "recall": recall.tolist(),
            "precision": precision.tolist(),
        },
        "ablation_mean_profile": {
            "accuracy": acc_mean,
            "recall": recall_mean.tolist(),
            "description": "every observer's RT-quantile input replaced by the "
                           "across-observer mean profile; NOT in-distribution for "
                           "the trained network (see function docstring)",
        },
        "ablation_shuffled": {
            "accuracy": acc_shuf,
            "recall": recall_shuf.tolist(),
            "shuffle_seed": shuffle_seed,
            "description": "RT-quantile inputs permuted across observers, "
                           "preserving the marginal RT distribution but breaking "
                           "matrix-RT correspondence",
        },
    }
