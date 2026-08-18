"""
provenance.py — the checkpoint manifest external review asked for.

WHY THIS EXISTS: the currently-shipped `results/models/npe_model.pt` (dated
2026-07-15) predates `data/simulated/grt_dataset.npz` (regenerated 2026-07-24, the
same day `TRIAL_RANGE` was set to its current (1, 1000) in a large repo
restructuring commit). The dataset that trained the shipped checkpoint cannot be
identified from anything saved alongside it -- the checkpoint dict only carries
architecture hyperparameters (`hidden`, `dropout`, `activation`, `comparison`), not
data/config/generator identity. That ambiguity is exactly what this module exists
to make impossible going forward: every future `scripts/train.py` run should embed
a full manifest in the saved checkpoint, not just enough to rebuild the network.

Call `build_manifest()` right before saving and stash the result under a
`"provenance"` key in the checkpoint dict (see scripts/train.py). Deliberately
does NOT hash the exported ONNX/TorchScript files or validation artifacts here --
those don't exist yet at train time; `scripts/export_onnx.py` and the validation
suite should each read this manifest back out of the checkpoint, add their own
hash under a new key, and re-save alongside their own output (not silently drop
the training-time provenance).
"""
import hashlib
import json
import os
import subprocess
import sys

from src.config import (
    DATASET_FILE, Z_MAX, R_MAX, TRIAL_RANGE, TRIAL_IMBALANCE, DATA_SEED,
    N_PER_CLASS, HIDDEN_LAYERS, ACTIVATION, DROPOUT, NPE_HEAD, N_INPUT,
    VAL_SPLIT, EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE, MIN_DELTA,
    RLRP_FACTOR, RLRP_PATIENCE, RLRP_MIN_LR, TRAIN_SEED,
    RT_DATASET_FILE, RT_HIDDEN_LAYERS, RT_DROPOUT, RT_DRIFT_SD,
)


def _sha256_file(path, chunk_size=1 << 20):
    """Hash file CONTENTS, not the path -- so a renamed-but-identical dataset
    still matches, and a same-named-but-regenerated one doesn't silently pass."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(dirty_ok=True):
    """The generator/training-code commit, plus a dirty-tree flag -- a hash alone
    is misleading if the working tree had uncommitted changes at train time."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL, text=True).strip())
        if dirty and not dirty_ok:
            raise RuntimeError("git working tree is dirty; commit before training a "
                               "checkpoint you intend to keep, or pass dirty_ok=True "
                               "knowingly")
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": None}  # not a git checkout, or git unavailable


def build_manifest(dataset_file=DATASET_FILE, extra=None):
    """Everything the external review's manifest list asked for, computed fresh
    at call time rather than trusted from memory. `extra` merges in caller-specific
    fields (e.g. the actual best-epoch validation loss) without editing this file."""
    manifest = {
        "schema": 1,
        "dataset_file": os.path.relpath(dataset_file),
        "dataset_sha256": _sha256_file(dataset_file),
        "generator": _git_commit(),
        "prior": {"z_max": Z_MAX, "r_max": R_MAX,
                  "trial_range": list(TRIAL_RANGE), "trial_imbalance": TRIAL_IMBALANCE,
                  "n_per_class": N_PER_CLASS, "data_seed": DATA_SEED},
        "architecture": {"hidden": list(HIDDEN_LAYERS), "activation": ACTIVATION,
                         "dropout": DROPOUT, "head": NPE_HEAD, "n_input": N_INPUT},
        "optimisation": {"val_split": VAL_SPLIT, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                         "learning_rate": LEARNING_RATE, "patience": PATIENCE,
                         "min_delta": MIN_DELTA, "rlrp_factor": RLRP_FACTOR,
                         "rlrp_patience": RLRP_PATIENCE, "rlrp_min_lr": RLRP_MIN_LR,
                         "train_seed": TRAIN_SEED},
        "python": sys.version,
    }
    try:
        import torch
        # str(): torch.__version__ is a torch.torch_version.TorchVersion, a str
        # subclass -- cast to plain str so checkpoints stay loadable with PyTorch
        # >=2.6's weights_only=True default (its safe unpickler doesn't allowlist
        # TorchVersion). Checkpoints saved before this fix need weights_only=False
        # explicitly at load time (see src/api.py, src/inference/predict_rt.py).
        manifest["torch_version"] = str(torch.__version__)
    except ImportError:
        pass
    if extra:
        manifest.update(extra)
    return manifest


def build_rt_manifest(dataset_file=RT_DATASET_FILE, extra=None):
    """The RT-model counterpart to build_manifest() -- same manifest shape and
    same hashing/git-commit machinery, but reading the RT_* config values
    (architecture, dataset) instead of the counts-only ones, and adding
    drift_sd to the prior block (the one generative-model parameter the RT
    pipeline has that the counts-only one doesn't). Training optimisation
    settings (epochs, batch size, ...) are shared with the counts-only model
    -- scripts/train_rt.py imports the same globals from src.config, not a
    separate RT_EPOCHS/RT_BATCH_SIZE/etc., confirmed by reading that script
    rather than assumed -- so this reuses the same values build_manifest()
    does rather than duplicating a second copy that could drift out of sync."""
    manifest = {
        "schema": 1,
        "model": "rt",
        "dataset_file": os.path.relpath(dataset_file),
        "dataset_sha256": _sha256_file(dataset_file),
        "generator": _git_commit(),
        "prior": {"z_max": Z_MAX, "r_max": R_MAX,
                  "trial_range": list(TRIAL_RANGE), "trial_imbalance": TRIAL_IMBALANCE,
                  "n_per_class": N_PER_CLASS, "data_seed": DATA_SEED,
                  "drift_sd": RT_DRIFT_SD},
        "architecture": {"hidden": list(RT_HIDDEN_LAYERS), "dropout": RT_DROPOUT,
                         "n_input": 100},
        "optimisation": {"val_split": VAL_SPLIT, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                         "learning_rate": LEARNING_RATE, "patience": PATIENCE,
                         "min_delta": MIN_DELTA, "rlrp_factor": RLRP_FACTOR,
                         "rlrp_patience": RLRP_PATIENCE, "rlrp_min_lr": RLRP_MIN_LR,
                         "train_seed": TRAIN_SEED},
        "python": sys.version,
    }
    try:
        import torch
        # str(): torch.__version__ is a torch.torch_version.TorchVersion, a str
        # subclass -- cast to plain str so checkpoints stay loadable with PyTorch
        # >=2.6's weights_only=True default (its safe unpickler doesn't allowlist
        # TorchVersion). Checkpoints saved before this fix need weights_only=False
        # explicitly at load time (see src/api.py, src/inference/predict_rt.py).
        manifest["torch_version"] = str(torch.__version__)
    except ImportError:
        pass
    if extra:
        manifest.update(extra)
    return manifest


def verify_manifest(checkpoint_path, dataset_file=DATASET_FILE):
    """Sanity check an EXISTING checkpoint against the CURRENT dataset file --
    exactly the check that would have caught the July 15/24 mismatch immediately,
    instead of it surfacing only via an external literature review. Returns a
    dict of {"ok": bool, "reason": str}; never raises on a mismatch, only on a
    genuinely unreadable checkpoint."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "provenance" not in ckpt:
        return {"ok": False, "reason": "no provenance manifest in this checkpoint "
                "(pre-dates this module, or was saved by a script that doesn't call "
                "build_manifest) -- its training dataset cannot be verified"}
    saved = ckpt["provenance"]
    current_hash = _sha256_file(dataset_file)
    if saved.get("dataset_sha256") != current_hash:
        return {"ok": False, "reason": f"checkpoint was trained on a dataset hashing to "
                f"{saved.get('dataset_sha256')}, but {dataset_file} currently hashes to "
                f"{current_hash} -- these are not the same file, do not assume the "
                f"checkpoint's trial range/prior/imbalance rule still apply"}
    return {"ok": True, "reason": "checkpoint's recorded dataset hash matches the current "
            "dataset file"}


if __name__ == "__main__":
    # `python -m src.provenance` -- print the manifest that WOULD be embedded if you
    # trained right now, and separately check the currently-shipped checkpoints.
    print("=== counts-only ===")
    print(json.dumps(build_manifest(), indent=2))
    print()
    from src.config import MODEL_FILE
    if os.path.exists(MODEL_FILE):
        print(f"verifying {MODEL_FILE} against {DATASET_FILE}:")
        print(json.dumps(verify_manifest(MODEL_FILE), indent=2))

    print("\n=== RT ===")
    print(json.dumps(build_rt_manifest(), indent=2))
    print()
    from src.config import RT_MODEL_FILE
    if os.path.exists(RT_MODEL_FILE):
        print(f"verifying {RT_MODEL_FILE} against {RT_DATASET_FILE}:")
        print(json.dumps(verify_manifest(RT_MODEL_FILE, RT_DATASET_FILE), indent=2))
