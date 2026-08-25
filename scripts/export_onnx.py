"""Export a trained GRIN model to ONNX for in-browser / package inference.

    python scripts/export_onnx.py                        # export to results/models/
    python scripts/export_onnx.py --install --version 1.0.0
    python scripts/export_onnx.py --rt --install --version 1.0.0

Inputs : counts (B,16), trials (B,4)
Outputs: mean (B,12), std (B,12)           — posterior mean + marginal SD, parameter space
         p_corr (B,3)                      — P(PI), P(RHO1), P(free)   [comparison head]
         p_sep  (B,2)                      — P(separable A), P(separable B)

The construct probabilities now come from the TRAINED comparison heads rather than being
re-derived in JavaScript from the regression output.

--install writes the artifact into web/assets/models/<id>/ under a VERSIONED filename
and stamps the site manifest with the file's real sha256.

Both halves of that matter, for different reasons:

  * Versioned filename. .github/workflows/deploy.yaml ships .onnx with
    `max-age=31536000, immutable`. CloudFront invalidation clears the CDN edge but
    never a visitor's own browser cache, so overwriting npe_model.onnx in place
    leaves returning visitors silently running last release's weights for up to a
    year, with no error anywhere. A new filename per release makes that impossible
    rather than merely unlikely. grin-model.js loads `${dir}/${mf.file}`, so the
    manifest is the only thing that needs to know the name.

  * Real sha256. The field existed before this script wrote it, which meant it was
    hand-maintained -- and a hand-copied hash that silently stops matching the file
    is strictly worse than no hash at all, because it looks like verification. It
    is now computed here and gated in CI.
"""
import json
import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.config import MODEL_FILE, MODELS_DIR
from src.api import load_model
from src.models.network import featurize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release_provenance import PROJECT_ROOT, git_state, sha256_file, utc_now  # noqa: E402

WEB_MODELS_DIR = os.path.join(PROJECT_ROOT, "web", "assets", "models")

# (site model id, exported basename stem, exported_by string for the manifest)
EXPORT_SPECS = {
    False: ("cm", "npe_model", "scripts/export_onnx.py"),
    True: ("cmrt", "npe_rt_model", "scripts/export_onnx.py --rt"),
}


class OnnxWrapper(nn.Module):
    def __init__(self, model, dim=12):
        super().__init__()
        self.m = model; self.dim = dim
        self.has_cmp = getattr(model, "comparison", False)
        ti, _ = torch.tril_indices(dim, dim, offset=-1)
        M = torch.zeros(ti.shape[0], dim); M[torch.arange(ti.shape[0]), ti] = 1.0
        self.register_buffer("rowmap", M)

    def forward(self, counts, trials):
        h = self.m.encoder(featurize(counts, trials))
        mean_train = self.m.head.mean(h)
        diag = Fn.softplus(self.m.head.diag(h)) + 1e-5
        lower = self.m.head.lower(h)
        var = diag ** 2 + (lower ** 2) @ self.rowmap
        std_train = var.clamp_min(1e-12).sqrt()
        rho = torch.tanh(mean_train[..., 8:12])
        mean_params = torch.cat([mean_train[..., :8], rho], dim=-1)
        std_params = torch.cat([std_train[..., :8], (1 - rho ** 2) * std_train[..., 8:12]], dim=-1)
        if self.has_cmp:
            p_corr = torch.softmax(self.m.corr_head(h), -1)
            p_sep = torch.cat([torch.softmax(self.m.sepA_head(h), -1)[:, 1:2],
                               torch.softmax(self.m.sepB_head(h), -1)[:, 1:2]], dim=-1)
        else:                                    # graceful fallback
            p_corr = torch.zeros(counts.shape[0], 3) + 1.0 / 3
            p_sep = torch.zeros(counts.shape[0], 2) + 0.5
        return mean_params, std_params, p_corr, p_sep


class RTOnnxWrapper(nn.Module):
    """RT model: counts + RT quantiles + trials -> params, sd, constructs, architecture, LBA."""
    def __init__(self, model, lba_mu, lba_sd, dim=12):
        super().__init__()
        self.m = model
        self.register_buffer("lba_mu", lba_mu)
        self.register_buffer("lba_sd", lba_sd)
        ti, _ = torch.tril_indices(dim, dim, offset=-1)
        M = torch.zeros(ti.shape[0], dim); M[torch.arange(ti.shape[0]), ti] = 1.0
        self.register_buffer("rowmap", M)

    def forward(self, counts, rtq, trials):
        c = counts.reshape(-1, 4, 4)
        t = trials.clamp(min=1)
        props = (c / t.unsqueeze(-1)).reshape(-1, 16)
        x = torch.cat([props, torch.log10(t), rtq], dim=-1)
        h = self.m.encoder(x)
        mean_train = self.m.head.mean(h)
        diag = Fn.softplus(self.m.head.diag(h)) + 1e-5
        lower = self.m.head.lower(h)
        std_train = (diag ** 2 + (lower ** 2) @ self.rowmap).clamp_min(1e-12).sqrt()
        rho = torch.tanh(mean_train[..., 8:12])
        mean_params = torch.cat([mean_train[..., :8], rho], dim=-1)
        std_params = torch.cat([std_train[..., :8], (1 - rho ** 2) * std_train[..., 8:12]], dim=-1)
        p_corr = torch.softmax(self.m.corr_head(h), -1)
        p_sep = torch.cat([torch.softmax(self.m.sepA_head(h), -1)[:, 1:2],
                           torch.softmax(self.m.sepB_head(h), -1)[:, 1:2]], dim=-1)
        p_arch = torch.softmax(self.m.arch_head(h), -1)
        lba = self.m.lba_head(h) * self.lba_sd + self.lba_mu
        return mean_params, std_params, p_corr, p_sep, p_arch, lba


# --------------------------------------------------------------------------- #
# Provenance stamping
# --------------------------------------------------------------------------- #
RELEASE_NOTE = (
    "Release checkpoint. The filename carries the version, so CloudFront's "
    "immutable cache can never serve these weights in place of a later release; "
    "artifact_sha256 is computed by scripts/export_onnx.py --install and verified "
    "in .github/workflows/deploy.yaml before the file is allowed to ship."
)


def install_to_web(rt, exported_path, version, status="release", note=None, prune=True,
                   checkpoint_path=None):
    """Copy an exported .onnx into web/assets/models/<id>/ under a versioned name
    and rewrite that model's manifest.json so it describes the file that is
    actually there.

    Returns the destination path. Raises rather than half-writing: the manifest is
    only rewritten once the copy is on disk and hashed.
    """
    model_id, stem, exported_by = EXPORT_SPECS[bool(rt)]
    dest_dir = os.path.join(WEB_MODELS_DIR, model_id)
    manifest_path = os.path.join(dest_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"no manifest to stamp at {manifest_path}")

    filename = f"{stem}.v{version}.onnx"
    dest = os.path.join(dest_dir, filename)
    shutil.copyfile(exported_path, dest)
    digest = sha256_file(dest)

    with open(manifest_path, encoding="utf-8") as fh:
        mf = json.load(fh)
    previous = mf.get("file")

    mf["file"] = filename
    mf["version"] = version
    mf["artifact_sha256"] = digest
    mf["exported_by"] = exported_by
    mf["exported_utc"] = utc_now()
    mf["source_commit"] = git_state()["commit"]

    training = mf.setdefault("training", {})
    training["provenance_status"] = status

    # Chain the TRAIN-time manifest through to the site. src/provenance.py embeds
    # one in every checkpoint (dataset hash, prior, architecture, optimiser) and
    # its docstring asks the exporters to carry it forward rather than silently
    # drop it -- without that, a shipped .onnx cannot be traced back to the
    # dataset that trained it, which is the exact ambiguity that module was
    # written to end.
    if checkpoint_path and os.path.isfile(checkpoint_path):
        training["checkpoint_file"] = os.path.relpath(checkpoint_path, PROJECT_ROOT).replace(os.sep, "/")
        training["checkpoint_sha256"] = sha256_file(checkpoint_path)
        try:
            import torch as _t
            ckpt = _t.load(checkpoint_path, map_location="cpu", weights_only=False)
            prov = ckpt.get("provenance") if isinstance(ckpt, dict) else None
            if prov:
                training["checkpoint_provenance"] = prov
                training.pop("checkpoint_provenance_missing", None)
            else:
                training["checkpoint_provenance_missing"] = (
                    "this checkpoint carries no provenance manifest -- it predates "
                    "src/provenance.py or was saved by a script that does not call "
                    "build_manifest(); its training dataset cannot be verified"
                )
                print("  ! checkpoint has no embedded provenance manifest")
        except Exception as exc:
            training["checkpoint_provenance_missing"] = f"unreadable ({type(exc).__name__}: {exc})"
            print(f"  ! could not read checkpoint provenance: {type(exc).__name__}: {exc}")
    elif checkpoint_path:
        print(f"  ! checkpoint not found at {checkpoint_path}; manifest will not carry train-time provenance")
    training["note"] = note or (RELEASE_NOTE if status == "release" else training.get("note", ""))
    # Promote the "intended" ranges to what the run actually trained on. Leaving a
    # field named `intended_release_*` on a shipped release is how a placeholder
    # becomes a permanent lie.
    try:
        from src.config import TRIAL_RANGE
        training["trained_trial_range"] = list(TRIAL_RANGE)
        training.pop("intended_release_trial_range", None)
    except Exception:
        pass

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(mf, fh, indent=2)
        fh.write("\n")

    # Drop superseded weights. Without this every release doubles the repo's model
    # payload and the S3 --delete sync keeps shipping dead files.
    removed = []
    if prune:
        for name in sorted(os.listdir(dest_dir)):
            if name.endswith(".onnx") and name != filename:
                os.remove(os.path.join(dest_dir, name))
                removed.append(name)

    print(f"  installed  {os.path.relpath(dest, PROJECT_ROOT)}")
    print(f"  sha256     {digest}")
    print(f"  manifest   file: {previous!r} -> {filename!r}   version: {version}   status: {status}")
    for name in removed:
        print(f"  pruned     {name}  (superseded)")
    return dest


def main(rt=False, version=None, install=False, status="release", note=None, prune=True,
         stamp_existing=False):
    if (install or stamp_existing) and not version:
        raise SystemExit("--install/--stamp-existing requires --version (e.g. --version 1.0.0)")

    # --stamp-existing: version and hash the .onnx that is ALREADY in web/, without
    # re-exporting. For weights that are staying exactly as they are but whose
    # manifest is wrong -- e.g. a superseded model that must keep working on the
    # site while being labelled honestly. No checkpoint and no GPU needed, so it
    # runs on a laptop. It deliberately cannot change the weights: the bytes are
    # copied, not regenerated, so "stamped" never silently means "retrained".
    if stamp_existing:
        model_id, _, _ = EXPORT_SPECS[bool(rt)]
        dest_dir = os.path.join(WEB_MODELS_DIR, model_id)
        with open(os.path.join(dest_dir, "manifest.json"), encoding="utf-8") as fh:
            current = json.load(fh)
        src = os.path.join(dest_dir, current["file"])
        if not os.path.isfile(src):
            raise SystemExit(f"manifest names {current['file']!r} but it is not on disk")
        print(f"stamping existing weights: {os.path.relpath(src, PROJECT_ROOT)}")
        install_to_web(rt, src, version, status=status, note=note, prune=prune)
        return

    if rt:
        from src.config import RT_MODEL_FILE
        from src.inference.predict_rt import load_rt_model
        m = load_rt_model(device="cpu")
        w = RTOnnxWrapper(m, m._lba_mu.cpu(), m._lba_sd.cpu()).eval()
        n_q = w.m.encoder[0].in_features - 20            # 80 RT quantile features
        args = (torch.zeros(1, 16), torch.zeros(1, n_q), torch.full((1, 4), 100.0))
        out = os.path.join(MODELS_DIR, "npe_rt_model.onnx")
        names = ["mean", "std", "p_corr", "p_sep", "p_arch", "lba"]
        torch.onnx.export(w, args, out,
                          input_names=["counts", "rtq", "trials"], output_names=names,
                          dynamic_axes={k: {0: "batch"} for k in
                                        ["counts", "rtq", "trials"] + names},
                          opset_version=17, dynamo=False)
        ckpt_path = RT_MODEL_FILE
        print(f"exported RT model -> {out}")
    else:
        wrapper = OnnxWrapper(load_model(MODEL_FILE, device="cpu")).eval()
        out = os.path.join(MODELS_DIR, "npe_model.onnx")
        torch.onnx.export(
            wrapper, (torch.zeros(1, 16), torch.full((1, 4), 100.0)), out,
            input_names=["counts", "trials"],
            output_names=["mean", "std", "p_corr", "p_sep"],
            dynamic_axes={k: {0: "batch"} for k in
                          ("counts", "trials", "mean", "std", "p_corr", "p_sep")},
            opset_version=17, dynamo=False)
        ckpt_path = MODEL_FILE
        print(f"exported -> {out}  (with comparison heads: {wrapper.has_cmp})")

    if install:
        install_to_web(rt, out, version, status=status, note=note, prune=prune,
                       checkpoint_path=ckpt_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt", action="store_true", help="export the RT model instead")
    ap.add_argument("--install", action="store_true",
                    help="also install into web/assets/models/ and stamp the manifest")
    ap.add_argument("--version", help="release version for the filename + manifest, e.g. 1.0.0")
    ap.add_argument("--status", default="release",
                    choices=["release", "preview", "unverified_legacy_checkpoint"],
                    help="manifest training.provenance_status (default: release)")
    ap.add_argument("--note", help="override the manifest training.note")
    ap.add_argument("--stamp-existing", action="store_true",
                    help="version+hash the .onnx already in web/ instead of re-exporting "
                         "(no checkpoint or GPU needed; cannot change the weights)")
    ap.add_argument("--no-prune", action="store_true",
                    help="keep superseded .onnx files in the site directory")
    a = ap.parse_args()
    main(rt=a.rt, version=a.version, install=a.install, status=a.status,
         note=a.note, prune=not a.no_prune, stamp_existing=a.stamp_existing)
