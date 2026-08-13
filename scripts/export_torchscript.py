"""Export a trained GRIN model to TorchScript for native R inference (via the R
`torch` package / libtorch -- no Python or reticulate required on the R side).
    python scripts/export_torchscript.py
Produces results/models/npe_model_ts.pt.

Same wrapper as scripts/export_onnx.py (kept import-compatible so the two exports
can never silently drift), traced instead of onnx-exported. Verifies the traced
module reproduces the eager wrapper's output exactly before writing anything, so a
divergent trace fails loudly here rather than silently in R.
"""
import os
import torch

from src.config import MODEL_FILE, MODELS_DIR
from src.api import load_model
from scripts.export_onnx import OnnxWrapper, RTOnnxWrapper


def _verify(wrapper, traced, args, atol=1e-6):
    with torch.no_grad():
        eager_out = wrapper(*args)
        traced_out = traced(*args)
    for e, t in zip(eager_out, traced_out):
        assert torch.allclose(e, t, atol=atol), "traced module diverges from eager wrapper"


def main(rt=False):
    if rt:
        from src.inference.predict_rt import load_rt_model
        m = load_rt_model(device="cpu")
        wrapper = RTOnnxWrapper(m, m._lba_mu.cpu(), m._lba_sd.cpu()).eval()
        n_q = wrapper.m.encoder[0].in_features - 20        # 80 RT quantile features
        args = (torch.zeros(1, 16), torch.zeros(1, n_q), torch.full((1, 4), 100.0))
        out = os.path.join(MODELS_DIR, "npe_rt_model_ts.pt")
    else:
        wrapper = OnnxWrapper(load_model(MODEL_FILE, device="cpu")).eval()
        args = (torch.zeros(1, 16), torch.full((1, 4), 100.0))
        out = os.path.join(MODELS_DIR, "npe_model_ts.pt")

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, args)
    _verify(wrapper, traced, args)

    # re-check on random (non-degenerate) inputs, not just the zero/constant trace
    # example -- tracing can silently bake in shapes/values seen only at trace time.
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        counts = torch.randint(0, 50, (4, 16), generator=g).float()
        trials = counts.reshape(4, 4, 4).sum(-1)
        if rt:
            rtq = torch.randn(4, args[1].shape[1], generator=g)
            _verify(wrapper, traced, (counts, rtq, trials))
        else:
            _verify(wrapper, traced, (counts, trials))

    traced.save(out)
    print(f"exported -> {out}  (traced + verified against eager wrapper)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", action="store_true", help="export the RT model instead")
    main(rt=ap.parse_args().rt)
