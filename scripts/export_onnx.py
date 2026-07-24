"""Export a trained GRIN model to ONNX for in-browser / package inference.
    python scripts/export_onnx.py
Produces results/models/npe_model.onnx.

Inputs : counts (B,16), trials (B,4)
Outputs: mean (B,12), std (B,12)           — posterior mean + marginal SD, parameter space
         p_corr (B,3)                      — P(PI), P(RHO1), P(free)   [comparison head]
         p_sep  (B,2)                      — P(separable A), P(separable B)

The construct probabilities now come from the TRAINED comparison heads rather than being
re-derived in JavaScript from the regression output.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.config import MODEL_FILE, MODELS_DIR
from src.api import load_model
from src.models.network import featurize


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


def main(rt=False):
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
        print(f"exported RT model -> {out}")
        return

    wrapper = OnnxWrapper(load_model(MODEL_FILE, device="cpu")).eval()
    out = os.path.join(MODELS_DIR, "npe_model.onnx")
    torch.onnx.export(
        wrapper, (torch.zeros(1, 16), torch.full((1, 4), 100.0)), out,
        input_names=["counts", "trials"],
        output_names=["mean", "std", "p_corr", "p_sep"],
        dynamic_axes={k: {0: "batch"} for k in
                      ("counts", "trials", "mean", "std", "p_corr", "p_sep")},
        opset_version=17, dynamo=False)
    print(f"exported -> {out}  (with comparison heads: {wrapper.has_cmp})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", action="store_true", help="export the RT model instead")
    main(rt=ap.parse_args().rt)
