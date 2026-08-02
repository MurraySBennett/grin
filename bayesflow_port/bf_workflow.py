"""
bf_workflow.py — the BayesFlow inference machinery for the GRIN single-matrix port.

This is the part that DIFFERS from the bespoke stack. Everything upstream
(the generative model in bf_simulator.py) is shared; here we swap the
hand-rolled Gaussian/Cholesky NPE head + custom training loop + custom SBC for
BayesFlow's adapter -> inference network (FlowMatching or CouplingFlow) ->
BasicWorkflow.

Design decisions (see README.md for the full log):
  * Uses `bf.BasicWorkflow`, not a hand-assembled `ContinuousApproximator` +
    `OfflineDataset`. This is what gives `workflow.simulate`,
    `workflow.fit_offline(..., validation_data=...)`, and the built-in
    `compute_default_diagnostics` / `plot_default_diagnostics` for free.
  * rho in (-1,1) is mapped to unconstrained space via the adapter's
    `constrain` (a logit-style transform). The bespoke stack uses atanh
    (Fisher-z). These are different but equivalent unconstraining maps; the
    network learns in whichever coordinate it is given. We use BayesFlow's
    idiomatic `constrain` here and note the difference rather than forcing
    atanh, so the comparison reflects each framework's natural choice.
  * z-scores pass through real-valued; `standardize="all"` handles scaling for
    both z and x inside the approximator.
  * Dtype conversion (float64 -> float32) happens in the adapter via
    `.convert_dtype`, not by hand in bf_simulator.simulate — the adapter owns
    every train/inference-time transform so the two code paths cannot
    silently diverge (references/adapter.md convention).
  * No summary network for the base (single-matrix) port: the condition is
    already a fixed 20-d featurised vector with meaningful element order —
    exactly the "simple vector" case in the amortized-workflow skill's
    conditioning decision table — so it goes straight in as
    inference_conditions. The summary/set network enters for the
    multi-participant extension, see multiparticipant_workflow.py.
  * Network capacity is pinned to a named tier from references/model-sizes.md
    (Base by default). FlowMatching/Base is the primary recommendation — the
    skill prefers free-form/diffusion-like inference nets over coupling flows
    unless there's a reason not to. CouplingFlow (spline) stays available via
    kind="coupling": it's the transform validated against a real affine-flow
    NaN bug (see README.md) and has cheaper single-step-ish sampling latency,
    relevant to the adaptive-loop use case.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import bayesflow as bf


# --------------------------------------------------------------------------- #
def build_adapter():
    """Shared adapter: (z, rho, x) -> (inference_variables, inference_conditions).
    rho is unconstrained from (-1,1); z passes through; x is the 20-d condition.
    Order: structural -> constrain -> dtype -> concatenate/rename, per
    references/adapter.md."""
    return (
        bf.Adapter()
        .constrain("rho", lower=-1.0, upper=1.0)
        .convert_dtype("float64", "float32")
        .concatenate(["z", "rho"], into="inference_variables")
        .rename("x", "inference_conditions")
        .keep(["inference_variables", "inference_conditions"])
    )


# --------------------------------------------------------------------------- #
# Named tiers, copied verbatim from references/model-sizes.md so a future
# capacity change is a one-line tier swap, not a guess.
_COUPLING_BASE = dict(depth=4, transform="spline", subnet="mlp",
                       subnet_kwargs=dict(widths=(128, 128), activation="gelu"))
_COUPLING_LARGE = dict(depth=6, transform="spline", subnet="mlp",
                        subnet_kwargs=dict(widths=(256, 256), activation="gelu"))
# FlowMatching's default subnet ("time_mlp") must NOT be overridden to "mlp" —
# that swaps in a subnet that doesn't accept FlowMatching's (xz, time,
# conditions) build signature and breaks with a cryptic shape error deep in
# Keras. Only subnet_kwargs is set; subnet is left at its default.
_FLOWMATCHING_BASE = dict(subnet_kwargs=dict(widths=(256, 256, 256, 256),
                                              time_embedding_dim=32))
_FLOWMATCHING_LARGE = dict(subnet_kwargs=dict(widths=(512, 512, 512, 512, 512),
                                               time_embedding_dim=64))

_TIERS = {
    ("coupling", "base"): _COUPLING_BASE,
    ("coupling", "large"): _COUPLING_LARGE,
    ("flow_matching", "base"): _FLOWMATCHING_BASE,
    ("flow_matching", "large"): _FLOWMATCHING_LARGE,
}


def build_inference_network(kind="flow_matching", size="base", **overrides):
    """kind in {'flow_matching', 'coupling'}, size in {'base', 'large'}.

    Defaults to FlowMatching/Base. Pass kind='coupling' to use the spline
    coupling flow instead (the choice already validated against the
    affine-transform NaN bug, see README.md). Only move to size='large' after
    diagnostics show poor recovery/calibration at Base — never start there.
    """
    key = (kind, size)
    if key not in _TIERS:
        raise ValueError(f"unknown (kind, size): {key}")
    cfg = dict(_TIERS[key])
    cfg.update(overrides)
    if kind == "coupling":
        return bf.networks.CouplingFlow(**cfg)
    return bf.networks.FlowMatching(**cfg)


# --------------------------------------------------------------------------- #
def build_workflow(results_dir, kind="flow_matching", size="base",
                    learning_rate=1e-3, simulator=None, **net_kwargs):
    """Build a BasicWorkflow: explicit adapter (rho needs `.constrain`, so the
    naming shorthand won't do), standardize="all" (z and x are both already
    reasonably scaled real-valued quantities, so blanket standardization is
    safe), checkpoint_filepath=results_dir so the run is self-contained.

    Pass `simulator=` (a bf_simulator.GrinSimulator instance) if you want
    `workflow.simulate(...)` / `fit_online` available; omit it for a purely
    offline pipeline driven by pre-generated dicts.
    """
    os.makedirs(results_dir, exist_ok=True)
    inference_network = build_inference_network(kind, size, **net_kwargs)
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        inference_network=inference_network,
        adapter=build_adapter(),
        standardize="all",
        checkpoint_filepath=os.path.join(results_dir, "checkpoints"),
    )
    workflow.approximator.compile(optimizer=keras.optimizers.Adam(learning_rate))
    return workflow


def sample_posterior(workflow, x, num_samples=2000):
    """Draw posterior samples for a batch of featurised matrices x (B,20).
    Returns theta_samples (B, num_samples, 12) in ORIGINAL param space
    [zx0..3, zy0..3, rho0..3] — the adapter's inverse transform already
    mapped rho back to (-1,1). workflow.sample() returns original parameter
    names ("z", "rho"), never "inference_variables".
    """
    import numpy as np
    samples = workflow.sample(conditions={"x": np.asarray(x, dtype=np.float32)},
                               num_samples=num_samples)
    z = np.asarray(samples["z"])       # (B, num_samples, 8)
    rho = np.asarray(samples["rho"])   # (B, num_samples, 4)
    return np.concatenate([z, rho], axis=-1)
