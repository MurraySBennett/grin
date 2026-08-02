"""
multiparticipant_workflow.py — milestone 2: a permutation-invariant
SetTransformer summary network pooling across a SET of participants'
confusion matrices to recover a shared group-level perceptual structure, the
amortized analogue of GRT-wIND (Design Record §10: "the route to decisional
separability: constrain group-level perceptual structure, let individuals vary
in dimensional attention"). This is the capability bf_workflow.py's
single-matrix port cannot provide, and the reason this BayesFlow branch exists
at all (see README.md).

This is a SKELETON, not a finished model — the exact hierarchical
parameterization (what varies by participant, what's shared) is a modeling
decision to confirm, not something to infer from the code. The concrete choice
made here, to make the skeleton runnable and directly testable:

  - Group-level (SHARED, what we infer): the canonical 12-parameter identified
    vector (zx, zy, rho) — one PS/PI/RHO1/etc. class template per session,
    exactly `src/grt_model.py`'s existing prior. This is the group template
    GRT-wIND fixes structurally.
  - Participant-level (NUISANCE, sampled but NOT inferred by this skeleton):
    per-participant dimensional-attention scalars k_A, k_B (log-normal around
    1) that scale the group's zx/zy before the forward map — same idea as the
    existing `k_A`/`k_B` drift scaling in `src/data/rt_lba_generator.py`, reused
    here rather than invented fresh.
  - Each of the N participants in a "session" contributes one 20-d featurised
    confusion-matrix vector (the SAME featurization as the single-matrix port).
    A session's N participant vectors are the exchangeable SET —
    `summary_variables` + `SetTransformer`, per the amortized-workflow skill's
    decision table (N i.i.d. observations -> summary_variables, never flattened
    into inference_conditions).

Why the attention scalars are NOT inferred here: recovering per-participant
nuisance parameters jointly with a shared group template from a *set* target is
a hierarchical amortization problem beyond what `BasicWorkflow`'s default
(fixed-shape `inference_variables`) does out of the box. The practical
composition that stays inside the existing toolkit:

  1. Train THIS workflow to recover the group template from the pooled set.
  2. Re-run the EXISTING single-matrix workflow (`bf_workflow.py`) per
     participant, but now condition it on the recovered group template
     (concatenate the group posterior mean into `inference_conditions`
     alongside `x`) to recover each participant's attention scalars with the
     group ambiguity already resolved.

That two-stage design reuses both pieces already built rather than requiring
new BayesFlow machinery, and is the concrete next step to validate before
committing to it as the final milestone-2 architecture.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
sys.path.insert(0, os.path.dirname(__file__))                              # bayesflow_port/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))          # repo root

import numpy as np
# keras/bayesflow are imported lazily inside build_adapter()/build_workflow() so
# that the simulator (simulate_sessions) and its shape smoke-test below can run
# and be checked independently of the BayesFlow-specific code.

import src.grt_model as gm
from bf_simulator import (_sample_trial_counts, _multinomial_counts, featurize_np,
                           TRIAL_RANGE, Z_MAX, R_MAX)


# --------------------------------------------------------------------------- #
def simulate_sessions(n_sessions, n_participants=8, rng=None, z_max=Z_MAX, r_max=R_MAX,
                       trial_range=TRIAL_RANGE, attention_sd=0.25):
    """Draw `n_sessions` groups, each with `n_participants` exchangeable
    participants sharing one group-level identified template but varying in
    dimensional attention (k_A, k_B). Fixed N per call — see module docstring
    for the meta_fn extension needed for variable-N sessions."""
    rng = np.random.default_rng() if rng is None else rng
    n_cls = len(gm.MODEL_NAMES)
    cls = rng.integers(0, n_cls, size=n_sessions)

    group_theta = np.empty((n_sessions, 12), dtype=np.float64)
    x_participants = np.empty((n_sessions, n_participants, 20), dtype=np.float32)
    k_A = np.empty((n_sessions, n_participants), dtype=np.float64)
    k_B = np.empty((n_sessions, n_participants), dtype=np.float64)

    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        name = gm.MODEL_NAMES[c]
        m = len(idx)
        zx, zy, rho = gm.sample_prior(name, m, rng, z_max=z_max, r_max=r_max)  # each (m,4)
        group_theta[idx] = gm.pack(zx, zy, rho)

        kA = np.exp(rng.normal(0, attention_sd, size=(m, n_participants)))
        kB = np.exp(rng.normal(0, attention_sd, size=(m, n_participants)))
        k_A[idx], k_B[idx] = kA, kB

        # broadcast the group template across participants, scale by attention
        zx_p = zx[:, None, :] * kA[:, :, None]                      # (m, P, 4)
        zy_p = zy[:, None, :] * kB[:, :, None]                      # (m, P, 4)
        rho_p = np.broadcast_to(rho[:, None, :], (m, n_participants, 4))

        probs = gm.forward_probabilities(zx_p.reshape(-1, 4), zy_p.reshape(-1, 4),
                                          rho_p.reshape(-1, 4))       # (m*P, 4, 4)
        trials = _sample_trial_counts(m * n_participants, rng, trial_range)
        counts = _multinomial_counts(probs, trials, rng)
        x_flat = featurize_np(counts.reshape(m * n_participants, 16), trials)  # (m*P, 20)
        x_participants[idx] = x_flat.reshape(m, n_participants, 20).astype(np.float32)

    return {
        "group_z":   group_theta[:, :8].astype(np.float32),
        "group_rho": group_theta[:, 8:].astype(np.float32),
        "x_participants": x_participants,           # (n_sessions, P, 20) — the set
        "k_A": k_A.astype(np.float32),               # nuisance, carried for diagnostics only
        "k_B": k_B.astype(np.float32),
        "cls": cls.astype(np.int32),
    }


class GrinSessionSimulator:
    """Fixed-N session simulator, matching bf_simulator.GrinSimulator's protocol."""

    def __init__(self, n_participants=8, seed=None, **kw):
        self.n_participants = n_participants
        self.rng = np.random.default_rng(seed)
        self.kw = kw

    def sample(self, batch_shape):
        n = int(np.prod(batch_shape))
        return simulate_sessions(n, self.n_participants, self.rng, **self.kw)


# --------------------------------------------------------------------------- #
def build_adapter():
    """(group_z, group_rho, x_participants) -> (inference_variables, summary_variables).

    x_participants is already (batch, P, 20) — 3D with a meaningful trailing
    feature axis — so `.as_set` is not needed (that call is only for 1D
    sequences missing a trailing axis; here it already has one). group_rho
    still needs the same (-1,1) unconstraining as the single-matrix port.
    """
    import bayesflow as bf
    return (
        bf.Adapter()
        .constrain("group_rho", lower=-1.0, upper=1.0)
        .convert_dtype("float64", "float32")
        .concatenate(["group_z", "group_rho"], into="inference_variables")
        .rename("x_participants", "summary_variables")
        .keep(["inference_variables", "summary_variables"])
    )


# SetTransformer Base tier (references/model-sizes.md), summary_dim set via the
# skill's heuristic: 3x the number of inferred parameters (12 group params -> 36).
# mlp_depths/mlp_widths aren't specified per-tier in model-sizes.md, but
# SetTransformer requires them to match embed_dims/num_heads in length (its
# default is length-2, which silently mismatches a length-3 Base embed_dims
# tuple and raises ValueError at construction) — extended to length 3 at the
# class default per-element values (depth=2, width=128).
_SET_TRANSFORMER_BASE = dict(embed_dims=(64, 64, 64), num_heads=(4, 4, 4),
                              mlp_depths=(2, 2, 2), mlp_widths=(128, 128, 128),
                              num_seeds=8, summary_dim=36)


def build_workflow(results_dir, learning_rate=1e-3,
                    simulator=None, inference_kind="flow_matching"):
    import keras
    import bayesflow as bf
    os.makedirs(results_dir, exist_ok=True)
    summary_network = bf.networks.SetTransformer(**_SET_TRANSFORMER_BASE)
    if inference_kind == "flow_matching":
        # FlowMatching's default subnet ("time_mlp") must not be overridden to
        # "mlp" here — see the note in bf_workflow.py's _FLOWMATCHING_BASE.
        inference_network = bf.networks.FlowMatching(
            subnet_kwargs=dict(widths=(256, 256, 256, 256), time_embedding_dim=32))
    else:
        inference_network = bf.networks.CouplingFlow(
            depth=4, transform="spline", subnet="mlp",
            subnet_kwargs=dict(widths=(128, 128), activation="gelu"))

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        inference_network=inference_network,
        summary_network=summary_network,
        adapter=build_adapter(),
        standardize="all",
        checkpoint_filepath=os.path.join(results_dir, "checkpoints"),
    )
    workflow.approximator.compile(optimizer=keras.optimizers.Adam(learning_rate))
    return workflow


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Shape smoke-test only — no training. Confirms the simulator + adapter
    # produce tensors of the shapes BasicWorkflow expects before spending any
    # GPU time. Run with: python multiparticipant_workflow.py
    rng = np.random.default_rng(0)
    out = simulate_sessions(n_sessions=64, n_participants=8, rng=rng)
    for k, v in out.items():
        print(f"{k:16s} {v.shape} {v.dtype}")
    assert out["x_participants"].shape == (64, 8, 20)
    assert out["group_z"].shape == (64, 8)
    assert out["group_rho"].shape == (64, 4)
    print("\nShapes OK. Next: build_workflow(...) + workflow.fit_offline(...) "
          "following the pattern in train_multiparticipant.py.")
