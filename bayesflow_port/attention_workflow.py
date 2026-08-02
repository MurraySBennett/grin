"""
attention_workflow.py -- milestone 2, stage 2: recover per-participant
dimensional-attention scalars (k_A, k_B) CONDITIONED on a group-level identified
template (zx, zy, rho), completing the two-stage design documented in
`multiparticipant_workflow.py`'s module docstring:

  1. multiparticipant_workflow.py: pool a SET of participants' confusion
     matrices via a SetTransformer to recover the SHARED group template.
  2. attention_workflow.py (here): condition the EXISTING single-matrix-style
     machinery on that group template, alongside the participant's own
     confusion matrix, to recover their individual attention scalars.

Design decisions locked in with the user before building this (see chat log,
not inferred from the code):

  * Training uses the TRUE simulated group template as the condition, not a
    stage-1 posterior draw. This is the standard, cheap way to train a
    plug-in two-stage amortizer: `simulate_sessions(n_participants=1, ...)`
    already draws an exact (zx, zy, rho) template per row alongside that one
    participant's attention scalars and matrix -- reused here VERBATIM
    (squeezed to drop the length-1 participant axis), per the "reuse the
    simulator" rule. No new generative-model code was written for this file.
  * At APPLICATION time on real (or in-silico held-out) data, the group
    template is NOT known exactly -- it is itself a posterior from stage 1.
    Uncertainty is propagated by MONTE CARLO MIXTURE: run this stage-2
    workflow once per stage-1 posterior draw (e.g. 50) and pool the resulting
    (log_k_A, log_k_B) samples across draws. See `sample_attention_propagated`
    below. This is NOT full joint hierarchical inference -- individual-level
    fit cannot feed back and correct the group-level estimate (a "cut" in the
    Bayesian-workflow sense). That is a known, accepted limitation of the
    plug-in design, not an oversight; note it in any report built on this.
  * Targets are log(k_A), log(k_B), not k_A/k_B directly -- this matches the
    generative draw (`k = exp(Normal(0, attention_sd))`) exactly, so no
    `.constrain()` is needed on the inference side, and diagnostics/calibration
    are on the same scale the prior is defined in.
  * No summary network: the condition is `x_participant` (20-d featurised
    matrix) concatenated with `group_theta` (12-d, the group template) --
    still a single fixed-length vector with meaningful element order, the
    "simple vector" case in the amortized-workflow skill's conditioning table.
  * Network capacity reuses bf_workflow.build_inference_network's named tiers
    directly (same Base/Large FlowMatching/CouplingFlow configs as the
    single-matrix port) rather than redefining new tiers -- one source of
    truth for capacity, per the skill's "start with Base" rule.

Real-data goal (not built here yet, deliberately out of scope for this pass):
the user's real data is trial-by-trial with response times, reducible to the
same confusion-matrix format used throughout this port (for backwards
comparison with historical GRT analyses) OR usable at the raw trial level.
Raw-trial-level conditioning would need a set/time-series summary network
over trials directly (a different architecture from the fixed-length
featurised vector used here and throughout the rest of this port) -- flagged
as a later milestone, not attempted in this pass. This pass is
simulation-only, per the user's explicit request to validate recovery first.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import sys
sys.path.insert(0, os.path.dirname(__file__))                              # bayesflow_port/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))          # repo root

import numpy as np
from multiparticipant_workflow import simulate_sessions
# keras/bayesflow/bf_workflow are imported lazily inside build_adapter()/
# build_workflow() so simulate_participants() below stays smoke-testable
# without BayesFlow installed, matching multiparticipant_workflow.py's pattern.


# --------------------------------------------------------------------------- #
def simulate_participants(n, rng=None, attention_sd=0.25, **kw):
    """One participant per row, group template TRUE (plug-in) rather than
    recovered. Reuses `simulate_sessions(n_participants=1, ...)` VERBATIM and
    squeezes the length-1 participant axis -- do NOT reimplement the
    attention-scaling forward model here; that logic lives in exactly one
    place (multiparticipant_workflow.simulate_sessions)."""
    rng = np.random.default_rng() if rng is None else rng
    sessions = simulate_sessions(n, n_participants=1, rng=rng, attention_sd=attention_sd, **kw)
    return {
        "x":         sessions["x_participants"][:, 0, :],   # (n, 20) this participant's matrix
        "group_z":   sessions["group_z"],                   # (n, 8)  TRUE group template (plug-in)
        "group_rho": sessions["group_rho"],                 # (n, 4)
        "log_k_A":   np.log(sessions["k_A"]),                # (n, 1) target, log-space
        "log_k_B":   np.log(sessions["k_B"]),                # (n, 1) target, log-space
        "cls":       sessions["cls"],
    }


class GrinParticipantSimulator:
    """Fixed protocol matching GrinSimulator/GrinSessionSimulator: sample(batch_shape)."""

    def __init__(self, seed=None, attention_sd=0.25, **kw):
        self.rng = np.random.default_rng(seed)
        self.attention_sd = attention_sd
        self.kw = kw

    def sample(self, batch_shape):
        n = int(np.prod(batch_shape))
        return simulate_participants(n, self.rng, attention_sd=self.attention_sd, **self.kw)


# --------------------------------------------------------------------------- #
def build_adapter():
    """(x, group_z, group_rho) -> inference_conditions (32-d fixed vector);
    (log_k_A, log_k_B) -> inference_variables (2-d, unconstrained log-space).
    group_z/group_rho are a CONDITION here (the plug-in-true or, at
    application time, stage-1-recovered group template), not something being
    flowed, so they need no `.constrain()` -- `standardize="all"` in
    build_workflow handles their scaling like any other condition."""
    import bayesflow as bf
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["log_k_A", "log_k_B"], into="inference_variables")
        .concatenate(["x", "group_z", "group_rho"], into="inference_conditions")
        .keep(["inference_variables", "inference_conditions"])
    )


def build_workflow(results_dir, kind="flow_matching", size="base",
                    learning_rate=1e-3, simulator=None, **net_kwargs):
    """Reuses bf_workflow.build_inference_network's named Base/Large tiers --
    same capacity choices as the single-matrix port, one source of truth."""
    import keras
    import bayesflow as bf
    from bf_workflow import build_inference_network

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


# --------------------------------------------------------------------------- #
def sample_attention_propagated(workflow, x_obs, group_theta_draws, num_samples_per_draw=40):
    """Apply stage 2 to REAL (or held-out) participant data while propagating
    stage 1's group-template uncertainty, per the "full posterior propagation"
    design confirmed with the user: run stage 2 once per stage-1 posterior
    draw and pool the resulting samples, rather than plugging in a single
    point estimate (which would silently understate stage-2 uncertainty) or
    retraining with injected noise (more complex, not needed here since
    inference is cheap -- see the single-matrix port's ~ms-scale latency).

    This is a custom composition of two workflows, NOT workflow.sample() used
    directly -- there is no single BasicWorkflow call that does this.

    Parameters
    ----------
    workflow : the stage-2 BasicWorkflow (built by build_workflow above).
    x_obs : (B, 20) featurised confusion-matrix rows, one per participant.
    group_theta_draws : (M, 12) M posterior draws of the GROUP template
        [zx0..3, zy0..3, rho0..3], e.g. from concatenating
        `samples["z"][0, :M]` and `samples["rho"][0, :M]` after calling
        `multiparticipant_workflow.build_workflow(...).sample(...)` on that
        session's pooled participant data.
    num_samples_per_draw : posterior draws of (log_k_A, log_k_B) per group draw.

    Returns
    -------
    log_k_A, log_k_B : (B, M * num_samples_per_draw) pooled posterior samples.
    """
    x_obs = np.asarray(x_obs, dtype=np.float32)
    group_theta_draws = np.asarray(group_theta_draws, dtype=np.float32)
    B = x_obs.shape[0]
    M = group_theta_draws.shape[0]

    all_log_kA, all_log_kB = [], []
    for m in range(M):
        group_z = np.broadcast_to(group_theta_draws[m, :8], (B, 8)).astype(np.float32)
        group_rho = np.broadcast_to(group_theta_draws[m, 8:], (B, 4)).astype(np.float32)
        samples = workflow.sample(
            conditions={"x": x_obs, "group_z": group_z, "group_rho": group_rho},
            num_samples=num_samples_per_draw,
        )
        all_log_kA.append(np.asarray(samples["log_k_A"]))  # (B, S, 1)
        all_log_kB.append(np.asarray(samples["log_k_B"]))

    log_k_A = np.concatenate(all_log_kA, axis=1)[..., 0]  # (B, M*S)
    log_k_B = np.concatenate(all_log_kB, axis=1)[..., 0]
    return log_k_A, log_k_B


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Shape smoke-test only -- no training, no BayesFlow import required.
    rng = np.random.default_rng(0)
    out = simulate_participants(2000, rng)
    for k, v in out.items():
        print(f"{k:10s} {v.shape} {v.dtype}")
    assert out["x"].shape == (2000, 20)
    assert out["group_z"].shape == (2000, 8)
    assert out["group_rho"].shape == (2000, 4)
    assert out["log_k_A"].shape == (2000, 1)
    assert out["log_k_B"].shape == (2000, 1)
    print("\nShapes OK. Next: build_workflow(...) + workflow.fit_offline(...) "
          "following the pattern in train_attention.py.")
