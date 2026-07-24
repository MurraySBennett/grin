"""
engine.py — online adaptive inference + stimulus selection.

An AdaptiveSession accumulates (stimulus, response) trials into a confusion matrix
and, at any point, returns the current posterior via the amortized NPE (fast enough
to run inside the trial loop). Stimulus-selection policies decide what to present
next; `info_gain` uses the network's own predicted covariance to pick the stimulus
that is expected to shrink posterior uncertainty the most.

Uncertainty for selection is read directly from the Gaussian head's covariance
(diag of L L^T in train space) — no sampling — so evaluating all 16 hypothetical
next-outcomes is a single batched forward pass per trial.
"""
import numpy as np
import torch

from ..models.network import featurize
from ..inference.predict import predict_point

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm


def simulate_response(theta, stim, rng):
    """Sample one response for `stim` from a true parameter vector theta (12,)."""
    zx, zy, rho = gm.unpack(np.asarray(theta)[None])
    p = gm.forward_probabilities(zx, zy, rho)[0][stim]
    return int(rng.choice(4, p=p / p.sum()))


class AdaptiveSession:
    def __init__(self, model, init_counts=None):
        self.model = model
        self.counts = (np.zeros((4, 4), dtype=np.int64)
                       if init_counts is None else np.asarray(init_counts, np.int64).copy())

    def uncertainty(self):
        return float(_train_space_std(self.model, self.counts.reshape(1, 16),
                                      self.trials.reshape(1, 4))[0])

    def add_trial(self, stim, resp):
        self.counts[stim, resp] += 1

    @property
    def trials(self):
        return self.counts.sum(1)

    def estimate(self):
        return predict_point(self.model, self.counts.reshape(1, 16),
                             self.trials.reshape(1, 4))[0].numpy()


@torch.no_grad()
def _train_space_std(model, counts_batch, trials_batch):
    """Mean marginal posterior std (train space) per matrix — deterministic, batched."""
    device = next(model.parameters()).device
    x = featurize(torch.as_tensor(counts_batch), torch.as_tensor(trials_batch)).to(device)
    _, L = model(x)
    var = (L ** 2).sum(-1)                     # diag of L L^T -> (B, 12)
    return var.sqrt().mean(-1).cpu().numpy()


def info_gain_select(model, session):
    """Pick the stimulus with the largest expected reduction in posterior std."""
    C, T = session.counts, session.trials
    base = _train_space_std(model, C.reshape(1, 16), T.reshape(1, 4))[0]
    zx, zy, rho = gm.unpack(session.estimate()[None])
    pred = gm.forward_probabilities(zx, zy, rho)[0]           # p(response | stimulus)

    hypo_C, sr = [], []
    for s in range(4):
        for r in range(4):
            Cp = C.copy(); Cp[s, r] += 1
            hypo_C.append(Cp.reshape(16)); sr.append((s, r))
    hypo_C = np.array(hypo_C)
    hypo_T = hypo_C.reshape(-1, 4, 4).sum(2)
    stds = _train_space_std(model, hypo_C, hypo_T)            # (16,)

    exp_std = np.zeros(4)
    for k, (s, r) in enumerate(sr):
        exp_std[s] += pred[s, r] * stds[k]
    return int(np.argmax(base - exp_std))                    # max expected reduction


def select(policy, model, session, t, rng):
    if policy == "round_robin":
        return t % 4
    if policy == "random":
        return int(rng.integers(4))
    if policy == "info_gain":
        return info_gain_select(model, session)
    raise ValueError(policy)


def run_session(model, theta_true, policy, n_trials, rng, init_counts=None):
    """Run an adaptive session; return (session, trajectory of (total_trials, mean_abs_error))."""
    sess = AdaptiveSession(model, init_counts=init_counts)
    if init_counts is None:
        for s in range(4):                                  # warm-up: one trial each
            sess.add_trial(s, simulate_response(theta_true, s, rng))
    traj = []
    for t in range(n_trials):
        s = select(policy, model, sess, t, rng)
        sess.add_trial(s, simulate_response(theta_true, s, rng))
        if (t + 1) % 10 == 0:
            err = np.abs(sess.estimate() - theta_true).mean()
            traj.append((int(sess.trials.sum()), err))
    return sess, np.array(traj)
