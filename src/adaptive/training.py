"""
training.py — targeted perceptual-training loop (expert-target engine).

Closed loop: a simulated learner has a current perceptual representation theta;
we (1) ASSESS it in real time via the NPE, (2) DECIDE what to train, (3) apply
TRAINING that moves the representation toward an expert target, and repeat until
mastery.

*** MODELLING CAVEAT ***  Unlike the rest of GRIN, this loop rests on an ASSUMED
perceptual-learning dynamic (`apply_training`): training a dimension moves its
sensitivity toward the target on an exponential approach, fastest when the training
difficulty sits near the learner's CURRENT sensitivity (the perceptual-learning
"sweet spot"). This demonstrates the *value of real-time representation tracking*
for training; real-world gains depend on real learning curves. It is a
proof-of-concept for the mechanism, not a validated training protocol.

The mechanism it showcases: the optimal training difficulty is a MOVING target
(as the learner improves, harder training becomes productive). Only a controller
that tracks the learner's representation online can follow it; a fixed curriculum
cannot.
"""
import numpy as np

from ..inference.predict import predict_point
from .engine import simulate_response

try:
    from src import grt_model as gm
except ImportError:
    import grt_model as gm


def ideal_expert(sensitivity=2.5):
    """Idealised expert: well-separated (high sensitivity), perceptually independent."""
    zx = np.array([-1., -1., 1., 1.]) * sensitivity
    zy = np.array([-1., 1., -1., 1.]) * sensitivity
    rho = np.zeros(4)
    return gm.pack(zx[None], zy[None], rho[None])[0]


def sensitivity(theta, dim):
    zx, zy, _ = gm.unpack(np.asarray(theta)[None])
    return float(np.abs(zx[0]).mean() if dim == "A" else np.abs(zy[0]).mean())


def apply_training(theta, dim, difficulty, n_train, target, lr=0.03, tau=0.8, rho_lr=0.015):
    """Update the learner's representation after n_train training trials on `dim`."""
    zx, zy, rho = (a[0] for a in gm.unpack(np.asarray(theta)[None]))
    tzx, tzy, trho = (a[0] for a in gm.unpack(np.asarray(target)[None]))
    s = np.abs(zx).mean() if dim == "A" else np.abs(zy).mean()
    gate = np.exp(-((difficulty - s) ** 2) / (2 * tau ** 2))     # sweet-spot gate
    frac = 1 - np.exp(-lr * gate * n_train)                       # exponential approach
    if dim == "A":
        zx = zx + frac * (tzx - zx)
    else:
        zy = zy + frac * (tzy - zy)
    rho = rho + (1 - np.exp(-rho_lr * n_train)) * (trho - rho)    # independence emerges
    return gm.pack(zx[None], zy[None], rho[None])[0]


def assess(model, theta_true, n_assess, rng):
    """Short assessment block -> NPE estimate of the learner's current representation."""
    counts = np.zeros((4, 4), dtype=int)
    for t in range(n_assess):
        s = t % 4
        counts[s, simulate_response(theta_true, s, rng)] += 1
    return predict_point(model, counts.reshape(1, 16), counts.sum(1).reshape(1, 4))[0].numpy()


def run_training(model, theta0, target, policy, rng, n_assess=40, n_train=80,
                 eps=0.3, max_rounds=200, fixed_difficulty=1.0, lr=0.0018, tau=0.7):
    """
    Run a training session. policy 'adaptive' tracks the learner via the NPE and
    trains the furthest-from-target dimension at the moving sweet spot; 'uniform'
    alternates dimensions at a fixed difficulty with no feedback.
    Returns (trajectory of (trials, sA, sB), total_trials, mastered_bool).
    """
    theta = np.asarray(theta0, float).copy()
    t_sens = sensitivity(target, "A")
    trials = 0
    traj = []
    for rnd in range(max_rounds):
        est = assess(model, theta, n_assess, rng); trials += n_assess
        sA, sB = sensitivity(theta, "A"), sensitivity(theta, "B")
        traj.append((trials, sA, sB))
        if sA >= t_sens - eps and sB >= t_sens - eps:
            return np.array(traj), trials, True
        if policy == "adaptive":
            sA_e, sB_e = sensitivity(est, "A"), sensitivity(est, "B")
            dim = "A" if (t_sens - sA_e) >= (t_sens - sB_e) else "B"
            difficulty = sensitivity(est, dim)                   # sweet spot = current estimate
        else:                                                    # uniform curriculum
            dim = "A" if rnd % 2 == 0 else "B"
            difficulty = fixed_difficulty
        theta = apply_training(theta, dim, difficulty, n_train, target, lr=lr, tau=tau); trials += n_train
    return np.array(traj), trials, False
