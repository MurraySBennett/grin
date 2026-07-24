"""losses.py — NPE negative log-likelihood (exact, via the Gaussian head)."""
def npe_nll(model, x, target_train_space):
    """Mean negative log-likelihood of the targets under the predicted posterior."""
    return -model.distribution(x).log_prob(target_train_space).mean()


def joint_loss(model, x, target_train_space, corr_lbl, sepA_lbl, sepB_lbl, w_cls=1.0):
    """NPE regression NLL + cross-entropy for the three comparison heads (shared encoder)."""
    import torch
    import torch.nn.functional as F
    mean, L, cl, al, bl = model.forward_all(x)
    nll = -torch.distributions.MultivariateNormal(mean, scale_tril=L).log_prob(target_train_space).mean()
    ce = (F.cross_entropy(cl, corr_lbl) + F.cross_entropy(al, sepA_lbl) + F.cross_entropy(bl, sepB_lbl))
    return nll + w_cls * ce, nll.item(), ce.item()
