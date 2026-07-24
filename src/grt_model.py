"""
grt_model.py  —  single source of truth for GRIN's GRT parameterization.

GRIN works in the *identified* coordinates of the 2x2 GRT identification model
under decisional separability (DS): per-stimulus marginal z-scores (sensitivities)
plus within-stimulus correlations, with the decision bounds fixed at 0 and unit
variances (both without loss of generality once DS + a reference are assumed).
This is exactly the coordinate system that mdsdt fits, and in it the fully
unconstrained model has 12 parameters — precisely the 12 independent numbers in a
4x4 confusion matrix. So every model in the family is identifiable from a single
matrix; there is no structural null space to average over. (The only GRT
non-identifiability, the perceptual-vs-decisional decomposition of Silbert &
Thomas 2013, is resolved here by the DS convention: bounds fixed, marginal
position attributed to the perceptual mean. It would only re-appear if DS were
relaxed, which is out of scope for the current model family.)

Canonical vector (what the network predicts), 12 values:
    [zx_0..zx_3, zy_0..zy_3, rho_0..rho_3]
where stimulus order is the mdsdt/grtools order:
    s0=A1B1, s1=A1B2, s2=A2B1, s3=A2B2   (dimension A = x, dimension B = y)

Sign convention (design-consistent): level-1 of a dimension sits below its bound
(negative), level-2 above it (positive). Magnitudes are the sensitivities. Response
BIAS is represented by asymmetric magnitudes about the fixed bound (not by sign
flips), so nothing is lost by fixing the signs to the design. This is a modelling
choice for the prior (identification experiments have ordered, correctly-signed
levels); it is documented here so it can be revisited.
"""

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Stimulus / response layout (canonical = mdsdt/grtools order)
# --------------------------------------------------------------------------- #
STIMULUS_ORDER = ["A1B1", "A1B2", "A2B1", "A2B2"]
RESPONSE_ORDER = ["a1b1", "a1b2", "a2b1", "a2b2"]
A_LEVEL = np.array([0, 0, 1, 1])   # 0 = A1, 1 = A2  (dimension A = x)
B_LEVEL = np.array([0, 1, 0, 1])   # 0 = B1, 1 = B2  (dimension B = y)
SIGN_X = np.where(A_LEVEL == 0, -1.0, 1.0)   # A1 below bound, A2 above
SIGN_Y = np.where(B_LEVEL == 0, -1.0, 1.0)   # B1 below bound, B2 above

PARAM_NAMES = ([f"zx_{i}" for i in range(4)] +
               [f"zy_{i}" for i in range(4)] +
               [f"rho_{i}" for i in range(4)])

# --------------------------------------------------------------------------- #
# Model-class definitions:  name -> (correlation structure, PS on A, PS on B)
#   corr: 'pi'   -> all correlations 0
#         'rho1' -> one shared (equal) correlation across stimuli
#         'free' -> correlations differ across stimuli
#   ps_x (dimension A separable): zx invariant across B-level -> zx_0=zx_1, zx_2=zx_3
#   ps_y (dimension B separable): zy invariant across A-level -> zy_0=zy_2, zy_1=zy_3
# --------------------------------------------------------------------------- #
MODEL_SPECS = {
    "pi_ps_ds":    ("pi",   True,  True),
    "pi_psa_ds":   ("pi",   True,  False),
    "pi_psb_ds":   ("pi",   False, True),
    "rho1_ps_ds":  ("rho1", True,  True),
    "rho1_psa_ds": ("rho1", True,  False),
    "rho1_psb_ds": ("rho1", False, True),
    "pi_ds":       ("pi",   False, False),
    "ps_ds":       ("free", True,  True),
    "rho1_ds":     ("rho1", False, False),
    "psa_ds":      ("free", True,  False),
    "psb_ds":      ("free", False, True),
    "ds":          ("free", False, False),
}
MODEL_NAMES = list(MODEL_SPECS.keys())

# Data degrees of freedom in a 4x4 confusion matrix (4 rows x 3 free probs).
DATA_DF = 12


def n_free_params(model_name):
    """Number of identified free parameters for a model class (<= DATA_DF)."""
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    n_x = 2 if ps_x else 4
    n_y = 2 if ps_y else 4
    n_rho = {"pi": 0, "rho1": 1, "free": 4}[corr]
    return n_x + n_y + n_rho


def free_param_table():
    """Return {model_name: (n_free, identified?)} for all classes."""
    return {m: (n_free_params(m), n_free_params(m) <= DATA_DF) for m in MODEL_NAMES}


# --------------------------------------------------------------------------- #
# Bivariate normal CDF  Phi2(h, k; r) = P(Z1 <= h, Z2 <= k),  standard, corr r.
# Sheppard r-integration:  Phi2 = Phi(h)Phi(k) + int_0^r phi2(h,k;t) dt.
# Fully vectorised over arrays of (h, k, r); accurate for |r| up to ~0.99.
# --------------------------------------------------------------------------- #
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)  # on [-1, 1]


def bvn_cdf(h, k, r):
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    r = np.asarray(r, dtype=float)
    base = norm.cdf(h) * norm.cdf(k)
    # map GL nodes from [-1,1] to [0, r] (per-element upper limit)
    t = r[..., None] * (_GL_NODES + 1.0) / 2.0          # (..., n_nodes)
    jac = r[..., None] / 2.0
    hh = h[..., None]
    kk = k[..., None]
    omt2 = 1.0 - t * t
    dens = np.exp(-(hh * hh - 2.0 * t * hh * kk + kk * kk) / (2.0 * omt2)) \
        / (2.0 * np.pi * np.sqrt(omt2))
    integral = np.sum(_GL_WEIGHTS * dens * jac, axis=-1)
    return base + integral


# --------------------------------------------------------------------------- #
# Forward map:  identified params -> confusion-matrix probabilities.
# --------------------------------------------------------------------------- #
def forward_probabilities(zx, zy, rho):
    """
    (zx, zy, rho) each shape (..., 4)  ->  probabilities shape (..., 4, 4).

    For stimulus i with mean (zx_i, zy_i), unit variances, correlation rho_i, and
    decision bounds at 0, the four response probabilities (a1b1, a1b2, a2b1, a2b2)
    are the four quadrant masses. Response 'a1' on a dimension means the percept
    falls below its bound (0).
    """
    zx = np.asarray(zx, dtype=float)
    zy = np.asarray(zy, dtype=float)
    rho = np.asarray(rho, dtype=float)
    p_x1 = norm.cdf(-zx)                 # P(respond a1 on x) = P(X < 0)
    p_y1 = norm.cdf(-zy)                 # P(respond b1 on y) = P(Y < 0)
    p_both1 = bvn_cdf(-zx, -zy, rho)     # P(X < 0, Y < 0)
    p_a1b1 = p_both1
    p_a1b2 = p_x1 - p_both1
    p_a2b1 = p_y1 - p_both1
    p_a2b2 = 1.0 - p_x1 - p_y1 + p_both1
    probs = np.stack([p_a1b1, p_a1b2, p_a2b1, p_a2b2], axis=-1)  # (...,4,4)
    return np.clip(probs, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Prior over identified parameters (Option 1: explicit, correct-by-construction).
# --------------------------------------------------------------------------- #
def sample_prior(model_name, n, rng, z_max=3.0, r_max=0.9):
    """
    Draw n parameter sets from the class-specific prior.

    z-score magnitudes ~ U(0, z_max)  (d'-like sensitivities; z_max ~ 3 spans
      chance to near-ceiling identification per dimension), signed by design.
    correlations ~ U(-r_max, r_max) for the relevant structure. The near-zero
      band is deliberately NOT excluded, so PI-vs-weak-RHO1 lives on a continuum
      and the network can learn calibrated uncertainty at that boundary.

    Returns zx, zy, rho, each shape (n, 4), with the class constraints applied.
    """
    corr, ps_x, ps_y = MODEL_SPECS[model_name]

    # x-sensitivities (dimension A)
    if ps_x:                                   # zx invariant across B-level
        mA1 = rng.uniform(0, z_max, n)
        mA2 = rng.uniform(0, z_max, n)
        zx = np.stack([-mA1, -mA1, mA2, mA2], axis=1)
    else:
        m = rng.uniform(0, z_max, (n, 4))
        zx = SIGN_X[None, :] * m

    # y-sensitivities (dimension B)
    if ps_y:                                   # zy invariant across A-level
        mB1 = rng.uniform(0, z_max, n)
        mB2 = rng.uniform(0, z_max, n)
        zy = np.stack([-mB1, mB2, -mB1, mB2], axis=1)
    else:
        m = rng.uniform(0, z_max, (n, 4))
        zy = SIGN_Y[None, :] * m

    # correlations
    if corr == "pi":
        rho = np.zeros((n, 4))
    elif corr == "rho1":
        r = rng.uniform(-r_max, r_max, n)
        rho = np.repeat(r[:, None], 4, axis=1)
    else:  # free
        rho = rng.uniform(-r_max, r_max, (n, 4))

    return zx, zy, rho


# --------------------------------------------------------------------------- #
# Pack / unpack between structured form and the canonical 12-vector.
# --------------------------------------------------------------------------- #
def pack(zx, zy, rho):
    return np.concatenate([np.asarray(zx), np.asarray(zy), np.asarray(rho)], axis=-1)


def unpack(vec):
    vec = np.asarray(vec)
    return vec[..., 0:4], vec[..., 4:8], vec[..., 8:12]


# --------------------------------------------------------------------------- #
# Cross-reference transforms to the two R packages.
# --------------------------------------------------------------------------- #
def to_mdsdt(zx, zy, rho):
    """
    Exact (verified from mdsdt source): mdsdt uses the identical coordinates and
    the identical stimulus order. It reports per stimulus (mu_r, sd_r, mu_c, sd_c,
    rho) with sd fixed at 1, so mu_r = zx, mu_c = zy. Returns shape (..., 4, 5).
    """
    zx = np.asarray(zx, float); zy = np.asarray(zy, float); rho = np.asarray(rho, float)
    ones = np.ones_like(zx)
    return np.stack([zx, ones, zy, ones, rho], axis=-1)


def to_grtools(zx, zy, rho):
    """
    grtools uses a different but equivalent convention: it fixes the REFERENCE
    stimulus (s0 = A1B1) mean at the origin and reports the decision bounds (a1,a2)
    plus the other stimuli's means and covariances. The identified content is the
    same; only the frame differs. Our frame has the bounds at 0 and s0's mean at
    (zx_0, zy_0), so translating by -(zx_0, zy_0):

        means_i  = (zx_i - zx_0, zy_i - zy_0)     (s0 -> origin)
        bounds   = (a1, a2) = (-zx_0, -zy_0)
        cov_i    = [[1, rho_i], [rho_i, 1]]       (unit variances)

    The means and bounds map is exact (a rigid translation). The covariance map is
    PROVISIONAL: whether grtools additionally fixes s0's covariance to the identity
    depends on its internal reference convention, which is not verified here. This
    transform should be validated against actual grtools output on shared matrices
    (Phase-4 gate) before being relied on for numeric cross-checks.

    Returns dict with 'means' (...,4,2), 'bounds' (...,2), 'cov' (...,4,2,2).
    """
    zx = np.asarray(zx, float); zy = np.asarray(zy, float); rho = np.asarray(rho, float)
    means = np.stack([zx - zx[..., :1], zy - zy[..., :1]], axis=-1)   # (...,4,2)
    bounds = np.stack([-zx[..., 0], -zy[..., 0]], axis=-1)            # (...,2)
    cov = np.zeros(rho.shape + (2, 2))
    cov[..., 0, 0] = 1.0
    cov[..., 1, 1] = 1.0
    cov[..., 0, 1] = rho
    cov[..., 1, 0] = rho
    return {"means": means, "bounds": bounds, "cov": cov}


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def validate(zx, zy, rho, model_name, atol=1e-6):
    """Check that a parameter set satisfies its class constraints and is valid."""
    corr, ps_x, ps_y = MODEL_SPECS[model_name]
    zx = np.asarray(zx, float); zy = np.asarray(zy, float); rho = np.asarray(rho, float)
    problems = []
    if np.any(np.abs(rho) >= 1.0):
        problems.append("|rho| >= 1 (covariance not positive definite)")
    if ps_x and (np.any(np.abs(zx[..., 0] - zx[..., 1]) > atol) or
                 np.any(np.abs(zx[..., 2] - zx[..., 3]) > atol)):
        problems.append("PS(A) violated: zx not tied across B-level")
    if ps_y and (np.any(np.abs(zy[..., 0] - zy[..., 2]) > atol) or
                 np.any(np.abs(zy[..., 1] - zy[..., 3]) > atol)):
        problems.append("PS(B) violated: zy not tied across A-level")
    if corr == "pi" and np.any(np.abs(rho) > atol):
        problems.append("PI violated: rho != 0")
    if corr == "rho1" and np.any(np.abs(rho - rho[..., :1]) > atol):
        problems.append("RHO1 violated: correlations not equal across stimuli")
    return (len(problems) == 0), problems
