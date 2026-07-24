/**
 * grt-core.js — GRIN's GRT parameterization, in the browser.
 *
 * A direct port of `src/grt_model.py`. This is the single source of truth for
 * the JS side and must stay in lockstep with the Python: same identified
 * coordinates, same stimulus order, same forward map, same model-class specs.
 * `tests/core.test.js` checks it against Python output to ~1e-10.
 *
 * IDENTIFIED COORDINATES (decisional separability assumed):
 *   decision bounds fixed at 0, unit variances, per-stimulus marginal z-scores
 *   plus within-stimulus correlations. 12 parameters — exactly the 12 free
 *   numbers in a 4x4 confusion matrix, so every model in the family is
 *   identifiable from a single matrix.
 *
 * CANONICAL PARAMETER VECTOR (what the network predicts), length 12:
 *   [zx_0..zx_3, zy_0..zy_3, rho_0..rho_3]
 *
 * CANONICAL STIMULUS ORDER (mdsdt / grtools order):
 *   s0 = A1B1,  s1 = A1B2,  s2 = A2B1,  s3 = A2B2
 *   dimension A = x, dimension B = y
 *
 * SIGN CONVENTION: level 1 of a dimension sits below its bound (negative),
 * level 2 above it (positive). Magnitudes are the sensitivities.
 *
 * ES module. No dependencies.
 */

// --------------------------------------------------------------------------- //
// Stimulus / response layout
// --------------------------------------------------------------------------- //
export const STIMULUS_ORDER = ["A1B1", "A1B2", "A2B1", "A2B2"];
export const RESPONSE_ORDER = ["a1b1", "a1b2", "a2b1", "a2b2"];
export const A_LEVEL = [0, 0, 1, 1]; // 0 = A1, 1 = A2  (dimension A = x)
export const B_LEVEL = [0, 1, 0, 1]; // 0 = B1, 1 = B2  (dimension B = y)
export const SIGN_X = [-1, -1, 1, 1];
export const SIGN_Y = [-1, 1, -1, 1];

export const PARAM_NAMES = [
  "zx_0",
  "zx_1",
  "zx_2",
  "zx_3",
  "zy_0",
  "zy_1",
  "zy_2",
  "zy_3",
  "rho_0",
  "rho_1",
  "rho_2",
  "rho_3",
];
export const N_PARAMS = 12;

/** Data degrees of freedom in a 4x4 confusion matrix (4 rows x 3 free probs). */
export const DATA_DF = 12;

// --------------------------------------------------------------------------- //
// Model classes:  name -> {corr, psA, psB}
//   corr: "pi"   -> all correlations 0
//         "rho1" -> one shared correlation across stimuli
//         "free" -> correlations differ across stimuli
//   psA (dimension A separable): zx invariant across B-level -> zx0=zx1, zx2=zx3
//   psB (dimension B separable): zy invariant across A-level -> zy0=zy2, zy1=zy3
// --------------------------------------------------------------------------- //
export const MODEL_SPECS = {
  pi_ps_ds: { corr: "pi", psA: true, psB: true },
  pi_psa_ds: { corr: "pi", psA: true, psB: false },
  pi_psb_ds: { corr: "pi", psA: false, psB: true },
  rho1_ps_ds: { corr: "rho1", psA: true, psB: true },
  rho1_psa_ds: { corr: "rho1", psA: true, psB: false },
  rho1_psb_ds: { corr: "rho1", psA: false, psB: true },
  pi_ds: { corr: "pi", psA: false, psB: false },
  ps_ds: { corr: "free", psA: true, psB: true },
  rho1_ds: { corr: "rho1", psA: false, psB: false },
  psa_ds: { corr: "free", psA: true, psB: false },
  psb_ds: { corr: "free", psA: false, psB: true },
  ds: { corr: "free", psA: false, psB: false },
};
export const MODEL_NAMES = Object.keys(MODEL_SPECS);

/** Number of identified free parameters for a model class (<= DATA_DF). */
export function nFreeParams(name) {
  const { corr, psA, psB } = MODEL_SPECS[name];
  const nX = psA ? 2 : 4;
  const nY = psB ? 2 : 4;
  const nRho = { pi: 0, rho1: 1, free: 4 }[corr];
  return nX + nY + nRho;
}

/** Human-readable label for a model class, e.g. "PI · PS(A,B) · DS". */
export function modelLabel(name) {
  const { corr, psA, psB } = MODEL_SPECS[name];
  const c = { pi: "PI", rho1: "RHO1", free: "free ρ" }[corr];
  const s = psA && psB ? "PS(A,B)" : psA ? "PS(A)" : psB ? "PS(B)" : "no PS";
  return `${c} · ${s} · DS`;
}

// --------------------------------------------------------------------------- //
// Normal distribution — double precision.
//
// ncdf: Hart/West rational approximation, ~1e-15 absolute. (The old
// Abramowitz-Stegun 7.1.26 form was only good to 1.5e-7, which is not enough
// to match scipy in the tails where GRT lives at high d'.)
// --------------------------------------------------------------------------- //
export function ncdf(x) {
  const xa = Math.abs(x);
  let c;
  if (xa > 37) {
    c = 0;
  } else {
    const e = Math.exp((-xa * xa) / 2);
    if (xa < 7.07106781186547) {
      let b = 3.52624965998911e-2 * xa + 0.700383064443688;
      b = b * xa + 6.37396220353165;
      b = b * xa + 33.912866078383;
      b = b * xa + 112.079291497871;
      b = b * xa + 221.213596169931;
      b = b * xa + 220.206867912376;
      let d = 8.83883476483184e-2 * xa + 1.75566716318264;
      d = d * xa + 16.064177579207;
      d = d * xa + 86.7807322029461;
      d = d * xa + 296.564248779674;
      d = d * xa + 637.333633378831;
      d = d * xa + 793.826512519948;
      d = d * xa + 440.413735824752;
      c = (e * b) / d;
    } else {
      let b = xa + 0.65;
      b = xa + 4 / b;
      b = xa + 3 / b;
      b = xa + 2 / b;
      b = xa + 1 / b;
      c = e / (b * 2.506628274631);
    }
  }
  return x > 0 ? 1 - c : c;
}

export function npdf(z) {
  return Math.exp((-z * z) / 2) / Math.sqrt(2 * Math.PI);
}

export function erf(x) {
  return 2 * ncdf(x * Math.SQRT2) - 1;
}

/** Inverse standard-normal CDF: Acklam's rational approx + one Halley step. */
export function nppf(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pl = 0.02425;
  let x, q, r;
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= 1 - pl) {
    q = p - 0.5;
    r = q * q;
    x =
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    x =
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  // Halley refinement -> machine precision
  const e = ncdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  return x - u / (1 + (x * u) / 2);
}

// --------------------------------------------------------------------------- //
// Bivariate normal CDF, Sheppard r-integration:
//   Phi2(h,k;r) = Phi(h)Phi(k) + int_0^r phi2(h,k;t) dt
// Same 48-node Gauss-Legendre rule as numpy's leggauss(48) in grt_model.py.
// --------------------------------------------------------------------------- //
const GL_NODES = [
  -0.9987710072524261, -0.9935301722663508, -0.9841245837228269,
  -0.9705915925462473, -0.9529877031604308, -0.9313866907065543,
  -0.9058791367155696, -0.8765720202742479, -0.8435882616243935,
  -0.8070662040294426, -0.7671590325157403, -0.7240341309238146,
  -0.6778723796326639, -0.6288673967765136, -0.5772247260839727,
  -0.5231609747222331, -0.4669029047509584, -0.4086864819907167,
  -0.3487558862921607, -0.2873624873554556, -0.2247637903946891,
  -0.1612223560688917, -0.0970046992094627, -0.0323801709628694,
  0.0323801709628694, 0.0970046992094627, 0.1612223560688917,
  0.2247637903946891, 0.2873624873554556, 0.3487558862921607,
  0.4086864819907167, 0.4669029047509584, 0.5231609747222331,
  0.5772247260839727, 0.6288673967765136, 0.6778723796326639,
  0.7240341309238146, 0.7671590325157403, 0.8070662040294426,
  0.8435882616243935, 0.8765720202742479, 0.9058791367155696,
  0.9313866907065543, 0.9529877031604308, 0.9705915925462473,
  0.9841245837228269, 0.9935301722663508, 0.9987710072524261,
];
const GL_WEIGHTS = [
  0.0031533460523058, 0.0073275539012763, 0.0114772345792345,
  0.0155793157229438, 0.0196161604573555, 0.0235707608393244,
  0.0274265097083569, 0.0311672278327981, 0.0347772225647704,
  0.0382413510658307, 0.0415450829434647, 0.0446745608566943,
  0.0476166584924905, 0.0503590355538545, 0.0528901894851937,
  0.0551995036999842, 0.0572772921004032, 0.0591148396983956,
  0.0607044391658939, 0.0620394231598927, 0.063114192286254, 0.0639242385846482,
  0.0644661644359501, 0.0647376968126839, 0.0647376968126839,
  0.0644661644359501, 0.0639242385846482, 0.063114192286254, 0.0620394231598927,
  0.0607044391658939, 0.0591148396983956, 0.0572772921004032,
  0.0551995036999842, 0.0528901894851937, 0.0503590355538545,
  0.0476166584924905, 0.0446745608566943, 0.0415450829434647,
  0.0382413510658307, 0.0347772225647704, 0.0311672278327981,
  0.0274265097083569, 0.0235707608393244, 0.0196161604573555,
  0.0155793157229438, 0.0114772345792345, 0.0073275539012763,
  0.0031533460523058,
];

/** P(Z1 <= h, Z2 <= k) for standard bivariate normal with correlation r. */
export function bvnCdf(h, k, r) {
  const base = ncdf(h) * ncdf(k);
  if (r === 0) return base;
  const jac = r / 2;
  let integral = 0;
  for (let i = 0; i < GL_NODES.length; i++) {
    const t = (r * (GL_NODES[i] + 1)) / 2;
    const omt2 = 1 - t * t;
    const dens =
      Math.exp(-(h * h - 2 * t * h * k + k * k) / (2 * omt2)) /
      (2 * Math.PI * Math.sqrt(omt2));
    integral += GL_WEIGHTS[i] * dens * jac;
  }
  return base + integral;
}

// --------------------------------------------------------------------------- //
// Forward map: identified params -> confusion-matrix probabilities.
// --------------------------------------------------------------------------- //
/**
 * Response probabilities [a1b1, a1b2, a2b1, a2b2] for ONE stimulus.
 * Responding "a1" on a dimension means the percept fell below its bound (0).
 */
export function quad(zx, zy, rho) {
  const pX1 = ncdf(-zx); // P(respond a1)
  const pY1 = ncdf(-zy); // P(respond b1)
  const pBoth = bvnCdf(-zx, -zy, rho);
  const p = [pBoth, pX1 - pBoth, pY1 - pBoth, 1 - pX1 - pY1 + pBoth];
  return p.map((v) => Math.min(1, Math.max(0, v)));
}

/**
 * Full 4x4 predicted confusion matrix from a 12-vector (or {zx,zy,rho}).
 * Rows = stimuli, columns = responses, both in canonical order.
 */
export function forwardProbabilities(params) {
  const { zx, zy, rho } = unpack(params);
  return [0, 1, 2, 3].map((i) => quad(zx[i], zy[i], rho[i]));
}

// --------------------------------------------------------------------------- //
// Pack / unpack
// --------------------------------------------------------------------------- //
export function pack(zx, zy, rho) {
  return [...zx, ...zy, ...rho];
}

export function unpack(vec) {
  if (!Array.isArray(vec) && vec && vec.zx) return vec; // already structured
  const v = Array.from(vec);
  return { zx: v.slice(0, 4), zy: v.slice(4, 8), rho: v.slice(8, 12) };
}

/** Apply a model class's constraints to a free 12-vector (projection). */
export function constrain(params, name) {
  const { corr, psA, psB } = MODEL_SPECS[name];
  let { zx, zy, rho } = unpack(params);
  zx = zx.slice();
  zy = zy.slice();
  rho = rho.slice();
  if (psA) {
    const g1 = (zx[0] + zx[1]) / 2;
    const g2 = (zx[2] + zx[3]) / 2;
    zx = [g1, g1, g2, g2];
  }
  if (psB) {
    const g1 = (zy[0] + zy[2]) / 2;
    const g2 = (zy[1] + zy[3]) / 2;
    zy = [g1, g2, g1, g2];
  }
  if (corr === "pi") {
    rho = [0, 0, 0, 0];
  } else if (corr === "rho1") {
    const m = (rho[0] + rho[1] + rho[2] + rho[3]) / 4;
    rho = [m, m, m, m];
  }
  return pack(zx, zy, rho);
}

/** Check that a parameter set satisfies its class constraints. */
export function validate(params, name, atol = 1e-6) {
  const { corr, psA, psB } = MODEL_SPECS[name];
  const { zx, zy, rho } = unpack(params);
  const problems = [];
  if (rho.some((r) => Math.abs(r) >= 1))
    problems.push("|rho| >= 1 (covariance not positive definite)");
  if (psA && (Math.abs(zx[0] - zx[1]) > atol || Math.abs(zx[2] - zx[3]) > atol))
    problems.push("PS(A) violated: zx not tied across B-level");
  if (psB && (Math.abs(zy[0] - zy[2]) > atol || Math.abs(zy[1] - zy[3]) > atol))
    problems.push("PS(B) violated: zy not tied across A-level");
  if (corr === "pi" && rho.some((r) => Math.abs(r) > atol))
    problems.push("PI violated: rho != 0");
  if (corr === "rho1" && rho.some((r) => Math.abs(r - rho[0]) > atol))
    problems.push("RHO1 violated: correlations not equal across stimuli");
  return { ok: problems.length === 0, problems };
}

// --------------------------------------------------------------------------- //
// Likelihood
// --------------------------------------------------------------------------- //
/** Multinomial log-likelihood of a 4x4 count matrix under a 12-vector (drops the
 *  combinatorial constant, which is identical across models). */
export function logLik(counts, params) {
  const P = forwardProbabilities(params);
  let ll = 0;
  for (let s = 0; s < 4; s++) {
    for (let r = 0; r < 4; r++) {
      const n = counts[s][r];
      if (n > 0) ll += n * Math.log(Math.max(P[s][r], 1e-300));
    }
  }
  return ll;
}

/** Saturated (perfect-fit) log-likelihood: the empirical proportions. */
export function logLikSaturated(counts) {
  let ll = 0;
  for (let s = 0; s < 4; s++) {
    const T = counts[s].reduce((a, b) => a + b, 0);
    if (T === 0) continue;
    for (let r = 0; r < 4; r++) {
      const n = counts[s][r];
      if (n > 0) ll += n * Math.log(n / T);
    }
  }
  return ll;
}

/** Chi-square survival function (upper tail), for the G^2 goodness-of-fit p-value. */
export function chi2sf(x, df) {
  if (x <= 0) return 1;
  return 1 - lowerGamma(df / 2, x / 2);
}

// regularized lower incomplete gamma P(a, x)
function lowerGamma(a, x) {
  if (x < 0 || a <= 0) return NaN;
  if (x < a + 1) {
    // series
    let ap = a;
    let sum = 1 / a;
    let del = sum;
    for (let n = 0; n < 500; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-14) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
  }
  // continued fraction for Q(a,x)
  const FPMIN = 1e-300;
  let b = x + 1 - a;
  let c = 1 / FPMIN;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  const q = Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
  return 1 - q;
}

export function lgamma(z) {
  const g = 7;
  const C = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1;
  let x = C[0];
  for (let i = 1; i < g + 2; i++) x += C[i] / (z + i);
  const t = z + g + 0.5;
  return (
    0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
  );
}
