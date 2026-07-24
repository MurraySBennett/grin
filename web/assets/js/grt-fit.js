/**
 * grt-fit.js — maximum-likelihood GRT fitting and AIC/BIC model selection,
 * in the browser.
 *
 * A port of `src/inference/mle.py`: same identified coordinates, same expansion
 * of each class's free parameters, same warm start from the confusion-matrix
 * marginals, same multinomial likelihood, same AIC/BIC. This is the estimator
 * mdsdt implements.
 *
 * WHY THIS EXISTS IN THE WEB APP
 * ------------------------------
 * GRIN's whole claim is "as good as MLE, ~10,000x faster, with calibrated
 * uncertainty". The honest way to make that claim is to run BOTH, on the user's
 * own data, in front of them, and show the timings. So the Analyse page reports
 * the GRIN posterior AND this MLE fit side by side, and lets the user see where
 * they disagree. If they disagree a lot, that is information, not something to
 * hide.
 *
 * It also gives us exact G^2 goodness-of-fit and proper AIC/BIC model selection,
 * which replaces the ad-hoc heuristics the old prototype used.
 *
 * Optimizer is Nelder-Mead with restarts (derivative-free, no dependencies).
 * Verified against scipy's L-BFGS-B in tests/fit.test.mjs.
 *
 * ES module. Depends on grt-core.js.
 */

import {
  MODEL_SPECS,
  MODEL_NAMES,
  nFreeParams,
  DATA_DF,
  forwardProbabilities,
  pack,
  nppf,
  chi2sf,
  logLikSaturated,
} from "./grt-core.js";

// --------------------------------------------------------------------------- //
// Class parameterization (mirrors mle._expand)
//
// Correlations are fitted in Fisher-z space (tanh link), so the optimizer is
// unconstrained and rho can never leave (-1, 1).
// --------------------------------------------------------------------------- //
export function expand(name, free) {
  const { corr, psA, psB } = MODEL_SPECS[name];
  let i = 0;
  let zx, zy, rho;

  if (psA) {
    zx = [free[i], free[i], free[i + 1], free[i + 1]];
    i += 2;
  } else {
    zx = free.slice(i, i + 4);
    i += 4;
  }

  if (psB) {
    zy = [free[i], free[i + 1], free[i], free[i + 1]];
    i += 2;
  } else {
    zy = free.slice(i, i + 4);
    i += 4;
  }

  if (corr === "pi") {
    rho = [0, 0, 0, 0];
  } else if (corr === "rho1") {
    const r = Math.tanh(free[i]);
    rho = [r, r, r, r];
    i += 1;
  } else {
    rho = free.slice(i, i + 4).map(Math.tanh);
    i += 4;
  }

  return pack(zx, zy, rho);
}

/** Negative multinomial log-likelihood (drops the constant term). */
export function nll(free, name, counts) {
  const P = forwardProbabilities(expand(name, free));
  let s = 0;
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      const n = counts[i][j];
      if (n > 0) s += n * Math.log(Math.max(P[i][j], 1e-12));
    }
  return -s;
}

/** Warm start from the marginal response proportions (mdsdt-style). */
export function initFromData(name, counts, trials) {
  const clip = (v) => Math.min(1 - 1e-3, Math.max(1e-3, v));
  const zx = [],
    zy = [];
  for (let s = 0; s < 4; s++) {
    const T = Math.max(trials[s], 1);
    zx.push(-nppf(clip((counts[s][0] + counts[s][1]) / T))); // P(respond a1)
    zy.push(-nppf(clip((counts[s][0] + counts[s][2]) / T))); // P(respond b1)
  }
  const { corr, psA, psB } = MODEL_SPECS[name];
  const init = [];
  if (psA) init.push(0.5 * (zx[0] + zx[1]), 0.5 * (zx[2] + zx[3]));
  else init.push(...zx);
  if (psB) init.push(0.5 * (zy[0] + zy[2]), 0.5 * (zy[1] + zy[3]));
  else init.push(...zy);
  if (corr === "rho1") init.push(0);
  else if (corr === "free") init.push(0, 0, 0, 0);
  return init;
}

// --------------------------------------------------------------------------- //
// Nelder-Mead (with restarts). Derivative-free, dependency-free, and fast
// enough here: the worst case is 12 dimensions and a handful of milliseconds.
// --------------------------------------------------------------------------- //
export function nelderMead(f, x0, opts = {}) {
  const {
    maxIter = 4000,
    tolF = 1e-12,
    tolX = 1e-10,
    step = 0.35,
    restarts = 2,
  } = opts;
  const n = x0.length;
  if (n === 0) return { x: [], fx: f([]), iters: 0 };

  let best = { x: x0.slice(), fx: f(x0) };

  for (let attempt = 0; attempt <= restarts; attempt++) {
    // build simplex around the current best
    let simplex = [best.x.slice()];
    for (let i = 0; i < n; i++) {
      const p = best.x.slice();
      p[i] += (Math.abs(p[i]) > 1e-8 ? Math.abs(p[i]) * 0.05 : 0) + step;
      simplex.push(p);
    }
    let fv = simplex.map(f);

    for (let iter = 0; iter < maxIter; iter++) {
      // order
      const ord = fv.map((v, i) => i).sort((a, b) => fv[a] - fv[b]);
      simplex = ord.map((i) => simplex[i]);
      fv = ord.map((i) => fv[i]);

      // convergence
      const fSpread = Math.abs(fv[n] - fv[0]);
      let xSpread = 0;
      for (let i = 1; i <= n; i++)
        for (let j = 0; j < n; j++)
          xSpread = Math.max(xSpread, Math.abs(simplex[i][j] - simplex[0][j]));
      if (fSpread <= tolF * (Math.abs(fv[0]) + tolF) && xSpread <= tolX) break;

      // centroid of the best n
      const c = new Array(n).fill(0);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) c[j] += simplex[i][j] / n;

      const along = (t) => c.map((v, j) => v + t * (v - simplex[n][j]));

      // reflect
      const xr = along(1.0);
      const fr = f(xr);
      if (fr < fv[0]) {
        // expand
        const xe = along(2.0);
        const fe = f(xe);
        if (fe < fr) {
          simplex[n] = xe;
          fv[n] = fe;
        } else {
          simplex[n] = xr;
          fv[n] = fr;
        }
      } else if (fr < fv[n - 1]) {
        simplex[n] = xr;
        fv[n] = fr;
      } else {
        // contract
        const outside = fr < fv[n];
        const xc = along(outside ? 0.5 : -0.5);
        const fc = f(xc);
        if (fc < Math.min(fr, fv[n])) {
          simplex[n] = xc;
          fv[n] = fc;
        } else {
          // shrink
          for (let i = 1; i <= n; i++) {
            simplex[i] = simplex[i].map(
              (v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]),
            );
            fv[i] = f(simplex[i]);
          }
        }
      }
    }

    const ord = fv.map((v, i) => i).sort((a, b) => fv[a] - fv[b]);
    if (fv[ord[0]] < best.fx - 1e-12) {
      best = { x: simplex[ord[0]].slice(), fx: fv[ord[0]] };
    } else if (attempt > 0) {
      break; // restart bought us nothing; we're converged
    } else {
      best = { x: simplex[ord[0]].slice(), fx: fv[ord[0]] };
    }
  }
  return best;
}

// --------------------------------------------------------------------------- //
// Fitting
// --------------------------------------------------------------------------- //
/**
 * Fit ONE model class by maximum likelihood.
 * @returns {{model, params, loglik, k, aic, bic, g2, df, p, converged}}
 */
export function fitClass(counts, trials, name, init = null) {
  const x0 = init ?? initFromData(name, counts, trials);
  const t0 = now();
  const res = nelderMead((x) => nll(x, name, counts), x0);
  const ms = now() - t0;

  const params = expand(name, res.x);
  const loglik = -res.fx;
  const k = nFreeParams(name);
  const n = counts.flat().reduce((a, b) => a + b, 0);

  // G^2 goodness of fit against the saturated model
  const g2 = 2 * (logLikSaturated(counts) - loglik);
  const df = DATA_DF - k;

  return {
    model: name,
    params,
    loglik,
    k,
    n,
    aic: 2 * k - 2 * loglik,
    bic: k * Math.log(Math.max(n, 1)) - 2 * loglik,
    g2,
    df,
    p: df > 0 ? chi2sf(g2, df) : NaN,
    ms,
  };
}

/** Fit the saturated (fully free) model: the 12 identified parameters. */
export function fitFull(counts, trials) {
  return fitClass(counts, trials, "ds");
}

/**
 * Fit EVERY class and select. Returns {best, fits, ms} with fits sorted by the
 * criterion, plus Akaike weights (how much better is the winner, really?).
 */
export function fitAndSelect(counts, trials, criterion = "bic") {
  const t0 = now();
  const fits = MODEL_NAMES.map((m) => fitClass(counts, trials, m));

  // Akaike/Schwarz weights: exp(-delta/2) normalised. These are the honest way
  // to report model selection — "the winner" is often barely ahead.
  const vals = fits.map((f) => f[criterion]);
  const min = Math.min(...vals);
  const w = vals.map((v) => Math.exp(-0.5 * (v - min)));
  const wSum = w.reduce((a, b) => a + b, 0);
  fits.forEach((f, i) => {
    f.delta = vals[i] - min;
    f.weight = w[i] / wSum;
  });

  fits.sort((a, b) => a[criterion] - b[criterion]);
  return { best: fits[0], fits, criterion, ms: now() - t0 };
}

function now() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/**
 * Summarise a selection result into the three structural questions GRIN cares
 * about, by summing the weights of every class that asserts each construct.
 * This is the MLE-side analogue of the network's p_corr / p_sep heads, so the
 * two can be put next to each other honestly.
 */
export function constructWeights(fits) {
  const acc = { pi: 0, rho1: 0, free: 0, sepA: 0, sepB: 0 };
  for (const f of fits) {
    const { corr, psA, psB } = MODEL_SPECS[f.model];
    acc[corr] += f.weight;
    if (psA) acc.sepA += f.weight;
    if (psB) acc.sepB += f.weight;
  }
  return acc;
}

// --------------------------------------------------------------------------- //
// Checkpointed fitting: refit cumulatively as trials arrive, snapshotting the
// estimate every N trials. This is the shared core loop behind every
// "Dynamics" page idea (adaptive selection, early stopping, power planning,
// drift tracking) and behind the fading-trail figure — all of them are
// "refit repeatedly as data accumulates, watch what changes." Only the
// checkpoint SPACING differs between those uses (fixed-N here; a Dynamics
// page might stop early on a confidence threshold instead).
// --------------------------------------------------------------------------- //
/**
 * @param {{stimulus:number, response:number}[]} trials — in the order they
 *   occurred; only stimulus/response are used (rt, if present, is ignored —
 *   this checkpoints the counts-only model, matching Explore/Analyse's
 *   default; a caller doing an RT-aware version needs its own loop, since RT
 *   quantiles aren't meaningfully incremental the way counts are)
 * @param {Object} options
 *   every     default 20 — trials per checkpoint
 *   modelName default "ds" — which of the 12 classes to fit at each
 *             checkpoint; "ds" (fully free) matches what Explore/Analyse
 *             show as "the estimate" before any model selection
 *   minPerCheckpoint default 20 — skip checkpoints before every stimulus has
 *             at least this many trials. This isn't just about avoiding an
 *             empty row: an unconstrained multinomial MLE is a classic
 *             complete-separation case whenever a cell shows ZERO errors,
 *             which is common at low N even under perfectly ordinary
 *             sensitivities — the z-score for that cell is technically
 *             unbounded, and the optimiser just runs off toward whatever
 *             large-but-finite value it happens to stop at (verified
 *             directly: at 15 trials/stimulus, z-scores of 8-12 turned up
 *             routinely). Raising the floor doesn't eliminate the
 *             possibility, only makes it much rarer.
 *   windowSize default null — if set, only the most recent `windowSize`
 *             trials count toward each checkpoint's fit, a true sliding
 *             window rather than a running total. This is the mode that
 *             matters for DRIFT: a cumulative fit (windowSize unset) can
 *             never fully catch up to a representation that changed
 *             partway through, because every pre-change trial keeps
 *             dragging the estimate toward a blend of old and new truth
 *             forever, no matter how much post-change data arrives — that
 *             was verified directly, not assumed (see grt-sim.js's
 *             simulateDriftStream and the tests around it). A window
 *             correctly forgets old trials instead.
 * @returns {{trialCount, counts, stimuli:{zx,zy,rho}[], fit}[]}
 *   `stimuli`'s zx/zy are clamped to +/-Z_CLAMP for rendering ONLY, so one
 *   degenerate checkpoint can't draw an ellipse off the edge of the canvas.
 *   `fit.params` is left completely untouched (the real, unclamped numbers)
 *   for anyone who wants to inspect what the optimiser actually returned.
 */
const Z_CLAMP = 4; // well beyond the +/-3 sensitivity range anything on this
// site is trained or simulated with, so it only ever
// catches genuine degenerate fits, never a real value

/**
 * Which trials belong to which checkpoint — the part of checkpointed fitting
 * that has nothing to do with WHICH estimator does the fitting. Pulled out on
 * its own so a caller can drive GRIN through this exact same checkpoint
 * schedule (fast enough to run every checkpoint for real) while MLE-based
 * checkpointFits (below) reuses it unchanged for the classical comparison.
 *
 * @param {{stimulus:number, response:number}[]} trials
 * @param {Object} options — every, minPerCheckpoint, windowSize (see
 *   checkpointFits' docs below for what each means; identical here)
 * @returns {{trialCount, counts:number[][], trials:number[]}[]}
 */
export function checkpointSnapshots(trials, options = {}) {
  const { every = 20, minPerCheckpoint = 20, windowSize = null } = options;
  const counts = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  const queue = []; // only populated when windowSize is set
  const snapshots = [];

  trials.forEach((t, i) => {
    counts[t.stimulus][t.response] += 1;
    if (windowSize) {
      queue.push(t);
      if (queue.length > windowSize) {
        const old = queue.shift();
        counts[old.stimulus][old.response] -= 1;
      }
    }

    const trialCount = i + 1;
    if (trialCount % every !== 0 && trialCount !== trials.length) return;

    const rowTotals = counts.map((r) => r.reduce((a, b) => a + b, 0));
    if (Math.min(...rowTotals) < minPerCheckpoint) return;

    snapshots.push({
      trialCount,
      counts: counts.map((r) => r.slice()),
      trials: rowTotals,
    });
  });

  return snapshots;
}

export function checkpointFits(trials, options = {}) {
  const { modelName = "ds" } = options;
  const clamp = (v) => Math.max(-Z_CLAMP, Math.min(Z_CLAMP, v));
  return checkpointSnapshots(trials, options).map(
    ({ trialCount, counts, trials: rowTotals }) => {
      const fit = fitClass(counts, rowTotals, modelName);
      const { zx, zy, rho } = unpackParams(fit.params);
      return {
        trialCount,
        counts,
        stimuli: [0, 1, 2, 3].map((k) => ({
          zx: clamp(zx[k]),
          zy: clamp(zy[k]),
          rho: rho[k],
        })),
        fit,
      };
    },
  );
}

function unpackParams(v) {
  return { zx: v.slice(0, 4), zy: v.slice(4, 8), rho: v.slice(8, 12) };
}
