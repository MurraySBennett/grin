/**
 * grt-sim.js — the forward model, for the teaching pages.
 *
 * A port of `src/data/rt_lba_generator.py::_simulate_group` for a single
 * participant. This matters: the explorers must generate data from the SAME
 * process the network was trained to invert. If the demo's simulator drifts
 * from the training simulator, the demo is lying about how well GRIN works.
 *
 * Per trial, ONE perceptual sample is drawn, and that single sample determines
 * BOTH the response (which quadrant it fell in) AND the response time (its
 * distance from each bound drives the LBA drift rate). Counts and RTs are
 * matched by construction — that coupling is the entire reason RTs carry
 * information about the perceptual representation.
 *
 * Seeded (mulberry32) so "run the experiment" is reproducible when you want it
 * to be and fresh when you don't.
 *
 * ES module. Depends on grt-core.js (forward probabilities) and grt-io.js
 * (aggregation to network inputs).
 */

import { quad } from "./grt-core.js";
import { aggregate, TRAIN_BOUNDS } from "./grt-io.js";

/** SFT taxonomy: architecture x stopping rule. Coactive has no stopping-rule
 *  crossing (evidence pools into one accumulator), hence 5, not 6. */
export const ARCHITECTURES = [
  "serial_exhaustive",
  "serial_self_terminating",
  "parallel_exhaustive",
  "parallel_self_terminating",
  "coactive",
];

export const ARCH_LABELS = {
  serial_exhaustive: { name: "Serial, exhaustive", rt: "RT = t₀ + tₐ + t_b" },
  serial_self_terminating: {
    name: "Serial, self-terminating",
    rt: "RT = t₀ + t_first",
  },
  parallel_exhaustive: {
    name: "Parallel, exhaustive",
    rt: "RT = t₀ + max(tₐ, t_b)",
  },
  parallel_self_terminating: {
    name: "Parallel, self-terminating",
    rt: "RT = t₀ + min(tₐ, t_b)",
  },
  coactive: { name: "Coactive", rt: "RT = t₀ + A / (vₐ + v_b)" },
};

/**
 * The self-terminating models deserve a health warning, and the generator's own
 * docstring gives it: in an IDENTIFICATION task the response must name BOTH
 * levels, so stopping early means GUESSING the un-processed dimension. These
 * are not a normal processing mode — they are the participant who is not using
 * a dimension (incapacity, inattention, or strategy). A pathology to detect.
 */
export const SELF_TERMINATING = ARCHITECTURES.filter((a) =>
  a.includes("self_terminating"),
);

export const LBA_DEFAULTS = { t0: 0.3, threshold: 0.7, kA: 1.3, kB: 1.3 };
export const LBA_RANGES = {
  t0: [0.15, 0.45], // non-decision time (s)
  threshold: [0.35, 1.1], // caution
  kA: [0.6, 2.0], // drift scaling, dimension A
  kB: [0.6, 2.0], // drift scaling, dimension B
};
export const DRIFT_SD = 0.35; // config.RT_DRIFT_SD

// --------------------------------------------------------------------------- //
// RNG — seeded, so a demo can be replayed
// --------------------------------------------------------------------------- //
export function makeRNG(seed) {
  let a = (seed ?? Math.random() * 2 ** 32) >>> 0;
  const uniform = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u, v, s;
    do {
      u = 2 * uniform() - 1;
      v = 2 * uniform() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * f;
    return u * f;
  };
  return { uniform, normal };
}

// --------------------------------------------------------------------------- //
// Counts only: multinomial sampling from the exact forward probabilities
// --------------------------------------------------------------------------- //
/** Draw one multinomial(n, p) vector via sequential binomials. */
function multinomial(p, n, rng) {
  const c = [0, 0, 0, 0];
  let remaining = n;
  let cumP = 0;
  for (let k = 0; k < 3; k++) {
    if (remaining <= 0) break;
    const pk = Math.min(1, Math.max(0, p[k] / Math.max(1e-12, 1 - cumP)));
    let x = 0;
    for (let i = 0; i < remaining; i++) if (rng.uniform() < pk) x++;
    c[k] = x;
    remaining -= x;
    cumP += p[k];
  }
  c[3] = Math.max(0, remaining);
  return c;
}

/**
 * Simulate a counts-only identification experiment.
 * @param {{zx:number[], zy:number[], rho:number[]}} rep — the true representation
 * @param {number} nPerStimulus
 * @returns {{counts, trials, rtq:null, probs}}
 */
export function simulateCounts(rep, nPerStimulus, rng = makeRNG()) {
  const probs = [0, 1, 2, 3].map((i) => quad(rep.zx[i], rep.zy[i], rep.rho[i]));
  const counts = probs.map((p) => multinomial(p, nPerStimulus, rng));
  return {
    counts,
    trials: counts.map((r) => r.reduce((a, b) => a + b, 0)),
    rtq: null,
    probs,
  };
}

/**
 * A trial-level stream (not just aggregate counts) — needed anywhere ARRIVAL
 * ORDER matters, e.g. checkpointed/incremental fitting. simulateCounts can't
 * be reused for this: multinomial sampling gives you totals per cell, not an
 * order trials occurred in.
 *
 * Draws nPerStimulus trials for each of the 4 stimuli in turn, then shuffles
 * the WHOLE stream — matching a real experiment, which randomises stimulus
 * order across the session rather than blocking by stimulus.
 */
export function simulateTrialStream(rep, nPerStimulus, rng = makeRNG()) {
  const trials = [];
  for (let s = 0; s < 4; s++) {
    const p = quad(rep.zx[s], rep.zy[s], rep.rho[s]);
    for (let t = 0; t < nPerStimulus; t++)
      trials.push({ stimulus: s, response: sampleCategorical(p, rng) });
  }
  shuffleInPlace(trials, rng);
  return trials;
}

/**
 * A trial stream whose TRUE representation changes partway through — a
 * simulated participant who genuinely drifts (fatigue, practice, a shifted
 * criterion), not just noise around a fixed truth. The shift is a single
 * abrupt jump at the midpoint rather than a smooth interpolation: dramatic
 * enough to see clearly in a fading-trail figure, and unambiguous to verify
 * (the first half's checkpoints should track repBefore, the second half's
 * should track repAfter).
 *
 * Shuffles WITHIN each half (realistic local randomisation) but keeps the
 * two halves in temporal order — shuffling across the whole stream would
 * destroy the very drift signal this function exists to create.
 */
export function simulateDriftStream(
  repBefore,
  repAfter,
  totalPerStimulus,
  rng = makeRNG(),
) {
  const half = Math.floor(totalPerStimulus / 2);
  const first = simulateTrialStream(repBefore, half, rng);
  const second = simulateTrialStream(repAfter, totalPerStimulus - half, rng);
  return [...first, ...second];
}

/**
 * A trial stream whose separability shifts GRADUALLY starting at a given
 * intervention point, rather than jumping between two blocks. Before the
 * intervention trial, dA/dB are held at their starting values. From the
 * intervention trial onward, each trial's dA/dB creep toward the target by a
 * fixed fraction (alpha) of whatever gap remains — a Rescorla-Wagner-style
 * exponential approach: fast movement right after the intervention, flattening
 * out as it nears the target, never a linear ramp that only arrives at the
 * very last trial. mA/mB/rho stay fixed throughout, same as the abrupt version.
 *
 * This walks the session in genuine TEMPORAL order rather than generating a
 * block per stimulus and shuffling afterward (simulateTrialStream's approach)
 * — shuffling would scramble which representation was true "at trial i" and
 * destroy the very thing a gradual ramp is supposed to show. Instead, which
 * stimulus appears at which trial index is decided once, up front (a shuffled
 * bag, balanced across the 4 stimuli), and then walked in order, updating
 * dA/dB as we go.
 *
 * @param {Object} base — {mA, mB, rho, dAStart, dBStart, dATarget, dBTarget}
 * @param {number} totalPerStimulus
 * @param {number} interventionFrac — 0..1, where in the session (by trial
 *   COUNT, not calendar time) the drift begins
 * @param {number} alpha — learning rate; gap shrinks by this fraction each
 *   trial after the intervention point
 */
export function simulateGradualDriftStream(
  base,
  totalPerStimulus,
  interventionFrac,
  alpha,
  rng = makeRNG(),
) {
  const { mA, mB, rho, dAStart, dBStart, dATarget, dBTarget } = base;
  const bag = [];
  for (let s = 0; s < 4; s++)
    for (let k = 0; k < totalPerStimulus; k++) bag.push(s);
  shuffleInPlace(bag, rng);

  const interventionTrial = Math.round(interventionFrac * bag.length);
  const trials = [];
  let dA = dAStart,
    dB = dBStart;
  for (let i = 0; i < bag.length; i++) {
    if (i >= interventionTrial) {
      dA += alpha * (dATarget - dA);
      dB += alpha * (dBTarget - dB);
    }
    const s = bag[i];
    const rep = buildRepresentation({ mA, mB, rho, dA, dB });
    const p = quad(rep.zx[s], rep.zy[s], rep.rho[s]);
    trials.push({ stimulus: s, response: sampleCategorical(p, rng) });
  }
  return trials;
}

function sampleCategorical(p, rng) {
  const u = rng.uniform();
  let cum = 0;
  for (let r = 0; r < 3; r++) {
    cum += p[r];
    if (u < cum) return r;
  }
  return 3;
}

function shuffleInPlace(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng.uniform() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// --------------------------------------------------------------------------- //
// Counts + RT: the LBA / SFT simulator
// --------------------------------------------------------------------------- //
/**
 * Simulate a full identification experiment WITH response times.
 *
 * @param {{zx,zy,rho}} rep
 * @param {number} nPerStimulus
 * @param {string} arch — one of ARCHITECTURES
 * @param {{t0,threshold,kA,kB}} lba
 * @returns {{counts, trials, rtq, probs, trialList, cellRTs}}
 *   trialList: [{stimulus, response, rt}] — the raw trials, for plotting
 *   cellRTs:   [stimulus][response] -> sorted RTs, for the quantile display
 */
export function simulateRT(
  rep,
  nPerStimulus,
  arch,
  lba = LBA_DEFAULTS,
  rng = makeRNG(),
) {
  const { t0, threshold: A, kA, kB } = lba;
  const trialList = [];

  for (let s = 0; s < 4; s++) {
    const zx = rep.zx[s],
      zy = rep.zy[s],
      r = rep.rho[s];
    const sqrt1mr2 = Math.sqrt(Math.max(0, 1 - r * r));

    for (let t = 0; t < nPerStimulus; t++) {
      // ONE perceptual sample -> both the response and the RT
      const z1 = rng.normal();
      const z2 = rng.normal();
      const x = zx + z1;
      const y = zy + r * z1 + sqrt1mr2 * z2;

      // dimensional decisions (bounds at 0)
      const rx = x >= 0 ? 1 : 0;
      const ry = y >= 0 ? 1 : 0;

      // LBA drift rates: distance from the bound, scaled, plus noise
      const vx = Math.max(kA * Math.abs(x) + DRIFT_SD * rng.normal(), 0.05);
      const vy = Math.max(kB * Math.abs(y) + DRIFT_SD * rng.normal(), 0.05);
      const tx = A / vx;
      const ty = A / vy;

      const coin = rng.uniform() < 0.5 ? 0 : 1; // the guess, for self-terminating
      let rt, gx, gy;

      switch (arch) {
        case "serial_exhaustive":
          rt = t0 + tx + ty;
          gx = rx;
          gy = ry;
          break;

        case "serial_self_terminating": {
          const doX = rng.uniform() < 0.5;
          rt = t0 + (doX ? tx : ty);
          gx = doX ? rx : coin;
          gy = doX ? coin : ry;
          break;
        }
        case "parallel_exhaustive":
          rt = t0 + Math.max(tx, ty);
          gx = rx;
          gy = ry;
          break;

        case "parallel_self_terminating": {
          const firstX = tx <= ty;
          rt = t0 + Math.min(tx, ty);
          gx = firstX ? rx : coin;
          gy = firstX ? coin : ry;
          break;
        }
        case "coactive":
          rt = t0 + A / Math.max(vx + vy, 0.05);
          gx = rx;
          gy = ry;
          break;

        default:
          throw new Error(`Unknown architecture "${arch}".`);
      }

      const [lo, hi] = TRAIN_BOUNDS.rtSeconds;
      rt = Math.min(hi, Math.max(lo, rt));
      trialList.push({
        participant: "sim",
        stimulus: s,
        response: 2 * gx + gy,
        rt,
      });
    }
  }

  // Aggregate through the SAME code path the real app uses — so if the
  // aggregation is wrong, the demo breaks too, loudly, instead of the demo
  // quietly working while real uploads silently don't.
  const agg = aggregate(trialList, { hasRT: true });

  const cellRTs = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => []),
  );
  for (const t of trialList) cellRTs[t.stimulus][t.response].push(t.rt);
  cellRTs.forEach((row) => row.forEach((v) => v.sort((a, b) => a - b)));

  return {
    ...agg,
    probs: [0, 1, 2, 3].map((i) => quad(rep.zx[i], rep.zy[i], rep.rho[i])),
    trialList,
    cellRTs,
  };
}

// --------------------------------------------------------------------------- //
// Building a representation from the explorer's sliders
// --------------------------------------------------------------------------- //
/**
 * Turn the teaching controls into the canonical 12 parameters.
 *
 *   mA, mB     sensitivity on each dimension (the d'-like magnitude)
 *   rho        baseline perceptual correlation
 *   dA, dB     SYMMETRIC separability violation: shifts a matched PAIR of
 *              stimuli oppositely depending on the level of the other
 *              dimension (the textbook "PS(A) fails" pattern — zx_0=zx_1 and
 *              zx_2=zx_3 both move apart together).
 *   rhoSpread  INDEPENDENCE violation: gives the two "diagonal" stimuli a
 *              different correlation from the other two.
 *   nudge      {x:[4], y:[4]} — INDEPENDENT per-stimulus offsets, applied on
 *              top of everything above. This is the other, arguably more
 *              common way separability actually fails in real data: one
 *              specific stimulus reads oddly, not a symmetric pair moving in
 *              lockstep. dA/dB and nudge compose — you can have both at once.
 *
 * Stimulus i -> (A-level, B-level) as ai = i<2?0:1, bi = i%2 — the canonical
 * order (s0=A1B1, s1=A1B2, s2=A2B1, s3=A2B2).
 */
export function buildRepresentation({
  mA,
  mB,
  rho,
  dA = 0,
  dB = 0,
  rhoSpread = 0,
  nudge = null,
}) {
  const nx = nudge?.x ?? [0, 0, 0, 0];
  const ny = nudge?.y ?? [0, 0, 0, 0];
  const zx = [],
    zy = [],
    r = [];
  for (let i = 0; i < 4; i++) {
    const ai = i < 2 ? 0 : 1;
    const bi = i % 2;
    zx.push((ai === 0 ? -mA : mA) + (bi === 0 ? -dA / 2 : dA / 2) + nx[i]);
    zy.push((bi === 0 ? -mB : mB) + (ai === 0 ? -dB / 2 : dB / 2) + ny[i]);
    r.push(clampRho(rho + (ai === bi ? rhoSpread : -rhoSpread)));
  }
  return { zx, zy, rho: r };
}

function clampRho(v) {
  return Math.min(0.95, Math.max(-0.95, v));
}

/**
 * Which assumptions does a representation actually satisfy?
 *
 * Checks the BUILT vectors directly (zx/zy/rho), not the slider values that
 * produced them. This matters once nudges exist: dA===0 no longer implies
 * PS(A) holds, because a per-stimulus nudge can break the zx_0=zx_1 /
 * zx_2=zx_3 pairing on its own, and dA!==0 with a compensating nudge could in
 * principle restore it. Checking the actual numbers is the only version of
 * this that can't go stale as new ways to build a representation are added.
 */
export function trueAssumptions({ zx, zy, rho }, tol = 1e-9) {
  const psA = Math.abs(zx[0] - zx[1]) < tol && Math.abs(zx[2] - zx[3]) < tol;
  const psB = Math.abs(zy[0] - zy[2]) < tol && Math.abs(zy[1] - zy[3]) < tol;
  const pi = rho.every((v) => Math.abs(v) < tol);
  const rho1 = rho.every((v) => Math.abs(v - rho[0]) < tol);
  return { psA, psB, pi, rho1 };
}

/** Overall proportion correct implied by a representation (the diagonal). */
export function accuracy(probs) {
  return probs.reduce((a, row, i) => a + row[i], 0) / 4;
}
