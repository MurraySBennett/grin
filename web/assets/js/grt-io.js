/**
 * grt-io.js — getting the user's data into the exact shape the networks were
 * trained on.
 *
 * This module is the contract boundary. If it is wrong, the networks are being
 * fed off-distribution input and everything downstream is quietly meaningless,
 * so it mirrors `src/data/rt_lba_generator.py` line for line:
 *
 *   counts (16)  row-major 4x4, rows = stimuli, cols = responses, canonical order
 *   trials (4)   row sums
 *   rtq    (80)  4 stimuli x 4 responses x 5 quantiles [.1 .3 .5 .7 .9],
 *                RAW SECONDS, nearest-rank, 0.0 for empty cells
 *
 * Two accepted input formats:
 *
 *   LONG (preferred — what people actually collect)
 *     participant,stimulus,response,rt
 *     p01,A1B1,A1B1,0.612
 *   `participant` and `rt` are optional. Omit `rt` -> counts-only model.
 *
 *   MATRIX (a 4x4 block of counts, optionally with header row / label column)
 *   RT is impossible in this format; counts-only model.
 *
 * ES module. Depends on grt-core.js for the canonical order only.
 */

import { STIMULUS_ORDER } from "./grt-core.js";

export const QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9];
export const N_Q = QUANTILES.length;

/** Training-distribution bounds, from src/config.py + rt_lba_generator.py. */
export const TRAIN_BOUNDS = {
  trialsPerStimulus: [1, 1000], // config.TRIAL_RANGE
  rtSeconds: [0.1, 10.0], // generator clips to this
};

// --------------------------------------------------------------------------- //
// np.rint — round half to EVEN. Math.round rounds half UP and would disagree
// with the training-time quantile index whenever q*(k-1) lands on a half.
// --------------------------------------------------------------------------- //
export function rint(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1; // exact .5 -> nearest even
}

// --------------------------------------------------------------------------- //
// Label handling
// --------------------------------------------------------------------------- //
/**
 * Resolve a stimulus/response token to a canonical index 0..3.
 * Accepts: 0-3 (or 1-4), "A1B1"-style codes, or "Happy/Male"-style pairs
 * (delimiters / _ - | and whitespace), given a discovered level map.
 */
function splitPair(tok) {
  const t = String(tok).trim();
  const m = t.match(/^([A-Za-z]+)(\d)\s*[\/_\-|]?\s*([A-Za-z]+)(\d)$/); // A1B1
  if (m) return [`${m[1]}${m[2]}`, `${m[3]}${m[4]}`];
  const parts = t
    .split(/[\/_\-|]/)
    .map((s) => s.trim())
    .filter(Boolean);
  return parts.length === 2 ? parts : null;
}

/**
 * Scan all tokens and build a level map: which A-levels and B-levels exist,
 * in order of first appearance, and how each token maps to index 0..3.
 * Returns null (with reason) if the tokens don't form a clean 2x2.
 */
export function buildLevelMap(tokens) {
  const uniq = [...new Set(tokens.map((t) => String(t).trim()))].filter(
    Boolean,
  );

  // Case 1: plain integer codes 0-3 or 1-4 -> assume canonical order already
  const ints = uniq.map((t) => Number(t));
  if (ints.every((v) => Number.isInteger(v))) {
    const lo = Math.min(...ints);
    const hi = Math.max(...ints);
    if (lo >= 0 && hi <= 3) {
      return {
        kind: "index",
        offset: 0,
        levels: null,
        map: Object.fromEntries(uniq.map((t) => [t, Number(t)])),
      };
    }
    if (lo >= 1 && hi <= 4) {
      return {
        kind: "index",
        offset: 1,
        levels: null,
        map: Object.fromEntries(uniq.map((t) => [t, Number(t) - 1])),
      };
    }
    return {
      error: `Integer stimulus codes must be 0-3 or 1-4; found ${lo}-${hi}.`,
    };
  }

  // Case 2: paired labels
  const pairs = uniq.map((t) => [t, splitPair(t)]);
  const bad = pairs.filter(([, p]) => !p);
  if (bad.length)
    return {
      error:
        `Could not split "${bad[0][0]}" into two dimension levels. ` +
        `Use a code like "A1B1", "Happy/Male", or an integer 0-3.`,
    };

  const A = [],
    B = [];
  for (const [, [a, b]] of pairs) {
    if (!A.includes(a)) A.push(a);
    if (!B.includes(b)) B.push(b);
  }
  if (A.length !== 2 || B.length !== 2)
    return {
      error:
        `Expected exactly 2 levels per dimension; found ${A.length} ` +
        `on dimension A (${A.join(", ")}) and ${B.length} on dimension B (${B.join(", ")}).`,
    };

  A.sort();
  B.sort(); // deterministic: A1 before A2, "Happy" before "Sad"
  const map = {};
  for (const [t, [a, b]] of pairs) map[t] = 2 * A.indexOf(a) + B.indexOf(b);
  return { kind: "labels", levels: { A, B }, map };
}

// --------------------------------------------------------------------------- //
// CSV parsing
// --------------------------------------------------------------------------- //
function splitCSV(text) {
  return text
    .replace(/^\uFEFF/, "") // strip BOM
    .trim()
    .split(/\r?\n/)
    .filter((r) => r.trim().length)
    .map((r) => r.split(/[,\t;]/).map((v) => v.trim()));
}

const HEADER_ALIASES = {
  participant: ["participant", "subject", "subj", "id", "pid", "sub"],
  stimulus: ["stimulus", "stim", "signal", "presented", "target"],
  response: ["response", "resp", "answer", "choice", "reported"],
  rt: [
    "rt",
    "response_time",
    "responsetime",
    "reaction_time",
    "reactiontime",
    "latency",
    "time",
  ],
};

function matchHeader(cells) {
  const lower = cells.map((c) => c.toLowerCase().replace(/\s+/g, "_"));
  const idx = {};
  for (const [key, aliases] of Object.entries(HEADER_ALIASES)) {
    const i = lower.findIndex((c) => aliases.includes(c));
    if (i >= 0) idx[key] = i;
  }
  return idx;
}

/**
 * Parse a CSV into one of:
 *   { format: "long",   trials: [{participant, stimulus, response, rt}], hasRT, levels }
 *   { format: "matrix", counts: number[4][4], levels }
 *   { error: "..." }
 */
export function parseCSV(text) {
  const rows = splitCSV(text);
  if (!rows.length) return { error: "The file is empty." };

  const idx = matchHeader(rows[0]);
  const isLong = idx.stimulus !== undefined && idx.response !== undefined;

  if (isLong) return parseLong(rows, idx);
  return parseMatrix(rows);
}

function parseLong(rows, idx) {
  const body = rows.slice(1);
  if (!body.length) return { error: "The file has a header but no data rows." };

  const hasRT = idx.rt !== undefined;
  const hasPid = idx.participant !== undefined;

  const stimTokens = body.map((r) => r[idx.stimulus]);
  const respTokens = body.map((r) => r[idx.response]);
  const lm = buildLevelMap([...stimTokens, ...respTokens]);
  if (lm.error) return { error: lm.error };

  const trials = [];
  const problems = [];
  body.forEach((r, i) => {
    const line = i + 2; // 1-indexed, +1 for header
    const s = lm.map[String(r[idx.stimulus] ?? "").trim()];
    const q = lm.map[String(r[idx.response] ?? "").trim()];
    if (s === undefined || q === undefined) {
      if (problems.length < 5)
        problems.push(
          `line ${line}: unrecognised stimulus/response "${r[idx.stimulus]}"/"${r[idx.response]}"`,
        );
      return;
    }
    let rt = null;
    if (hasRT) {
      rt = Number(r[idx.rt]);
      if (!Number.isFinite(rt)) {
        if (problems.length < 5)
          problems.push(`line ${line}: RT "${r[idx.rt]}" is not a number`);
        return;
      }
    }
    trials.push({
      participant: hasPid ? String(r[idx.participant]).trim() : "all",
      stimulus: s,
      response: q,
      rt,
    });
  });

  if (!trials.length)
    return { error: "No usable rows. " + problems.join("; ") };

  return {
    format: "long",
    trials,
    hasRT,
    hasParticipant: hasPid,
    levels: lm.levels,
    warnings: problems.length
      ? [`${problems.length}+ rows were skipped: ${problems.join("; ")}`]
      : [],
  };
}

function parseMatrix(rows) {
  const isNum = (v) => v !== "" && !isNaN(Number(v));
  let counts = null,
    headers = null;

  // plain 4x4
  if (
    rows.length === 4 &&
    rows.every((r) => r.length === 4 && r.every(isNum))
  ) {
    counts = rows.map((r) => r.map(Number));
  }
  // header row + 4 data rows
  else if (
    rows.length === 5 &&
    rows[0].length === 4 &&
    rows.slice(1).every((r) => r.length === 4)
  ) {
    headers = rows[0];
    counts = rows.slice(1).map((r) => r.map((v) => Number(v) || 0));
  }
  // corner + header row + label column
  else if (
    rows.length === 5 &&
    rows[0].length === 5 &&
    rows.slice(1).every((r) => r.length === 5)
  ) {
    headers = rows[0].slice(1);
    counts = rows.slice(1).map((r) => r.slice(1).map((v) => Number(v) || 0));
  } else {
    return {
      error:
        "Could not read this file. Expected either a long/tidy file with " +
        "`stimulus`,`response` (and optionally `rt`,`participant`) columns, " +
        "or a plain 4x4 matrix of counts.",
    };
  }

  let levels = null;
  if (headers) {
    const lm = buildLevelMap(headers);
    if (!lm.error && lm.levels) levels = lm.levels;
    // NB: if headers are present we trust the file's column ORDER, not the
    // header text, to avoid silently permuting the matrix.
  }
  return { format: "matrix", counts, levels, hasRT: false, warnings: [] };
}

// --------------------------------------------------------------------------- //
// Aggregation: trials -> the exact network inputs
// --------------------------------------------------------------------------- //
/**
 * Aggregate a list of trials for ONE participant into network inputs.
 *
 * Mirrors rt_lba_generator._simulate_group exactly:
 *   - RTs clipped to [0.1, 10] s
 *   - per (stimulus, response) cell: sort ascending, take index rint(q*(k-1))
 *   - empty cells -> quantiles are all 0.0
 *
 * @returns {{counts:number[][], trials:number[], rtq:number[]|null}}
 */
export function aggregate(trials, { hasRT = false } = {}) {
  const counts = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  const cellRTs = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => []),
  );

  for (const t of trials) {
    counts[t.stimulus][t.response] += 1;
    if (hasRT && t.rt !== null) {
      const clipped = Math.min(
        TRAIN_BOUNDS.rtSeconds[1],
        Math.max(TRAIN_BOUNDS.rtSeconds[0], t.rt),
      );
      cellRTs[t.stimulus][t.response].push(clipped);
    }
  }
  const rowTrials = counts.map((r) => r.reduce((a, b) => a + b, 0));

  if (!hasRT) return { counts, trials: rowTrials, rtq: null };

  const rtq = new Array(4 * 4 * N_Q).fill(0);
  for (let s = 0; s < 4; s++) {
    for (let r = 0; r < 4; r++) {
      const v = cellRTs[s][r];
      if (!v.length) continue; // stays 0.0, as in training
      v.sort((a, b) => a - b);
      const k = v.length;
      for (let qi = 0; qi < N_Q; qi++) {
        const j = Math.min(k - 1, Math.max(0, rint(QUANTILES[qi] * (k - 1))));
        rtq[(s * 4 + r) * N_Q + qi] = v[j];
      }
    }
  }
  return { counts, trials: rowTrials, rtq };
}

/** Split parsed long-format trials by participant. Returns a Map. */
export function byParticipant(parsed) {
  const m = new Map();
  for (const t of parsed.trials) {
    if (!m.has(t.participant)) m.set(t.participant, []);
    m.get(t.participant).push(t);
  }
  return m;
}

// --------------------------------------------------------------------------- //
// Validation / out-of-distribution checks
//
// The networks are amortized over the prior they were trained on. Input outside
// that support gets a confident answer that means nothing, so we say so loudly.
// --------------------------------------------------------------------------- //
/**
 * @param {{counts, trials, rtq}} agg
 * @param {{hasRT:boolean, rawRTs?:number[]}} opts
 * @returns {{errors:string[], warnings:string[], notes:string[]}}
 */
export function checkInputs(agg, { hasRT = false, rawRTs = [] } = {}) {
  const errors = [],
    warnings = [],
    notes = [];
  const [tMin, tMax] = TRAIN_BOUNDS.trialsPerStimulus;

  const total = agg.trials.reduce((a, b) => a + b, 0);
  if (total === 0) errors.push("The matrix is empty — no trials at all.");

  agg.trials.forEach((n, s) => {
    if (n === 0)
      errors.push(
        `Stimulus ${STIMULUS_ORDER[s]} has no trials. GRIN needs all four stimuli.`,
      );
    else if (n < tMin)
      warnings.push(`Stimulus ${STIMULUS_ORDER[s]} has only ${n} trial(s).`);
    else if (n > tMax)
      warnings.push(
        `Stimulus ${STIMULUS_ORDER[s]} has ${n} trials, above the ${tMax} the network ` +
          `was trained on. Estimates should still be fine (more data is easier), but the ` +
          `uncertainty may be slightly conservative.`,
      );
  });

  // accuracy sanity: below-chance identification is not in the prior (signs are
  // fixed to the design), so it will be mis-fit rather than flagged by the net.
  if (total > 0) {
    const correct = [0, 1, 2, 3].reduce((a, s) => a + agg.counts[s][s], 0);
    const acc = correct / total;
    if (acc < 0.25)
      warnings.push(
        `Overall accuracy is ${(100 * acc).toFixed(0)}%, below chance (25%). GRIN's prior ` +
          `assumes correctly-signed levels, so check that your stimulus and response ` +
          `columns are not swapped or mislabelled.`,
      );
    notes.push(`${total} trials, ${(100 * acc).toFixed(1)}% correct.`);
  }

  if (hasRT) {
    const [rMin, rMax] = TRAIN_BOUNDS.rtSeconds;
    const finite = rawRTs.filter(Number.isFinite);
    if (finite.length) {
      const sorted = [...finite].sort((a, b) => a - b);
      const med = sorted[Math.floor(sorted.length / 2)];
      // Millisecond detection: a median RT of 600 is not 600 seconds.
      if (med > 20)
        errors.push(
          `Median RT is ${med.toFixed(0)}, which looks like MILLISECONDS. GRIN's RT model ` +
            `was trained on SECONDS. Divide your RT column by 1000 and try again.`,
        );
      const nLow = finite.filter((v) => v < rMin).length;
      const nHigh = finite.filter((v) => v > rMax).length;
      if (nLow || nHigh)
        warnings.push(
          `${nLow + nHigh} RT(s) fell outside the trained range [${rMin}, ${rMax}] s and ` +
            `were clipped to it (as the training simulator does).`,
        );
      notes.push(`RT median ${med.toFixed(3)} s.`);
    }
    // sparse cells make the RT quantiles noisy; the net saw this during training
    // (cells with 0 trials were zeroed) but it's worth telling the user.
    let sparse = 0;
    for (let s = 0; s < 4; s++)
      for (let r = 0; r < 4; r++)
        if (agg.counts[s][r] > 0 && agg.counts[s][r] < 5) sparse++;
    if (sparse)
      warnings.push(
        `${sparse} of 16 cells have fewer than 5 trials, so their RT quantiles are very ` +
          `noisy. The architecture inference in particular will be uncertain.`,
      );
  }

  return { errors, warnings, notes };
}

// --------------------------------------------------------------------------- //
// Template / export helpers
// --------------------------------------------------------------------------- //
export function templateCSV({ withRT = true } = {}) {
  const head = withRT
    ? "participant,stimulus,response,rt"
    : "participant,stimulus,response";
  const rows = [head];
  for (const s of STIMULUS_ORDER)
    rows.push(withRT ? `p01,${s},${s},0.650` : `p01,${s},${s}`);
  rows.push(
    "# one row per TRIAL. stimulus/response: A1B1|A1B2|A2B1|A2B2, your own",
  );
  rows.push(
    "# labels like Happy/Male, or integers 0-3. rt in SECONDS. participant",
  );
  rows.push("# and rt are optional; drop rt to use the counts-only model.");
  return rows.join("\n") + "\n";
}

export function countsToCSV(counts, levels = null) {
  const lab = levels
    ? [0, 1, 2, 3].map((i) => `${levels.A[i < 2 ? 0 : 1]}/${levels.B[i % 2]}`)
    : STIMULUS_ORDER;
  const rows = [["", ...lab].join(",")];
  counts.forEach((row, i) => rows.push([lab[i], ...row].join(",")));
  return rows.join("\n") + "\n";
}
