/**
 * live.js — run an actual 2x2 identification task and watch the model fit update
 * between trials.
 *
 * This is the paper's amortisation argument made concrete: inference costs a single
 * forward pass, so the perceptual space, the construct probabilities and the posterior
 * widths can all be recomputed after every response rather than after the session.
 * Nothing here is simulated except the stimuli themselves -- the responses are the
 * visitor's, and the fit is the same ONNX graph the packages ship.
 *
 * The stimulus is a rectangle varying in width (dimension A) and height (dimension B),
 * two levels each. Per-trial jitter is what makes the task imperfect: without it a
 * visitor would sit at ceiling, which is precisely the regime the paper shows is
 * least informative about the correlations. The jitter slider therefore doubles as a
 * demonstration of the accuracy-band result.
 */
import { loadModelCached } from "../grin-model.js";
import * as Plot from "../grt-plot.js";

const $ = (id) => document.getElementById(id);

// canonical order: A1B1, A1B2, A2B1, A2B2 (B varies fastest)
const STIM = [
  { a: 0, b: 0, label: "narrow / short" },
  { a: 0, b: 1, label: "narrow / tall" },
  { a: 1, b: 0, label: "wide / short" },
  { a: 1, b: 1, label: "wide / tall" },
];
const BASE = { w: 92, h: 92 };   // px, level 0
const STEP = { w: 30, h: 30 };   // px added at level 1

let model = null;
let counts = [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
];
let trialStim = null;
let running = false;
let nTrials = 0;
let history = [];              // posterior SD after each fit, for the sparkline
let awaiting = false;

// --------------------------------------------------------------------------- //
// Stimulus
// --------------------------------------------------------------------------- //
function jitter() {
  return +$("jitter").value;
}

function drawStimulus(s) {
  const c = $("stim");
  const g = c.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = 260;
  c.width = W * dpr; c.height = H * dpr;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  // Gaussian jitter on each dimension independently. This is the perceptual noise
  // the GRT model is meant to recover, so it is applied to the RENDERED size rather
  // than to the response: the participant genuinely sees an ambiguous stimulus.
  const j = jitter();
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const w = BASE.w + s.a * STEP.w + gauss() * j;
  const h = BASE.h + s.b * STEP.h + gauss() * j;

  const css = getComputedStyle(document.documentElement);
  g.fillStyle = css.getPropertyValue("--slate").trim() || "#456";
  const x = (W - w) / 2, y = (H - h) / 2;
  const r = 6;
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
  g.fill();
}

function nextTrial() {
  trialStim = Math.floor(Math.random() * 4);
  drawStimulus(STIM[trialStim]);
  awaiting = true;
  $("prompt").textContent = "Which was it?";
}

// --------------------------------------------------------------------------- //
// Response
// --------------------------------------------------------------------------- //
async function respond(r) {
  if (!running || !awaiting) return;
  awaiting = false;
  counts[trialStim][r] += 1;
  nTrials += 1;

  const btn = $(`resp-${r}`);
  if (btn) {
    btn.classList.add("flash");
    setTimeout(() => btn.classList.remove("flash"), 140);
  }

  await refit();
  if (nTrials % 4 === 0) checkStop();
  setTimeout(nextTrial, 90);
}

// --------------------------------------------------------------------------- //
// Fit
// --------------------------------------------------------------------------- //
async function refit() {
  const trials = counts.map((row) => row.reduce((a, b) => a + b, 0));
  $("n-trials").textContent = nTrials;

  // The network needs some data in every row before its output means anything;
  // showing a fit from three trials would be theatre rather than demonstration.
  if (trials.some((t) => t < 2)) {
    $("fit-status").textContent =
      "Collecting — every stimulus needs at least a couple of trials.";
    return;
  }

  const t0 = performance.now();
  const out = await model.predict({ counts, trials });
  const ms = performance.now() - t0;

  Plot.renderSpace($("space"), {
    stimuli: Plot.toStimuli(out.mean),
    showMarginals: false,
    title: null,
  });

  const sd = out.std;
  const meanSD = sd.reduce((a, b) => a + b, 0) / sd.length;
  history.push(meanSD);
  drawSparkline(history);

  const pct = (x) => `${Math.round(100 * x)}%`;
  $("constructs").innerHTML = `
    <div class="est"><span>Perceptual independence</span><strong>${pct(out.corr.pi)}</strong></div>
    <div class="est"><span>Separability, dimension A</span><strong>${pct(out.sep.A)}</strong></div>
    <div class="est"><span>Separability, dimension B</span><strong>${pct(out.sep.B)}</strong></div>`;

  $("fit-status").innerHTML =
    `Refit in <strong>${ms.toFixed(1)} ms</strong> · mean posterior SD ` +
    `<strong>${meanSD.toFixed(3)}</strong>`;

  updateAccuracy();
}

// Observed per-dimension accuracy, and where it sits relative to the band the
// identifiability analysis identifies. This is the design result from the paper,
// shown live on the visitor's own data.
function updateAccuracy() {
  let aOK = 0, bOK = 0, n = 0;
  for (let s = 0; s < 4; s++) {
    for (let r = 0; r < 4; r++) {
      const c = counts[s][r];
      if (!c) continue;
      n += c;
      if (s >> 1 === r >> 1) aOK += c;
      if ((s & 1) === (r & 1)) bOK += c;
    }
  }
  if (!n) return;
  const acc = 0.5 * (aOK / n + bOK / n);
  const inBand = acc >= 0.6 && acc <= 0.8;
  $("accuracy").innerHTML =
    `Per-dimension accuracy <strong>${Math.round(100 * acc)}%</strong> ` +
    (inBand
      ? `<span class="pill ok">in the informative band</span>`
      : acc > 0.8
        ? `<span class="pill warn">above the band — correlations get harder to see</span>`
        : `<span class="pill warn">below the band — sensitivities get hard to pin down</span>`);
}

function drawSparkline(vals) {
  const c = $("spark");
  if (!c || vals.length < 2) return;
  const g = c.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = 54;
  c.width = W * dpr; c.height = H * dpr;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  const css = getComputedStyle(document.documentElement);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;
  g.strokeStyle = css.getPropertyValue("--slate").trim() || "#456";
  g.lineWidth = 1.8;
  g.beginPath();
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * (W - 4) + 2;
    const y = H - 4 - ((v - lo) / span) * (H - 10);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();

  const target = +$("sd-target").value;
  if (target >= lo && target <= hi) {
    const y = H - 4 - ((target - lo) / span) * (H - 10);
    g.strokeStyle = css.getPropertyValue("--rose-deep").trim() || "#c86a93";
    g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
    g.setLineDash([]);
  }
}

// --------------------------------------------------------------------------- //
// Stopping rule — the same criterion the packages expose, evaluated in-task
// --------------------------------------------------------------------------- //
function checkStop() {
  if (!history.length) return;
  const target = +$("sd-target").value;
  const cur = history[history.length - 1];
  if (cur <= target) {
    running = false;
    awaiting = false;
    $("prompt").textContent = "Target reached.";
    $("stop-banner").hidden = false;
    $("stop-banner").innerHTML =
      `<h4>Stopped after ${nTrials} trials</h4>
       <p>Mean posterior SD reached ${cur.toFixed(3)}, at or below your target of
       ${target.toFixed(2)}. A fixed design would have had to guess this number in
       advance, and guess it for the least precise participant in the sample.</p>`;
    $("start").textContent = "Run again";
    $("start").disabled = false;
  }
}

// --------------------------------------------------------------------------- //
// Wiring
// --------------------------------------------------------------------------- //
function reset() {
  counts = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  nTrials = 0;
  history = [];
  $("stop-banner").hidden = true;
  $("constructs").innerHTML = "";
  $("accuracy").textContent = "";
  $("fit-status").textContent = "";
  $("n-trials").textContent = "0";
  const sp = $("spark");
  if (sp) sp.getContext("2d").clearRect(0, 0, sp.width, sp.height);
}

async function start() {
  $("start").disabled = true;
  $("start").textContent = "Loading model…";
  try {
    if (!model) model = await loadModelCached("./assets/models/cm", (m) => {
      $("start").textContent = m || "Loading model…";
    });
  } catch (e) {
    $("start").textContent = "Start";
    $("start").disabled = false;
    $("fit-status").innerHTML =
      `<span class="pill bad">Couldn't load the model (${e.message}).</span>`;
    return;
  }
  reset();
  running = true;
  $("start").textContent = "Running";
  $("task").hidden = false;
  nextTrial();
}

$("start").addEventListener("click", start);
[0, 1, 2, 3].forEach((r) => {
  const b = $(`resp-${r}`);
  if (b) b.addEventListener("click", () => respond(r));
});
$("sd-target").addEventListener("input", () => {
  $("sd-target-val").textContent = (+$("sd-target").value).toFixed(2);
  drawSparkline(history);
});
$("jitter").addEventListener("input", () => {
  $("jitter-val").textContent = $("jitter").value;
});

// keyboard: 1-4 map to the four responses, in the on-screen order
window.addEventListener("keydown", (e) => {
  const i = ["1", "2", "3", "4"].indexOf(e.key);
  if (i >= 0) respond(i);
});
