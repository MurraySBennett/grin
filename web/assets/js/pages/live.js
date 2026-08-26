/**
 * live.js — a short 2x2 identification task, with the model refitting between trials.
 *
 * This is the paper's amortisation argument made concrete: inference costs one forward
 * pass, so the perceptual space, the construct probabilities and the posterior widths
 * can be recomputed after every response rather than after the session. Nothing is
 * simulated except the stimulus; the responses are the visitor's.
 *
 * DESIGN NOTES, because a demo that misrepresents the method is worse than no demo.
 *
 *  * Response keys are E / F / J / I, laid out to match the stimulus grid: the key's
 *    VERTICAL position codes dimension A (E, I = high; F, J = low) and its HORIZONTAL
 *    position codes dimension B (E, F = low; J, I = high). This is the mapping used in
 *    the lab, and it matters here because an arbitrary mapping adds response-selection
 *    noise that GRT would then attribute to perception.
 *  * Errors are produced by brief presentation and a response deadline, not by making
 *    the stimuli trivially similar. A speeded task with a real deadline is how an
 *    identification experiment normally lands in the 60-85% range the paper shows is
 *    informative; simply blurring the stimuli produces the same accuracy through a
 *    different mechanism.
 *  * Timeouts are DISCARDED rather than scored as errors. A missed deadline is not a
 *    confusion, and folding it into the matrix would bias every parameter.
 *  * The instruction screen shows all four stimuli. Identification assumes the observer
 *    knows the response set; without that the early trials measure learning instead.
 */
import { loadModelCached } from "../grin-model.js";
import * as Plot from "../grt-plot.js";

const $ = (id) => document.getElementById(id);

// canonical order: A1B1, A1B2, A2B1, A2B2 -- B varies fastest
const STIM = [
  { a: 0, b: 0, key: "f", name: "low / low" },
  { a: 0, b: 1, key: "j", name: "low / high" },
  { a: 1, b: 0, key: "e", name: "high / low" },
  { a: 1, b: 1, key: "i", name: "high / high" },
];
const KEY_TO_STIM = Object.fromEntries(STIM.map((s, i) => [s.key, i]));

// Two stimulus sets. Colour is the more conventional GRT pairing (saturation and hue
// are separable-ish dimensions of one object); size is included because some visitors
// find it easier to see what the task is asking.
const SETS = {
  colour: {
    label: "Colour patch — saturation × hue",
    dimA: "saturation",
    dimB: "hue",
    draw(g, W, H, a, b, noise) {
      const sat = (a ? 62 : 34) + noise * 6;
      const hue = (b ? 38 : 8) + noise * 5;
      g.fillStyle = `hsl(${hue} ${sat}% 52%)`;
      const r = Math.min(W, H) * 0.3;
      g.beginPath();
      g.arc(W / 2, H / 2, r, 0, Math.PI * 2);
      g.fill();
    },
  },
  size: {
    label: "Rectangle — width × height",
    dimA: "width",
    dimB: "height",
    draw(g, W, H, a, b, noise) {
      const w = (a ? 122 : 92) + noise * 7;
      const h = (b ? 122 : 92) + noise * 7;
      const css = getComputedStyle(document.documentElement);
      g.fillStyle = css.getPropertyValue("--slate").trim() || "#456";
      g.fillRect((W - w) / 2, (H - h) / 2, w, h);
    },
  },
};

let model = null;
let counts = zeros();
let stimIdx = null;
let phase = "idle";        // idle | fixation | stimulus | response | feedback | done
let nTrials = 0, nTimeouts = 0;
let history = [];
let deadlineTimer = null, stimTimer = null;

function zeros() {
  return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
}
function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function setName() { return $("stim-set").value; }
function stimMs() { return +$("stim-ms").value; }
function deadlineMs() { return +$("deadline-ms").value; }

// --------------------------------------------------------------------------- //
// Drawing
// --------------------------------------------------------------------------- //
function ctx(canvas, H) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 320;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  return { g, W, H };
}

function drawFixation() {
  const { g, W, H } = ctx($("stim"), 260);
  const css = getComputedStyle(document.documentElement);
  g.strokeStyle = css.getPropertyValue("--steel").trim() || "#888";
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(W / 2 - 9, H / 2); g.lineTo(W / 2 + 9, H / 2);
  g.moveTo(W / 2, H / 2 - 9); g.lineTo(W / 2, H / 2 + 9);
  g.stroke();
}

function drawStimulus(i) {
  const { g, W, H } = ctx($("stim"), 260);
  SETS[setName()].draw(g, W, H, STIM[i].a, STIM[i].b, gauss());
}

function drawBlank() { ctx($("stim"), 260); }

// --------------------------------------------------------------------------- //
// Trial loop
// --------------------------------------------------------------------------- //
function runTrial() {
  if (phase === "done") return;
  phase = "fixation";
  drawFixation();
  $("prompt").textContent = "";
  clearTimeout(stimTimer);
  stimTimer = setTimeout(() => {
    stimIdx = Math.floor(Math.random() * 4);
    phase = "stimulus";
    drawStimulus(stimIdx);
    setTimeout(() => {
      if (phase !== "stimulus") return;
      drawBlank();
      phase = "response";
      $("prompt").textContent = "Which one?";
      clearTimeout(deadlineTimer);
      deadlineTimer = setTimeout(onTimeout, deadlineMs());
    }, stimMs());
  }, 420);
}

function onTimeout() {
  if (phase !== "response") return;
  phase = "feedback";
  nTimeouts += 1;
  $("prompt").innerHTML = `<span class="pill warn">too slow</span>`;
  $("timeouts").textContent = nTimeouts;
  setTimeout(runTrial, 420);
}

async function respond(r) {
  if (phase !== "response") return;
  clearTimeout(deadlineTimer);
  phase = "feedback";

  counts[stimIdx][r] += 1;
  nTrials += 1;
  $("n-trials").textContent = nTrials;

  const btn = $(`resp-${r}`);
  if (btn) { btn.classList.add("flash"); setTimeout(() => btn.classList.remove("flash"), 130); }

  await refit();
  if (phase === "done") return;
  setTimeout(runTrial, 220);
}

// --------------------------------------------------------------------------- //
// Fit
// --------------------------------------------------------------------------- //
async function refit() {
  const trials = counts.map((r) => r.reduce((a, b) => a + b, 0));
  if (trials.some((t) => t < 3)) {
    $("fit-status").textContent =
      "Collecting — each of the four stimuli needs a few trials before a fit means anything.";
    return;
  }

  const t0 = performance.now();
  const out = await model.predict({ counts, trials });
  const ms = performance.now() - t0;

  Plot.renderSpace($("space"), {
    stimuli: Plot.toStimuli(out.mean),
    showMarginals: false,
  });

  const meanSD = out.std.reduce((a, b) => a + b, 0) / out.std.length;
  history.push(meanSD);
  drawSparkline();

  const pct = (x) => `${Math.round(100 * x)}%`;
  $("constructs").innerHTML = `
    <div class="est"><span>Perceptual independence</span><strong>${pct(out.corr.pi)}</strong></div>
    <div class="est"><span>Separability, ${SETS[setName()].dimA}</span><strong>${pct(out.sep.A)}</strong></div>
    <div class="est"><span>Separability, ${SETS[setName()].dimB}</span><strong>${pct(out.sep.B)}</strong></div>`;
  $("fit-status").innerHTML =
    `Refit in <strong>${ms.toFixed(1)} ms</strong> · mean posterior SD <strong>${meanSD.toFixed(3)}</strong>`;

  updateAccuracy();
  checkStop(meanSD);
}

function updateAccuracy() {
  let aOK = 0, bOK = 0, n = 0;
  for (let s = 0; s < 4; s++) for (let r = 0; r < 4; r++) {
    const c = counts[s][r];
    if (!c) continue;
    n += c;
    if (s >> 1 === r >> 1) aOK += c;
    if ((s & 1) === (r & 1)) bOK += c;
  }
  if (!n) return;
  const acc = 0.5 * (aOK / n + bOK / n);
  $("accuracy").innerHTML =
    `Per-dimension accuracy <strong>${Math.round(100 * acc)}%</strong> ` +
    (acc >= 0.6 && acc <= 0.85
      ? `<span class="pill ok">informative range</span>`
      : acc > 0.85
        ? `<span class="pill warn">near ceiling — correlations get hard to see</span>`
        : `<span class="pill warn">low — sensitivities get hard to pin down</span>`);
}

function drawSparkline() {
  if (history.length < 2) return;
  const { g, W, H } = ctx($("spark"), 54);
  const css = getComputedStyle(document.documentElement);
  const lo = Math.min(...history), hi = Math.max(...history);
  const span = hi - lo || 1;
  g.strokeStyle = css.getPropertyValue("--slate").trim() || "#456";
  g.lineWidth = 1.8;
  g.beginPath();
  history.forEach((v, i) => {
    const x = (i / (history.length - 1)) * (W - 4) + 2;
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

function checkStop(meanSD) {
  const target = +$("sd-target").value;
  if (meanSD > target) return;
  phase = "done";
  clearTimeout(deadlineTimer); clearTimeout(stimTimer);
  drawBlank();
  $("prompt").textContent = "";
  $("stop-banner").hidden = false;
  $("stop-banner").innerHTML =
    `<h4>Stopped after ${nTrials} scored trials</h4>
     <p>Mean posterior SD reached ${meanSD.toFixed(3)}, at or below the target of
     ${target.toFixed(2)}${nTimeouts ? `. ${nTimeouts} trial${nTimeouts === 1 ? " was" : "s were"} discarded for missing the deadline` : ""}.
     A fixed design would have had to choose this trial count in advance, for the least
     precise participant in the sample.</p>`;
  $("start").textContent = "Run again";
  $("start").disabled = false;
}

// --------------------------------------------------------------------------- //
// Instructions and wiring
// --------------------------------------------------------------------------- //
function renderInstructions() {
  const set = SETS[setName()];
  $("legend-title").textContent =
    `${set.label} — press the key under each option`;
  const cells = [2, 3, 0, 1];   // high/low, high/high, low/low, low/high -> visual grid
  $("legend").innerHTML = cells.map((i) => {
    const s = STIM[i];
    return `<div class="legend-cell">
      <canvas class="legend-canvas" data-stim="${i}" height="76"></canvas>
      <div><kbd>${s.key.toUpperCase()}</kbd> <span class="cap">${set.dimA} ${s.a ? "high" : "low"},
      ${set.dimB} ${s.b ? "high" : "low"}</span></div>
    </div>`;
  }).join("");
  $("legend").querySelectorAll(".legend-canvas").forEach((c) => {
    const i = +c.dataset.stim;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth || 110, H = 76;
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    // no noise in the legend: this is the canonical form of each stimulus
    SETS[setName()].draw(g, W, H, STIM[i].a, STIM[i].b, 0);
  });
  // response buttons mirror the same grid
  cells.forEach((i) => {
    const b = $(`resp-${i}`);
    if (b) b.innerHTML = `${STIM[i].name} <kbd>${STIM[i].key.toUpperCase()}</kbd>`;
  });
}

function reset() {
  counts = zeros(); nTrials = 0; nTimeouts = 0; history = [];
  $("stop-banner").hidden = true;
  $("constructs").innerHTML = "";
  $("accuracy").textContent = "";
  $("fit-status").textContent = "";
  $("n-trials").textContent = "0";
  $("timeouts").textContent = "0";
  ctx($("spark"), 54);
}

async function start() {
  $("start").disabled = true;
  $("start").textContent = "Loading model…";
  try {
    if (!model) model = await loadModelCached("./assets/models/cm",
      (m) => { $("start").textContent = m || "Loading model…"; });
  } catch (e) {
    $("start").textContent = "Start";
    $("start").disabled = false;
    $("fit-status").innerHTML = `<span class="pill bad">Couldn't load the model (${e.message}).</span>`;
    return;
  }
  reset();
  $("start").textContent = "Running";
  $("task").hidden = false;
  runTrial();
}

$("start").addEventListener("click", start);
$("stim-set").addEventListener("change", renderInstructions);
[0, 1, 2, 3].forEach((i) => {
  const b = $(`resp-${i}`);
  if (b) b.addEventListener("click", () => respond(i));
});
["stim-ms", "deadline-ms", "sd-target"].forEach((id) => {
  const el = $(id);
  if (el) el.addEventListener("input", () => {
    $(`${id}-val`).textContent = el.value;
    if (id === "sd-target") drawSparkline();
  });
});
window.addEventListener("keydown", (e) => {
  const i = KEY_TO_STIM[e.key.toLowerCase()];
  if (i !== undefined) { e.preventDefault(); respond(i); }
});

renderInstructions();
