/**
 * live.js — a short 2x2 identification task run as a real experiment, with the model
 * refitting between blocks.
 *
 * Structure: configure -> overlay opens -> welcome -> instructions -> blocks of trials,
 * with a diagnostics break between each -> results -> optional handoff to Analyse.
 *
 * DESIGN NOTES, because a demo that misrepresents the method is worse than no demo.
 *
 *  * DIAGNOSTICS ARE HIDDEN DURING TRIALS BY DEFAULT. A live-updating plot beside a
 *    200 ms stimulus is an attention confound, and showing a participant their own
 *    fit mid-task is performance feedback that changes behaviour. The fit is shown at
 *    block breaks, which is a natural pause and is also where an adaptive design would
 *    actually make its stopping decision. A toggle can force it on for demonstration;
 *    it says plainly that this is not how you would run the task.
 *  * Response keys E / F / J / I map to high-low, low-low, low-high, high-high, laid
 *    out so the key's row codes dimension A and its column dimension B. An arbitrary
 *    mapping adds response-selection noise GRT would read as perceptual.
 *  * Errors come from brief presentation and a response deadline, not from making the
 *    stimuli nearly identical -- a speeded task is how identification experiments
 *    normally land in the informative accuracy range.
 *  * Timeouts are DISCARDED, not scored as errors. A missed deadline is not a
 *    confusion; folding it into the matrix would bias every parameter.
 *  * Trial-level data is kept so it can be handed to the Analyse page in the same
 *    long format the tutorials document.
 */
import { loadModelCached } from "../grin-model.js";
import * as Plot from "../grt-plot.js";
import * as IO from "../grt-io.js";
import * as Store from "../session-store.js";

const $ = (id) => document.getElementById(id);

// canonical order: A1B1, A1B2, A2B1, A2B2 -- B varies fastest
const STIM = [
  { a: 0, b: 0, key: "f", name: "low / low" },
  { a: 0, b: 1, key: "j", name: "low / high" },
  { a: 1, b: 0, key: "e", name: "high / low" },
  { a: 1, b: 1, key: "i", name: "high / high" },
];
const KEY_TO_STIM = Object.fromEntries(STIM.map((s, i) => [s.key, i]));
const GRID = [2, 3, 0, 1];        // visual/keyboard 2x2: E I over F J

const SETS = {
  colour: {
    label: "Colour patch", dimA: "saturation", dimB: "hue",
    draw(g, W, H, a, b, n) {
      g.fillStyle = `hsl(${(b ? 38 : 8) + n * 5} ${(a ? 62 : 34) + n * 6}% 52%)`;
      g.beginPath();
      g.arc(W / 2, H / 2, Math.min(W, H) * 0.3, 0, Math.PI * 2);
      g.fill();
    },
  },
  size: {
    label: "Rectangle", dimA: "width", dimB: "height",
    draw(g, W, H, a, b, n) {
      const w = (a ? 122 : 92) + n * 7, h = (b ? 122 : 92) + n * 7;
      g.fillStyle = getComputedStyle(document.documentElement)
        .getPropertyValue("--slate").trim() || "#456";
      g.fillRect((W - w) / 2, (H - h) / 2, w, h);
    },
  },
};

let model = null;
let counts, trialLog, checkpoints, history;
let stimIdx = null, phase = "idle", nTrials = 0, nTimeouts = 0;
let blockN = 0, inBlock = 0, stimOnAt = 0;
let timers = [];

const cfg = () => ({
  set: $("stim-set").value,
  stimMs: +$("stim-ms").value,
  deadlineMs: +$("deadline-ms").value,
  target: +$("sd-target").value,
  block: +$("block-n").value,
  maxTrials: +$("max-trials").value,
  liveDiag: $("live-diag").checked,
});

function reset() {
  counts = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  trialLog = []; checkpoints = []; history = [];
  nTrials = 0; nTimeouts = 0; blockN = 0; inBlock = 0;
}
function clearTimers() { timers.forEach(clearTimeout); timers = []; }
function after(ms, fn) { timers.push(setTimeout(fn, ms)); }
function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function ctx(c, H) {
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 320;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  return { g, W, H };
}

// --------------------------------------------------------------------------- //
// Screens
// --------------------------------------------------------------------------- //
function show(screen) {
  ["welcome", "instructions", "trial", "break", "results"].forEach((s) => {
    const el = $(`screen-${s}`);
    if (el) el.hidden = s !== screen;
  });
}

function openOverlay() {
  $("overlay").hidden = false;
  document.body.style.overflow = "hidden";
}
function closeOverlay() {
  clearTimers();
  phase = "idle";
  $("overlay").hidden = true;
  document.body.style.overflow = "";
}

function renderStimulusGrid(container, size) {
  container.innerHTML = GRID.map((i) => `
    <div class="legend-cell">
      <canvas class="legend-canvas" data-stim="${i}" height="76"></canvas>
      <div><kbd>${STIM[i].key.toUpperCase()}</kbd>
        <span class="cap">${size.dimA} ${STIM[i].a ? "high" : "low"},
        ${size.dimB} ${STIM[i].b ? "high" : "low"}</span></div>
    </div>`).join("");
  container.querySelectorAll(".legend-canvas").forEach((c) => {
    const { g, W, H } = ctx(c, 76);
    size.draw(g, W, H, STIM[+c.dataset.stim].a, STIM[+c.dataset.stim].b, 0);
  });
}

// --------------------------------------------------------------------------- //
// Trials
// --------------------------------------------------------------------------- //
function runTrial() {
  const c = cfg();
  if (nTrials >= c.maxTrials) return finish("reached the trial ceiling");
  phase = "fixation";
  const { g, W, H } = ctx($("stim"), 300);
  const css = getComputedStyle(document.documentElement);
  g.strokeStyle = css.getPropertyValue("--steel").trim() || "#888";
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(W/2 - 9, H/2); g.lineTo(W/2 + 9, H/2);
  g.moveTo(W/2, H/2 - 9); g.lineTo(W/2, H/2 + 9);
  g.stroke();
  $("trial-prompt").textContent = "";

  after(420, () => {
    stimIdx = Math.floor(Math.random() * 4);
    phase = "stimulus";
    const s = ctx($("stim"), 300);
    SETS[c.set].draw(s.g, s.W, s.H, STIM[stimIdx].a, STIM[stimIdx].b, gauss());
    after(c.stimMs, () => {
      if (phase !== "stimulus") return;
      ctx($("stim"), 300);
      phase = "response";
      stimOnAt = performance.now();
      $("trial-prompt").textContent = "Which one?";
      after(c.deadlineMs, onTimeout);
    });
  });
}

function onTimeout() {
  if (phase !== "response") return;
  phase = "feedback";
  nTimeouts += 1;
  $("trial-prompt").innerHTML = `<span class="pill warn">too slow</span>`;
  updateBar();
  after(420, nextAfterTrial);
}

function respond(r) {
  if (phase !== "response") return;
  clearTimers();
  phase = "feedback";
  const rt = (performance.now() - stimOnAt) / 1000;

  counts[stimIdx][r] += 1;
  nTrials += 1; inBlock += 1;
  trialLog.push({
    trial: nTrials,
    stimulus: `${STIM[stimIdx].a ? "a2" : "a1"}/${STIM[stimIdx].b ? "b2" : "b1"}`,
    response: `${STIM[r].a ? "a2" : "a1"}/${STIM[r].b ? "b2" : "b1"}`,
    rt: rt.toFixed(3),
  });
  updateBar();
  if (cfg().liveDiag) {
    refit().then((f) => {
      if (phase !== "idle") paintDiagnostics("trial", f);
    });
  }
  after(180, nextAfterTrial);
}

function nextAfterTrial() {
  if (inBlock >= cfg().block) return endBlock();
  runTrial();
}

function updateBar() {
  const c = cfg();
  $("bar-trials").textContent = nTrials;
  $("bar-timeouts").textContent = nTimeouts;
  $("bar-fill").style.width = `${Math.min(100, 100 * nTrials / c.maxTrials)}%`;
}

// --------------------------------------------------------------------------- //
// Fit
// --------------------------------------------------------------------------- //
async function refit() {
  const trials = counts.map((r) => r.reduce((a, b) => a + b, 0));
  if (trials.some((t) => t < 3)) return null;
  const out = await model.predict({ counts, trials });
  const meanSD = out.std.reduce((a, b) => a + b, 0) / out.std.length;
  history.push(meanSD);
  checkpoints.push({ stimuli: Plot.toStimuli(out.mean), n: nTrials });
  return { out, meanSD };
}

function accuracy() {
  let a = 0, b = 0, n = 0;
  for (let s = 0; s < 4; s++) for (let r = 0; r < 4; r++) {
    const c = counts[s][r];
    if (!c) continue;
    n += c;
    if (s >> 1 === r >> 1) a += c;
    if ((s & 1) === (r & 1)) b += c;
  }
  return n ? 0.5 * (a / n + b / n) : null;
}

// At most this many layers in the fade trail. With the live panel on we refit every
// trial, so an uncapped trail would be hundreds of near-identical ellipses -- slow to
// draw and impossible to read. Thin evenly and always keep the most recent fit.
const MAX_TRAIL = 24;
function thinned(cps) {
  if (cps.length <= MAX_TRAIL) return cps;
  const step = (cps.length - 1) / (MAX_TRAIL - 1);
  return Array.from({ length: MAX_TRAIL }, (_, i) => cps[Math.round(i * step)]);
}

function paintDiagnostics(prefix, fitted) {
  const canvas = $(`${prefix}-space`);
  if (!canvas || !canvas.clientWidth) return;   // not laid out yet
  if (!fitted) {
    const st = $(`${prefix}-status`);
    if (st) st.textContent =
      "Waiting for a few trials of each stimulus before the fit means anything.";
    return;
  }
  const { out, meanSD } = fitted;
  // The fade trail is the honest picture of convergence: every earlier fit stays on
  // the canvas, dimmed, so the reader sees the estimate settling rather than a single
  // confident-looking ellipse set.
  // Solid + thick = the current fit; dashed + thin = where it has been. The floor is
  // lifted off the default 0.08 because the dashing now carries the distinction, so the
  // history can stay visible instead of nearly disappearing.
  Plot.renderFadeTrail(canvas, thinned(checkpoints), { showMarginals: false },
                       { minAlpha: 0.18, maxAlpha: 1.0, curve: "exp" });
  const pct = (x) => `${Math.round(100 * x)}%`;
  const set = SETS[cfg().set];
  const el = $(`${prefix}-constructs`);
  if (el) el.innerHTML = `
    <div class="est"><span>Perceptual independence</span><strong>${pct(out.corr.pi)}</strong></div>
    <div class="est"><span>Separability, ${set.dimA}</span><strong>${pct(out.sep.A)}</strong></div>
    <div class="est"><span>Separability, ${set.dimB}</span><strong>${pct(out.sep.B)}</strong></div>`;
  const acc = accuracy();
  const st = $(`${prefix}-status`);
  if (st) st.innerHTML =
    `Mean posterior SD <strong>${meanSD.toFixed(3)}</strong> ` +
    `(target ${cfg().target.toFixed(2)}) · accuracy <strong>${Math.round(100*acc)}%</strong> ` +
    (acc >= 0.6 && acc <= 0.85 ? `<span class="pill ok">informative range</span>`
      : `<span class="pill warn">${acc > 0.85 ? "near ceiling" : "low"}</span>`);
}

// --------------------------------------------------------------------------- //
// Blocks
// --------------------------------------------------------------------------- //
async function endBlock() {
  clearTimers();
  blockN += 1; inBlock = 0;
  phase = "break";
  const fitted = await refit();
  if (fitted && fitted.meanSD <= cfg().target) return finish("precision target reached");
  if (nTrials >= cfg().maxTrials) return finish("reached the trial ceiling");

  show("break");
  $("break-title").textContent = `Block ${blockN} complete — ${nTrials} trials so far`;
  paintDiagnostics("break", fitted);
  $("break-note").textContent = fitted
    ? "This is where an adaptive design makes its decision: carry on, or stop."
    : "Not enough data yet for a stable fit — carry on.";
  phase = "await-space-block";
}

function finish(why) {
  clearTimers();
  phase = "done";
  show("results");
  refit().then((fitted) => {
    // Persist before painting: if anything below throws, the data still survives.
    const saved = Store.save({
      counts,
      matrixCSV: IO.countsToCSV(counts),
      trialsCSV: trialCSV(),
      nTrials, nTimeouts, blocks: blockN,
      accuracy: accuracy(),
      set: cfg().set,
      why,
    });
    const dl = $("results-analyse");
    if (dl) dl.disabled = !saved;
    if (!saved) {
      const d = $("results-detail");
      if (d) d.innerHTML =
        `<p class="cap">This browser will not let the page store data, so the
         hand-off to Analyse is unavailable. Download the trials instead.</p>`;
    }
    $("results-title").textContent = `Finished — ${why}`;
    paintDiagnostics("results", fitted);
    const acc = accuracy();
    $("results-detail").innerHTML =
      `<p>${nTrials} scored trials over ${blockN} block${blockN === 1 ? "" : "s"}` +
      (nTimeouts ? `, plus ${nTimeouts} discarded for missing the deadline` : "") +
      `. Per-dimension accuracy ${Math.round(100 * acc)}%.</p>` +
      `<p class="cap">Every earlier fit is still on the plot, dimmed. The estimate
       settling as trials accumulate is the thing a fixed trial budget cannot see.</p>`;
  });
}

// --------------------------------------------------------------------------- //
// Handoff to Analyse
// --------------------------------------------------------------------------- //
function trialCSV() {
  const rows = ["participant,trial,stimulus,response,rt"];
  trialLog.forEach((t) =>
    rows.push(`live,${t.trial},${t.stimulus},${t.response},${t.rt}`));
  return rows.join("\n") + "\n";
}

function sendToAnalyse() {
  // The session is already in localStorage; this only asks the Analyse page to open
  // it on arrival rather than merely offering it. A missed flag is now harmless --
  // the data is still there to be loaded by hand.
  try { sessionStorage.setItem("grin.openSession", "1"); } catch (e) {}
  window.location.href = "./analyse.html";
}

function downloadCSV() {
  const blob = new Blob([trialCSV()], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "grin_live_task.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

// --------------------------------------------------------------------------- //
// Wiring
// --------------------------------------------------------------------------- //
async function begin() {
  $("start").disabled = true;
  $("start").textContent = "Loading model…";
  try {
    if (!model) model = await loadModelCached("./assets/models/cm",
      (m) => { $("start").textContent = m || "Loading…"; });
  } catch (e) {
    $("start").textContent = "Start";
    $("start").disabled = false;
    $("config-status").innerHTML =
      `<span class="pill bad">Couldn't load the model (${e.message}).</span>`;
    return;
  }
  $("start").textContent = "Start";
  $("start").disabled = false;
  reset();
  openOverlay();
  show("welcome");
  phase = "await-space-welcome";
  const live = cfg().liveDiag;
  $("bar-diag").hidden = !live;
  $("trial-diag").hidden = !live;
  $("trial-layout").classList.toggle("with-diag", live);
  updateBar();
}

$("start").addEventListener("click", begin);
$("overlay-quit").addEventListener("click", () => {
  if (nTrials > 0) finish("stopped early"); else closeOverlay();
});
$("results-close").addEventListener("click", closeOverlay);
$("results-analyse").addEventListener("click", sendToAnalyse);
$("results-download").addEventListener("click", downloadCSV);
$("stim-set").addEventListener("change", () =>
  renderStimulusGrid($("config-legend"), SETS[cfg().set]));
["stim-ms", "deadline-ms", "sd-target", "block-n", "max-trials"].forEach((id) => {
  const el = $(id);
  if (el) el.addEventListener("input", () => { $(`${id}-val`).textContent = el.value; });
});

window.addEventListener("keydown", (e) => {
  if ($("overlay").hidden) return;
  if (e.key === "Escape") { e.preventDefault(); return $("overlay-quit").click(); }
  if (e.code === "Space") {
    e.preventDefault();
    if (phase === "await-space-welcome") {
      show("instructions");
      renderStimulusGrid($("instr-legend"), SETS[cfg().set]);
      phase = "await-space-instructions";
    } else if (phase === "await-space-instructions" || phase === "await-space-block") {
      show("trial");
      runTrial();
    }
    return;
  }
  const i = KEY_TO_STIM[e.key.toLowerCase()];
  if (i !== undefined && phase === "response") { e.preventDefault(); respond(i); }
});

renderStimulusGrid($("config-legend"), SETS[cfg().set]);

// A previous run is still in this browser: say so, rather than silently discarding it
// when the next task starts.
(function offerPrevious() {
  const prev = Store.load();
  const box = $("prev-session");
  if (!prev || !box) return;
  box.hidden = false;
  $("prev-detail").textContent =
    `${prev.nTrials} trials, finished ${Store.describeAge(prev.savedAt)}` +
    (prev.accuracy ? ` at ${Math.round(100 * prev.accuracy)}% per-dimension accuracy` : "") + ".";
  $("prev-analyse").addEventListener("click", () => {
    try { sessionStorage.setItem("grin.openSession", "1"); } catch (e) {}
    window.location.href = "./analyse.html";
  });
  $("prev-clear").addEventListener("click", () => {
    Store.clear();
    box.hidden = true;
  });
})();
