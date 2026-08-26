import * as IO from "../grt-io.js";
import * as Fit from "../grt-fit.js";
import * as Core from "../grt-core.js";
import * as Plot from "../grt-plot.js";
import * as Store from "../session-store.js";
import { loadModelCached } from "../grin-model.js";

const $ = (id) => document.getElementById(id);

// Lazy-load display fonts only when a figure palette that needs them is chosen.
const FIGURE_FONT_HREF =
  "https://fonts.googleapis.com/css2?family=Rye&family=VT323&family=Kalam:wght@400;700&family=Bungee&family=Orbitron:wght@600&family=Playfair+Display:wght@600&display=swap";
let figureFontsLoading = null;
function ensureFigureFonts() {
  if (document.getElementById("grin-figure-fonts")) return Promise.resolve();
  if (figureFontsLoading) return figureFontsLoading;
  figureFontsLoading = new Promise((resolve) => {
    const link = document.createElement("link");
    link.id = "grin-figure-fonts";
    link.rel = "stylesheet";
    link.href = FIGURE_FONT_HREF;
    link.onload = () => resolve();
    link.onerror = () => resolve();
    document.head.appendChild(link);
  });
  return figureFontsLoading;
}


// --------------------------------------------------------------------------- //
// State
// --------------------------------------------------------------------------- //
let parsed = null; // result of IO.parseCSV
let groups = null; // Map participant -> trials[]  (long format)
// The response-time model is withdrawn from the site: its weights come from a
// generator retired on 2026-08-14 (docs/dynamic_grt_rt_design.md), and the
// replacement is still in validation. RT columns in an uploaded file are parsed and
// then ignored rather than rejected, so an existing file still analyses on counts.
let hasRT = false;
let rtColumnsSeen = false;
let cmModel = null;
let currentLabels = {
  aName: "Dimension A",
  a1: "A1",
  a2: "A2",
  bName: "Dimension B",
  b1: "B1",
  b2: "B2",
};
let currentParticipant = null; // whoever the figure/label controls currently act on
const cache = new Map(); // participant id -> { agg, checks, grin, mleFull, mleSel, ms }

async function getModel() {
  const status = (msg) => {
    const el = $("analysis-status") || $("batch-status") || $("file-status");
    if (el && msg) el.textContent = msg;
  };
  if (!cmModel) {
    cmModel = await loadModelCached("./assets/models/cm", status);
    // Reveal the toggle only if this build actually ships the scale factors, so the
    // control can never promise something the model cannot do.
    const row = $("calib-row");
    if (row && cmModel.canCalibrate) row.hidden = false;
  }
  cmModel.setCalibrated(!!($("calib-toggle") && $("calib-toggle").checked));
  return cmModel;
}

// --------------------------------------------------------------------------- //
// Manual matrix entry
// --------------------------------------------------------------------------- //
function buildManualGrid() {
  const lab = Core.STIMULUS_ORDER;
  let html = `<tr><th></th>${lab.map((l) => `<th>${l}</th>`).join("")}</tr>`;
  for (let i = 0; i < 4; i++) {
    html += `<tr><th>${lab[i]}</th>`;
    for (let j = 0; j < 4; j++)
      html += `<td><input type="number" min="0" step="1" value="0" data-i="${i}" data-j="${j}"></td>`;
    html += "</tr>";
  }
  $("manual-grid").innerHTML = html;
}
buildManualGrid();

$("use-manual").addEventListener("click", () => {
  const counts = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  document.querySelectorAll("#manual-grid input").forEach((el) => {
    counts[+el.dataset.i][+el.dataset.j] = Math.max(
      0,
      Math.round(+el.value) || 0,
    );
  });
  loadParsed(
    {
      format: "matrix",
      counts,
      levels: null,
      hasRT: false,
      warnings: [],
    },
    "typed matrix",
  );
});

// --------------------------------------------------------------------------- //
// Published examples, real confusion matrices from the GRT literature. Not
// here to claim GRIN is faster or better; here so the GRIN-vs-MLE comparison
// above can be run on data someone else already published, with a citation
// attached, so "different methods land on the same answer" is checkable
// against a real result rather than only simulated ones.
// --------------------------------------------------------------------------- //
const EXAMPLES = {
  soto: {
    label: "Emotion / Gender, Soto et al. (2017)",
    matrix: [
      [140, 34, 36, 40],
      [85, 90, 5, 70],
      [89, 4, 91, 66],
      [20, 8, 59, 163],
    ],
    Aname: "Emotion",
    A: ["Happy", "Sad"],
    Bname: "Gender",
    B: ["Male", "Female"],
    citation: `Soto, F. A., Zheng, E., Fonseca, J., &amp; Ashby, F. G. (2017).
Testing separability and independence of perceptual dimensions with general
recognition theory: A tutorial and new R package (<em>grtools</em>).
<em>Frontiers in Psychology</em>, 8, 696.
<a href="https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.00696/full"
   target="_blank" rel="noopener">View paper</a>
, data reproduced from Table 1 (emotion/gender face identification).`,
  },
  silbert_FD: {
    label: "Frequency / Duration, Silbert et al. (2009)",
    matrix: [
      [159, 33, 46, 12],
      [20, 186, 5, 39],
      [21, 9, 191, 29],
      [3, 22, 22, 203],
    ],
    Aname: "Frequency",
    A: ["490\u20131490Hz", "510\u20131510Hz"],
    Bname: "Duration",
    B: ["250ms", "300ms"],
    citation: `Silbert, N. H., Townsend, J. T., &amp; Lentz, J. J. (2009).
Independence and separability in the perception of complex nonspeech sounds.
<em>Attention, Perception, &amp; Psychophysics</em>, 71(8), 1900\u20131915.
<a href="https://link.springer.com/article/10.3758/APP.71.8.1900"
   target="_blank" rel="noopener">View paper</a>
, reproduced from their frequency/duration identification experiment. Values
transcribed manually; worth a spot check against the original table before citing.`,
  },
  silbert_PT: {
    label: "Pitch / Timbre, Silbert et al. (2009)",
    matrix: [
      [186, 22, 41, 1],
      [16, 180, 26, 28],
      [42, 32, 149, 27],
      [1, 28, 13, 208],
    ],
    Aname: "Pitch",
    A: ["150Hz", "152Hz"],
    Bname: "Timbre",
    B: ["850Hz", "1050Hz"],
    citation: `Silbert, N. H., Townsend, J. T., &amp; Lentz, J. J. (2009).
Independence and separability in the perception of complex nonspeech sounds.
<em>Attention, Perception, &amp; Psychophysics</em>, 71(8), 1900\u20131915.
<a href="https://link.springer.com/article/10.3758/APP.71.8.1900"
   target="_blank" rel="noopener">View paper</a>
, reproduced from their pitch/timbre identification experiment. Values
transcribed manually; worth a spot check against the original table before citing.`,
  },
};

$("load-example").addEventListener("click", () => {
  const ex = EXAMPLES[$("example-select").value];
  loadParsed(
    {
      format: "matrix",
      counts: ex.matrix,
      levels: { A: ex.A, B: ex.B, Aname: ex.Aname, Bname: ex.Bname },
      hasRT: false,
      warnings: [],
    },
    ex.label,
    `<div class="note"><h4>Source</h4><p style="margin-bottom:0">${ex.citation}</p></div>`,
  );
});

// --------------------------------------------------------------------------- //
// File / paste input
// --------------------------------------------------------------------------- //
const dz = $("dropzone");
dz.addEventListener("click", () => $("file-input").click());
dz.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") $("file-input").click();
});
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => {
    e.preventDefault();
    dz.classList.add("is-active");
  }),
);
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, () => {
    dz.classList.remove("is-active");
  }),
);
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files?.[0];
  if (f) readFile(f);
});
$("file-input").addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) readFile(f);
});
function readFile(f) {
  const reader = new FileReader();
  reader.onload = () => loadParsed(IO.parseCSV(reader.result), f.name);
  reader.onerror = () => {
    $("file-status").innerHTML =
      `<span class="pill bad">Could not read the file.</span>`;
  };
  reader.readAsText(f);
}
// A live-task session stored in this browser. Two entry points: arriving from the
// task with the open flag set (load it straight away), or arriving cold with a session
// still saved (offer it, do not assume). The data lives in localStorage rather than
// being handed over once, so a stale cached copy of this page can no longer lose it and
// there is a way back to your own results after navigating away.
(function offerStoredSession() {
  const sess = Store.load();
  const box = $("stored-session");
  if (!sess || !sess.matrixCSV) return;

  const open = () => {
    try {
      loadParsed(IO.parseCSV(sess.matrixCSV), "your live-task session");
      if (box) box.hidden = true;
    } catch (e) {
      $("file-status").innerHTML =
        `<span class="pill bad">Could not load the stored session (${e.message}).</span>`;
    }
  };

  let asked = false;
  try { asked = sessionStorage.getItem("grin.openSession") === "1"; } catch (e) {}
  if (asked) {
    try { sessionStorage.removeItem("grin.openSession"); } catch (e) {}
    open();
    return;
  }

  if (!box) return;
  box.hidden = false;
  $("stored-detail").textContent =
    `${sess.nTrials} trials from the live task, finished ${Store.describeAge(sess.savedAt)}.`;
  $("stored-open").addEventListener("click", open);
  $("stored-clear").addEventListener("click", () => {
    Store.clear();
    box.hidden = true;
  });
})();

$("parse-paste").addEventListener("click", () => {
  const txt = $("paste-area").value;
  if (!txt.trim()) return;
  loadParsed(IO.parseCSV(txt), "pasted text");
});
const calibToggle = $("calib-toggle");
if (calibToggle) {
  calibToggle.addEventListener("change", () => {
    if (cmModel) cmModel.setCalibrated(calibToggle.checked);
    // Re-run whichever view is currently showing; point estimates do not change,
    // but every interval and error bar does.
    if (typeof analyseAll === "function" && !$("results").hidden) analyseAll();
  });
}

$("dl-template-cm").addEventListener("click", () =>
  downloadText(
    IO.templateCSV({ withRT: false }),
    "grin_template_counts.csv",
  ),
);

function downloadText(text, filename) {
  const blob = new Blob([text], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// --------------------------------------------------------------------------- //
// Loading a parse result into state
// --------------------------------------------------------------------------- //
function loadParsed(result, sourceName, citationHTML = null) {
  cache.clear();
  $("results").hidden = true;
  $("cm-card").hidden = true;
  $("figure-card").hidden = true;
  $("compare-card").hidden = true;
  $("batch-card").hidden = true;
  $("citation-box").innerHTML = citationHTML || "";

  if (result.error) {
    $("file-status").innerHTML =
      `<span class="pill bad">${escapeHTML(result.error)}</span>`;
    return;
  }
  parsed = result;
  rtColumnsSeen = !!result.hasRT;
  hasRT = false;   // counts-only model; see the note at the top of this file
  if (result.levels)
    currentLabels = {
      aName: result.levels.Aname || "Dimension A",
      a1: result.levels.A[0],
      a2: result.levels.A[1],
      bName: result.levels.Bname || "Dimension B",
      b1: result.levels.B[0],
      b2: result.levels.B[1],
    };

  const nBadge =
    result.format === "long"
      ? `${result.trials.length} trials`
      : "1 matrix";
  $("file-status").innerHTML =
    `<span class="pill ok">Parsed ${escapeHTML(sourceName)}, ${nBadge}</span>` +
    (rtColumnsSeen
      ? `<span class="pill">response-time columns ignored \u2014 counts-only model</span>`
      : "") +
    (result.warnings?.length
      ? `<br><span class="pill warn" style="margin-top:.4rem">${result.warnings.length} warning(s)</span>`
      : "");

  if (result.format === "matrix") {
    groups = new Map([["matrix", { __matrix: result.counts }]]);
  } else {
    groups = IO.byParticipant(result);
  }

  const sel = $("participant-select");
  sel.innerHTML = [...groups.keys()]
    .map(
      (id) =>
        `<option value="${escapeAttr(id)}">${escapeHTML(id)}</option>`,
    )
    .join("");
  const multi = groups.size > 1;
  sel.style.display = multi ? "" : "none";
  $("analyse-all").style.display = multi ? "" : "none";

  renderValidationSummary(result);
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth", block: "nearest" });

  // convenience: analyse immediately if there's exactly one group
  if (groups.size === 1) analyseOne([...groups.keys()][0]);
}

function renderValidationSummary(result) {
  const el = $("validation-card");
  el.innerHTML = trustNoteHTML() +
    (result.warnings?.length
      ? `<span class="eyebrow">Parsing notes</span>` +
        result.warnings
          .map((w) => `<div class="note warn"><p>${escapeHTML(w)}</p></div>`)
          .join("")
      : "");
}

function trustNoteHTML() {
  return `<span class="eyebrow">Before interpreting the fit</span>
    <div class="note">
      <h4>No warning does not mean “the model must be right”</h4>
      <p>These checks catch file problems and some observable departures from the training regime. They cannot show that GRT generated the data, or diagnose lapses, drift, mixtures, or changing response criteria. Read GRIN's probabilities as conditional on its model and training prior; pay attention to wide intervals, weak factorized support, and disagreement with the MLE reference.</p>
    </div>`;
}

// --------------------------------------------------------------------------- //
// Per-participant aggregation + analysis
// --------------------------------------------------------------------------- //
function aggFor(id) {
  const g = groups.get(id);
  if (g.__matrix) {
    const counts = g.__matrix;
    return {
      counts,
      trials: counts.map((r) => r.reduce((a, b) => a + b, 0)),
      rtq: null,
    };
  }
  return IO.aggregate(g, { hasRT });
}

async function analyseOne(id) {
  $("analysis-status").textContent = `Analysing ${id}…`;
  const agg = aggFor(id);
  const checks = IO.checkInputs(agg, {
    hasRT,
    rawRTs:
      hasRT && !groups.get(id).__matrix
        ? groups
            .get(id)
            .map((t) => t.rt)
            .filter((v) => v !== null)
        : [],
  });

  renderChecks(checks);
  if (checks.errors.length) {
    $("cm-card").hidden = true;
    $("figure-card").hidden = true;
    $("compare-card").hidden = true;
    $("analysis-status").innerHTML =
      `<span class="pill bad">Cannot analyse ${escapeHTML(id)}, see notes above.</span>`;
    return;
  }

  const t0 = performance.now();
  const m = await getModel();
  const grin = await m.predict(agg);
  const t1 = performance.now();
  const mleFull = Fit.fitClass(agg.counts, agg.trials, "ds");
  const mleSel = Fit.fitAndSelect(agg.counts, agg.trials, "bic");
  const mleMs = performance.now() - t1;

  cache.set(id, { agg, checks, grin, mleFull, mleSel, mleMs });
  renderParticipant(id);
  $("analysis-status").innerHTML =
    `<span class="pill ok">${escapeHTML(id)}: GRIN ${grin.ms.toFixed(2)} ms · MLE (12 classes) ${mleMs.toFixed(1)} ms</span>`;
}

function renderChecks(checks) {
  const parts = [];
  for (const e of checks.errors)
    parts.push(`<div class="note bad"><p>${escapeHTML(e)}</p></div>`);
  for (const w of checks.warnings)
    parts.push(`<div class="note warn"><p>${escapeHTML(w)}</p></div>`);
  $("validation-card").innerHTML =
    trustNoteHTML() +
    (parsed.warnings?.length
      ? `<span class="eyebrow">Parsing notes</span>` +
        parsed.warnings
          .map(
            (w) => `<div class="note warn"><p>${escapeHTML(w)}</p></div>`,
          )
          .join("")
      : "") +
    (parts.length
      ? `<span class="eyebrow">This participant</span>${parts.join("")}`
      : "");
}

function renderParticipant(id) {
  currentParticipant = id;
  const { agg, grin, mleFull, mleSel, mleMs } = cache.get(id);
  const L = currentLabels;

  // keep the label-editing inputs in sync with whatever labels are active,
  // without clobbering an edit the user is mid-typing on a different field
  if (document.activeElement?.id?.indexOf("fig-") !== 0) {
    $("fig-aName").value = L.aName;
    $("fig-aLevels").value = `${L.a1}, ${L.a2}`;
    $("fig-bName").value = L.bName;
    $("fig-bLevels").value = `${L.b1}, ${L.b2}`;
  }

  // confusion matrix
  const props = agg.counts.map((r, i) =>
    r.map((v) => v / Math.max(1, agg.trials[i])),
  );
  Plot.renderCM($("obs-cm"), props, L, { counts: agg.counts });
  const acc = Plot.cmAccuracy(props);
  const total = agg.trials.reduce((a, b) => a + b, 0);
  $("cm-stats").innerHTML = `
    <div class="est"><div class="lbl">Total trials</div><div class="track"></div><div class="num">${total}</div></div>
    <div class="est"><div class="lbl">Accuracy</div>
<div class="track"><div class="ci" style="left:0;width:${(100 * acc).toFixed(1)}%;opacity:.8"></div></div>
<div class="num">${(100 * acc).toFixed(1)}%</div></div>`;
  $("cm-card").hidden = false;

  renderFigure(id);

  // GRIN column
  const names = grin.mean.length === 12 ? Core.PARAM_NAMES : [];
  $("grin-ms").textContent = `${grin.ms.toFixed(2)} ms`;
  $("grin-ests").innerHTML = names
    .map((n, i) => {
      const range = i < 8 ? [-3, 3] : [-1, 1];
      return Plot.estRow(n, grin.mean[i], grin.std[i], range);
    })
    .join("");
  $("grin-constructs").innerHTML =
    Plot.pbar("A separable", grin.sep.A) +
    Plot.pbar("B separable", grin.sep.B) +
    Plot.pbar("Independent (PI)", grin.corr.pi) +
    Plot.pbar("One shared ρ", grin.corr.rho1) +
    Plot.pbar("ρ varies by stimulus", grin.corr.free);
  const mc = grin.modelClass;
  $("grin-verdict").innerHTML =
    `<div class="note ${mc.factorizedSupport >= 0.5 ? "" : "warn"}">
    <h4>Best-supported class: <code>${mc.name}</code> <span class="cap">(${Core.modelLabel(mc.name)})</span></h4>
    <p>Factorized support ${mc.factorizedSupport.toFixed(2)}.${
mc.factorizedSupport < 0.5
  ? " Not decisive, treat the structural conclusion as tentative."
  : ""
    } This is the product of three marginal outputs, not a separately calibrated joint probability.</p>
  </div>`;

  if (hasRT && grin.archBest) {
    $("grin-rt-block").innerHTML = `
<h3 style="font-size:.68rem;color:var(--steel)">Processing architecture</h3>
${Object.entries(grin.arch)
  .map(([k, v]) => Plot.pbar(k.replace(/_/g, " "), v))
  .join("")}
<div class="note">
  <h4>Self-terminating architectures: ${grin.selfTerminatingProbability.toFixed(2)}</h4>
  <p>Total probability assigned to the serial and parallel self-terminating models. In this simulator, one randomly selected dimension is processed and the other is guessed; this is not a diagnosis of stable dimension neglect.</p>
</div>`;
  } else {
    $("grin-rt-block").innerHTML = "";
  }

  // MLE column: full (unconstrained) fit, comparable term-for-term to GRIN's posterior
  $("mle-ms").textContent = `${mleFull.ms.toFixed(2)} ms`;
  $("mle-ests").innerHTML = mleFull.params
    .map(
      (v, i) =>
        `<div class="est"><div class="lbl">${Core.PARAM_NAMES[i]}</div>
<div class="track"></div><div class="num">${v.toFixed(3)}</div></div>`,
    )
    .join("");

  // MLE model selection table
  const rows = mleSel.fits
    .slice(0, 6)
    .map(
      (f) => `<tr>
    <td style="text-align:left;font-family:Inter,sans-serif">${f.model}</td>
    <td>${f.k}</td><td>${f.loglik.toFixed(1)}</td>
    <td>${f.bic.toFixed(1)}</td><td>${f.weight.toFixed(3)}</td>
    <td>${Number.isFinite(f.p) ? f.p.toFixed(3) : ", "}</td>
  </tr>`,
    )
    .join("");
  $("mle-table").innerHTML =
    `<tr><th style="text-align:left">Model</th><th>k</th>
    <th>log L</th><th>BIC</th><th>weight</th><th>G² p</th></tr>${rows}`;
  $("mle-rt-note").textContent = hasRT
    ? "MLE has no response-time model here, grtools and mdsdt don't fit RT either. This comparison is counts-only by nature."
    : "";

  // honest agreement check: GRIN posterior mean vs the unconstrained MLE fit
  let maxDiff = 0,
    worst = "";
  Core.PARAM_NAMES.forEach((n, i) => {
    const d = Math.abs(grin.mean[i] - mleFull.params[i]);
    if (d > maxDiff) {
      maxDiff = d;
      worst = n;
    }
  });
  const nSD =
    maxDiff / Math.max(1e-6, grin.std[Core.PARAM_NAMES.indexOf(worst)]);
  $("agree-note").innerHTML = `<div class="note ${nSD > 2 ? "warn" : ""}">
    <h4>Agreement</h4>
    <p>
Largest gap between GRIN's posterior mean and the unconstrained MLE fit is on
<strong>${worst}</strong>: ${maxDiff.toFixed(3)} (${nSD.toFixed(1)} GRIN posterior SDs).
${
  nSD > 2
    ? "That's a bigger gap than GRIN's own uncertainty would predict, worth a second look, especially if trial counts are low or this matrix is unusual."
    : "That's well within GRIN's stated uncertainty, which is what you want to see."
}
    </p>
  </div>`;

  $("compare-card").hidden = false;
}

// --------------------------------------------------------------------------- //
// The classic GRT plot: GRIN's per-stimulus estimate (solid), against a chosen
// reference (dashed), either the unconstrained MLE fit (the "different methods,
// same answer" comparison) or GRIN's own estimate projected onto its best-fitting
// model class (how much adjustment getting to a clean structural model cost).
// Redraws from cache only, never refits, so toggling the overlay or editing
// labels is instant.
// --------------------------------------------------------------------------- //
// --------------------------------------------------------------------------- //
const PALETTE_NOTES = {
  site: "",
  custom: "",
  blackOnWhite:
    "All-black lines, stimuli are told apart by position, not colour.",
  whiteOnBlack:
    "All-white lines, stimuli are told apart by position, not colour.",
  grayscalePrint:
    "Four distinguishable greys. Photocopies and prints cleanly.",
  colorblindSafe:
    "Okabe–Ito palette, safe under deuteranopia, protanopia, and tritanopia.",
  apaStyle: "Muted, high-contrast, minimal decoration.",
  trueGrit: "Sepia + a wanted-poster title font. Purely for fun.",
  matrixConsole:
    "Neon terminal green on black, with scanlines. Purely for fun.",
  synthwave: "Uses the site's own arcade font. Purely for fun.",
  chalkboard:
    "Chalk-on-slate, with a handwritten title font. Purely for fun.",
  blueprint: "White-on-blue technical drafting look.",
  independenceDay:
    "Red, white, and blue. Try downloading a PNG with this one on.",
  senseAndSeparability:
    "A Regency-era reading of separability. Purely for fun.",
  perceptualSpace: "A perceptual space, in space. Purely for fun.",
  spursModern: "Go Spurs Go. Silver and black.",
  spursRetroFiesta: "Go Spurs Go. The 90s Fiesta colours.",
};

// --------------------------------------------------------------------------- //
// A small easter egg: typing "Go Spurs Go" (case-insensitive) as the figure
// title unlocks a Spurs palette + two image-based rendering options. Once
// unlocked in a session it stays unlocked, even if the title is changed
// afterward, this is a one-time reveal, not something that flickers on and
// off as the user types.
// --------------------------------------------------------------------------- //
let spursUnlocked = false;
const spursImages = { spur: null };

/** Loads once, reused on every subsequent render. Resolves to null (never
 * rejects) on a missing file or decode error, so a caller can just check
 * truthiness rather than wrapping every call in try/catch. */
function loadImageOnce(cacheKey, src) {
  if (spursImages[cacheKey])
    return Promise.resolve(spursImages[cacheKey]);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      spursImages[cacheKey] = img;
      resolve(img);
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

function checkSpursUnlock() {
  if (spursUnlocked) return;
  if ($("fig-title").value.trim().toLowerCase() !== "go spurs go") return;
  spursUnlocked = true;
  $("opt-spurs-modern").hidden = false;
  $("opt-spurs-retro").hidden = false;
  $("spurs-extra").hidden = false;
  $("spurs-unlock-note").innerHTML =
    `<span class="pill ok">🏀 Go Spurs Go!, Spurs palettes unlocked below.</span>`;
}

/** Read the "Style this export" panel into renderSpace-ready options. */
function figureStyle() {
  const palette = $("fig-palette").value;
  let themeOverride = null;
  if (palette === "custom") {
    const stim = $("fig-color-stim").value;
    themeOverride = {
      stim: [stim, stim, stim, stim],
      predicted: $("fig-color-pred").value,
    };
  } else if (palette !== "site") {
    // CLONE, never reference directly: FIGURE_PALETTES entries are shared
    // module-level objects reused on every render, on every page. Handing out
    // a live reference and then mutating .stim below (for the mono-colour
    // toggle) would permanently corrupt that preset for everyone, forever.
    const preset = Plot.FIGURE_PALETTES[palette];
    themeOverride = preset ? { ...preset, stim: [...preset.stim] } : null;
  }

  if ($("fig-mono-stim").checked) {
    const base = themeOverride
      ? themeOverride.stim[0]
      : Plot.theme().stim[0];
    themeOverride = {
      ...(themeOverride || {}),
      stim: [base, base, base, base],
    };
  }

  let background = $("fig-background").value;
  if (background === "palette") {
    // "Match palette": resolve to that theme's own paper colour. For "site" (no
    // override) this falls back to transparent, since the live site theme's
    // --paper already shows through the page CSS behind the canvas anyway.
    background = themeOverride?.paper ?? "transparent";
  }

  return {
    theme: themeOverride,
    background,
    title: $("fig-title").value.trim() || null,
    showAxisNames: $("fig-show-axis-names").checked,
    showLevelTicks: $("fig-show-ticks").checked,
    bakeLegend: $("fig-legend-on-png").checked,
  };
}

// Rapid palette-switching can fire several renders whose font-preload promises
// resolve out of order; a token guard makes sure only the LAST one triggered
// ever actually paints, so the canvas can't flicker back to a stale style.
let figureRenderToken = 0;

async function renderFigure(id) {
  const myToken = ++figureRenderToken;
  const entry = cache.get(id);
  if (!entry) return;
  const { grin, mleFull } = entry;
  const L = currentLabels;

  const stimuli = Plot.toStimuli(grin.mean);
  const showPredicted = $("fig-show-predicted").checked;
  $("overlay-select").disabled = !showPredicted;

  const overlay = $("overlay-select").value;
  const predicted = showPredicted
    ? overlay === "mle"
      ? Plot.toStimuli(mleFull.params)
      : Plot.toStimuli(Core.constrain(grin.mean, grin.modelClass.name))
    : null;
  const predictedLabel =
    overlay === "mle"
      ? "MLE (unconstrained)"
      : `Best model (${grin.modelClass.name})`;

  $("overlay-cap").textContent = !showPredicted
    ? "Comparison ellipses are hidden, just GRIN's own estimate below."
    : overlay === "mle"
      ? "How far GRIN's posterior mean sits from the classical maximum-likelihood fit on the same data."
      : "How much GRIN's free per-stimulus estimate had to move to satisfy its own best-fitting structural model.";

  const style = figureStyle();
  $("fig-palette-note").textContent =
    PALETTE_NOTES[$("fig-palette").value] || "";

  // A webfont used ONLY on canvas (never applied via CSS to any DOM element)
  // is not guaranteed to be loaded yet, fillText() would silently fall back
  // to the default font for that draw, with no error. Explicitly request it
  // first; if the load fails (offline, blocked font host, older browser
  // without the Font Loading API), draw anyway with whatever font resolves.
  // Skip this for a palette like "blueprint" whose titleFont is already a
  // system font stack (contains a comma), there's nothing to fetch.
  const wf = style.theme?.titleFont;
  if (wf && !wf.includes(",") && document.fonts?.load) {
    try {
      await document.fonts.load(`16px "${wf}"`);
    } catch {
      /* fall back silently */
    }
  }
  if (myToken !== figureRenderToken) return; // a newer render started meanwhile

  // Same pattern as the font preload just above: an image needs to be loaded
  // before drawImage() will show it (a not-yet-loaded Image just draws
  // nothing, silently), request it first, respect the render token after.
  let stimMarkerImage = null;
  if (spursUnlocked && $("fig-spurs-marker").checked) {
    stimMarkerImage = await loadImageOnce(
      "spur",
      "./assets/images/spur-icon.svg",
    );
  }
  if (myToken !== figureRenderToken) return; // a newer render started meanwhile

  Plot.renderSpace($("space"), {
    stimuli,
    predicted,
    labels: L,
    showMarginals: $("fig-marginals").checked,
    theme: style.theme,
    background: style.background,
    title: style.title,
    showAxisNames: style.showAxisNames,
    showLevelTicks: style.showLevelTicks,
    stimMarkerImage,
    legend: style.bakeLegend
      ? {
          stimuli: [0, 1, 2, 3].map((i) => Plot.stimLabel(L, i)),
          predictedLabel: showPredicted ? predictedLabel : null,
        }
      : null,
  });
  // the DOM legend below the canvas always shows (cheap, always readable on
  // screen), independent of whether it's ALSO baked into the exported PNG
  Plot.buildLegend($("space-legend"), L, {
    predictedLabel,
    theme: style.theme,
  });
  $("figure-card").hidden = false;
}

function applyFigureLabels() {
  const [a1, a2] = Plot.parseLevels(
    $("fig-aLevels").value,
    currentLabels.a1
      ? [currentLabels.a1, currentLabels.a2]
      : ["A1", "A2"],
  );
  const [b1, b2] = Plot.parseLevels(
    $("fig-bLevels").value,
    currentLabels.b1
      ? [currentLabels.b1, currentLabels.b2]
      : ["B1", "B2"],
  );
  currentLabels = {
    aName: $("fig-aName").value || "Dimension A",
    a1,
    a2,
    bName: $("fig-bName").value || "Dimension B",
    b1,
    b2,
  };
  if (currentParticipant && cache.has(currentParticipant))
    renderParticipant(currentParticipant);
}
["fig-aName", "fig-aLevels", "fig-bName", "fig-bLevels"].forEach((id) =>
  $(id).addEventListener("input", applyFigureLabels),
);

$("overlay-select").addEventListener("change", () => {
  if (currentParticipant) renderFigure(currentParticipant);
});
$("fig-show-predicted").addEventListener("change", () => {
  if (currentParticipant) renderFigure(currentParticipant);
});
$("fig-marginals").addEventListener("change", () => {
  if (currentParticipant) renderFigure(currentParticipant);
});
// --------------------------------------------------------------------------- //
// An easter egg. Plays the real recording; if it's missing, blocked, or fails
// to decode for any reason, does nothing, no synthesized substitute.
// --------------------------------------------------------------------------- //
const EAGLE_AUDIO_SRC = "./assets/audio/eagle-screech.mp3";
let eagleAudio = null;

function playEagleScreech() {
  try {
    if (!eagleAudio) {
      eagleAudio = new Audio(EAGLE_AUDIO_SRC);
      eagleAudio.volume = 0.65;
      eagleAudio.preload = "auto";
    }
    eagleAudio.currentTime = 0;
    const p = eagleAudio.play();
    if (p?.catch) p.catch(() => {}); // missing file, autoplay block, decode error, just skip it
  } catch {
    /* no audio support, it's an easter egg, not a feature */
  }
}

/** A brief on-screen firework flourish. Lives entirely outside the exported
 * PNG, the PNG gets its own static spark decoration via the "fireworks"
 * texture; this is purely a viewport celebration, and removes itself. Fades
 * through navy rather than clearing to transparent each frame, which leaves
 * short glowing trails behind each spark instead of hard-edged dots. */
function celebrateFireworks() {
  try {
    const canvas = document.createElement("canvas");
    canvas.style.cssText =
      "position:fixed;inset:0;pointer-events:none;z-index:9999;";
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    document.body.appendChild(canvas);
    const g = canvas.getContext("2d");
    const colors = ["#ff5a5a", "#5a8dff", "#ffe14d", "#ffffff"];
    let particles = [];

    const burst = (x, y) => {
      const color = colors[Math.floor(Math.random() * colors.length)];
      for (let i = 0; i < 40; i++) {
        const ang = (i / 40) * Math.PI * 2;
        const speed = 2 + Math.random() * 3;
        particles.push({
          x,
          y,
          vx: Math.cos(ang) * speed,
          vy: Math.sin(ang) * speed,
          life: 1,
          color,
        });
      }
    };
    burst(canvas.width * 0.3, canvas.height * 0.3);
    setTimeout(
      () => burst(canvas.width * 0.7, canvas.height * 0.25),
      250,
    );
    setTimeout(
      () => burst(canvas.width * 0.5, canvas.height * 0.42),
      500,
    );

    const start = performance.now();
    function frame(t) {
      g.globalAlpha = 0.2;
      g.fillStyle = "#002868";
      g.fillRect(0, 0, canvas.width, canvas.height);
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.04;
        p.life -= 0.012;
        g.globalAlpha = Math.max(0, p.life);
        g.fillStyle = p.color;
        g.beginPath();
        g.arc(p.x, p.y, 2, 0, Math.PI * 2);
        g.fill();
      }
      g.globalAlpha = 1;
      particles = particles.filter((p) => p.life > 0);
      if (t - start < 2200) requestAnimationFrame(frame);
      else canvas.remove();
    }
    requestAnimationFrame(frame);
  } catch {
    /* also just an easter egg */
  }
}

$("export-png").addEventListener("click", () => {
  const name = currentParticipant
    ? `grin_${currentParticipant}_space.png`
    : "grin_space.png";
  Plot.exportPNG($("space"), name);
  if ($("fig-palette").value === "independenceDay") {
    playEagleScreech();
    celebrateFireworks();
  }
});

// --- figure style panel -------------------------------------------------- //
$("fig-palette").addEventListener("change", () => {
  const val = $("fig-palette").value;
  const needsFonts = [
    "trueGrit",
    "matrixConsole",
    "chalkboard",
    "independenceDay",
    "senseAndSeparability",
    "perceptualSpace",
    "spursRetroFiesta",
  ].includes(val);
  if (needsFonts) ensureFigureFonts();
  $("fig-custom-colors").hidden = val !== "custom";
  // Themed palettes (True GRiT's sepia, Matrix's black, etc.) look wrong on a
  // transparent or mismatched background, so suggest the matching one, but
  // only as a default; the user can still pick something else afterward.
  if (val !== "site" && val !== "custom")
    $("fig-background").value = "palette";
  // Picking a Spurs palette is a pretty strong signal you want the spur
  // marker too, suggest it on, but leave it fully overridable.
  if (
    spursUnlocked &&
    (val === "spursModern" || val === "spursRetroFiesta")
  ) {
    $("fig-spurs-marker").checked = true;
  }
  if (currentParticipant) renderFigure(currentParticipant);
});
[
  "fig-background",
  "fig-legend-on-png",
  "fig-show-axis-names",
  "fig-show-ticks",
  "fig-mono-stim",
  "fig-color-stim",
  "fig-color-pred",
  "fig-spurs-marker",
].forEach((id) =>
  $(id).addEventListener("input", () => {
    if (currentParticipant) renderFigure(currentParticipant);
  }),
);

$("fig-title").addEventListener("input", () => {
  checkSpursUnlock();
  if (currentParticipant) renderFigure(currentParticipant);
});

$("fig-style-reset").addEventListener("click", () => {
  $("fig-palette").value = "site";
  $("fig-custom-colors").hidden = true;
  $("fig-background").value = "transparent";
  $("fig-title").value = "";
  $("fig-mono-stim").checked = false;
  $("fig-legend-on-png").checked = true;
  $("fig-show-axis-names").checked = true;
  $("fig-show-ticks").checked = true;
  if (spursUnlocked) {
    $("fig-spurs-marker").checked = false;
  }
  if (currentParticipant) renderFigure(currentParticipant);
});

$("analyse-one").addEventListener("click", () =>
  analyseOne($("participant-select").value),
);

// --------------------------------------------------------------------------- //
// Batch mode
// --------------------------------------------------------------------------- //
async function analyseAll() {
  const ids = [...groups.keys()];
  $("batch-card").hidden = false;
  $("batch-card").scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
  $("batch-status").textContent = `Analysing 0 / ${ids.length}…`;
  const rows = [];
  let skipped = 0;

  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    const agg = aggFor(id);
    const checks = IO.checkInputs(agg, { hasRT });
    if (checks.errors.length) {
      skipped++;
      $("batch-status").textContent =
        `Analysing ${i + 1} / ${ids.length}… (${skipped} skipped)`;
      continue;
    }

    const m = await getModel();
    const grin = await m.predict(agg);
    const mleFull = Fit.fitClass(agg.counts, agg.trials, "ds");
    const mleSel = Fit.fitAndSelect(agg.counts, agg.trials, "bic");
    cache.set(id, {
      agg,
      checks,
      grin,
      mleFull,
      mleSel,
      mleMs: mleSel.ms,
    });

    const total = agg.trials.reduce((a, b) => a + b, 0);
    const acc =
      [0, 1, 2, 3].reduce((s, k) => s + agg.counts[k][k], 0) /
      Math.max(1, total);
    rows.push({ id, total, acc, grin, mleFull, mleSel });
    $("batch-status").textContent = `Analysing ${i + 1} / ${ids.length}…`;
    // yield to the UI thread so the status text actually paints
    await new Promise((r) => setTimeout(r, 0));
  }

  renderBatch(rows, skipped);
}

function renderBatch(rows, skipped) {
  const header = `<tr><th style="text-align:left">Participant</th><th>n</th><th>Acc.</th>
    <th>GRIN class</th><th>Factorized support</th><th>GRIN ms</th>
    <th>MLE class (BIC)</th><th>MLE weight</th><th>MLE ms</th></tr>`;
  const body = rows
    .map(
      (r) => `<tr>
    <td style="text-align:left"><a href="#" class="batch-row-link" data-id="${escapeAttr(r.id)}"
  style="font-family:Inter,sans-serif">${escapeHTML(r.id)}</a></td>
    <td>${r.total}</td><td>${(100 * r.acc).toFixed(1)}%</td>
    <td>${r.grin.modelClass.name}</td><td>${r.grin.modelClass.factorizedSupport.toFixed(2)}</td>
    <td>${r.grin.ms.toFixed(2)}</td>
    <td>${r.mleSel.best.model}</td><td>${r.mleSel.best.weight.toFixed(2)}</td>
    <td>${r.mleSel.ms.toFixed(1)}</td>
  </tr>`,
    )
    .join("");
  $("batch-table").innerHTML = header + body;
  $("batch-status").innerHTML =
    `<span class="pill ok">${rows.length} analysed</span>` +
    (skipped
      ? ` <span class="pill warn">${skipped} skipped (see per-participant notes)</span>`
      : "");
  window.__grinBatchRows = rows; // for CSV export

  // batch mode already populated `cache` for every non-skipped participant, so
  // jumping to one's figure is a cache-only redraw, no refit.
  document.querySelectorAll(".batch-row-link").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const id = a.dataset.id;
      $("participant-select").value = id;
      renderParticipant(id);
      $("figure-card").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  });
}

$("analyse-all").addEventListener("click", analyseAll);
$("dl-batch-csv").addEventListener("click", () => {
  const rows = window.__grinBatchRows || [];
  if (!rows.length) return;
  const cols = [
    "participant",
    "n_trials",
    "accuracy",
    "grin_model",
    "grin_factorized_support",
    "grin_ms",
    "mle_bic_model",
    "mle_bic_weight",
    "mle_ms",
    ...Core.PARAM_NAMES.map((n) => "grin_" + n),
    ...Core.PARAM_NAMES.map((n) => "mle_" + n),
  ];
  const lines = [cols.join(",")];
  for (const r of rows) {
    lines.push(
      [
        csvEscape(r.id),
        r.total,
        r.acc.toFixed(4),
        r.grin.modelClass.name,
        r.grin.modelClass.factorizedSupport.toFixed(3),
        r.grin.ms.toFixed(3),
        r.mleSel.best.model,
        r.mleSel.best.weight.toFixed(3),
        r.mleSel.ms.toFixed(2),
        ...r.grin.mean.map((v) => v.toFixed(4)),
        ...r.mleFull.params.map((v) => v.toFixed(4)),
      ].join(","),
    );
  }
  downloadText(lines.join("\n") + "\n", "grin_batch_results.csv");
});

// --------------------------------------------------------------------------- //
// Small helpers
// --------------------------------------------------------------------------- //
function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c],
  );
}
function escapeAttr(s) {
  return escapeHTML(s);
}
function csvEscape(s) {
  const t = String(s);
  return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
}

Plot.onThemeChange(() => {
  const id = $("participant-select").value;
  if (id && cache.has(id)) renderParticipant(id);
});
    
