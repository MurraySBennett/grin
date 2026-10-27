  import * as Core from "../grt-core.js";
  import * as Sim from "../grt-sim.js";
  import * as Plot from "../grt-plot.js";
  import { loadModelCached } from "../grin-model.js";

  const $ = (id) => document.getElementById(id);
  const L = Plot.DEFAULT_LABELS;

  function ctxFor(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.maxWidth = "100%";
    canvas.style.height = "auto";
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return g;
  }

  // Section 1: two spaces
  function repVec(rho) {
    const rep = Sim.buildRepresentation({
      mA: 1.2,
      mB: 1.2,
      rho,
      dA: 0,
      dB: 0,
    });
    return [...rep.zx, ...rep.zy, ...rep.rho];
  }

  function renderCompare() {
    const rho = +$("rhoA").value;
    const n = +$("nA").value;
    $("v-rhoA").textContent = rho.toFixed(2);
    $("v-nA").textContent = n;
    $("cm-corr-cap").textContent = `Correlated (ρ = ${rho.toFixed(2)})`;

    const v0 = repVec(0);
    const vr = repVec(rho);
    const P0 = Core.forwardProbabilities(v0);
    const Pr = Core.forwardProbabilities(vr);

    Plot.renderSpace($("space-indep"), {
      stimuli: Plot.toStimuli(v0),
      labels: L,
      title: "ρ = 0",
    });
    Plot.renderSpace($("space-corr"), {
      stimuli: Plot.toStimuli(vr),
      labels: L,
      title: `ρ = ${rho.toFixed(2)}`,
    });
    Plot.renderCM($("cm-indep"), P0, L);
    Plot.renderCM($("cm-corr"), Pr, L);

    let maxd = 0;
    let pAt = 0;
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        const d = Math.abs(P0[i][j] - Pr[i][j]);
        if (d > maxd) {
          maxd = d;
          pAt = Pr[i][j];
        }
      }
    const noise95 = 1.96 * Math.sqrt((pAt * (1 - pAt)) / n);
    const tellable = maxd > noise95;
    $("compare-note").innerHTML = `
<div class="note ${tellable ? "" : "warn"}">
  <h4>${tellable ? "Distinguishable" : "Hard to tell apart"} at ${n} trials per stimulus</h4>
  <p>The largest change any cell shows between ρ = 0 and ρ = ${rho.toFixed(2)} is
  ${(100 * maxd).toFixed(1)} percentage points. At ${n} trials per stimulus, ordinary
  sampling noise on that cell is about &plusmn;${(100 * noise95).toFixed(1)} percentage points.
  ${
    tellable
      ? "The change is larger than the noise. At this trial count, one participant's data can begin to show the correlation."
      : "The change is smaller than the noise, so the two matrices look the same in the data and the correlation cannot be read with any confidence."
  }</p>
</div>`;
  }

  // Section 2: frontier
  let model = null;
  async function getModel() {
    const status = (msg) => {
      const el = $("statusB");
      if (el && msg) el.textContent = msg;
    };
    if (!model) model = await loadModelCached("./assets/models/cm", status);
    return model;
  }

  let frontierData = null;

  function drawFrontier(canvas, pts, T) {
    const W = 560,
      H = 320,
      m = 44,
      pad = 16;
    const g = ctxFor(canvas, W, H);
    const x0 = m,
      y0 = H - m,
      plotW = W - m - pad,
      plotH = H - m - pad;
    const X = (v) => x0 + (v / 0.9) * plotW;
    const Y = (v) => y0 - v * plotH;

    g.strokeStyle = T.line;
    g.lineWidth = 1;
    g.strokeRect(x0, pad, plotW, plotH);

    // y gridlines at 0.5 and 1
    // g.strokeStyle = T.line;
    // g.setLineDash([2, 4]);
    // [0.5].forEach((yy) => {
    //   g.beginPath();
    //   g.moveTo(x0, Y(yy));
    //   g.lineTo(x0 + plotW, Y(yy));
    //   g.stroke();
    // });
    // g.setLineDash([]);

    // curve
    g.strokeStyle = T.predicted;
    g.fillStyle = T.predicted;
    g.lineWidth = 2;
    g.beginPath();
    pts.forEach(([px, py], i) =>
      i ? g.lineTo(X(px), Y(py)) : g.moveTo(X(px), Y(py)),
    );
    g.stroke();
    for (const [px, py] of pts) {
      g.beginPath();
      g.arc(X(px), Y(py), 3, 0, Math.PI * 2);
      g.fill();
    }

    // axes labels
    g.fillStyle = T.mute;
    g.font = "11px Inter, system-ui, sans-serif";
    g.textAlign = "center";
    ["0", "0.3", "0.6", "0.9"].forEach((t) =>
      g.fillText(t, X(+t), H - m + 15),
    );
    g.fillText("true |ρ|", (x0 + X(0.9)) / 2, H - 6);
    g.textAlign = "right";
    g.fillText("100%", x0 - 6, Y(1) + 3);
    g.fillText("50%", x0 - 6, Y(0.5) + 3);
    g.fillText("0%", x0 - 6, Y(0) + 3);
    g.save();
    g.translate(12, (pad + y0) / 2);
    g.rotate(-Math.PI / 2);
    g.textAlign = "center";
    g.fillText("reports a correlation", 0, 0);
    g.restore();
  }

  function renderFrontier() {
    if (!frontierData) return;
    const T = Plot.theme();
    const edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    const pts = [];
    for (let b = 0; b < edges.length - 1; b++) {
      const lo = edges[b],
        hi = edges[b + 1];
      const inBin = frontierData.data.filter(
        ([r]) => Math.abs(r) >= lo && Math.abs(r) < hi,
      );
      if (inBin.length) {
        const rate = inBin.reduce((a, [, v]) => a + v, 0) / inBin.length;
        pts.push([(lo + hi) / 2, rate]);
      }
    }
    drawFrontier($("frontier"), pts, T);

    // Every sentence below is computed from this run. Rows are
    // [rho, reportedCorrelation, nonSeparable].
    const D = frontierData.data;
    const pct = (rows) =>
      (100 * rows.reduce((a, r) => a + r[1], 0)) / rows.length;
    const people = (k) => `${k} participant${k === 1 ? "" : "s"}`;
    const near = D.filter(([r]) => Math.abs(r) < 0.05);
    const strong = D.filter(([r]) => Math.abs(r) >= 0.6);
    const parts = [];
    parts.push(
      near.length
        ? `When the true correlation was within 0.05 of zero (${people(near.length)}), GRIN called the dimensions independent in ${(100 - pct(near)).toFixed(0)}% of cases.`
        : "When the true correlation was within 0.05 of zero (0 participants), there was no case to score.",
    );
    parts.push(
      strong.length
        ? `When the correlation was 0.6 or more in size (${people(strong.length)}), GRIN reported a correlation in ${pct(strong).toFixed(0)}% of cases.`
        : "When the correlation was 0.6 or more in size (0 participants), there was no case to score.",
    );
    const cross = pts.find(([, rate]) => rate >= 0.5);
    parts.push(
      cross
        ? `Detection first reached 50% in the band where |ρ| is between ${(cross[0] - 0.05).toFixed(1)} and ${(cross[0] + 0.05).toFixed(1)}.`
        : "Detection did not reach 50% in any band.",
    );
    const nonSep = D.filter((r) => r[2]);
    parts.push(
      `${nonSep.length} of the ${D.length} spaces had at least one dimension that was not separable.`,
    );
    const strongSep = strong.filter((r) => !r[2]);
    const strongNon = strong.filter((r) => r[2]);
    if (strongSep.length && strongNon.length)
      parts.push(
        `For correlations of 0.6 or more in size, GRIN reported a correlation in ${pct(strongSep).toFixed(0)}% of separable spaces (${strongSep.length}) and ${pct(strongNon).toFixed(0)}% of non-separable spaces (${strongNon.length}).`,
      );
    $("frontier-note").innerHTML = `
<div class="note">
  <h4>Results from ${people(frontierData.M)} at ${frontierData.n} trials per stimulus</h4>
  <p>${parts.join(" ")}</p>
</div>`;
  }

  async function runFrontier() {
    const n = +$("nB").value;
    const M = +$("mB").value;
    let mdl;
    try {
      mdl = await getModel();
    } catch (e) {
      $("statusB").innerHTML =
        `<span class="pill bad">Could not load the network (${e.message}).</span>`;
      return;
    }
    $("runB").disabled = true;
    $("statusB").textContent = "Simulating…";
    $("progressB").style.width = "0%";
    const t0 = performance.now();

    const rng = Sim.makeRNG();
    const U = (lo, hi) => lo + (hi - lo) * rng.uniform();
    const data = [];
    for (let k = 0; k < M; k++) {
      // One shared correlation per participant, drawn over GRIN's training
      // range. Each dimension is non-separable with probability 0.5, the rate
      // in GRIN's training prior (6 of its 12 model classes break each one).
      // The shift is at most 1 in size, so with levels at least 0.6 from the
      // bound no stimulus crosses it.
      const rho = U(-0.9, 0.9);
      const dA = rng.uniform() < 0.5 ? U(-1, 1) : 0;
      const dB = rng.uniform() < 0.5 ? U(-1, 1) : 0;
      const rep = Sim.buildRepresentation({
        mA: U(0.6, 2.0),
        mB: U(0.6, 2.0),
        rho,
        dA,
        dB,
      });
      const sim = Sim.simulateCounts(rep, n, rng);
      const res = await mdl.predict(sim);
      const notPI = !(
        res.corr.pi >= res.corr.rho1 && res.corr.pi >= res.corr.free
      );
      data.push([rho, notPI ? 1 : 0, dA !== 0 || dB !== 0]);
      if (k % 25 === 0) {
        $("statusB").textContent = `Simulating… ${k} / ${M}`;
        $("progressB").style.width = `${((100 * k) / M).toFixed(0)}%`;
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    $("progressB").style.width = "100%";
    frontierData = { n, M, data };
    renderFrontier();
    $("runB").disabled = false;
    const secs = (performance.now() - t0) / 1000;
    $("statusB").innerHTML =
      `<span class="pill ok">${M} participants, ${n} trials per stimulus each. Done in ${secs.toFixed(1)} seconds.</span>`;
  }

  // ---------------------------------------------------------------- wiring
  ["rhoA", "nA"].forEach((id) =>
    $(id).addEventListener("input", renderCompare),
  );
  // Readouts always come from the input's actual value, so they cannot drift
  // from what a run uses.
  function syncBatchReadouts() {
    $("v-nB").textContent = $("nB").value;
    $("v-mB").textContent = $("mB").value;
    $("nB-warn").hidden = !(+$("nB").value > 500);
  }
  $("nB").addEventListener("input", syncBatchReadouts);
  $("mB").addEventListener("input", syncBatchReadouts);
  syncBatchReadouts();
  $("runB").addEventListener("click", runFrontier);
  Plot.onThemeChange(() => {
    renderCompare();
    renderFrontier();
  });

  renderCompare();

