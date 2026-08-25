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
  <h4>${tellable ? "Just about distinguishable" : "Hard to tell apart"} at ${n} trials per stimulus</h4>
  <p>The largest change any cell shows between ρ = 0 and ρ = ${rho.toFixed(2)} is
  ${(100 * maxd).toFixed(1)} percentage points. At ${n} trials per stimulus, ordinary
  sampling noise on that cell is about &plusmn;${(100 * noise95).toFixed(1)} points.
  ${
    tellable
      ? "The signal is a little larger than the noise here, so with enough participants the correlation is recoverable."
      : "The signal sits inside the noise, so the two matrices look the same to the data and the correlation cannot be read off with any confidence."
  }</p>
</div>`;
  }

  // Section 2: frontier
  let model = null;
  async function getModel() {
    const status = (msg) => {
      const el = $("status") || $("run-status") || $("fit-status");
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
      const inBin = frontierData.data.filter(([r]) => r >= lo && r < hi);
      if (inBin.length) {
        const rate = inBin.reduce((a, [, v]) => a + v, 0) / inBin.length;
        pts.push([(lo + hi) / 2, rate]);
      }
    }
    drawFrontier($("frontier"), pts, T);

    const truePI = frontierData.data.filter(([r]) => r < 0.05);
    const spec = truePI.length
      ? 1 - truePI.reduce((a, [, v]) => a + v, 0) / truePI.length
      : NaN;
    const strong = frontierData.data.filter(([r]) => r >= 0.6);
    const det = strong.length
      ? strong.reduce((a, [, v]) => a + v, 0) / strong.length
      : NaN;
    $("frontier-note").innerHTML = `
<div class="note">
  <h4>Read from ${frontierData.M} participants at ${frontierData.n} trials per stimulus</h4>
  <p>When the dimensions truly are independent, GRIN kept that call about
  ${(100 * spec).toFixed(0)}% of the time. By |ρ| around 0.6 and above it reported the
  correlation about ${(100 * det).toFixed(0)}% of the time. The climb between the two is
  where weak correlations live, and they are close to invisible at this sample size.
  More trials per stimulus shift that climb slowly, which is the point.</p>
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
        `<span class="pill bad">Couldn't load the model (${e.message}).</span>`;
      return;
    }
    $("runB").disabled = true;
    $("statusB").textContent = "Simulating…";
    $("progressB").style.width = "0%";

    const rng = Sim.makeRNG();
    const U = (lo, hi) => lo + (hi - lo) * rng.uniform();
    const data = [];
    for (let k = 0; k < M; k++) {
      const rho = U(0, 0.9);
      const rep = Sim.buildRepresentation({
        mA: U(0.6, 2.0),
        mB: U(0.6, 2.0),
        rho,
        dA: 0,
        dB: 0,
      });
      const sim = Sim.simulateCounts(rep, n, rng);
      const res = await mdl.predict(sim);
      const notPI = !(
        res.corr.pi >= res.corr.rho1 && res.corr.pi >= res.corr.free
      );
      data.push([rho, notPI ? 1 : 0]);
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
    $("statusB").innerHTML =
      `<span class="pill ok">${M} participants &times; ${n} trials/stimulus.</span>`;
  }

  // ---------------------------------------------------------------- wiring
  ["rhoA", "nA"].forEach((id) =>
    $(id).addEventListener("input", renderCompare),
  );
  $("nB").addEventListener(
    "input",
    () => ($("v-nB").textContent = $("nB").value),
  );
  $("mB").addEventListener(
    "input",
    () => ($("v-mB").textContent = $("mB").value),
  );
  $("runB").addEventListener("click", runFrontier);
  Plot.onThemeChange(() => {
    renderCompare();
    renderFrontier();
  });

  renderCompare();

