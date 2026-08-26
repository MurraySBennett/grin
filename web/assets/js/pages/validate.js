  import * as Sim from "../grt-sim.js";
  import * as Core from "../grt-core.js";
  import * as Plot from "../grt-plot.js";
  import { loadModelCached } from "../grin-model.js";

  const $ = (id) => document.getElementById(id);

  let model = null;
  async function getModel() {
    const status = (msg) => {
      const el = $("status") || $("run-status") || $("fit-status");
      if (el && msg) el.textContent = msg;
    };
    if (!model) model = await loadModelCached("./assets/models/cm", status);
    return model;
  }

  let batch = null;

  // --- sampling a random, in-range true representation ---------------------
  // Uses the same builder the Space Builder uses, with randomised controls, so
  // the truth always lands inside the range the network was trained on.
  function sampleTruth(rng) {
    const U = (lo, hi) => lo + (hi - lo) * rng.uniform();
    return Sim.buildRepresentation({
      mA: U(0.3, 2.2),
      mB: U(0.3, 2.2),
      rho: rng.uniform() < 0.3 ? 0 : U(-0.85, 0.85),
      dA: rng.uniform() < 0.4 ? 0 : U(-1.0, 1.0),
      dB: rng.uniform() < 0.4 ? 0 : U(-1.0, 1.0),
      rhoSpread: rng.uniform() < 0.5 ? 0 : U(0, 0.35),
    });
  }

  // --- stats helpers -------------------------------------------------------
  function pearson(x, y) {
    const n = x.length;
    let sx = 0,
      sy = 0,
      sxx = 0,
      syy = 0,
      sxy = 0;
    for (let i = 0; i < n; i++) {
      sx += x[i];
      sy += y[i];
      sxx += x[i] * x[i];
      syy += y[i] * y[i];
      sxy += x[i] * y[i];
    }
    const cov = n * sxy - sx * sy;
    const dx = Math.sqrt(n * sxx - sx * sx);
    const dy = Math.sqrt(n * syy - sy * sy);
    return dx && dy ? cov / (dx * dy) : NaN;
  }

  /** Empirical coverage of the central `level` interval, assuming the
   * Gaussian marginal the network actually returns (mean ± z·sd). */
  function coverageAt(tru, est, sd, level) {
    const z = Core.nppf(0.5 + level / 2);
    let c = 0;
    for (let i = 0; i < tru.length; i++)
      if (Math.abs(tru[i] - est[i]) <= z * sd[i]) c++;
    return c / tru.length;
  }

  // --- tiny self-contained canvas plotting  --------
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

  function drawScatter(canvas, xs, ys, lo, hi, color, T) {
    const W = 300,
      H = 300,
      m = 38,
      pad = 12;
    const g = ctxFor(canvas, W, H);
    const x0 = m,
      y0 = H - m,
      plotW = W - m - pad,
      plotH = H - m - pad;
    const X = (v) => x0 + ((v - lo) / (hi - lo)) * plotW;
    const Y = (v) => y0 - ((v - lo) / (hi - lo)) * plotH;

    // frame
    g.strokeStyle = T.line;
    g.lineWidth = 1;
    g.strokeRect(x0, pad, plotW, plotH);

    // identity line
    g.strokeStyle = T.mute;
    g.setLineDash([5, 4]);
    g.lineWidth = 1.2;
    g.beginPath();
    g.moveTo(X(lo), Y(lo));
    g.lineTo(X(hi), Y(hi));
    g.stroke();
    g.setLineDash([]);

    // points
    g.fillStyle = color;
    g.globalAlpha = xs.length > 1200 ? 0.22 : 0.38;
    for (let i = 0; i < xs.length; i++) {
      g.beginPath();
      g.arc(X(xs[i]), Y(ys[i]), 2.1, 0, Math.PI * 2);
      g.fill();
    }
    g.globalAlpha = 1;

    // axis ticks + labels
    g.fillStyle = T.mute;
    g.font = "11px Inter, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText(String(lo), X(lo), H - m + 16);
    g.fillText(String(hi), X(hi), H - m + 16);
    g.fillText("true", (x0 + X(hi)) / 2, H - 6);
    g.save();
    g.translate(12, (pad + y0) / 2);
    g.rotate(-Math.PI / 2);
    g.fillText("recovered", 0, 0);
    g.restore();
    g.textAlign = "right";
    g.fillText(String(hi), x0 - 6, Y(hi) + 3);
    g.fillText(String(lo), x0 - 6, Y(lo) + 3);
  }

  function drawCalib(canvas, series, T) {
    const W = 320,
      H = 300,
      m = 40,
      pad = 14;
    const g = ctxFor(canvas, W, H);
    const x0 = m,
      y0 = H - m,
      plotW = W - m - pad,
      plotH = H - m - pad;
    const X = (v) => x0 + v * plotW;
    const Y = (v) => y0 - v * plotH;

    g.strokeStyle = T.line;
    g.lineWidth = 1;
    g.strokeRect(x0, pad, plotW, plotH);

    g.strokeStyle = T.mute;
    g.setLineDash([5, 4]);
    g.lineWidth = 1.2;
    g.beginPath();
    g.moveTo(X(0), Y(0));
    g.lineTo(X(1), Y(1));
    g.stroke();
    g.setLineDash([]);

    for (const s of series) {
      g.strokeStyle = s.color;
      g.fillStyle = s.color;
      g.lineWidth = 1.8;
      g.beginPath();
      s.pts.forEach(([nx, ny], i) =>
        i ? g.lineTo(X(nx), Y(ny)) : g.moveTo(X(nx), Y(ny)),
      );
      g.stroke();
      for (const [nx, ny] of s.pts) {
        g.beginPath();
        g.arc(X(nx), Y(ny), 2.6, 0, Math.PI * 2);
        g.fill();
      }
    }

    // axis labels
    g.fillStyle = T.mute;
    g.font = "11px Inter, system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText("stated confidence", (x0 + X(1)) / 2, H - 6);
    ["0", "0.5", "1"].forEach((t, i) =>
      g.fillText(t, X(i / 2), H - m + 15),
    );
    g.save();
    g.translate(12, (pad + y0) / 2);
    g.rotate(-Math.PI / 2);
    g.fillText("actual coverage", 0, 0);
    g.restore();

    // legend
    g.textAlign = "left";
    let ly = pad + 6;
    for (const s of series) {
      g.fillStyle = s.color;
      g.fillRect(X(0) + 8, ly - 7, 10, 3);
      g.fillStyle = T.ink;
      g.fillText(s.label, X(0) + 22, ly - 2);
      ly += 15;
    }
  }

  function renderAll() {
    if (!batch) return;
    const T = Plot.theme();
    const cSens = T.slate || T.stim[1];
    const cCorr = T.predicted;

    drawScatter(
      $("sc-sens"),
      batch.sensTrue,
      batch.sensRec,
      -3,
      3,
      cSens,
      T,
    );
    drawScatter(
      $("sc-corr"),
      batch.corrTrue,
      batch.corrRec,
      -1,
      1,
      cCorr,
      T,
    );

    const rS = pearson(batch.sensTrue, batch.sensRec);
    const rC = pearson(batch.corrTrue, batch.corrRec);
    $("rec-summary").innerHTML = `
<div class="note">
  <h4>Recovery over ${batch.M} participants at ${batch.n} trials/stimulus</h4>
  <p>Sensitivities recover closely (r &asymp; <strong>${rS.toFixed(2)}</strong>).
  Correlations are recovered less sharply (r &asymp; <strong>${rC.toFixed(2)}</strong>) , 
  a weak within-stimulus correlation leaves only a faint trace in one confusion
  matrix, so it is genuinely harder to pin down. This split is a property of the
  data rather than the method.</p>
</div>`;

    const levels = [0.5, 0.7, 0.8, 0.9, 0.95];
    const sSens = levels.map((L) => [
      L,
      coverageAt(batch.sensTrue, batch.sensRec, batch.sensStd, L),
    ]);
    const sCorr = levels.map((L) => [
      L,
      coverageAt(batch.corrTrue, batch.corrRec, batch.corrStd, L),
    ]);
    drawCalib(
      $("calib"),
      [
        { label: "sensitivities", color: cSens, pts: sSens },
        { label: "correlations", color: cCorr, pts: sCorr },
      ],
      T,
    );

    // Report the two families SEPARATELY. Pooling them averages miscalibrations that
    // run in opposite directions -- the sensitivity intervals are wider than nominal and
    // the correlation intervals narrower -- so a single pooled number lands near 90% and
    // describes neither family. This page used to headline that pooled number.
    const sens90 = coverageAt(batch.sensTrue, batch.sensRec, batch.sensStd, 0.9);
    const corr90 = coverageAt(batch.corrTrue, batch.corrRec, batch.corrStd, 0.9);
    const pct = (x) => `${(100 * x).toFixed(0)}%`;
    const verdict = (x) =>
      x > 0.915 ? "wider than nominal" : x < 0.885 ? "narrower than nominal" : "on target";
    $("calib-headline").innerHTML = `
<div class="note">
  <h4>90% intervals: sensitivities ${pct(sens90)}, correlations ${pct(corr90)}</h4>
  <p>The two families are calibrated differently, so they are reported separately.
  Here the sensitivity intervals came out ${verdict(sens90)} and the correlation
  intervals ${verdict(corr90)}. Averaging the two would give a number close to 90%
  that describes neither.</p>
  <p>Across large held-out sets the sensitivities cover about 94% and the correlations
  about 84%, so a nominal 90% interval on a within-stimulus correlation behaves more
  like an 84% one. The released packages ship an optional correction for this
  (<code>calibrated = TRUE</code> in R, <code>calibrated=True</code> in Python); it is
  off by default and does not change point estimates.</p>
  <p class="cap">Measured live on the batch above, so it will wobble run to run and
  shift as you change the trial count.</p>
</div>`;
  }

  // --- run -----------------------------------------------------------------
  async function run() {
    const n = +$("n").value;
    const M = +$("m").value;

    let mdl;
    try {
      mdl = await getModel();
    } catch (e) {
      $("status").innerHTML =
        `<span class="pill bad">Couldn't load the model (${e.message}).</span>`;
      return;
    }

    $("run").disabled = true;
    $("status").textContent = "Simulating…";
    $("progress").style.width = "0%";

    const rng = Sim.makeRNG();
    const sensTrue = [],
      sensRec = [],
      sensStd = [];
    const corrTrue = [],
      corrRec = [],
      corrStd = [];

    for (let k = 0; k < M; k++) {
      const rep = sampleTruth(rng);
      const truth = [...rep.zx, ...rep.zy, ...rep.rho];
      const data = Sim.simulateCounts(rep, n, rng);
      const res = await mdl.predict(data);
      for (let i = 0; i < 8; i++) {
        sensTrue.push(truth[i]);
        sensRec.push(res.mean[i]);
        sensStd.push(res.std[i]);
      }
      for (let i = 8; i < 12; i++) {
        corrTrue.push(truth[i]);
        corrRec.push(res.mean[i]);
        corrStd.push(res.std[i]);
      }
      if (k % 25 === 0) {
        $("status").textContent = `Simulating… ${k} / ${M}`;
        $("progress").style.width = `${((100 * k) / M).toFixed(0)}%`;
        await new Promise((r) => setTimeout(r, 0)); // let the UI breathe
      }
    }

    $("progress").style.width = "100%";
    batch = {
      n,
      M,
      sensTrue,
      sensRec,
      sensStd,
      corrTrue,
      corrRec,
      corrStd,
    };
    renderAll();
    $("run").disabled = false;
    $("status").innerHTML =
      `<span class="pill ok">${M} participants &times; ${n} trials/stimulus.</span>`;
  }

  $("n").addEventListener(
    "input",
    () => ($("v-n").textContent = $("n").value),
  );
  $("m").addEventListener(
    "input",
    () => ($("v-m").textContent = $("m").value),
  );
  $("run").addEventListener("click", run);

  Plot.onThemeChange(renderAll);

