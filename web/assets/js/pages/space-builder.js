    import * as Plot from "../grt-plot.js";
    import * as Sim from "../grt-sim.js";
    import { loadModelCached } from "../grin-model.js";
    import { cmAccuracy } from "../grt-plot.js";
    import {
      forwardProbabilities,
      modelWords,
      STIMULUS_ORDER,
    } from "../grt-core.js";

    const $ = (id) => document.getElementById(id);
    const SLIDERS = ["mA", "mB", "rho", "dA", "dB", "rhoSpread", "n"];
    // mA and mB sliders are in d′, the distance between the two levels of a
    // dimension. The model places the levels at ±d′/2 around the bound at 0,
    // so readControls halves them before they reach buildRepresentation.
    const DEFAULTS = {
      mA: 2,
      mB: 2,
      rho: 0,
      dA: 0,
      dB: 0,
      rhoSpread: 0,
      n: 100,
    };

    let nudge = { x: [0, 0, 0, 0], y: [0, 0, 0, 0] };

    let model = null;
    let lastRun = null;

    async function getModel() {
      const status = (msg) => {
        const el = $("status") || $("run-status") || $("fit-status");
        if (el && msg) el.textContent = msg;
      };
      if (model) return model;
      model = await loadModelCached("./assets/models/cm", status);
      return model;
    }

    /** Raw slider values, in display units (d′ for mA and mB). */
    function readSliders() {
      const o = {};
      for (const k of SLIDERS) o[k] = parseFloat($(k).value);
      return o;
    }

    /** Slider values in model units: mA and mB are half of d′. */
    function readControls() {
      const o = readSliders();
      o.mA = o.mA / 2;
      o.mB = o.mB / 2;
      return o;
    }

    // GRIN was trained on 1 to 1,000 trials per stimulus but tested on 5 to 500.
    const TESTED_MAX_TRIALS = 500;
    // Below this separability support the verdict does not call that result firm.
    // Matches the threshold the verdict box already used for its warning style.
    const FIRM_SEPARABILITY = 0.75;

    /** The actual representation on screen: sliders + the independent nudges. */
    function currentRep() {
      return Sim.buildRepresentation({ ...readControls(), nudge });
    }

    /** Parameter labels in the same form as the Analyse page. */
    function paramLabels(L) {
      const named =
        L.a1 !== "A1" || L.a2 !== "A2" || L.b1 !== "B1" || L.b2 !== "B2";
      const stim = (i) => (named ? Plot.stimLabel(L, i) : STIMULUS_ORDER[i]);
      const dimA = L.aName && L.aName !== "Dimension A" ? L.aName : "A";
      const dimB = L.bName && L.bName !== "Dimension B" ? L.bName : "B";
      const out = [];
      for (let i = 0; i < 4; i++) out.push(`${dimA} sensitivity, ${stim(i)}`);
      for (let i = 0; i < 4; i++) out.push(`${dimB} sensitivity, ${stim(i)}`);
      for (let i = 0; i < 4; i++) out.push(`Correlation, ${stim(i)}`);
      return out;
    }

    function escapeHTML(str) {
      return String(str).replace(
        /[&<>"']/g,
        (ch) =>
          ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
            ch
          ],
      );
    }

    function nudgeSummary() {
      const active = [0, 1, 2, 3].filter(
        (i) => nudge.x[i] !== 0 || nudge.y[i] !== 0,
      );
      if (!active.length) return "No single stimulus moved.";
      const names = ["A1/B1", "A1/B2", "A2/B1", "A2/B2"];
      return active
        .map(
          (i) =>
            `${names[i]} moved ${nudge.x[i].toFixed(2)} on A and ${nudge.y[i].toFixed(2)} on B.`,
        )
        .join(" ");
    }

    function labels() {
      const [a1, a2] = Plot.parseLevels($("aLevels").value, ["A1", "A2"]);
      const [b1, b2] = Plot.parseLevels($("bLevels").value, ["B1", "B2"]);
      return {
        aName: $("aName").value || "Dimension A",
        a1,
        a2,
        bName: $("bName").value || "Dimension B",
        b1,
        b2,
      };
    }

    /**
     * @param {boolean} invalidate, true when the CONTROLS changed (the previous run
     * no longer describes the space on screen, so it must be cleared). False when we
     * are only repainting, e.g. on a theme change, repainting must never destroy
     * the user's results.
     */
    function draw(invalidate = true) {
      const c = readSliders();
      for (const k of SLIDERS)
        $("v-" + k) &&
          ($("v-" + k).textContent = k === "n" ? c[k] : c[k].toFixed(2));
      $("n-warn").hidden = !(c.n > TESTED_MAX_TRIALS);
      $("nudge-summary").textContent = nudgeSummary();

      const rep = currentRep();
      const L = labels();
      const stimuli = [0, 1, 2, 3].map((i) => ({
        zx: rep.zx[i],
        zy: rep.zy[i],
        rho: rep.rho[i],
      }));

      Plot.renderSpace($("space"), {
        stimuli,
        labels: L,
        showMarginals: $("showMarginals").checked,
      });
      Plot.buildLegend($("legend"), L);

      const P = forwardProbabilities([...rep.zx, ...rep.zy, ...rep.rho]);
      Plot.renderCM($("cm"), P, L);

      const acc = cmAccuracy(P);
      $("acc").innerHTML = `<div class="est">
  <div class="lbl">Accuracy</div>
  <div class="track"><div class="ci" style="left:0;width:${(100 * acc).toFixed(1)}%;opacity:.8"></div></div>
  <div class="num">${(100 * acc).toFixed(1)}%</div>
</div>`;

      // what is TRUE of this space, stated before any inference, so the user can
      // check the model's answer against a known truth rather than trusting it.
      // Checked against the actual built vectors (rep), not the sliders (c): a
      // nudge can break PS(A) on its own even when dA is 0, so c alone can no
      // longer tell you the truth.
      const t = Sim.trueAssumptions(rep);
      const pill = (ok, yes, no) =>
        `<span class="pill ${ok ? "ok" : "warn"}">${ok ? yes : no}</span>`;
      $("truth").innerHTML = `
  <p class="cap" style="margin-bottom:.35rem"><strong>Ground truth for this space:</strong></p>
  <div class="row" style="margin-top:0">
    ${pill(t.psA, "A separable", "A not separable")}
    ${pill(t.psB, "B separable", "B not separable")}
    ${pill(t.pi, "independent", t.rho1 ? "one shared correlation" : "correlation varies by stimulus")}
  </div>`;

      // any change to the space invalidates the previous run
      if (invalidate && lastRun) {
        $("recovery-card").hidden = true;
        lastRun = null;
        $("status").textContent = "Space changed. Run the experiment again.";
      }
    }

    async function run() {
      const rep = currentRep();
      const L = labels();
      const n = parseFloat($("n").value);

      $("status").textContent = "Simulating…";
      const data = Sim.simulateCounts(rep, n);

      const m = await getModel();
      const res = await m.predict(data);
      lastRun = { data, res, rep, L, n };

      // the observed (noisy) matrix now replaces the exact one
      const props = data.counts.map((r, i) =>
        r.map((v) => v / Math.max(1, data.trials[i])),
      );
      Plot.renderCM($("cm"), props, L, { counts: data.counts });
      $("cm-cap").textContent =
        `Observed from ${n} trials per stimulus, with counts in small type. ` +
        `These counts include sampling noise.`;

      // recovered space
      const stimuli = [0, 1, 2, 3].map((i) => ({
        zx: rep.zx[i],
        zy: rep.zy[i],
        rho: rep.rho[i],
      }));
      Plot.renderSpace($("space2"), {
        stimuli,
        predicted: Plot.toStimuli(res.mean),
        labels: L,
        showMarginals: false,
      });
      Plot.buildLegend($("legend2"), L, { predictedLabel: "recovered" });

      // estimates with intervals, against the truth
      const names = paramLabels(L);
      const truth = [...rep.zx, ...rep.zy, ...rep.rho];
      let html = "";
      for (let i = 0; i < 12; i++) {
        const range = i < 8 ? [-3, 3] : [-1, 1];
        html += Plot.estRow(
          `${escapeHTML(names[i])} <span style="color:var(--mute-soft)">(true value ${truth[i].toFixed(2)})</span>`,
          res.mean[i],
          res.std[i],
          range,
        );
      }
      $("ests").innerHTML = html;

      // structural conclusions vs ground truth
      const t = Sim.trueAssumptions(rep);
      const yn = (b) => (b ? "yes" : "no");
      $("constructs").innerHTML =
        Plot.pbar(`A separable. True answer: ${yn(t.psA)}`, res.sep.A) +
        Plot.pbar(`B separable. True answer: ${yn(t.psB)}`, res.sep.B) +
        Plot.pbar(`Independent. True answer: ${yn(t.pi)}`, res.corr.pi) +
        Plot.pbar(
          `One shared correlation. True answer: ${yn(t.rho1 && !t.pi)}`,
          res.corr.rho1,
        ) +
        Plot.pbar(
          `Correlation varies by stimulus. True answer: ${yn(!t.rho1)}`,
          res.corr.free,
        );

      const mc = res.modelClass;
      // The factorized support is the product of three marginal decisions:
      // separability on A, separability on B, and the correlation class. The
      // first two are usually well identified; whether rho is 0 (perceptual
      // independence) is information-limited, so it caps the joint. Report the
      // parts, not just the product, so low support reads as "PI is hard"
      // rather than "the fit failed".
      const sepConf = Math.min(
        Math.max(res.sep.A, 1 - res.sep.A),
        Math.max(res.sep.B, 1 - res.sep.B),
      );
      const corrConf = Math.max(res.corr.pi, res.corr.rho1, res.corr.free);
      // Compare the value as shown, so a displayed 0.75 is never called weak.
      const sepFirm = +sepConf.toFixed(2) >= FIRM_SEPARABILITY;
      const corrNote =
        corrConf >= 0.6
          ? "Here the data carry enough to lean one way on the correlation as well."
          : "Lower support for the correlation is expected. Whether the tilt is zero leaves only a faint mark on the table of answers, and more trials help only slowly.";
      const sepNote = sepFirm
        ? corrConf >= 0.6
          ? ""
          : " The separability result is the firm one here."
        : corrConf >= 0.6
          ? " The separability answer is not firm, so treat this class as tentative."
          : " The separability answer is not firm either, so treat this class as tentative.";
      $("verdict").innerHTML = `
  <div class="note ${sepFirm ? "" : "warn"}">
    <h4>Best-supported model class: ${escapeHTML(modelWords(mc.name))}</h4>
    <p>
      Support for the separability answer is ${sepConf.toFixed(2)}.
      Support for the best correlation answer is ${corrConf.toFixed(2)}.
      Together they give this class an overall support of ${mc.factorizedSupport.toFixed(2)}.
      That overall figure is the separate answers multiplied together. It has not
      been checked as a probability in its own right.
      ${corrNote}${sepNote}
    </p>
  </div>`;

      $("ntrials-echo").textContent = n;
      $("recovery-card").hidden = false;
      $("what-happened").open = true;
      $("status").textContent = `Done in ${res.ms.toFixed(1)} milliseconds.`;
      $("recovery-card").scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }

    for (const k of SLIDERS) $(k).addEventListener("input", () => draw(true));
    // Renaming a dimension or toggling the marginals does NOT change the space
    ["aName", "aLevels", "bName", "bLevels", "showMarginals"].forEach((id) =>
      $(id).addEventListener("input", () => {
        draw(false);
        if (lastRun) redrawRun();
      }),
    );
    $("run").addEventListener("click", run);
    $("reset").addEventListener("click", () => {
      for (const [k, v] of Object.entries(DEFAULTS)) $(k).value = v;
      nudge = { x: [0, 0, 0, 0], y: [0, 0, 0, 0] };
      $("nudgeX").value = 0;
      $("nudgeY").value = 0;
      $("nudgeStim").value = "0";
      $("recovery-card").hidden = true;
      lastRun = null;
      draw();
      $("status").textContent =
        "Reset. Adjust the space, then run an experiment.";
    });

    // --- nudge controls: the dropdown selects WHICH stimulus the two sliders
    // below it are currently editing
    function syncNudgeSliders() {
      const i = +$("nudgeStim").value;
      $("nudgeX").value = nudge.x[i];
      $("nudgeY").value = nudge.y[i];
      $("v-nudgeX").textContent = nudge.x[i].toFixed(2);
      $("v-nudgeY").textContent = nudge.y[i].toFixed(2);
    }
    $("nudgeStim").addEventListener("change", syncNudgeSliders);
    $("nudgeX").addEventListener("input", () => {
      nudge.x[+$("nudgeStim").value] = parseFloat($("nudgeX").value);
      $("v-nudgeX").textContent = (+$("nudgeX").value).toFixed(2);
      draw(true);
    });
    $("nudgeY").addEventListener("input", () => {
      nudge.y[+$("nudgeStim").value] = parseFloat($("nudgeY").value);
      $("v-nudgeY").textContent = (+$("nudgeY").value).toFixed(2);
      draw(true);
    });
    $("nudge-reset").addEventListener("click", () => {
      nudge = { x: [0, 0, 0, 0], y: [0, 0, 0, 0] };
      syncNudgeSliders();
      draw(true);
    });

    // A theme change must only REDRAW. Re-running run() here would silently
    // resample the experiment, so toggling dark mode would change the results.
    Plot.onThemeChange(() => {
      draw(false);
      if (lastRun) redrawRun();
    });

    function redrawRun() {
      const { data, res, rep, L, n } = lastRun;
      const props = data.counts.map((r, i) =>
        r.map((v) => v / Math.max(1, data.trials[i])),
      );
      Plot.renderCM($("cm"), props, L, { counts: data.counts });
      $("cm-cap").textContent =
        `Observed from ${n} trials per stimulus, with counts in small type. ` +
        `These counts include sampling noise.`;
      Plot.renderSpace($("space2"), {
        stimuli: [0, 1, 2, 3].map((i) => ({
          zx: rep.zx[i],
          zy: rep.zy[i],
          rho: rep.rho[i],
        })),
        predicted: Plot.toStimuli(res.mean),
        labels: L,
        showMarginals: false,
      });
      Plot.buildLegend($("legend2"), L, { predictedLabel: "recovered" });
      $("recovery-card").hidden = false;
    }

    draw();
  
