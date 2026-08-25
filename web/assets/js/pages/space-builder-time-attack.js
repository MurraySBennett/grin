    import * as Plot from "../grt-plot.js";
    import * as Sim from "../grt-sim.js";
    import { loadModelCached } from "../grin-model.js";
    import { forwardProbabilities } from "../grt-core.js";

    const $ = (id) => document.getElementById(id);
    const SPACE = ["mA", "mB", "rho", "dA", "dB", "rhoSpread"];
    const CLOCK = ["t0", "threshold", "kA", "kB"];
    const ALL = [...SPACE, ...CLOCK, "n"];

    // Independent per-stimulus nudges, layered on top of the symmetric dA/dB/
    // rhoSpread sliders. Kept as separate state (not read fresh from the DOM like
    // the other sliders) because only one stimulus's Δx/Δy is visible at a time.
    let nudge = { x: [0, 0, 0, 0], y: [0, 0, 0, 0] };

    const L = {
      aName: "Dimension A",
      a1: "A1",
      a2: "A2",
      bName: "Dimension B",
      b1: "B1",
      b2: "B2",
    };

    let model = null,
      cmModel = null,
      arch = "parallel_exhaustive",
      lastRun = null;

    async function getModel() {
      const status = (msg) => {
        const el = $("status") || $("run-status") || $("fit-status");
        if (el && msg) el.textContent = msg;
      };
      if (model) return model;
      model = await loadModelCached("./assets/models/cmrt", status);
      return model;
    }

    /** For the "what did RT actually change" comparison, the CM-only model run
     * on the SAME simulated data, ignoring its response times, so the only thing
     * that differs between the two predictions is whether RT was available. */
    async function getCmModel() {
      const status = (msg) => {
        const el = $("status") || $("run-status") || $("fit-status");
        if (el && msg) el.textContent = msg;
      };
      if (cmModel) return cmModel;
      cmModel = await loadModelCached("./assets/models/cm", status);
      return cmModel;
    }

    // --- architecture picker -----------------------------------------------------
    $("arch-choices").innerHTML = Sim.ARCHITECTURES.map((a) => {
      const d = Sim.ARCH_LABELS[a];
      return `<label class="choice" data-a="${a}">
  <input type="radio" name="arch" value="${a}" ${a === arch ? "checked" : ""}>
  <span><span class="what">${d.name}</span><span class="how">${d.rt}</span></span>
</label>`;
    }).join("");

    function syncChoices() {
      document
        .querySelectorAll(".choice")
        .forEach((el) => el.classList.toggle("on", el.dataset.a === arch));
    }
    $("arch-choices").addEventListener("change", (e) => {
      arch = e.target.value;
      syncChoices();
      invalidate("Architecture changed. Run again.");
      drawSpace();
    });
    syncChoices();

    // --- controls ----------------------------------------------------------------
    function readControls() {
      const o = {};
      for (const k of ALL) o[k] = parseFloat($(k).value);
      return o;
    }
    function lbaFrom(c) {
      return { t0: c.t0, threshold: c.threshold, kA: c.kA, kB: c.kB };
    }

    /** The actual representation on screen: sliders + the independent nudges. */
    function currentRep() {
      return Sim.buildRepresentation({ ...readControls(), nudge });
    }

    function nudgeSummary() {
      const active = [0, 1, 2, 3].filter(
        (i) => nudge.x[i] !== 0 || nudge.y[i] !== 0,
      );
      if (!active.length) return "No stimuli nudged.";
      const names = ["A1/B1", "A1/B2", "A2/B1", "A2/B2"];
      return (
        "Nudged: " +
        active
          .map(
            (i) =>
              `${names[i]} (Δx ${nudge.x[i].toFixed(2)}, Δy ${nudge.y[i].toFixed(2)})`,
          )
          .join("; ")
      );
    }

    function invalidate(msg) {
      lastRun = null;
      $("rt-card").hidden = true;
      $("infer-card").hidden = true;
      $("rt-value-card").hidden = true;
      $("compare-card").hidden = true;
      if (msg) $("status").textContent = msg;
    }

    function drawSpace() {
      const c = readControls();
      for (const k of ALL)
        $("v-" + k).textContent = k === "n" ? c[k] : c[k].toFixed(2);
      $("nudge-summary").textContent = nudgeSummary();

      const rep = currentRep();
      const stimuli = [0, 1, 2, 3].map((i) => ({
        zx: rep.zx[i],
        zy: rep.zy[i],
        rho: rep.rho[i],
      }));
      Plot.renderSpace($("space"), {
        stimuli,
        labels: L,
        showMarginals: true,
      });
      Plot.buildLegend($("legend"), L);

      if (!lastRun) {
        Plot.renderCM(
          $("cm"),
          forwardProbabilities([...rep.zx, ...rep.zy, ...rep.rho]),
          L,
        );
        $("cm-cap").textContent =
          "Exact response probabilities implied by the space.";
      }

      // Checked against the built vectors (rep), not the sliders (c): a nudge can
      // break PS(A) on its own even when dA is 0.
      const t = Sim.trueAssumptions(rep);
      const pill = (ok, yes, no) =>
        `<span class="pill ${ok ? "ok" : "warn"}">${ok ? yes : no}</span>`;
      $("truth").innerHTML = `<div class="row">
  ${pill(t.psA, "A separable", "A NOT separable")}
  ${pill(t.psB, "B separable", "B NOT separable")}
  ${pill(t.pi, "independent", t.rho1 ? "correlated (one ρ)" : "correlated (ρ varies)")}
</div>`;
      return rep;
    }

    // --- run ---------------------------------------------------------------------
    async function run() {
      const c = readControls();
      const rep = currentRep();
      $("status").textContent = "Simulating…";

      const data = Sim.simulateRT(rep, c.n, arch, lbaFrom(c));
      const m = await getModel();
      const res = await m.predict(data);

      // Same data, ignoring RT
      const cm = await getCmModel();
      const resCm = await cm.predict({
        counts: data.counts,
        trials: data.trials,
        rtq: null,
      });

      lastRun = { c, rep, data, res, resCm };

      drawObserved();
      drawRT();
      drawInference();
      drawRtValue();

      $("status").textContent = `Done in ${res.ms.toFixed(1)} ms.`;
      $("rt-card").hidden = false;
      $("infer-card").hidden = false;
      $("rt-value-card").hidden = false;
      $("compare-card").hidden = true;
    }

    function drawObserved() {
      const { data, c } = lastRun;
      const props = data.counts.map((r, i) =>
        r.map((v) => v / Math.max(1, data.trials[i])),
      );
      Plot.renderCM($("cm"), props, L, { counts: data.counts });
      $("cm-cap").innerHTML =
        `Observed from <strong>${c.n} trials per stimulus</strong>.`;
    }

    function drawRT() {
      const { data } = lastRun;
      Plot.renderRT($("rt"), {
        cellRTs: data.cellRTs,
        counts: data.counts,
        labels: L,
      });
      Plot.buildLegend($("legend-rt"), L);

      const all = data.trialList.map((t) => t.rt).sort((a, b) => a - b);
      const med = all[all.length >> 1];
      const correct = data.trialList.filter((t) => t.stimulus === t.response);
      const acc = correct.length / data.trialList.length;
      const cMed =
        correct.map((t) => t.rt).sort((a, b) => a - b)[correct.length >> 1] ??
        NaN;

      $("rt-stats").innerHTML = `
  <div class="est"><div class="lbl">Accuracy</div>
    <div class="track"><div class="ci" style="left:0;width:${(100 * acc).toFixed(1)}%;opacity:.8"></div></div>
    <div class="num">${(100 * acc).toFixed(1)}%</div></div>
  <div class="est"><div class="lbl">Median RT</div>
    <div class="track"><div class="ci" style="left:0;width:${Math.min(100, (med / 3) * 100).toFixed(1)}%;opacity:.8"></div></div>
    <div class="num">${med.toFixed(3)} s</div></div>
  <div class="est"><div class="lbl">Median RT (correct)</div>
    <div class="track"><div class="ci" style="left:0;width:${Math.min(100, (cMed / 3) * 100).toFixed(1)}%;opacity:.8"></div></div>
    <div class="num">${cMed.toFixed(3)} s</div></div>`;
    }

    function drawInference() {
      const { res, c } = lastRun;
      const m = model;

      // architecture posterior, with the TRUTH marked
      $("arch-probs").innerHTML = Sim.ARCHITECTURES.map((a) => {
        const truth = a === arch ? ' <span class="pill ok">true</span>' : "";
        return Plot.pbar(Sim.ARCH_LABELS[a].name + truth, res.arch[a]);
      }).join("");

      const pST = res.selfTerminatingProbability;
      const isST = Sim.SELF_TERMINATING.includes(arch);
      $("neglect").innerHTML = `
  <div class="note">
    <h4>Self-terminating architectures: ${pST.toFixed(2)}</h4>
    <p>
      Total probability assigned to the serial and parallel self-terminating
      models. In this simulation, one dimension is selected at random on each
      trial and the other is guessed. ${
        isST
          ? "That is the architecture you simulated. It is not evidence that the participant consistently neglected one particular dimension."
          : "You simulated exhaustive processing, so this should be low."
      }
    </p>
  </div>`;

      $("constructs").innerHTML =
        Plot.pbar("A is separable", res.sep.A) +
        Plot.pbar("B is separable", res.sep.B) +
        Plot.pbar("Independent (ρ = 0)", res.corr.pi) +
        Plot.pbar("One shared ρ", res.corr.rho1) +
        Plot.pbar("ρ varies by stimulus", res.corr.free);

      const truthLBA = {
        t0: c.t0,
        threshold_A: c.threshold,
        drift_k_A: c.kA,
        drift_k_B: c.kB,
      };
      const rows = Object.entries(res.lbaParams)
        .map(([k, v]) => {
          const t = truthLBA[k];
          const shown = Number.isFinite(v) ? v.toFixed(2) : ", ";
          return `<div class="est">
    <div class="lbl">${k}</div>
    <div class="track"></div>
    <div class="num">${shown} <span style="color:var(--mute-soft)">/ ${t.toFixed(2)}</span></div>
  </div>`;
        })
        .join("");
      $("lba").innerHTML =
        `<span class="eyebrow" style="font-size:.55rem;margin-top:.8rem">Accumulator (recovered / true)</span>${rows}`;
    }

    function drawRtValue() {
      const { res, resCm } = lastRun;

      $("rt-value-cm").innerHTML =
        Plot.pbar("A is separable", resCm.sep.A) +
        Plot.pbar("B is separable", resCm.sep.B) +
        Plot.pbar("Independent (ρ = 0)", resCm.corr.pi);
      $("rt-value-cmrt").innerHTML =
        Plot.pbar("A is separable", res.sep.A) +
        Plot.pbar("B is separable", res.sep.B) +
        Plot.pbar("Independent (ρ = 0)", res.corr.pi);

      const diffs = [
        ["A separable", Math.abs(res.sep.A - resCm.sep.A)],
        ["B separable", Math.abs(res.sep.B - resCm.sep.B)],
        ["Independent", Math.abs(res.corr.pi - resCm.corr.pi)],
      ];
      const maxDiff = diffs.reduce((a, b) => (b[1] > a[1] ? b : a), diffs[0]);
      const meaningfulShift = maxDiff[1] > 0.15;

      $("rt-value-verdict").innerHTML = `
  <div class="note ${meaningfulShift ? "" : "warn"}">
    <h4>Biggest shift: ${maxDiff[0]}, Δ${maxDiff[1].toFixed(2)}</h4>
    <p>
      ${
        meaningfulShift
          ? `Adding response times moved this specific conclusion meaningfully for this run, the
           same counts, on their own, would have left you less (or differently) sure about it.`
          : `On this run, the two models land in close agreement on the GRT structure itself. That's
           not a null result, it's the answer to "does RT change the GRT conclusions
           themselves, or mostly add architecture on top of them" for this particular simulated
           dataset. Try a different trial count or architecture and see if that holds.`
      }
    </p>
  </div>`;
    }

    // --- the demonstration -------------------------------------------------------
    function compareAll() {
      const c = readControls();
      const rep = currentRep();
      const N = Math.max(400, c.n);
      const rows = Sim.ARCHITECTURES.map((a) => {
        const d = Sim.simulateRT(
          rep,
          N,
          a,
          lbaFrom(c),
          Sim.makeRNG(20260711),
        );
        const rts = d.trialList.map((t) => t.rt).sort((x, y) => x - y);
        const acc =
          [0, 1, 2, 3].reduce((s, i) => s + d.counts[i][i], 0) /
          d.trialList.length;
        return {
          a,
          acc,
          q10: rts[Math.round(0.1 * (rts.length - 1))],
          med: rts[Math.round(0.5 * (rts.length - 1))],
          q90: rts[Math.round(0.9 * (rts.length - 1))],
          row0: d.counts[0].map((v) => v / N),
        };
      });

      let html = `<tr>
  <th style="text-align:left">Architecture</th><th>Accuracy</th>
  <th>Confusion matrix, row 1</th>
  <th>RT 10th</th><th>RT median</th><th>RT 90th</th></tr>`;
      for (const r of rows) {
        html += `<tr>
    <td style="text-align:left;font-family:Inter,sans-serif">${Sim.ARCH_LABELS[r.a].name}</td>
    <td>${(100 * r.acc).toFixed(1)}%</td>
    <td style="letter-spacing:.02em">${r.row0.map((v) => v.toFixed(2)).join("  ")}</td>
    <td>${r.q10.toFixed(2)}s</td>
    <td><strong>${r.med.toFixed(2)}s</strong></td>
    <td>${r.q90.toFixed(2)}s</td>
  </tr>`;
      }
      $("compare-table").innerHTML = html;

      // Quantify the claim
      const exh = rows.filter((r) => !Sim.SELF_TERMINATING.includes(r.a));
      const accSpread =
        Math.max(...exh.map((r) => r.acc)) -
        Math.min(...exh.map((r) => r.acc));
      const rtRatio =
        Math.max(...exh.map((r) => r.med)) /
        Math.min(...exh.map((r) => r.med));

      $("compare-verdict").innerHTML = `
  <div class="note">
    <h4>Serial-exhaustive, parallel-exhaustive and coactive are the same experiment</h4>
    <p>
      Across those three, accuracy differs by <strong>${(100 * accSpread).toFixed(2)} percentage
      points</strong>, and that is <em>sampling noise</em>, not signal. They don't just look similar, 
      they have <strong>identical response probabilities by construction</strong>. Every one of them
      processes both dimensions and responds with what it found. No confusion matrix, at
      any sample size, can separate them.
    </p>
    <p>
      Meanwhile the slowest of the three takes <strong>${rtRatio.toFixed(1)}×</strong> as long as the
      fastest. The information was never missing from the participant, it was missing
      from the <em>measure</em>.
    </p>
    <p style="margin-bottom:0">
      The two self-terminating rows are different in kind: they <em>do</em> move the confusion
      matrix, because the simulated process selects one dimension and guesses the other.
      That makes self-terminating versus exhaustive processing partly visible in the counts;
      it does not establish stable neglect of either dimension.
    </p>
  </div>`;

      $("compare-card").hidden = false;
      $("compare-card").scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
      $("status").textContent =
        `Compared 5 architectures × ${N} trials/stimulus.`;
    }

    // --- wiring ------------------------------------------------------------------
    for (const k of ALL)
      $(k).addEventListener("input", () => {
        if (lastRun) invalidate("Settings changed. Run again.");
        drawSpace();
      });
    $("run").addEventListener("click", run);
    $("compare").addEventListener("click", compareAll);

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
      if (lastRun) invalidate("Settings changed. Run again.");
      drawSpace();
    });
    $("nudgeY").addEventListener("input", () => {
      nudge.y[+$("nudgeStim").value] = parseFloat($("nudgeY").value);
      $("v-nudgeY").textContent = (+$("nudgeY").value).toFixed(2);
      if (lastRun) invalidate("Settings changed. Run again.");
      drawSpace();
    });
    $("nudge-reset").addEventListener("click", () => {
      nudge = { x: [0, 0, 0, 0], y: [0, 0, 0, 0] };
      syncNudgeSliders();
      if (lastRun) invalidate("Settings changed. Run again.");
      drawSpace();
    });

    Plot.onThemeChange(() => {
      drawSpace();
      if (lastRun) {
        drawObserved();
        drawRT();
      }
    });

    drawSpace();
  
