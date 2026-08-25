    import * as Plot from "../grt-plot.js";
    import * as Sim from "../grt-sim.js";
    import * as Fit from "../grt-fit.js";
    import * as Core from "../grt-core.js";
    import { loadModelCached } from "../grin-model.js";

    const $ = (id) => document.getElementById(id);
    const L = {
      aName: "Dimension A",
      a1: "A1",
      a2: "A2",
      bName: "Dimension B",
      b1: "B1",
      b2: "B2",
    };

    let cmModel = null;
    async function getModel() {
      const status = (msg) => {
        const el = $("status") || $("run-status") || $("fit-status");
        if (el && msg) el.textContent = msg;
      };
      if (!cmModel) cmModel = await loadModelCached("./assets/models/cm", status);
      return cmModel;
    }
    function factorizedSupportForClass(res, className) {
      const spec = Core.MODEL_SPECS[className];
      const pCorr = res.corr[spec.corr];
      const pSepA = spec.psA ? res.sep.A : 1 - res.sep.A;
      const pSepB = spec.psB ? res.sep.B : 1 - res.sep.B;
      return pCorr * pSepA * pSepB;
    }

    const TRAIL_MAX_CHECKPOINTS = 40; // caps worst-case work: 40 x one Nelder-Mead

    function syncTrailControls() {
      $("v-rho").textContent = (+$("rho").value).toFixed(2);
      $("v-dA").textContent = (+$("dA").value).toFixed(2);
      $("v-dB").textContent = (+$("dB").value).toFixed(2);
      $("v-dA2").textContent = (+$("dA2").value).toFixed(2);
      $("v-dB2").textContent = (+$("dB2").value).toFixed(2);
      $("v-intervention").textContent = `${$("intervention").value}%`;
      $("v-alpha").textContent = (+$("alpha").value).toFixed(2);
      $("v-total").textContent = $("total").value;
      $("v-every").textContent = $("every").value;
      $("drift-controls").hidden = !$("drift-on").checked;
      $("window-size-row").hidden = !$("use-window").checked;

      const total = +$("total").value,
        every = +$("every").value;
      const wouldBe = Math.ceil((total * 4) / every);
      $("trail-checkpoint-note").textContent =
        wouldBe > TRAIL_MAX_CHECKPOINTS
          ? `${wouldBe} checkpoints requested, capped to ${TRAIL_MAX_CHECKPOINTS} by widening the spacing, to keep this responsive.`
          : `${wouldBe} checkpoints at this setting.`;

      const windowEl = $("window");
      const newMax = total;
      const newMin = Math.min(10, total);
      windowEl.min = newMin;
      windowEl.max = newMax;
      if (+windowEl.value > newMax) windowEl.value = newMax;
      if (+windowEl.value < newMin) windowEl.value = newMin;
      $("v-window").textContent = windowEl.value;
      $("window-range-note").textContent =
        `Range adjusts with the total above: ${newMin}–${newMax} trials/stimulus.`;
    }
    [
      "rho",
      "dA",
      "dB",
      "dA2",
      "dB2",
      "intervention",
      "alpha",
      "total",
      "every",
      "window",
    ].forEach((id) => $(id).addEventListener("input", syncTrailControls));
    $("drift-on").addEventListener("change", () => {
      if ($("drift-on").checked) {
        $("use-window").checked = true;
      }
      syncTrailControls();
    });
    $("use-window").addEventListener("change", syncTrailControls);
    syncTrailControls();

    function boundedEvery(totalPerStimulus, requestedEvery, maxCheckpoints) {
      const totalTrials = totalPerStimulus * 4;
      return Math.max(
        requestedEvery,
        Math.ceil(totalTrials / maxCheckpoints),
      );
    }

    let trailAbort = null;

    async function runTrail() {
      trailAbort?.abort();
      trailAbort = new AbortController();
      const { signal } = trailAbort;

      const total = +$("total").value;
      const every = boundedEvery(
        total,
        +$("every").value,
        TRAIL_MAX_CHECKPOINTS,
      );
      const drift = $("drift-on").checked;
      const useWindow = $("use-window").checked;
      const windowSize = useWindow ? +$("window").value * 4 : null;
      const rho = +$("rho").value;
      const dA = +$("dA").value,
        dB = +$("dB").value;

      $("trail-status").textContent = "Simulating…";
      $("trail-progress").style.width = "0%";
      const trials = drift
        ? Sim.simulateGradualDriftStream(
            {
              mA: 1.3,
              mB: 1.3,
              rho,
              dAStart: dA,
              dBStart: dB,
              dATarget: +$("dA2").value,
              dBTarget: +$("dB2").value,
            },
            total,
            +$("intervention").value / 100,
            +$("alpha").value,
            Sim.makeRNG(),
          )
        : Sim.simulateTrialStream(
            Sim.buildRepresentation({ mA: 1.3, mB: 1.3, rho, dA, dB }),
            total,
            Sim.makeRNG(),
          );

      const model = await getModel();
      if (signal.aborted) return;
      const snapshots = Fit.checkpointSnapshots(trials, {
        every,
        windowSize: windowSize || undefined,
      });
      const checkpoints = [];
      for (const snap of snapshots) {
        if (signal.aborted) return;
        const res = await model.predict({
          counts: snap.counts,
          trials: snap.trials,
          rtq: null,
        });
        checkpoints.push({
          trialCount: snap.trialCount,
          counts: snap.counts,
          stimuli: Plot.toStimuli(res.mean),
          grin: res,
        });
      }
      if (!checkpoints.length) {
        $("trail-status").innerHTML =
          `<span class="pill bad">Not enough trials yet for a single checkpoint, raise the total or lower "checkpoint every".</span>`;
        return;
      }
      if (signal.aborted) return;

      $("trail-status").textContent =
        `Fitting ${checkpoints.length} checkpoints…`;
      await Plot.animateFadeTrail(
        $("trail-canvas"),
        checkpoints,
        { labels: L, showMarginals: false },
        { curve: "exp", minAlpha: 0.06 },
        {
          delayMs: 60,
          signal,
          onProgress: (i, n) => {
            $("trail-progress").style.width =
              `${((100 * i) / n).toFixed(0)}%`;
          },
        },
      );
      if (signal.aborted) return;

      const last = checkpoints[checkpoints.length - 1];
      const zx = [
        last.stimuli[0].zx,
        last.stimuli[1].zx,
        last.stimuli[2].zx,
        last.stimuli[3].zx,
      ];
      const zy = [
        last.stimuli[0].zy,
        last.stimuli[1].zy,
        last.stimuli[2].zy,
        last.stimuli[3].zy,
      ];
      const dAest = (
        Math.abs(zx[0] - zx[1]) + Math.abs(zx[2] - zx[3])
      ).toFixed(2);
      const dBest = (
        Math.abs(zy[0] - zy[2]) + Math.abs(zy[1] - zy[3])
      ).toFixed(2);
      const totalMs = checkpoints.reduce((a, c) => a + (c.grin.ms || 0), 0);
      $("trail-status").innerHTML =
        `<span class="pill ok">${checkpoints.length} GRIN checkpoints in ${totalMs.toFixed(1)} ms total, ${trials.length} trials. Latest estimated separability violation: A≈${dAest}, B≈${dBest}.</span>`;

      if (drift) {
        const interventionPct = $("intervention").value;
        const interventionPerStim = Math.round(
          (+interventionPct / 100) * total,
        );
        const trueBefore = Math.abs(dA),
          trueAfter = Math.abs(+$("dA2").value);
        const windowPerStim = useWindow ? +$("window").value : null;
        $("trail-insight").innerHTML = useWindow
          ? `<div class="note">
        <h4>The window is catching up</h4>
        <p>Separability creeps from |Δ|≈${trueBefore.toFixed(2)} toward |Δ|≈${trueAfter.toFixed(2)}
        starting around trial ${interventionPerStim} (per stimulus, ${interventionPct}% through the
        session), gradually. A sliding window only "remembers" the last
        ${windowPerStim} trials/stimulus, so once enough post-shift data has entered the window,
        the estimate can track the new truth as it settles. Try turning the window off to see
        what happens without it.</p>
      </div>`
          : `<div class="note warn">
        <h4>This is the lag</h4>
        <p>This is a <strong>running total</strong>, not a window, every trial from before the
        shift is still in there pulling the estimate toward a blend of old and new
        truth. No matter how long you kept running, a pure running total would never fully catch
        up to the new value the representation is settling toward. Turn on the sliding window
        (left) to see the fix.</p>
      </div>`;
      } else {
        $("trail-insight").innerHTML = "";
      }
    }
    $("run-trail").addEventListener("click", runTrail);
    Plot.onThemeChange(() => {});

    const STOP_MAX_CHECKPOINTS = 40;

    function syncStopControls() {
      $("v-stop-rho").textContent = (+$("stop-rho").value).toFixed(2);
      $("v-stop-dA").textContent = (+$("stop-dA").value).toFixed(2);
      $("v-stop-dB").textContent = (+$("stop-dB").value).toFixed(2);
      $("v-thresh").textContent = (+$("thresh").value).toFixed(2);
      $("v-stop-total").textContent = $("stop-total").value;
      $("v-stop-every").textContent = $("stop-every").value;

      const total = +$("stop-total").value,
        every = +$("stop-every").value;
      const wouldBe = Math.ceil((total * 4) / every);
      $("stop-checkpoint-note").textContent =
        wouldBe > STOP_MAX_CHECKPOINTS
          ? `${wouldBe} checkpoints requested, capped to ${STOP_MAX_CHECKPOINTS} by widening the spacing, to keep this responsive.`
          : `${wouldBe} checkpoints at this setting.`;

      const construct = $("construct").value;
      $("construct-note").textContent =
        construct === "full"
          ? "The largest factorized support assigned to any of the 12 model classes."
          : "The larger probability assigned to either answer for that one question; the simulation truth is not used by the rule.";
    }
    [
      "stop-rho",
      "stop-dA",
      "stop-dB",
      "thresh",
      "stop-total",
      "stop-every",
      "construct",
    ].forEach((id) => $(id).addEventListener("input", syncStopControls));
    syncStopControls();

    function trueClassName(rho, dA, dB) {
      const corr = rho === 0 ? "pi" : "rho1";
      const psA = dA === 0,
        psB = dB === 0;
      return Object.keys(Core.MODEL_SPECS).find((m) => {
        const spec = Core.MODEL_SPECS[m];
        return spec.corr === corr && spec.psA === psA && spec.psB === psB;
      });
    }

    function decisionFor(construct, res) {
      if (construct === "full") {
        const ranked = Object.keys(Core.MODEL_SPECS)
          .map((name) => ({
            decision: name,
            confidence: factorizedSupportForClass(res, name),
          }))
          .sort((a, b) => b.confidence - a.confidence);
        return ranked[0];
      }
      const pYes =
        construct === "pi"
          ? res.corr.pi
          : construct === "sepA"
            ? res.sep.A
            : res.sep.B;
      return {
        decision: pYes >= 0.5,
        confidence: Math.max(pYes, 1 - pYes),
      };
    }

    let stopAbort = null;

    async function runEarlyStopping() {
      stopAbort?.abort();
      stopAbort = new AbortController();
      const { signal } = stopAbort;

      const construct = $("construct").value;
      const threshold = +$("thresh").value;
      const total = +$("stop-total").value;
      const every = boundedEvery(
        total,
        +$("stop-every").value,
        STOP_MAX_CHECKPOINTS,
      );

      const truth = {
        rho: +$("stop-rho").value,
        dA: +$("stop-dA").value,
        dB: +$("stop-dB").value,
      };
      const trueClass = trueClassName(truth.rho, truth.dA, truth.dB);

      $("stop-status").textContent = "Simulating…";
      $("stop-verdict").innerHTML = "";
      $("stop-track").innerHTML = "";
      $("stop-mle-slot").innerHTML = "";
      const rep = Sim.buildRepresentation({ mA: 1.3, mB: 1.3, ...truth });
      const trials = Sim.simulateTrialStream(rep, total, Sim.makeRNG());

      const model = await getModel();
      if (signal.aborted) return;
      const snapshots = Fit.checkpointSnapshots(trials, {
        every,
        minPerCheckpoint: 4,
      });

      const rows = [];
      let stoppedAt = null;
      let totalGrinMs = 0;
      for (const snap of snapshots) {
        if (signal.aborted) return;
        const res = await model.predict({
          counts: snap.counts,
          trials: snap.trials,
          rtq: null,
        });
        totalGrinMs += res.ms || 0;
        const decision = decisionFor(construct, res);
        rows.push({ n: snap.trialCount, ...decision });

        const firstHitIdx = rows.findIndex((r) => r.confidence >= threshold);
        $("stop-track").innerHTML = rows
          .map((r, idx) =>
            Plot.pbar(`${Math.round(r.n / 4)} trials/stim.`, r.confidence, {
              dimBelow: threshold,
              highlight: idx === firstHitIdx,
            }),
          )
          .join("");
        $("stop-status").textContent =
          `Checkpoint ${rows.length} (${Math.round(snap.trialCount / 4)} trials/stimulus)…`;
        $("stop-progress").style.width =
          `${((100 * snap.trialCount) / trials.length).toFixed(0)}%`;

        if (stoppedAt === null && decision.confidence >= threshold)
          stoppedAt = snap.trialCount;
      }
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (signal.aborted) return;

      const finalSnap = snapshots[snapshots.length - 1];
      const mleSel = Fit.fitAndSelect(
        finalSnap.counts,
        finalSnap.trials,
        "bic",
      );
      const mleAgrees = mleSel.best.model === trueClass;
      $("stop-mle-slot").innerHTML = `<div class="note">
  <h4>Checked against a classical MLE fit (one fit on the final dataset)</h4>
  <p>For reference, maximum likelihood's own best-supported class on the full session is
  <code>${mleSel.best.model}</code> (${mleAgrees ? "agrees with" : "differs from"} the true
  <code>${trueClass}</code>). This comparison uses the known simulation truth only after the
  run; the stopping decision itself does not get to see it.</p>
</div>`;

      const firstHit = rows.find((r) => r.confidence >= threshold);
      const trueDecision =
        construct === "full"
          ? trueClass
          : construct === "pi"
            ? truth.rho === 0
            : construct === "sepA"
              ? truth.dA === 0
              : truth.dB === 0;
      const stopWasCorrect = firstHit
        ? firstHit.decision === trueDecision
        : null;
      const labelFor = construct === "full"
        ? `factorized support for <code>${firstHit?.decision ?? "the leading class"}</code>`
        : `support for the ${firstHit?.decision ? "yes" : "no"} answer`;
      const savedPct = stoppedAt
        ? Math.round(100 * (1 - stoppedAt / trials.length))
        : 0;
      const stoppedPerStim = stoppedAt ? Math.round(stoppedAt / 4) : null;
      const budgetPerStim = Math.round(trials.length / 4);
      $("stop-verdict").innerHTML = stoppedAt
        ? `<div class="note">
      <h4>The illustrative rule first crossed at ${stoppedPerStim} trials/stimulus</h4>
      <p>${labelFor} first crossed ${threshold.toFixed(2)} there (highlighted below). In this
      simulated run the decision was <strong>${stopWasCorrect ? "correct" : "wrong"}</strong>
      relative to the known truth, leaving ${savedPct}% of the planned session unused. That is a
      description of this run, not evidence that the rule controls false stops or saves that much
      on average.</p>
    </div>`
        : `<div class="note warn">
      <h4>Never reached ${threshold.toFixed(2)} confidence</h4>
      <p>Across the whole ${budgetPerStim}-trial-per-stimulus budget, no side crossed the
      threshold. This one run may be weakly informative, but it does not tell us the operating
      characteristics of the rule. Those need a proper repeated simulation before real use.</p>
    </div>`;
      $("stop-status").innerHTML = `<span class="pill ok">Done.</span>`;
    }
    $("run-stop").addEventListener("click", runEarlyStopping);
  
