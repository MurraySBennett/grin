/**
 * grin-model.js — load and run the trained GRIN networks in the browser.
 *
 * THE CONTRACT IS THE MANIFEST, NOT THE CODE.
 * Every ONNX artifact ships with a manifest.json declaring its input names and
 * shapes, its output names and dimensions, the parameter order, and the prior it
 * was trained under. Nothing about the network's layout is hardcoded here. That
 * means (a) the counts-only and RT models are the same code path with different
 * manifests, and (b) if the Python side changes the parameterization, the app
 * fails loudly at load instead of silently mis-decoding the output vector.
 *
 * Contract (from scripts/export_onnx.py):
 *
 *   CM model     in:  counts (B,16), trials (B,4)
 *                out: mean (B,12), std (B,12), p_corr (B,3), p_sep (B,2)
 *
 *   CM+RT model  in:  counts (B,16), rtq (B,80), trials (B,4)
 *                out: mean, std, p_corr, p_sep, p_arch (B,5), lba (B,4)
 *
 * STUB BACKEND
 * ------------
 * Until the .onnx files exist, `createStub()` returns an object with the same
 * interface, backed by the in-browser MLE fitter. It is NOT the network and
 * says so (`result.backend === "stub"`); the UI must surface that. It exists so
 * every page can be built and tested against a realistic response shape today.
 *
 * ES module. Depends on grt-fit.js (stub only) and, at runtime, onnxruntime-web.
 */

import { fitAndSelect, constructWeights } from "./grt-fit.js";
import { MODEL_SPECS } from "./grt-core.js";

/** Where the vendored onnxruntime-web files live (see web/assets/vendor/ort/).
 *  Only the non-threaded SIMD wasm backend is vendored — no COOP/COEP headers,
 *  no SharedArrayBuffer dependency, which matters for a plain S3/CloudFront
 *  deploy with no custom response-header policy. An MLP this small doesn't
 *  need threads; single-threaded WASM inference is sub-millisecond. */
export const ORT_DIR = "./assets/vendor/ort/";
export const ORT_SCRIPT = ORT_DIR + "ort.min.js";

let ortLoading = null;
function loadOrtScript() {
  if (typeof ort !== "undefined") return Promise.resolve();
  if (ortLoading) return ortLoading;
  ortLoading = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = ORT_SCRIPT;
    s.onload = () => resolve();
    s.onerror = () =>
      reject(
        new Error(
          `Could not load ${ORT_SCRIPT}. Check that web/assets/vendor/ort/ was deployed.`,
        ),
      );
    document.head.appendChild(s);
  });
  return ortLoading;
}

// --------------------------------------------------------------------------- //
// Manifest validation — fail loudly, early, and with a message a human can act on
// --------------------------------------------------------------------------- //
function assertManifest(mf) {
  const need = ["id", "file", "inputs", "outputs", "params", "training"];
  for (const k of need)
    if (!(k in mf))
      throw new Error(`Model manifest "${mf.id ?? "?"}" is missing "${k}".`);
  if (mf.params.names.length !== mf.outputs.mean.dim)
    throw new Error(
      `Manifest "${mf.id}": params.names has ${mf.params.names.length} entries but ` +
        `outputs.mean.dim is ${mf.outputs.mean.dim}. These must match.`,
    );
  return mf;
}

function assertSession(session, mf) {
  const want = mf.inputs.map((i) => i.name);
  const have = session.inputNames;
  for (const n of want)
    if (!have.includes(n))
      throw new Error(
        `Model "${mf.id}": the ONNX file expects inputs [${have.join(", ")}] but the ` +
          `manifest declares [${want.join(", ")}]. The manifest and the .onnx are out of ` +
          `sync — re-run scripts/export_onnx.py.`,
      );
}

// --------------------------------------------------------------------------- //
// Loading
// --------------------------------------------------------------------------- //
/**
 * @param {string} dir e.g. "./assets/models/cm"
 * @returns {Promise<GrinModel>}
 */
export async function loadModel(dir) {
  const mf = assertManifest(await (await fetch(`${dir}/manifest.json`)).json());

  await loadOrtScript();
  if (typeof ort === "undefined")
    throw new Error(
      "onnxruntime-web loaded but did not define `ort` on window.",
    );

  // numThreads: 1 is not just a performance choice — it is what makes the ORT
  // loader request "ort-wasm-simd.wasm" instead of the threaded build, which
  // needs SharedArrayBuffer and therefore COOP/COEP response headers. We don't
  // vendor the threaded build, so this must stay 1.
  ort.env.wasm.wasmPaths = ORT_DIR;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  const session = await ort.InferenceSession.create(`${dir}/${mf.file}`, {
    executionProviders: ["wasm"],
  });
  assertSession(session, mf);
  return new GrinModel(mf, session);
}

/** A drop-in stand-in backed by MLE. Same interface, honestly labelled. */
export function createStub(mf) {
  return new GrinModel(assertManifest(mf), null);
}

// --------------------------------------------------------------------------- //
// The model
// --------------------------------------------------------------------------- //
export class GrinModel {
  constructor(manifest, session) {
    this.manifest = manifest;
    this.session = session;
    this.backend = session ? "onnx" : "stub";
    this.needsRT = manifest.inputs.some((i) => i.name === "rtq");
  }

  get paramNames() {
    return this.manifest.params.names;
  }

  /**
   * Run inference on ONE participant.
   * @param {{counts:number[][], trials:number[], rtq?:number[]|null}} agg
   * @returns {Promise<Result>}
   */
  async predict(agg) {
    if (this.needsRT && !agg.rtq)
      throw new Error(
        `Model "${this.manifest.id}" needs response times, but none were given.`,
      );

    const t0 = performance.now();
    const out = this.session ? await this._runOnnx(agg) : this._runStub(agg);
    out.ms = performance.now() - t0;
    out.backend = this.backend;
    out.modelId = this.manifest.id;
    return this._decorate(out);
  }

  /** Batch over many participants. Sequential; each call is sub-millisecond. */
  async predictMany(aggs) {
    const res = [];
    for (const a of aggs) res.push(await this.predict(a));
    return res;
  }

  async _runOnnx(agg) {
    const feeds = {};
    for (const inp of this.manifest.inputs) {
      let data;
      if (inp.name === "counts") data = agg.counts.flat();
      else if (inp.name === "trials") data = agg.trials;
      else if (inp.name === "rtq") data = agg.rtq;
      else throw new Error(`Unknown manifest input "${inp.name}".`);
      if (data.length !== inp.dim)
        throw new Error(
          `Input "${inp.name}": expected ${inp.dim} values, got ${data.length}.`,
        );
      feeds[inp.name] = new ort.Tensor("float32", Float32Array.from(data), [
        1,
        inp.dim,
      ]);
    }
    const raw = await this.session.run(feeds);
    const out = {};
    for (const name of Object.keys(this.manifest.outputs)) {
      if (!raw[name])
        throw new Error(
          `ONNX output "${name}" declared in the manifest is missing.`,
        );
      out[name] = Array.from(raw[name].data);
    }
    return out;
  }

  /**
   * MLE-backed stand-in. Posterior mean := the saturated MLE. Posterior SD :=
   * a crude large-sample approximation (NOT the network's calibrated SD — the
   * calibration is precisely what the network provides and this does not).
   * Construct probabilities := BIC weights.
   */
  _runStub(agg) {
    const sel = fitAndSelect(agg.counts, agg.trials, "bic");
    const full = sel.fits.find((f) => f.model === "ds");
    const w = constructWeights(sel.fits);

    // se ~ 1/sqrt(n) scaled by how extreme the z is; deliberately rough.
    const std = full.params.map((v, i) => {
      const n = agg.trials[i % 4] || 1;
      const base = 1 / Math.sqrt(n);
      return i < 8
        ? base * (1 + 0.35 * Math.abs(v))
        : Math.min(0.6, base * 2.5);
    });

    const out = {
      mean: full.params,
      std,
      p_corr: [w.pi, w.rho1, w.free],
      p_sep: [w.sepA, w.sepB],
      _stubSelection: sel,
    };
    if (this.manifest.outputs.p_arch) {
      const k = this.manifest.outputs.p_arch.dim;
      out.p_arch = new Array(k).fill(1 / k); // the stub genuinely knows nothing here
    }
    if (this.manifest.outputs.lba)
      out.lba = this.manifest.outputs.lba.names.map(() => NaN);
    return out;
  }

  /** Attach names and derived quantities so the UI never re-derives anything. */
  _decorate(out) {
    const mf = this.manifest;
    out.params = Object.fromEntries(
      mf.params.names.map((n, i) => [n, out.mean[i]]),
    );
    out.paramsSD = Object.fromEntries(
      mf.params.names.map((n, i) => [n, out.std[i]]),
    );

    if (out.p_corr) {
      const names = mf.outputs.p_corr.names; // ["pi","rho1","free"]
      out.corr = Object.fromEntries(names.map((n, i) => [n, out.p_corr[i]]));
    }
    if (out.p_sep) out.sep = { A: out.p_sep[0], B: out.p_sep[1] };

    if (out.p_arch) {
      const names = mf.outputs.p_arch.names;
      out.arch = Object.fromEntries(names.map((n, i) => [n, out.p_arch[i]]));
      out.archBest = names[out.p_arch.indexOf(Math.max(...out.p_arch))];
      // Total probability assigned to the two self-terminating architectures.
      // In this simulator they process one randomly selected dimension and guess
      // the other; this is not evidence of stable neglect of a particular dimension.
      out.selfTerminatingProbability = names.reduce(
        (a, n, i) => a + (n.includes("self_terminating") ? out.p_arch[i] : 0),
        0,
      );
    }
    if (out.lba)
      out.lbaParams = Object.fromEntries(
        mf.outputs.lba.names.map((n, i) => [n, out.lba[i]]),
      );

    // Most probable model class, from the factorized heads. Reported WITH its
    // probability, because a 0.34 winner is not a finding.
    if (out.corr && out.sep) {
      const corrBest = Object.entries(out.corr).sort((a, b) => b[1] - a[1])[0];
      const psA = out.sep.A >= 0.5;
      const psB = out.sep.B >= 0.5;
      const name = Object.keys(MODEL_SPECS).find((m) => {
        const s = MODEL_SPECS[m];
        return s.corr === corrBest[0] && s.psA === psA && s.psB === psB;
      });
      out.modelClass = {
        name,
        pCorr: corrBest[1],
        pSepA: out.sep.A,
        pSepB: out.sep.B,
        // Product of separately trained marginal heads. This is useful as a
        // compact support score, but is not a directly calibrated joint posterior.
        factorizedSupport:
          corrBest[1] *
          (psA ? out.sep.A : 1 - out.sep.A) *
          (psB ? out.sep.B : 1 - out.sep.B),
      };
    }
    return out;
  }
}
