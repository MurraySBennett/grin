# GRIN web app

The browser front-end for **GRIN** (General Recognition Inference Network) —
amortised Bayesian inference for General Recognition Theory. Static site: no
build step, no bundler. ES modules and one stylesheet.

## Run locally

ES modules need an origin — opening the files over `file://` will not work.

    cd web && python3 -m http.server 8000
    # open http://localhost:8000/

## Pages

    index.html                       landing
    explore.html                     hub for interactive teaching / preview tools
    space-builder.html               teaching tier, counts only — build a space,
                                       run a virtual experiment, watch it recover
    space-builder-time-attack.html   research preview, counts + response times ("+RT")
                                       — processing architecture under a simplified
                                       ballistic timing model
    independence.html                demonstrating the difficulty of identifying
                                       perceptual independence
    analyse.html                     bring your own data (upload CSV, paste, or type
                                       a matrix); GRIN beside a maximum-likelihood fit
    validate.html                    recovery + interval calibration, simulated live
                                       in-browser over many participants
    dynamics.html                    research playground: drift tracking and an
                                       illustrative, not yet validated stopping rule
    learn.html                       primer, glossary, FAQ, limitations, references

Each page has a distinct role and tries not to duplicate another: Space Builder builds
and recovers one space; Analyse works on real/uploaded data with an MLE cross-check;
Validate checks recovery and calibration on simulated data where the truth is known.

## Assets

    assets/css/grin.css       design tokens + shell, matching the main site

    assets/js/grt-core.js       port of the GRT model (forward map, params, classes)
    assets/js/grt-io.js         CSV -> counts / trials / RT quantiles
    assets/js/grt-fit.js        maximum-likelihood baseline (Nelder-Mead + selection)
    assets/js/grt-sim.js        the forward simulator (counts, trial streams,
                                  simplified ballistic RT timing)
    assets/js/grt-plot.js       canvas + DOM rendering, theme, palettes
    assets/js/grin-model.js     manifest-driven ONNX wrapper (loadModel / loadModelCached)
    assets/js/grin-shell.js     theme, shared nav/footer injection, konami
    assets/js/pages/*.js        per-page application logic (ES modules)

    components/nav.html         fetched and injected by grin-shell.js
    components/footer.html      same

    assets/models/cm/           manifest.json + npe_model.onnx        (counts-only network)
    assets/models/cmrt/         manifest.json + npe_rt_model.onnx     (counts + RT network)
    assets/vendor/ort/          pinned onnxruntime-web, non-threaded SIMD only
                                    (no COOP/COEP headers needed)

## Inference wiring

Pages load a network with `loadModelCached("./assets/models/cm")` (or `.../cmrt`).
The **manifest is the contract**: each `manifest.json` declares the network's input
and output names, shapes, parameter order, and the prior it was trained under, so
nothing about the layout is hardcoded in the JS. If a manifest and its `.onnx` fall
out of sync, `grin-model.js` fails loudly at load rather than silently mis-decoding
the output. Single-threaded SIMD WASM keeps a fit at roughly a millisecond without
needing SharedArrayBuffer or special response headers. The ORT runtime is fetched
only on first inference, not on page paint.

If you're building a *different* experiment (jsPsych, lab.js, or similar) and want
this same live in-browser inference in it, `grin-model.js` is reusable as example
code — copy it in and adapt it, it's not a published package. See
[`docs/data_collection.md`](../docs/data_collection.md) for the worked pattern.

## Theme

One class on `<html>` (`is-dark`) drives everything. A tiny synchronous inline
`<script>` in each page's `<head>` (right after `<meta charset>`) applies it before
first paint to avoid flash, reading the shared `msb-dark-mode` key; `grin-shell.js`
re-applies the same logic afterwards and wires the toggle. The CSS and the canvas
drawing both read from that one class, so they can never disagree.

## Deploy

See [DEPLOYMENT.md](./DEPLOYMENT.md) and `.github/workflows/deploy.yaml`.
