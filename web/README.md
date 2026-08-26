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
    independence.html                demonstrating the difficulty of identifying
                                       perceptual independence
    analyse.html                     bring your own data (upload CSV, paste, or type
                                       a matrix); GRIN beside a maximum-likelihood fit
    validate.html                    recovery + interval calibration, simulated live
                                       in-browser over many participants
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
    assets/vendor/ort/          pinned onnxruntime-web, non-threaded SIMD only
                                    (no COOP/COEP headers needed)

## Inference wiring

Page scripts avoid names matching `analy*`: content blockers treat such script
requests as analytics and return `ERR_BLOCKED_BY_CLIENT`, which breaks the page for a
large share of visitors with nothing in the server logs to show for it. The Analyse
page's module is therefore `assets/js/pages/matrix-fit.js`, while the document stays
`analyse.html`.

Pages load the network with `loadModelCached("./assets/models/cm")`. If
`assets/models/cm/recalibration.json` is present the Analyse page offers an
off-by-default "calibrated intervals" toggle; if it is absent the toggle stays
hidden and the raw posterior is used, which is the correct fallback.

The `cmrt` response-time model was withdrawn pending validation of its
replacement (`docs/dynamic_grt_rt_design.md`). An `rt` column in uploaded data
is parsed and then ignored rather than rejected.
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
