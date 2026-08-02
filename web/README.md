# GRIN web app

The browser front-end for **GRIN** (General Recognition Inversion Network) —
amortised Bayesian inference for General Recognition Theory. Static site: no
build step, no bundler. ES modules and one stylesheet.

## Run locally

ES modules need a real origin — opening the files over `file://` will not work.

    cd web && python3 -m http.server 8000
    # open http://localhost:8000/

## Pages

    index.html                       landing
    space-builder.html               teaching tier, counts only — build a space,
                                       run a virtual experiment, watch it recover
    space-builder-time-attack.html   teaching tier, counts + response times ("+RT")
                                       — processing architecture and accumulators
    analyse.html                     bring your own data (upload CSV, paste, or type
                                       a matrix); GRIN beside a maximum-likelihood fit
    validate.html                    recovery + interval calibration, simulated live
                                       in-browser over many participants
    dynamics.html                    fitting as a process: drift tracking and early
                                       stopping, made practical by ~1 ms refits
    learn.html                       primer, glossary, FAQ, limitations, references

Each page has a distinct role and does not duplicate another: Space Builder builds
and recovers one space; Analyse works on real/uploaded data with an MLE cross-check;
Validate checks recovery and calibration on simulated data where the truth is known.

## Assets

    assets/css/grin.css       design tokens + shell, matching the main site

    assets/js/grt-core.js     port of the GRT model (forward map, params, classes)
    assets/js/grt-io.js       CSV -> counts / trials / RT quantiles
    assets/js/grt-fit.js      maximum-likelihood baseline (Nelder-Mead + selection)
    assets/js/grt-sim.js      the forward simulator (counts, trial streams, RT/LBA)
    assets/js/grt-plot.js     canvas + DOM rendering, theme, palettes
    assets/js/grin-model.js   manifest-driven ONNX wrapper (loadModel)
    assets/js/grin-shell.js   theme, shared nav injection, konami

    assets/components/nav.html            one nav, fetched and injected by grin-shell.js

    assets/models/cm/     manifest.json + npe_model.onnx        (counts-only network)
    assets/models/cmrt/   manifest.json + npe_rt_model.onnx     (counts + RT network)
    assets/vendor/ort/    pinned onnxruntime-web, non-threaded SIMD only
                           (no COOP/COEP headers needed)

## Inference wiring

Pages load a network with `loadModel("./assets/models/cm")` (or `.../cmrt`). The
**manifest is the contract**: each `manifest.json` declares the network's input and
output names, shapes, parameter order, and the prior it was trained under, so nothing
about the layout is hardcoded in the JS. If a manifest and its `.onnx` fall out of
sync, `grin-model.js` fails loudly at load rather than silently mis-decoding the
output. Single-threaded SIMD WASM keeps a fit at roughly a millisecond without needing
SharedArrayBuffer or special response headers.

## Theme

One class on `<html>` (`is-dark`) drives everything. A tiny synchronous inline
`<script>` in each page's `<head>` (right after `<meta charset>`) applies it before
first paint to avoid a flash, reading the shared `msb-dark-mode` key; `grin-shell.js`
re-applies the same logic afterwards and wires the toggle. The CSS and the canvas
drawing both read from that one class, so they can never disagree.

## Deploy

    scripts/deploy_s3.sh   aws s3 sync with correct MIME types + cache headers,
                            then an optional CloudFront invalidation. Run from
                            repo root:
                                bash scripts/deploy_s3.sh <bucket> [distribution-id]

## Tests

    bash ../tests/run.sh
