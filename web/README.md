# GRIN web app

Static. No build step, no bundler. ES modules + one stylesheet.

## Run locally

ES modules need a real origin — `file://` will not work.

    cd web && python3 -m http.server 8000
    # open http://localhost:8000/

## Layout

    index.html          landing
    explore.html        teaching tier, counts only
    explore-rt.html     teaching tier, counts + response times
    analyse.html        upload CSV / paste / type a matrix; GRIN vs MLE
    validate.html       (stage 4) benchmarks
    learn.html          (stage 4) primer + glossary

    assets/css/grin.css       design tokens + shell, matching the main site
    assets/js/grt-core.js     port of src/grt_model.py      (tested vs scipy)
    assets/js/grt-io.js       CSV -> counts/trials/rtq      (tested vs numpy)
    assets/js/grt-fit.js      port of src/inference/mle.py  (tested vs scipy)
    assets/js/grt-sim.js      port of rt_lba_generator      (the forward model)
    assets/js/grt-plot.js     canvas + DOM rendering
    assets/js/grin-model.js   manifest-driven ONNX wrapper (+ MLE stub)
    assets/js/grin-shell.js   theme, nav, konami

    assets/models/cm/manifest.json     + npe_model.onnx      <- drop weights here
    assets/models/cmrt/manifest.json   + npe_rt_model.onnx   <- and here
    assets/vendor/ort/                 pinned onnxruntime-web, non-threaded SIMD only
                                        (no COOP/COEP headers needed — see
                                        assets/vendor/ort/LICENSE-NOTICE.txt)

    scripts/deploy_s3.sh   aws s3 sync with correct MIME types + cache headers,
                            then an optional CloudFront invalidation. Run from
                            repo root:
                                bash scripts/deploy_s3.sh <bucket> [distribution-id]

## Wiring in the real weights

Every page currently calls `createStub(manifest)`. Search for `TODO(weights)`;
each is a one-line swap to `loadModel("./assets/models/cm")`. The stub
self-identifies (`result.backend === "stub"`) and the pages render a loud banner
saying so, so a half-wired deploy is impossible to miss.

## Tests

    bash ../tests/run.sh
