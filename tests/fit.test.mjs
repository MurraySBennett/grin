// Checks web/assets/js/grt-fit.js's Nelder-Mead MLE against the Python reference
// (tests/get_fit_reference.py, which fits with the project's real scipy-backed
// optimiser in src/inference/mle.py). Different optimisers, so this is NOT a
// bit-exact check like core.test.mjs -- both should converge close to the same
// optimum, and usually agree on which model BIC prefers.
//
//   node tests/fit.test.mjs
//
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { fitClass, fitAndSelect } from "../web/assets/js/grt-fit.js";
import { MODEL_NAMES } from "../web/assets/js/grt-core.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(HERE, "fit_reference.json"), "utf8"));

test(`MLE reference has cases (got ${cases.length})`, () => {
  assert.ok(cases.length > 0, "run `python tests/get_fit_reference.py` first");
});

test("fitClass log-likelihood is within 1% of the Python optimum, every class", () => {
  let checked = 0;
  for (const c of cases) {
    for (const name of MODEL_NAMES) {
      const ref = c.fits[name];
      if (!ref) continue;
      const got = fitClass(c.counts, c.trials, name);
      // JS log-lik should not be MEANINGFULLY worse than Python's (both maximise
      // the same objective); a small negative slack covers optimiser noise, but a
      // JS fit that's clearly worse indicates a real divergence (wrong likelihood,
      // wrong expansion, ...), not just a different local step.
      const slack = Math.max(1e-3, 0.01 * Math.abs(ref.loglik));
      assert.ok(
        got.loglik >= ref.loglik - slack,
        `${c.true_model}/n=${c.n_trials} class ${name}: JS loglik ${got.loglik} ` +
          `notably worse than Python's ${ref.loglik}`,
      );
      checked++;
    }
  }
  assert.ok(checked > 0);
});

test("fitAndSelect's BIC winner usually agrees with Python's", () => {
  let agree = 0;
  for (const c of cases) {
    const sel = fitAndSelect(c.counts, c.trials, "bic");
    if (sel.best.model === c.best_bic) agree++;
  }
  // Not bit-exact (different optimisers can land on different sides of a close
  // call), but should agree on most cases -- if this drifts low it's a real signal.
  assert.ok(
    agree / cases.length >= 0.7,
    `BIC winner agreement ${agree}/${cases.length} is suspiciously low`,
  );
});
