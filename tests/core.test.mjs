// Checks web/assets/js/grt-core.js -- a hand-ported copy of src/grt_model.py --
// against the real Python package. Regenerate the reference with
// `python tests/gen_reference.py` whenever grt_model.py's public interface changes.
//
//   node tests/core.test.mjs
//
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  forwardProbabilities,
  pack,
  unpack,
  logLik,
  validate,
  MODEL_NAMES,
  nFreeParams,
} from "../web/assets/js/grt-core.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(HERE, "reference.json"), "utf8"));

test(`forward-model reference has cases (got ${cases.length})`, () => {
  assert.ok(cases.length > 0, "run `python tests/gen_reference.py` first");
});

test("forwardProbabilities matches Python to 1e-9", () => {
  for (const c of cases) {
    const got = forwardProbabilities(c.params);
    for (let s = 0; s < 4; s++) {
      for (let r = 0; r < 4; r++) {
        assert.ok(
          Math.abs(got[s][r] - c.probs[s][r]) < 1e-9,
          `${c.model} stimulus ${s} response ${r}: got ${got[s][r]}, want ${c.probs[s][r]}`,
        );
      }
    }
  }
});

test("pack/unpack round-trips exactly", () => {
  for (const c of cases) {
    const { zx, zy, rho } = unpack(c.params);
    assert.deepEqual(pack(zx, zy, rho), c.params);
  }
});

test("logLik matches Python to 1e-6 (relative)", () => {
  for (const c of cases) {
    const got = logLik(c.counts, c.params);
    const tol = Math.max(1e-6, 1e-9 * Math.abs(c.loglik));
    assert.ok(
      Math.abs(got - c.loglik) < tol,
      `${c.model}: got ${got}, want ${c.loglik}`,
    );
  }
});

test("validate agrees with Python on class-constraint satisfaction", () => {
  for (const c of cases) {
    const { ok } = validate(c.params, c.model);
    assert.equal(ok, c.valid, `${c.model}: params ${JSON.stringify(c.params)}`);
  }
});

test("every model class round-trips through nFreeParams within DATA_DF", () => {
  for (const name of MODEL_NAMES) {
    assert.ok(nFreeParams(name) <= 12, `${name} has more than 12 free params`);
  }
});
