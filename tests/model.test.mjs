// Checks web/assets/js/grin-model.js's STUB backend (MLE-based stand-in used
// before/without the ONNX network). The real ONNX path (loadModel) touches
// `fetch`/`document`/onnxruntime-web and is a browser integration concern, not a
// Node unit-test one; createStub()'s whole purpose is to make the rest of this
// module's logic (result shape, decoration, model-class summary) testable without
// a browser, so that is what this file exercises.
//
//   node tests/model.test.mjs
//
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { createStub } from "../web/assets/js/grin-model.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(
  readFileSync(join(HERE, "..", "web", "assets", "models", "cm", "manifest.json"), "utf8"),
);

const M = [
  [71, 17, 9, 5],
  [20, 67, 5, 9],
  [13, 6, 63, 20],
  [5, 10, 15, 71],
];

test("createStub builds a model backed by the manifest, not ONNX", () => {
  const model = createStub(manifest);
  assert.equal(model.backend, "stub");
  assert.equal(model.needsRT, false);
  assert.deepEqual(model.paramNames, manifest.params.names);
});

test("predict() on the stub returns a well-formed, decorated result", async () => {
  const model = createStub(manifest);
  const out = await model.predict({ counts: M, trials: M.map((r) => r.reduce((a, b) => a + b, 0)) });
  assert.equal(out.backend, "stub");
  assert.equal(out.modelId, "cm");
  assert.equal(Object.keys(out.params).length, 12);
  assert.equal(Object.keys(out.paramsSD).length, 12);
  assert.ok(out.corr.pi >= 0 && out.corr.pi <= 1);
  assert.ok(["A", "B"].every((k) => out.sep[k] >= 0 && out.sep[k] <= 1));
  assert.ok(out.modelClass.name, "expected a most-probable model class to be named");
  assert.ok(out.ms >= 0, "expected a timing to be recorded");
});

test("a manifest missing a required key is rejected loudly", () => {
  const bad = { ...manifest };
  delete bad.outputs;
  assert.throws(() => createStub(bad), /missing "outputs"/);
});

test("a manifest whose params/outputs disagree in length is rejected loudly", () => {
  const bad = JSON.parse(JSON.stringify(manifest));
  bad.params.names.push("extra_param");
  assert.throws(() => createStub(bad), /must match/);
});

test("predictMany matches sequential predict() calls", async () => {
  const model = createStub(manifest);
  const agg = { counts: M, trials: M.map((r) => r.reduce((a, b) => a + b, 0)) };
  const many = await model.predictMany([agg, agg]);
  assert.equal(many.length, 2);
  assert.deepEqual(many[0].mean, many[1].mean);
});
