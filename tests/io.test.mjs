// Checks web/assets/js/grt-io.js: the aggregate() RT-quantile step against a
// Python reference (tests/gen_io_reference.py), plus self-contained unit tests for
// CSV parsing / level-map resolution / input validation, which have no Python
// counterpart to compare against (grt-io.js's own contract, not a port).
//
//   node tests/io.test.mjs
//
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  rint,
  buildLevelMap,
  parseCSV,
  aggregate,
  checkInputs,
  templateCSV,
} from "../web/assets/js/grt-io.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(HERE, "io_reference.json"), "utf8"));

test("rint rounds half to even, matching numpy", () => {
  assert.equal(rint(0.5), 0);
  assert.equal(rint(1.5), 2);
  assert.equal(rint(2.5), 2);
  assert.equal(rint(-0.5), 0);
  assert.equal(rint(2.4), 2);
  assert.equal(rint(2.6), 3);
});

test(`RT aggregation reference has cases (got ${cases.length})`, () => {
  assert.ok(cases.length > 0, "run `python tests/gen_io_reference.py` first");
});

test("aggregate() counts and RT quantiles match the Python reference", () => {
  for (const c of cases) {
    const { counts, rtq } = aggregate(c.trials, { hasRT: true });
    assert.deepEqual(counts.flat(), c.counts.flat(2));
    for (let i = 0; i < rtq.length; i++) {
      assert.ok(
        Math.abs(rtq[i] - c.rtq[i]) < 1e-9,
        `rtq[${i}]: got ${rtq[i]}, want ${c.rtq[i]}`,
      );
    }
  }
});

test("aggregate() row sums equal trial counts", () => {
  for (const c of cases) {
    const { counts, trials } = aggregate(c.trials, { hasRT: true });
    counts.forEach((row, s) => {
      assert.equal(row.reduce((a, b) => a + b, 0), trials[s]);
    });
  }
});

test("buildLevelMap resolves A1B1-style and slash-separated tokens the same way", () => {
  const m1 = buildLevelMap(["A1B1", "A1B2", "A2B1", "A2B2"]);
  const m2 = buildLevelMap(["Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"]);
  assert.equal(m1.error, undefined);
  assert.equal(m2.error, undefined);
  assert.equal(Object.keys(m1.map).length, 4);
  assert.equal(Object.keys(m2.map).length, 4);
});

test("buildLevelMap rejects a dimension with more than 2 levels", () => {
  const m = buildLevelMap(["A1B1", "A2B1", "A3B1"]); // 3 levels on dimension A
  assert.ok(m.error, "expected an error for a non-binary dimension");
});

test("buildLevelMap rejects a token that doesn't split into two levels", () => {
  const m = buildLevelMap(["A1B1", "A1B2", "not a valid code"]);
  assert.ok(m.error, "expected an error for an unparseable token");
});

test("parseCSV reads templateCSV's header + example rows (long format, with RT)", () => {
  // templateCSV() appends trailing human-readable instructions after the header
  // + example rows, for a person editing the file in a spreadsheet -- not meant
  // to be machine-parsed as-is. Feed it just the data portion, as a filled-in
  // template would look after the instructions are deleted.
  const lines = templateCSV({ withRT: true }).split("\n").slice(0, 5).join("\n");
  const parsed = parseCSV(lines);
  assert.equal(parsed.error, undefined);
  assert.equal(parsed.format, "long");
  assert.equal(parsed.hasRT, true);
  assert.equal(parsed.trials.length, 4);
});

test("parseCSV reads a bare 4x4 matrix", () => {
  const csv = "71,17,9,5\n20,67,5,9\n13,6,63,20\n5,10,15,71";
  const parsed = parseCSV(csv);
  assert.equal(parsed.error, undefined);
  assert.equal(parsed.format, "matrix");
  assert.deepEqual(parsed.counts, [
    [71, 17, 9, 5], [20, 67, 5, 9], [13, 6, 63, 20], [5, 10, 15, 71],
  ]);
});

test("checkInputs flags trial counts outside the training distribution", () => {
  const agg = { counts: [[2000, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
               trials: [2000, 0, 0, 0], rtq: null };
  const report = checkInputs(agg, { hasRT: false });
  assert.ok(report.warnings.length > 0 || report.errors?.length > 0,
    "expected a warning/error for an out-of-range trial count and empty stimuli");
});
