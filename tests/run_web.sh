#!/usr/bin/env bash
# Regenerate the Python reference values, then check the JS against them.
# Run from the repo root:  bash tests/run.sh
set -e
python3 tests/gen_reference.py
python3 tests/gen_io_reference.py
python3 tests/gen_fit_reference.py
for t in core io fit model; do
  echo "=== $t ==="
  node "tests/$t.test.mjs"
done