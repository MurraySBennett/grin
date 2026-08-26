#!/usr/bin/env bash
# Re-fit the R baselines on THIS machine, so that every timing reported in the
# manuscript comes from one CPU instead of two.
#
# Run this when the laptop is otherwise idle and leave it alone. Timings are the
# point of the exercise, so anything else competing for CPU invalidates them:
# do not run it alongside training, the Python MLE timing, or the real-data script.
#
#   nohup bash scripts/run_baselines_overnight.sh > /tmp/grin_baselines.log 2>&1 &
#
# Expect roughly 7-9 hours on an idle machine (594 matrices; grtools fits a
# 12-model hierarchy with 10 restarts by default, which dominates).
# The previous Windows/RTX-A5000 results are preserved as
# results/mle_fits/baseline_fits.WINDOWS-A5000.csv and are NOT overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

export R_LIBS_USER="$HOME/R/library"
STAMP=$(date +%Y%m%d-%H%M)

if [[ -f results/mle_fits/baseline_fits.csv ]]; then
  cp -p results/mle_fits/baseline_fits.csv \
        "results/mle_fits/baseline_fits.pre-$STAMP.csv"
fi

echo "=== host ==="
uname -a
grep -m1 "model name" /proc/cpuinfo || true
nproc
echo "=== load at start (should be ~0) ==="
uptime
echo "=== R ==="
Rscript -e '.libPaths("'"$HOME"'/R/library"); cat(R.version.string, "\n");
            for (p in c("mdsdt","grtools")) cat(p, as.character(packageVersion(p)), "\n")'

echo "=== seed ==="
grep -m1 "BASELINE_SEED <-" scripts/R/fit_baselines.R

echo "=== fitting ==="
time Rscript -e '.libPaths("'"$HOME"'/R/library"); source("scripts/R/fit_baselines.R")'

echo "=== grtools seed stability (context for the convergence rate) ==="
Rscript -e '.libPaths("'"$HOME"'/R/library"); source("scripts/R/grtools_seed_stability.R")' || true
Rscript -e '.libPaths("'"$HOME"'/R/library"); source("scripts/R/grtools_seed_stability.R")' simulated || true

echo "=== done; refreshing downstream artifacts ==="
python_bin=".venv/bin/python"
export PYTHONPATH="$PWD"
$python_bin scripts/compare_to_r.py            || echo "(compare_to_r failed)"
$python_bin scripts/make_recovery_figures.py   || echo "(make_recovery_figures failed)"
$python_bin scripts/failure_subset_analysis.py || echo "(failure_subset failed)"
echo "=== finished at $(date) ==="
