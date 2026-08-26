# grtools_seed_stability.R — is grtools' convergence flag a property of the matrix,
# or of the random restarts?
#
#   Rscript scripts/R/grtools_seed_stability.R
#
# grt_hm_fit() searches from n_reps random starting points. The convergence code it
# reports belongs to whichever fit won, so the same matrix refitted under a different
# RNG seed can report a different code. This script refits a set of matrices under
# many seeds and records, per matrix: how often the code was zero, and whether the
# SELECTED MODEL changed.
#
# The distinction matters for how a convergence rate should be read. If the flag moves
# with the seed while the selected model does not, then a failure rate is partly a
# property of the search rather than of the data, and grtools' substantive output is
# more stable than its flag suggests.
#
# Writes results/mle_fits/grtools_seed_stability.csv

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tibble) })
if (!requireNamespace("grtools", quietly = TRUE)) stop("grtools is not installed")

# 10 seeds x 4 restart counts x 5 matrices is roughly 2 hours; this is meant to run
# inside run_baselines_overnight.sh, not interactively.
N_SEEDS  <- 10
N_REPS   <- 10
# Restart counts to sweep. grt_hm_fit()'s restart loop keeps whichever restart reaches
# the lowest negative log-likelihood REGARDLESS of its convergence code (see
# grtools:::fit_grt_models), so more restarts is not obviously more likely to yield a
# zero code: a better-fitting but abnormally-terminating restart displaces a cleanly
# converged one. This sweep tests that directly rather than assuming either way.
# n_reps=30 is dropped from the simulated arm: it costs three times the default run and
# the question -- does the flag move with the seed -- is already answered at 1/3/10.
# It is kept for the five real matrices, where the whole sweep is under 90 minutes.
REPS_SWEEP_REAL <- c(1L, 3L, 10L, 30L)
REPS_SWEEP_SIM  <- c(1L, 3L, 10L)
grtools_perm <- c(1, 3, 2, 4)

args <- commandArgs(trailingOnly = TRUE)
source_kind <- if (length(args) && args[1] == "simulated") "simulated" else "real"

mats <- list()
if (source_kind == "real") {
  for (nm in c("thomas01a", "thomas01b", "silbert09a", "silbert09b", "silbert12")) {
    ok <- tryCatch({ data(list = nm, package = "mdsdt", envir = environment()); TRUE },
                   error = function(e) FALSE)
    if (!ok) next
    cm <- tryCatch(as.matrix(get(nm)), error = function(e) NULL)
    if (!is.null(cm) && all(dim(cm) == c(4, 4))) mats[[nm]] <- cm
  }
} else {
  # a stratified handful from the simulated export, spanning the trial-count range
  d <- read_csv("data/simulated/test_set_for_R.csv", show_col_types = FALSE)
  # One matrix per trial band rather than two: nine cells is enough to see whether the
  # flag's seed-sensitivity depends on how much data an observer contributed.
  pick <- d %>% group_by(trial_bin) %>% slice_head(n = 1) %>% ungroup()
  for (i in seq_len(nrow(pick))) {
    r <- pick[i, ]
    cm <- matrix(as.numeric(r[paste0("cm_", rep(0:3, each = 4), rep(0:3, times = 4))]),
                 nrow = 4, byrow = TRUE)
    mats[[paste0("sim_", r$row_id, "_tps", round(r$tps))]] <- cm
  }
}

out <- list()
for (nm in names(mats)) {
  cg <- mats[[nm]][grtools_perm, grtools_perm]
  reps_sweep <- if (source_kind == "real") REPS_SWEEP_REAL else REPS_SWEEP_SIM
  for (nr in reps_sweep) {
  codes <- integer(0); models <- character(0)
  for (s in seq_len(N_SEEDS)) {
    set.seed(s)
    hm <- tryCatch(grtools::grt_hm_fit(cg, n_reps = nr), error = function(e) NULL)
    if (is.null(hm)) { codes <- c(codes, NA_integer_); models <- c(models, NA_character_); next }
    codes  <- c(codes, as.integer(hm$best_model$convergence[1]))
    models <- c(models, paste(hm$best_model$model, collapse = "/"))
  }
  out[[paste(nm, nr)]] <- tibble(
    dataset = nm, n_seeds = N_SEEDS, n_reps = nr,
    n_converged = sum(codes == 0, na.rm = TRUE),
    prop_converged = mean(codes == 0, na.rm = TRUE),
    n_distinct_codes = dplyr::n_distinct(codes, na.rm = TRUE),
    codes = paste(codes, collapse = " "),
    n_distinct_models = dplyr::n_distinct(models, na.rm = TRUE),
    modal_model = names(sort(table(models), decreasing = TRUE))[1],
    prop_modal_model = max(table(models)) / sum(!is.na(models)))
  r <- out[[paste(nm, nr)]]
  message(sprintf("%-22s n_reps=%2d  converged %2d/%d seeds; %d distinct model(s) (%s)",
                  nm, nr, r$n_converged, N_SEEDS, r$n_distinct_models, r$modal_model))
  }
}

res <- bind_rows(out)
dir.create("results/mle_fits", recursive = TRUE, showWarnings = FALSE)
write_csv(res, "results/mle_fits/grtools_seed_stability.csv")
message("\nconvergence flag varied with seed in ",
        sum(res$n_distinct_codes > 1), "/", nrow(res), " (matrix x n_reps) cells")
message("selected model varied with seed in ",
        sum(res$n_distinct_models > 1), "/", nrow(res), " cells")
message("\nmean proportion converged, by n_reps:")
agg <- aggregate(prop_converged ~ n_reps, data = res, FUN = mean)
for (i in seq_len(nrow(agg))) {
  message(sprintf("  n_reps=%2d  %.2f", agg$n_reps[i], agg$prop_converged[i]))
}
message("wrote results/mle_fits/grtools_seed_stability.csv")
