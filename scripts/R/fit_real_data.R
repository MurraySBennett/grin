# fit_real_data.R — export the REAL 2x2 confusion matrices shipped with mdsdt and fit them,
# so GRIN can be compared on real data (where there is NO ground truth).
#
#   Rscript scripts/R/fit_real_data.R      then:   python scripts/compare_real_data.py
#
# Writes: data/real/real_matrices.csv       (the matrices, for GRIN to read)
#         results/mle_fits/real_data_fits.csv  (mdsdt's model selection + parameters)
#
# Real 2x2 identification confusion matrices in mdsdt:
#   thomas01a, thomas01b   — face recognition, observers A and B (Thomas, 2001)
#   silbert09a, silbert09b, silbert12
# thomas15a/b are 3x3 and are skipped automatically (GRIN is 2x2 only).

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
})

if (!requireNamespace("mdsdt", quietly = TRUE)) {
  stop("mdsdt is not installed.  Run:  install.packages('mdsdt')")
}

candidates <- c(
  "thomas01a", "thomas01b", "silbert09a", "silbert09b", "silbert12",
  "thomas15a", "thomas15b"
) # the 3x3 ones are filtered out below
rows <- list()
fits <- list()

for (nm in candidates) {
  ok <- tryCatch(
    {
      data(list = nm, package = "mdsdt", envir = environment())
      TRUE
    },
    error = function(e) FALSE
  )
  if (!ok) {
    message("skip (not found): ", nm)
    next
  }

  cmat <- tryCatch(as.matrix(get(nm)), error = function(e) NULL)
  if (is.null(cmat) || !all(dim(cmat) == c(4, 4))) {
    message("skip (not a 4x4 / 2x2 design): ", nm, " [", paste(dim(cmat), collapse = "x"), "]")
    next
  }

  rows[[nm]] <- c(dataset = nm, as.numeric(t(cmat))) # row-major: s0r0..s3r3

  # each fit wrapped: one bad dataset must not kill the whole script
  safe_fit <- function(...) tryCatch(mdsdt::fit.grt(cmat, ...), error = function(e) NULL)
  safe_aic <- function(f) {
    if (is.null(f)) {
      NA_real_
    } else {
      tryCatch(as.numeric(mdsdt::GOF(f, teststat = "AIC")), error = function(e) NA_real_)
    }
  }

  fitlist <- list(
    full = safe_fit(),
    ps = safe_fit(PS_x = TRUE, PS_y = TRUE),
    pi = safe_fit(PI = "all"),
    pi_ps = safe_fit(PS_x = TRUE, PS_y = TRUE, PI = "all"),
    rho1 = safe_fit(PI = "same_rho")
  )
  aics <- vapply(fitlist, safe_aic, numeric(1))
  if (all(is.na(aics))) {
    message("skip (all fits failed): ", nm)
    next
  }

  best <- names(which.min(aics))
  fits[[nm]] <- tibble(
    dataset = nm, best_model = best,
    aic_full = aics[["full"]], aic_ps = aics[["ps"]],
    aic_pi = aics[["pi"]], aic_pi_ps = aics[["pi_ps"]],
    aic_rho1 = aics[["rho1"]],
    n_trials = sum(cmat)
  )
  message(nm, ": n=", sum(cmat), "  best (AIC) = ", best)
}

if (length(rows) == 0) stop("No usable 2x2 datasets found in mdsdt.")

dir.create("data/real", recursive = TRUE, showWarnings = FALSE)
dir.create("results/mle_fits", recursive = TRUE, showWarnings = FALSE)

mat <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
colnames(mat) <- c("dataset", paste0("cm_", rep(0:3, each = 4), rep(0:3, times = 4)))
write_csv(mat, "data/real/real_matrices.csv")
write_csv(bind_rows(fits), "results/mle_fits/real_data_fits.csv")

message("\nwrote data/real/real_matrices.csv (", nrow(mat), " real observers)")
message("wrote results/mle_fits/real_data_fits.csv")
message("next:  python scripts/compare_real_data.py")
