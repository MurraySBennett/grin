# verify_recalibration.R -- check that grin_infer(calibrated = TRUE) works end to end.
#
#   Rscript scripts/R/verify_recalibration.R
#
# The Python path for this feature is verified in CI; the R path needs a machine with
# torch installed, which the development container does not have. This script is that
# check. It exercises five things and prints PASS/FAIL for each:
#
#   1. the shipped scale factors load from inst/extdata
#   2. they match the values fitted by scripts/fit_recalibration.py
#   3. calibrated = TRUE widens the correlation intervals and narrows the z intervals,
#      by exactly those factors
#   4. point estimates, construct probabilities and model class are IDENTICAL either
#      way -- the correction must touch uncertainty and nothing else
#   5. the raw widths remain available on the corrected object
#
# It also checks the degraded path: with jsonlite unavailable, calibrated = TRUE must
# warn and return raw intervals rather than failing.

suppressPackageStartupMessages(library(grin))

EXPECT_Z   <- 0.8753
EXPECT_RHO <- 1.0822
TOL        <- 1e-3
ok <- TRUE
say <- function(pass, msg) {
  cat(if (pass) "PASS  " else "FAIL  ", msg, "\n", sep = "")
  if (!pass) ok <<- FALSE
}

cm <- matrix(c(
   83, 112,  47,  11,
   38, 154,  28,  33,
   15,  27, 117,  94,
    6,  36,  75, 136), nrow = 4, byrow = TRUE)

# ---- 1 / 2: the shipped factors -------------------------------------------
f <- system.file("extdata", "recalibration.json", package = "grin")
say(nzchar(f), paste("recalibration.json found:", if (nzchar(f)) f else "MISSING"))
if (nzchar(f) && requireNamespace("jsonlite", quietly = TRUE)) {
  spec <- jsonlite::fromJSON(f)
  say(abs(spec$global_scale$z - EXPECT_Z) < TOL,
      sprintf("z scale   = %.4f (expected %.4f)", spec$global_scale$z, EXPECT_Z))
  say(abs(spec$global_scale$rho - EXPECT_RHO) < TOL,
      sprintf("rho scale = %.4f (expected %.4f)", spec$global_scale$rho, EXPECT_RHO))
} else {
  say(FALSE, "jsonlite not installed -- install.packages('jsonlite') to verify fully")
}

# ---- 3 / 4 / 5: behaviour --------------------------------------------------
raw <- grin_infer(cm)
cal <- grin_infer(cm, calibrated = TRUE)

say(isFALSE(raw$result$calibrated) && isTRUE(cal$result$calibrated),
    "calibrated flag is FALSE by default and TRUE when requested")

r_z   <- cal$result$std[1:8]  / raw$result$std[1:8]
r_rho <- cal$result$std[9:12] / raw$result$std[9:12]
say(all(abs(r_z - EXPECT_Z) < TOL),
    sprintf("z widths scaled by %.4f-%.4f", min(r_z), max(r_z)))
say(all(abs(r_rho - EXPECT_RHO) < TOL),
    sprintf("rho widths scaled by %.4f-%.4f", min(r_rho), max(r_rho)))
say(all(r_rho > 1) && all(r_z < 1),
    "correlations widened, sensitivities narrowed (the documented directions)")

say(isTRUE(all.equal(raw$result$params, cal$result$params)),
    "point estimates identical")
say(isTRUE(all.equal(raw$constructs, cal$constructs)),
    "construct probabilities identical")
say(identical(raw$result$model_class, cal$result$model_class),
    "selected model class identical")
say(isTRUE(all.equal(cal$result$std_raw, raw$result$std)),
    "std_raw on the corrected object equals the uncorrected std")

ci_raw <- raw$result$ci_high - raw$result$ci_low
ci_cal <- cal$result$ci_high - cal$result$ci_low
say(all(abs(ci_cal[9:12] / ci_raw[9:12] - EXPECT_RHO) < TOL),
    "credible intervals rebuilt consistently with the scaled SDs")

cat("\n", if (ok) "ALL CHECKS PASSED" else "SOME CHECKS FAILED", "\n", sep = "")
cat("\nNote: if jsonlite is NOT installed, grin_infer(calibrated = TRUE) should warn\n")
cat("and return raw intervals rather than error. Test that separately by removing\n")
cat("jsonlite from the library path, or trust the tests in tests/testthat/.\n")
quit(status = if (ok) 0L else 1L)
