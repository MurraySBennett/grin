# The optional interval correction must change uncertainty and nothing else.
# See scripts/R/verify_recalibration.R for the standalone version of these checks.

cm <- matrix(c(83, 112, 47, 11, 38, 154, 28, 33,
               15, 27, 117, 94, 6, 36, 75, 136), nrow = 4, byrow = TRUE)

test_that("the shipped scale factors are present and have the fitted values", {
  skip_if_not_installed("jsonlite")
  f <- system.file("extdata", "recalibration.json", package = "grin")
  expect_true(nzchar(f))
  spec <- jsonlite::fromJSON(f)
  expect_equal(spec$global_scale$z, 0.8753, tolerance = 1e-3)
  expect_equal(spec$global_scale$rho, 1.0822, tolerance = 1e-3)
  # the two families must be corrected in OPPOSITE directions; a build in which both
  # scales sit on the same side of 1 means the fit has been regenerated wrongly
  expect_lt(spec$global_scale$z, 1)
  expect_gt(spec$global_scale$rho, 1)
})

test_that("calibrated = TRUE rescales widths and leaves everything else alone", {
  skip_if_not_installed("torch")
  skip_if_not_installed("jsonlite")
  raw <- grin_infer(cm)
  cal <- grin_infer(cm, calibrated = TRUE)

  expect_false(raw$result$calibrated)
  expect_true(cal$result$calibrated)

  expect_equal(cal$result$std[1:8] / raw$result$std[1:8], rep(0.8753, 8),
               tolerance = 1e-3)
  expect_equal(cal$result$std[9:12] / raw$result$std[9:12], rep(1.0822, 4),
               tolerance = 1e-3)

  expect_equal(raw$result$params, cal$result$params)
  expect_equal(raw$constructs, cal$constructs)
  expect_identical(raw$result$model_class, cal$result$model_class)
  expect_equal(cal$result$std_raw, raw$result$std)
})

test_that("the default is uncorrected, so published results do not depend on version", {
  skip_if_not_installed("torch")
  expect_false(grin_infer(cm)$result$calibrated)
  expect_equal(grin_infer(cm)$result$scale, rep(1, 12))
})
