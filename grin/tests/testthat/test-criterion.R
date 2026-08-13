# Port of grintools' test_grin_io.py stopping-decision cases: stubbed
# posterior/constructs, no model/network needed.

.stub_result <- function(std) {
  std <- as.numeric(std)
  list(names = PARAM_NAMES, std = std, ci_low = -std * 1.64, ci_high = std * 1.64)
}

tight <- .stub_result(rep(0.05, 12))
loose <- .stub_result(c(rep(0.05, 8), rep(0.40, 4)))   # rho block still wide

test_that("precision target: all-tight stops, wide-rho does not", {
  d1 <- grin_evaluate(grin_criterion(list(grin_target_precision(sd_max = 0.10))), tight)
  expect_true(d1$stop)
  d2 <- grin_evaluate(grin_criterion(list(grin_target_precision(sd_max = 0.10))), loose)
  expect_false(d2$stop)
})

test_that("precision target restricted to zx/zy ignores the wide rho block", {
  d <- grin_evaluate(grin_criterion(list(grin_target_precision(params = c("zx", "zy"), sd_max = 0.10))), loose)
  expect_true(d$stop)
})

test_that("grin_stop_on_precision convenience matches the explicit form", {
  d <- grin_stop_on_precision(tight, sd_max = 0.10)
  expect_true(d$stop)
})

decisive <- list(p_PI = 0.02, p_sep_A = 0.97, p_sep_B = 0.10,
                 evidence_PI = TRUE, evidence_sep_A = TRUE, evidence_sep_B = TRUE)
undecided_pi <- list(p_PI = 0.52, p_sep_A = 0.97, p_sep_B = 0.10,
                     evidence_PI = FALSE, evidence_sep_A = TRUE, evidence_sep_B = TRUE)

test_that("probability targets read the right side of the threshold", {
  d <- grin_evaluate(grin_criterion(list(grin_target_probability("PS_A", at_least = 0.9))), NULL, decisive)
  expect_true(d$stop)
  d <- grin_evaluate(grin_criterion(list(grin_target_probability("PS_B", at_least = 0.9))), NULL, decisive)
  expect_false(d$stop)
  d <- grin_evaluate(grin_criterion(list(grin_target_probability("PI_violated", at_least = 0.9))), NULL, decisive)
  expect_true(d$stop)   # 1 - 0.02 = 0.98
})

test_that("an undecidable construct never silently stops, and is reported blocked", {
  d <- grin_evaluate(grin_criterion(list(grin_target_probability("PI", at_least = 0.9))), NULL, undecided_pi)
  expect_false(d$stop)
  expect_equal(d$blocked_by, "PI")
})

test_that("combine 'all' and 'any' behave as documented", {
  crit_all <- grin_criterion(list(grin_target_probability("PS_A", at_least = 0.9),
                                  grin_target_precision(params = c("zx", "zy"), sd_max = 0.10)),
                             combine = "all")
  expect_true(grin_evaluate(crit_all, loose, decisive)$stop)

  crit_any <- grin_criterion(list(grin_target_probability("PS_B", at_least = 0.9),
                                  grin_target_precision(sd_max = 0.10)), combine = "any")
  expect_false(grin_evaluate(crit_any, loose, decisive)$stop)
  expect_true(.criterion_needs_constructs(crit_all))
})

test_that("Target constructors validate their arguments", {
  expect_error(grin_target_precision(sd_max = 0.1, ci_width_max = 0.2), "exactly one")
  expect_error(grin_target_precision(), "exactly one")
  expect_error(grin_target_probability("NOT_A_CONSTRUCT", at_least = 0.9), "unknown construct")
  expect_error(grin_criterion(list(), combine = "sometimes"), "all.*any")
})
