# Inference tests need a real libtorch runtime; skip gracefully where it isn't
# available (matches CRAN's own convention for packages built on 'torch').
skip_if_not_installed("torch")
skip_if_not(isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE)),
           "libtorch is not installed (torch::install_torch())")

M <- matrix(c(71, 17,  9,  5,
              20, 67,  5,  9,
              13,  6, 63, 20,
               5, 10, 15, 71), nrow = 4, byrow = TRUE)

test_that("the bundled model file exists and loads", {
  expect_true(file.exists(grin_default_model_path()))
  m <- grin_model()
  expect_s3_class(m, "grin_model")
})

test_that("grin_infer returns a well-formed result and constructs on a 4x4 matrix", {
  out <- grin_infer(M)
  expect_s3_class(out, "grin_inference")
  expect_s3_class(out$result, "grin_result")
  expect_length(out$result$params, 12)
  expect_length(out$result$std, 12)
  expect_true(all(out$result$std > 0))
  expect_equal(out$result$names, PARAM_NAMES)
  expect_true(all(c("p_PI", "p_sep_A", "p_sep_B", "p_corr",
                    "evidence_PI", "evidence_sep_A", "evidence_sep_B") %in% names(out$constructs)))
  expect_true(out$constructs$p_PI >= 0 && out$constructs$p_PI <= 1)
})

test_that("a length-16 row-major vector gives the same result as the equivalent 4x4 matrix", {
  v <- as.vector(t(M))
  out_mat <- grin_infer(M)
  out_vec <- grin_infer(v)
  expect_equal(out_mat$result$params, out_vec$result$params)
})

test_that("explicit trials override the row-sum default", {
  out_default <- grin_infer(M)
  out_explicit <- grin_infer(M, trials = rowSums(M))
  expect_equal(out_default$result$params, out_explicit$result$params)
})

test_that("the model is session-cached by path", {
  m1 <- grin_model()
  m2 <- grin_model()
  expect_identical(m1, m2)
})

test_that("the bundled example CSV is readable and infers cleanly", {
  csv_path <- system.file("extdata", "example_cm.csv", package = "grin")
  expect_true(file.exists(csv_path))
  ex <- as.matrix(read.csv(csv_path, header = FALSE))
  storage.mode(ex) <- "integer"
  out <- grin_infer(ex)
  expect_length(out$result$params, 12)
})
