skip_if_not_installed("torch")
skip_if_not(isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE)),
           "libtorch is not installed (torch::install_torch())")
skip_if_not_installed("ggplot2")

M1 <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
              13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
M2 <- matrix(c(50, 10, 15, 25, 12, 55, 20, 13,
               18, 22, 48, 12,  8, 14, 18, 60), nrow = 4, byrow = TRUE)

out1 <- grin_infer(M1)
out2 <- grin_infer(M2)
many <- list(a = out1, b = out2)

test_that("grin_tidy handles a single result and a named list", {
  one <- grin_tidy(out1)
  expect_equal(nrow(one), 1)
  expect_equal(one$id, "p1")

  td <- grin_tidy(many)
  expect_equal(nrow(td), 2)
  expect_equal(td$id, c("a", "b"))
  expect_true(all(c("model_class", "zx_0", "rho_3", "p_PI", "evidence_sep_A") %in% names(td)))
})

test_that("grin_tidy default IDs are p1, p2, ... for an unnamed list", {
  td <- grin_tidy(list(out1, out2))
  expect_equal(td$id, c("p1", "p2"))
})

test_that("individual plots build without error and return ggplot objects", {
  expect_s3_class(grin_plot_space(out1$result), "ggplot")
  expect_s3_class(grin_plot_params(out1$result), "ggplot")
  expect_s3_class(grin_plot_constructs(out1$result, out1$constructs), "ggplot")
})

test_that("group plots build without error and return ggplot objects", {
  expect_s3_class(grin_plot_space_group(many, facet = TRUE), "ggplot")
  expect_s3_class(grin_plot_space_group(many, facet = FALSE), "ggplot")
  expect_s3_class(grin_plot_params_group(many), "ggplot")
  expect_s3_class(grin_plot_model_classes(many), "ggplot")
  expect_s3_class(grin_plot_precision_group(many), "ggplot")
})

test_that("plots actually render (catches ggplot_build()-time errors, not just object construction)", {
  plots <- list(
    grin_plot_space(out1$result), grin_plot_params(out1$result),
    grin_plot_constructs(out1$result, out1$constructs),
    grin_plot_space_group(many), grin_plot_params_group(many),
    grin_plot_model_classes(many), grin_plot_precision_group(many)
  )
  for (p in plots) expect_silent(ggplot2::ggplot_build(p))
})

test_that("grin_plot_constructs flags a construct the data can't decide", {
  stub_constructs <- list(p_PI = 0.52, p_sep_A = 0.97, p_sep_B = 0.10,
                          p_corr = c(0.52, 0.30, 0.18),
                          evidence_PI = FALSE, evidence_sep_A = TRUE, evidence_sep_B = TRUE)
  p <- grin_plot_constructs(out1$result, stub_constructs)
  built <- ggplot2::ggplot_build(p)
  labels <- built$plot$data$label
  expect_true(any(labels == "insufficient evidence"))
})

test_that(".grin_long_params reshapes wide to long correctly", {
  td <- grin_tidy(many)
  long <- .grin_long_params(td)
  expect_equal(nrow(long), 12 * 2)  # 12 params x 2 participants
  expect_true(all(c("id", "group", "param", "estimate", "sd") %in% names(long)))
  expect_equal(long$estimate[long$id == "a" & long$param == "zx_0"], out1$result$params[1])
})
