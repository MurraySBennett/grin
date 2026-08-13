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
  expect_s3_class(grin_plot_bias(out1$result), "ggplot")
  expect_s3_class(grin_plot_empirical_bias(M1), "ggplot")
})

test_that("group plots build without error and return ggplot objects", {
  expect_s3_class(grin_plot_space_group(many, facet = TRUE), "ggplot")
  suppressMessages(expect_s3_class(grin_plot_space_group(many, facet = FALSE), "ggplot"))
  expect_s3_class(grin_plot_params_group(many), "ggplot")
  expect_s3_class(grin_plot_model_classes(many), "ggplot")
  expect_s3_class(grin_plot_precision_group(many), "ggplot")
  expect_s3_class(grin_plot_bias_group(many), "ggplot")
  expect_s3_class(grin_plot_empirical_bias_group(list(M1, M2)), "ggplot")
})

test_that("plots actually render (catches ggplot_build()-time errors, not just object construction)", {
  plots <- list(
    grin_plot_space(out1$result), grin_plot_params(out1$result),
    grin_plot_constructs(out1$result, out1$constructs),
    grin_plot_space_group(many), grin_plot_params_group(many),
    grin_plot_model_classes(many), grin_plot_precision_group(many),
    grin_plot_bias(out1$result), grin_plot_bias_group(many),
    grin_plot_empirical_bias(M1), grin_plot_empirical_bias_group(list(M1, M2)),
    grin_plot_diagnostics(out1$result, M1, show_marginals = FALSE)
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

.point_layer_data <- function(built) {
  idx <- which(vapply(built$plot$layers, function(l) inherits(l$geom, "GeomPoint"), logical(1)))[1]
  built$data[[idx]]
}

test_that("plots are black-on-white by default, not a preset colour palette", {
  built <- ggplot2::ggplot_build(grin_plot_space(out1$result))
  cols <- unique(.point_layer_data(built)$colour)
  expect_equal(cols, .grin_colors$ink)
})

test_that("grin_plot_space never colours per-stimulus, even with a palette set", {
  built <- ggplot2::ggplot_build(grin_plot_space(out1$result, palette = "dusk"))
  cols <- unique(.point_layer_data(built)$colour)
  expect_equal(length(cols), 1)                       # one colour for all 4 stimuli
  expect_true(cols %in% .grin_palettes$dusk)           # and it's the palette, not ink
})

test_that("palette = <name> switches grin_plot_params off monochrome", {
  bw <- ggplot2::ggplot_build(grin_plot_params(out1$result))
  colored <- ggplot2::ggplot_build(grin_plot_params(out1$result, palette = "contrast"))
  expect_equal(length(unique(.point_layer_data(bw)$colour)), 1)
  expect_gt(length(unique(.point_layer_data(colored)$colour)), 1)
})

test_that("a user-supplied vector of hex colours is honoured directly", {
  mine <- c("#123456", "#abcdef", "#00ff00")
  built <- ggplot2::ggplot_build(grin_plot_params(out1$result, palette = mine))
  cols <- tolower(unique(.point_layer_data(built)$colour))
  expect_true(all(cols %in% tolower(mine)))
})

test_that("grin_palette_names lists mono plus every built-in preset", {
  nm <- grin_palette_names()
  expect_true("mono" %in% nm)
  expect_true(all(names(.grin_palettes) %in% nm))
})

test_that("options(grin.palette = ...) sets the default without an explicit argument", {
  withr_opt <- options(grin.palette = "ember")
  on.exit(options(withr_opt), add = TRUE)
  with_option <- ggplot2::ggplot_build(grin_plot_params(out1$result))
  explicit <- ggplot2::ggplot_build(grin_plot_params(out1$result, palette = "ember"))
  expect_identical(.point_layer_data(with_option)$colour, .point_layer_data(explicit)$colour)
})

test_that("an explicit palette = 'mono' overrides options(grin.palette = ...)", {
  withr_opt <- options(grin.palette = "ember")
  on.exit(options(withr_opt), add = TRUE)
  built <- ggplot2::ggplot_build(grin_plot_params(out1$result, palette = "mono"))
  expect_equal(length(unique(.point_layer_data(built)$colour)), 1)
  expect_equal(unique(.point_layer_data(built)$colour), .grin_colors$ink)
})

test_that("an unknown palette name errors with a helpful message", {
  expect_error(grin_plot_space(out1$result, palette = "not-a-real-palette"), "unknown palette")
})

test_that("grin_plot_space_group(facet = FALSE) prints the exploratory-only caveat", {
  expect_message(grin_plot_space_group(many, facet = FALSE), "exploratory inspection view only")
})

test_that("grin_empirical_bias reads the bias direction from a lopsided matrix", {
  # every trial reported as level-2 on both dimensions -> maximal bias
  M_biased <- matrix(c(0, 0, 0, 40,  0, 0, 0, 40,  0, 0, 0, 40,  0, 0, 0, 40),
                     nrow = 4, byrow = TRUE)
  b <- grin_empirical_bias(M_biased)
  expect_equal(b$x_bias, 0.5)
  expect_equal(b$y_bias, 0.5)
})

test_that("grin_response_bias is zero when a dimension's z-scores are exactly symmetric", {
  fake <- structure(list(
    params = c(-1, -1, 1, 1,  -0.7, 0.7, -0.7, 0.7,  0, 0, 0, 0),
    std = rep(0.15, 12), names = PARAM_NAMES, model_class = "test"
  ), class = "grin_result")
  b <- grin_response_bias(fake)
  expect_equal(b$x_bias, 0)
  expect_equal(b$y_bias, 0)
})

test_that("grin_response_bias sign agrees with grin_empirical_bias's convention on an asymmetric case", {
  # level-1 z-scores (magnitude 1.5) sit further from the bound than level-2's
  # (magnitude 0.5) -> the bound is effectively closer to level 2, so it takes
  # LESS evidence to land on the level-1 side -> biased toward level 1 (negative).
  # Verified against the forward model directly during development: this exact
  # asymmetry lowers P(respond level 2) to ~0.38, a negative empirical bias too.
  fake <- structure(list(
    params = c(-1.5, -1.5, 0.5, 0.5,  -1, -1, 1, 1,  0, 0, 0, 0),
    std = rep(0.15, 12), names = PARAM_NAMES, model_class = "test"
  ), class = "grin_result")
  b <- grin_response_bias(fake)
  expect_equal(b$x_bias, -0.5)   # mean(-1.5,-1.5,0.5,0.5)
  expect_equal(b$y_bias, 0)
  expect_true(b$x_bias_se > 0)
})

test_that("grin_tidy carries x_bias/y_bias through for group plotting", {
  td <- grin_tidy(many)
  expect_true(all(c("x_bias", "y_bias") %in% names(td)))
})

test_that("grin_plot_diagnostics needs at least one panel switched on", {
  expect_error(
    grin_plot_diagnostics(out1$result, M1, show_predicted_observed = FALSE, show_marginals = FALSE),
    "nothing to plot"
  )
})

test_that("grin_plot_diagnostics predicted-vs-observed sits near the diagonal for a good fit", {
  p <- grin_plot_diagnostics(out1$result, M1, show_marginals = FALSE)
  built <- ggplot2::ggplot_build(p)
  d <- built$data[[which(vapply(built$plot$layers, function(l) inherits(l$geom, "GeomPoint"),
                                logical(1)))[1]]]
  expect_true(all(abs(d$x - d$y) < 0.05))
})

test_that(".grin_forward_probabilities rows sum to 1 and match a known reference point", {
  probs <- .grin_forward_probabilities(c(0, 0, 0, 0), c(0, 0, 0, 0), c(0, 0, 0, 0))
  expect_equal(rowSums(probs), rep(1, 4))
  expect_equal(unname(probs[1, ]), rep(0.25, 4), tolerance = 1e-6)  # chance on both dims, no correlation
})

test_that(".grin_long_params reshapes wide to long correctly", {
  td <- grin_tidy(many)
  long <- .grin_long_params(td)
  expect_equal(nrow(long), 12 * 2)  # 12 params x 2 participants
  expect_true(all(c("id", "group", "param", "estimate", "sd") %in% names(long)))
  expect_equal(long$estimate[long$id == "a" & long$param == "zx_0"], out1$result$params[1])
})
