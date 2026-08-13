# plot-individual.R: one-participant plots. All take a grin_result (e.g.
# grin_infer(M)$result) and, where relevant, its paired constructs list.

#' Plot one participant's inferred perceptual space
#'
#' Four stimulus means in the (zx, zy) plane, each with an ellipse showing its
#' within-stimulus correlation (rho), and the decision bounds -- always at 0 in
#' GRIN's identified coordinates, which is the whole point of working in them.
#'
#' @param result A `grin_result` (e.g. `grin_infer(M)$result`).
#' @param ci Confidence level for the ellipse radius (default 0.90, matching
#'   `result$ci_low`/`result$ci_high`'s own convention).
#' @return A ggplot object.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_plot_space(grin_infer(M)$result)
#' }
#' @export
grin_plot_space <- function(result, ci = 0.90) {
  stopifnot(inherits(result, "grin_result"))
  stim <- c("A1B1", "A1B2", "A2B1", "A2B2")
  k <- stats::qnorm(0.5 + ci / 2)
  p <- stats::setNames(as.numeric(result$params), result$names)

  centers <- data.frame(stimulus = stim,
                        zx = p[paste0("zx_", 0:3)], zy = p[paste0("zy_", 0:3)])
  ellipses <- do.call(rbind, lapply(seq_along(stim), function(i) {
    e <- .grin_ellipse_pts(centers$zx[i], centers$zy[i], p[[paste0("rho_", i - 1)]], k)
    e$stimulus <- stim[i]
    e
  }))

  ggplot2::ggplot() +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_path(data = ellipses, ggplot2::aes(.data$x, .data$y, color = .data$stimulus)) +
    ggplot2::geom_point(data = centers, ggplot2::aes(.data$zx, .data$zy, color = .data$stimulus),
                        size = 3) +
    ggplot2::scale_color_manual(values = .grin_stim_colors(), name = "stimulus") +
    ggplot2::coord_equal() +
    ggplot2::labs(x = "dimension A (zx)", y = "dimension B (zy)",
                  title = sprintf("Perceptual space (%s)", result$model_class),
                  subtitle = sprintf("%.0f%% ellipses; dashed lines are the decision bounds", 100 * ci)) +
    theme_grin()
}

#' Plot one participant's 12 parameter estimates with credible intervals
#'
#' A dot-and-whisker ("forest") plot, grouped by parameter type.
#'
#' @param result A `grin_result`.
#' @return A ggplot object.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_plot_params(grin_infer(M)$result)
#' }
#' @export
grin_plot_params <- function(result) {
  stopifnot(inherits(result, "grin_result"))
  group <- rep(c("zx", "zy", "rho"), each = 4)
  df <- data.frame(param = factor(result$names, levels = rev(result$names)),
                   group = group, estimate = result$params,
                   ci_low = result$ci_low, ci_high = result$ci_high)

  ggplot2::ggplot(df, ggplot2::aes(x = .data$estimate, y = .data$param, color = .data$group)) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$ci_low, xmax = .data$ci_high),
                           orientation = "y", width = 0) +
    ggplot2::geom_point(size = 2.5) +
    ggplot2::scale_color_manual(values = stats::setNames(.grin_palette[1:3], c("zx", "zy", "rho")),
                                name = NULL) +
    ggplot2::labs(x = "estimate (90% CI)", y = NULL, title = "Parameter estimates") +
    theme_grin()
}

#' Plot one participant's construct probabilities
#'
#' Two panels: correlation structure (P(PI), P(RHO1), P(free), which sum to 1)
#' and separability (P(separable A), P(separable B), each independent). Bars
#' for a construct the data cannot decide (`evidence_* == FALSE`) are flagged
#' rather than plotted as if they were informative -- the identifiability
#' limit is a property of the data, and this plot says so rather than hiding it.
#'
#' @param result A `grin_result`.
#' @param constructs The paired constructs list (e.g. `grin_infer(M)$constructs`).
#' @return A ggplot object.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' out <- grin_infer(M)
#' grin_plot_constructs(out$result, out$constructs)
#' }
#' @export
grin_plot_constructs <- function(result, constructs) {
  df <- data.frame(
    panel = c("correlation structure", "correlation structure", "correlation structure",
             "separability", "separability"),
    construct = factor(c("PI", "RHO1", "free", "separable A", "separable B"),
                       levels = c("PI", "RHO1", "free", "separable A", "separable B")),
    prob = c(constructs$p_corr, constructs$p_sep_A, constructs$p_sep_B),
    evidence = c(rep(constructs$evidence_PI, 3), constructs$evidence_sep_A, constructs$evidence_sep_B)
  )
  df$label <- ifelse(df$evidence, "", "insufficient evidence")

  ggplot2::ggplot(df, ggplot2::aes(x = .data$construct, y = .data$prob, fill = .data$panel)) +
    ggplot2::geom_col(ggplot2::aes(alpha = .data$evidence), width = 0.6) +
    ggplot2::geom_text(ggplot2::aes(label = .data$label), vjust = -0.4, size = 3,
                       color = .grin_colors$mute) +
    ggplot2::scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.35), guide = "none") +
    ggplot2::scale_fill_manual(values = stats::setNames(.grin_palette[1:2],
                                                        c("correlation structure", "separability")),
                               guide = "none") +
    ggplot2::facet_wrap(~panel, scales = "free_x") +
    ggplot2::ylim(0, 1.08) +
    ggplot2::labs(x = NULL, y = "P(construct)",
                  title = sprintf("Construct probabilities (%s)", result$model_class)) +
    theme_grin()
}
