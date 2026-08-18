# plot-individual.R: one-participant plots. All take a grin_result (e.g.
# grin_infer(M)$result) and, where relevant, its paired constructs list.
# Black-on-white by default; pass palette = "name" (see grin_palette_names())
# or your own vector of hex colours, or options(grin.palette = "name") for the
# session, to use colour instead.
#
# Every function returns a plain ggplot object. If a parameter below doesn't
# cover what you need, that is deliberate rather than an oversight: the
# returned object composes with ordinary ggplot2 calls
# (`grin_plot_space(x) + ggplot2::theme(...)`), which covers far more than
# this package could ever anticipate as an argument. See the package README's
# "Editing a figure further" section for worked examples in both base R and
# ggplot2.

#' Plot one participant's inferred perceptual space
#'
#' Four stimulus means in the (zx, zy) plane, each with an ellipse showing its
#' within-stimulus correlation (rho), and the decision bounds -- always at 0 in
#' GRIN's identified coordinates, which is the whole point of working in them.
#' Stimuli are told apart by a text label at each point by default rather than
#' by colour: with only one participant on the plot, four colours for four
#' points is more decoration than information, and the quadrant each stimulus
#' falls in is already fixed by the sign convention.
#'
#' The ellipse and the (optional) error bars show two different kinds of
#' uncertainty, and the plot never conflates them: the ellipse is the
#' predicted spread of a single trial's perceptual sample around the mean
#' (fixed at unit variance by the model, shaped by `rho`); the error bars, if
#' shown, are the *posterior* uncertainty about where that mean itself is,
#' given the data (`result$std`). A precisely-estimated participant can still
#' have a wide ellipse (they are just genuinely correlated/unseparated), and a
#' poorly-estimated one can have a narrow ellipse (few trials, but whatever
#' trials there were happened to look decisive).
#'
#' @param result A `grin_result` (e.g. `grin_infer(M)$result`).
#' @param ci Confidence level for the ellipse radius and (if shown) the error
#'   bars (default 0.90, matching `result$ci_low`/`result$ci_high`'s own
#'   convention).
#' @param palette Colour palette: `NULL` (default) defers to
#'   `options(grin.palette)`, itself default `"mono"` (black-on-white). Pass a
#'   name from [grin_palette_names()], or your own vector of hex colours.
#' @param title Plot title. `NULL` (default) uses
#'   `"Perceptual space (<model class>)"`; pass `""` for no title.
#' @param xlab,ylab Axis titles. Default to `"dimension A (zx)"` /
#'   `"dimension B (zy)"`; pass `NULL` to hide an axis title entirely (the
#'   ordinary ggplot2 convention).
#' @param stim_labels Length-4 character vector of labels for the four
#'   stimuli, in canonical order (A1B1, A1B2, A2B1, A2B2) -- e.g. the same
#'   labels you passed to [grin_to_confusion()]. Defaults to the canonical
#'   tokens themselves.
#' @param show_labels Draw the per-stimulus text labels? Default `TRUE`.
#' @param show_uncertainty Draw posterior-SD error bars (a horizontal and a
#'   vertical bar) through each stimulus mean, at the same confidence level as
#'   the ellipse? Default `TRUE` -- every estimate GRIN reports carries
#'   uncertainty, and the space plot shouldn't be the one place that hides it.
#'   Set `FALSE` if the crosshairs make a crowded plot harder to read.
#' @param show_marginals Add per-dimension marginal density strips (one curve
#'   per stimulus, `Normal(mean, 1)` under the model's own unit-variance
#'   convention) above and to the right of the main panel? Default `FALSE`.
#'   Requires the \pkg{patchwork} package. See also [grin_plot_diagnostics()],
#'   which pairs marginals with a predicted-vs-observed reconstruction panel
#'   for a fuller view (not a goodness-of-fit test of GRT itself -- see that
#'   function's documentation for why).
#' @param base_size Base font size in points (default 12); see
#'   [theme_grin()].
#' @return A `ggplot` object (or, if `show_marginals = TRUE`, a `patchwork`
#'   object -- both support `+` with further ggplot2 calls on their main
#'   panel is not guaranteed once composed; add customisation before setting
#'   `show_marginals = TRUE` if you need it applied to every panel).
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_plot_space(grin_infer(M)$result)
#' }
#' @export
grin_plot_space <- function(result, ci = 0.90, palette = NULL, title = NULL,
                            xlab = "dimension A (zx)", ylab = "dimension B (zy)",
                            stim_labels = CANON_STIM, show_labels = TRUE,
                            show_uncertainty = TRUE, show_marginals = FALSE,
                            base_size = 12) {
  stopifnot(inherits(result, "grin_result"))
  stopifnot(length(stim_labels) == 4)
  k <- stats::qnorm(0.5 + ci / 2)
  p <- stats::setNames(as.numeric(result$params), result$names)
  s <- stats::setNames(as.numeric(result$std), result$names)
  col <- .grin_group_colors(1, palette)

  centers <- data.frame(stimulus = stim_labels,
                        zx = p[paste0("zx_", 0:3)], zy = p[paste0("zy_", 0:3)],
                        sx = s[paste0("zx_", 0:3)], sy = s[paste0("zy_", 0:3)])
  ellipses <- do.call(rbind, lapply(seq_len(4), function(i) {
    e <- .grin_ellipse_pts(centers$zx[i], centers$zy[i], p[[paste0("rho_", i - 1)]], k)
    e$stimulus <- stim_labels[i]
    e
  }))

  g <- ggplot2::ggplot() +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute)

  if (show_uncertainty) {
    g <- g +
      ggplot2::geom_segment(data = centers, color = col, linewidth = 0.5,
        ggplot2::aes(x = .data$zx - k * .data$sx, xend = .data$zx + k * .data$sx,
                    y = .data$zy, yend = .data$zy)) +
      ggplot2::geom_segment(data = centers, color = col, linewidth = 0.5,
        ggplot2::aes(x = .data$zx, xend = .data$zx,
                    y = .data$zy - k * .data$sy, yend = .data$zy + k * .data$sy))
  }

  g <- g +
    ggplot2::geom_path(data = ellipses, ggplot2::aes(.data$x, .data$y, group = .data$stimulus),
                       color = col) +
    ggplot2::geom_point(data = centers, ggplot2::aes(.data$zx, .data$zy), color = col, size = 3)

  if (show_labels) {
    g <- g + ggplot2::geom_text(data = centers,
      ggplot2::aes(.data$zx, .data$zy, label = .data$stimulus), color = col, size = 3.2, vjust = -1.1)
  }

  g <- g +
    ggplot2::coord_equal() +
    ggplot2::labs(x = xlab, y = ylab,
                  title = .grin_title(title, sprintf("Perceptual space (%s)", result$model_class)),
                  subtitle = sprintf("%.0f%% ellipses%s; dashed lines are the decision bounds",
                                     100 * ci, if (show_uncertainty) " and error bars" else "")) +
    theme_grin(base_size)

  if (!show_marginals) return(g)
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    warning("show_marginals = TRUE needs the 'patchwork' package; returning the plot without marginals",
           call. = FALSE)
    return(g)
  }
  x_rng <- range(ellipses$x); y_rng <- range(ellipses$y)
  top <- .grin_marginal_strip(centers$zx, x_rng, "x", col, base_size)
  right <- .grin_marginal_strip(centers$zy, y_rng, "y", col, base_size)
  g <- g + ggplot2::coord_cartesian(xlim = x_rng, ylim = y_rng)
  patchwork::wrap_plots(top, patchwork::plot_spacer(), g, right,
                        ncol = 2, widths = c(4, 1), heights = c(1, 4))
}

#' @keywords internal
.grin_marginal_strip <- function(means, rng, orientation, col, base_size) {
  xs <- seq(rng[1], rng[2], length.out = 200)
  df <- do.call(rbind, lapply(seq_along(means), function(i) {
    data.frame(x = xs, density = stats::dnorm(xs, mean = means[i], sd = 1), stimulus = i)
  }))
  p <- ggplot2::ggplot(df, ggplot2::aes(.data$x, .data$density, group = .data$stimulus)) +
    ggplot2::geom_line(color = col) +
    ggplot2::labs(x = NULL, y = NULL) +
    theme_grin(base_size) +
    ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                  axis.line = ggplot2::element_blank(), panel.grid = ggplot2::element_blank())
  if (identical(orientation, "y")) p + ggplot2::coord_flip(xlim = rng) else p + ggplot2::coord_cartesian(xlim = rng)
}

#' Plot one participant's 12 parameter estimates with credible intervals
#'
#' A dot-and-whisker ("forest") plot, grouped by parameter type (position
#' already separates the groups, so colour is optional, not load-bearing).
#'
#' @param result A `grin_result`.
#' @param palette Colour palette (one colour per parameter group). `NULL`
#'   (default) defers to `options(grin.palette)`. See [grin_plot_space()].
#' @param title Plot title. `NULL` (default) uses `"Parameter estimates"`.
#' @param param_labels Named or length-12 character vector relabelling the
#'   y-axis parameter names (in `result$names` order). Defaults to
#'   `result$names` themselves (`zx_0`, ..., `rho_3`).
#' @param base_size Base font size in points (default 12).
#' @return A ggplot object.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_plot_params(grin_infer(M)$result)
#' }
#' @export
grin_plot_params <- function(result, palette = NULL, title = NULL,
                             param_labels = result$names, base_size = 12) {
  stopifnot(inherits(result, "grin_result"))
  stopifnot(length(param_labels) == length(result$names))
  group <- rep(c("zx", "zy", "rho"), each = 4)
  df <- data.frame(param = factor(param_labels, levels = rev(param_labels)),
                   group = group, estimate = result$params,
                   ci_low = result$ci_low, ci_high = result$ci_high)
  use_palette <- !is.null(palette) || !identical(getOption("grin.palette", "mono"), "mono")
  cols <- stats::setNames(.grin_group_colors(3, palette), c("zx", "zy", "rho"))

  ggplot2::ggplot(df, ggplot2::aes(x = .data$estimate, y = .data$param, color = .data$group)) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$ci_low, xmax = .data$ci_high),
                           orientation = "y", width = 0) +
    ggplot2::geom_point(size = 2.5) +
    ggplot2::scale_color_manual(values = cols, name = NULL,
                                guide = if (use_palette) "legend" else "none") +
    ggplot2::labs(x = "estimate (90% CI)", y = NULL,
                 title = .grin_title(title, "Parameter estimates")) +
    theme_grin(base_size)
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
#' @param palette Colour palette (one colour per panel). `NULL` (default)
#'   defers to `options(grin.palette)`. See [grin_plot_space()].
#' @param title Plot title. `NULL` (default) uses
#'   `"Construct probabilities (<model class>)"`.
#' @param base_size Base font size in points (default 12).
#' @return A ggplot object.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' out <- grin_infer(M)
#' grin_plot_constructs(out$result, out$constructs)
#' }
#' @export
grin_plot_constructs <- function(result, constructs, palette = NULL, title = NULL, base_size = 12) {
  df <- data.frame(
    panel = c("correlation structure", "correlation structure", "correlation structure",
             "separability", "separability"),
    construct = factor(c("PI", "RHO1", "free", "separable A", "separable B"),
                       levels = c("PI", "RHO1", "free", "separable A", "separable B")),
    prob = c(constructs$p_corr, constructs$p_sep_A, constructs$p_sep_B),
    evidence = c(rep(constructs$evidence_PI, 3), constructs$evidence_sep_A, constructs$evidence_sep_B)
  )
  df$label <- ifelse(df$evidence, "", "insufficient evidence")
  fills <- stats::setNames(.grin_group_colors(2, palette), c("correlation structure", "separability"))

  ggplot2::ggplot(df, ggplot2::aes(x = .data$construct, y = .data$prob, fill = .data$panel)) +
    ggplot2::geom_col(ggplot2::aes(alpha = .data$evidence), width = 0.6) +
    ggplot2::geom_text(ggplot2::aes(label = .data$label), vjust = -0.4, size = 3,
                       color = .grin_colors$mute) +
    ggplot2::scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.35), guide = "none") +
    ggplot2::scale_fill_manual(values = fills, guide = "none") +
    ggplot2::facet_wrap(~panel, scales = "free_x") +
    ggplot2::ylim(0, 1.08) +
    ggplot2::labs(x = NULL, y = "P(construct)",
                  title = .grin_title(title, sprintf("Construct probabilities (%s)", result$model_class))) +
    theme_grin(base_size)
}
