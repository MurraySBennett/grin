# bias.R: two different things people mean by "response bias," kept
# deliberately distinct rather than collapsed into one function.
#
# grin_empirical_bias() is a description of the DATA: how often a level-2
# response was given, relative to chance, straight from the raw counts. It
# needs no model fit and works even on a matrix GRIN can't otherwise fit.
#
# grin_response_bias() is the SDT-native quantity: GRT is a multidimensional
# extension of signal detection theory, and its decision bound is fixed at 0
# by convention (see the ordering contract in ?grin_infer) -- an unbiased
# observer's two levels on a dimension are mirror images about that bound, so
# their four identified z-scores average to exactly zero. A nonzero average
# is a shifted decision criterion in the classical SDT sense, read directly
# off parameters grin_infer() already estimated, no extra computation beyond
# an average. Confirmed numerically before shipping: a deliberately
# asymmetric z-score input and its raw-data response rate move together and
# agree in sign with grin_empirical_bias() on the same case.

#' Empirical response bias from a raw confusion matrix
#'
#' The signed tendency to report level 2 of a dimension more or less often
#' than level 1, averaged across the four stimuli: 0 is unbiased, positive
#' means the observer favours the "2" response on that dimension more than a
#' fair coin would, negative means they favour "1". A description of the
#' data, independent of any model fit -- it needs no trained network and
#' works even on a matrix GRIN can't otherwise fit. See [grin_response_bias()]
#' for the model-based decision-criterion counterpart.
#'
#' @param counts A canonical-order 4x4 matrix (see [grin_to_confusion()]), or
#'   a length-16 vector read row-major.
#' @param trials Optional per-stimulus trial totals (length 4); defaults to
#'   row sums.
#' @return A list: `x_bias`, `y_bias` (each in \eqn{[-0.5, 0.5]}), and
#'   `p_resp2`, a 4x2 matrix giving P(respond level 2) on each dimension for
#'   each of the four stimuli (so a systematic bias can be told apart from one
#'   driven by a single stimulus).
#' @examples
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_empirical_bias(M)
#' @export
grin_empirical_bias <- function(counts, trials = NULL) {
  cm <- .to_counts_matrix(counts)
  if (is.null(trials)) trials <- rowSums(cm)
  trials <- as.numeric(trials)[1:4]
  props <- cm / trials
  p_x2 <- props[, 3] + props[, 4]   # respond "A2" (canonical cols a2b1, a2b2)
  p_y2 <- props[, 2] + props[, 4]   # respond "B2" (canonical cols a1b2, a2b2)
  list(x_bias = mean(p_x2) - 0.5, y_bias = mean(p_y2) - 0.5,
      p_resp2 = cbind(x = p_x2, y = p_y2))
}

#' Parametric (decision-criterion) response bias from a fitted result
#'
#' The mean of a dimension's four identified z-scores: exactly zero when the
#' two levels are mirror images about the fixed decision bound (unbiased),
#' and equal to the criterion's effective offset from that symmetric point
#' otherwise -- the same sign convention as [grin_empirical_bias()] (positive
#' favours level 2), but read directly off `result$params` rather than raw
#' response counts. Unlike the empirical version, this needs a [grin_infer()]
#' fit, and carries the fit's own uncertainty forward.
#'
#' @param result A `grin_result` (e.g. `grin_infer(M)$result`).
#' @return A list: `x_bias`, `y_bias`, and `x_bias_se`/`y_bias_se`. The SEs
#'   are a marginal-independence approximation from `result$std` (the
#'   posterior's full covariance across parameters isn't exposed by
#'   [grin_infer()], only per-parameter marginal SDs, so this likely under-
#'   or overstates the true joint uncertainty somewhat; treat as
#'   approximate, not exact).
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_response_bias(grin_infer(M)$result)
#' }
#' @export
grin_response_bias <- function(result) {
  stopifnot(inherits(result, "grin_result"))
  p <- stats::setNames(as.numeric(result$params), result$names)
  s <- stats::setNames(as.numeric(result$std), result$names)
  x_nm <- paste0("zx_", 0:3); y_nm <- paste0("zy_", 0:3)
  list(x_bias = mean(p[x_nm]), y_bias = mean(p[y_nm]),
      x_bias_se = sqrt(sum(s[x_nm]^2)) / 4, y_bias_se = sqrt(sum(s[y_nm]^2)) / 4)
}

#' Plot decision-criterion response bias for one participant
#'
#' A two-bar summary of [grin_response_bias()], with error bars carrying its
#' (approximate) uncertainty forward. For the model-free alternative, see
#' [grin_plot_empirical_bias()].
#'
#' @param result A `grin_result` (e.g. `grin_infer(M)$result`).
#' @param ci Confidence level for the error bars (default 0.90).
#' @param palette Colour palette. `NULL` (default) defers to
#'   `options(grin.palette)`. See [grin_plot_space()].
#' @param title Plot title. `NULL` (default) uses `"Response bias"`.
#' @param dim_labels Length-2 character vector naming the two dimensions on
#'   the x-axis. Defaults to `c("A", "B")`.
#' @param base_size Base font size in points (default 12).
#' @return A ggplot object.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_plot_bias(grin_infer(M)$result)
#' }
#' @export
grin_plot_bias <- function(result, ci = 0.90, palette = NULL, title = NULL,
                           dim_labels = c("A", "B"), base_size = 12) {
  stopifnot(length(dim_labels) == 2)
  b <- grin_response_bias(result)
  k <- stats::qnorm(0.5 + ci / 2)
  df <- data.frame(dimension = factor(dim_labels, levels = dim_labels),
                   bias = c(b$x_bias, b$y_bias),
                   se = c(b$x_bias_se, b$y_bias_se))
  col <- .grin_group_colors(1, palette)

  ggplot2::ggplot(df, ggplot2::aes(x = .data$dimension, y = .data$bias)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_col(fill = col, width = 0.5) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = .data$bias - k * .data$se,
                                        ymax = .data$bias + k * .data$se), width = 0.15) +
    ggplot2::labs(x = NULL, y = "decision-criterion bias (mean z-score)",
                 title = .grin_title(title, "Response bias"),
                 subtitle = sprintf(
                   "%.0f%% CI; 0 = unbiased; positive favours level 2, negative favours level 1",
                   100 * ci)) +
    theme_grin(base_size)
}

#' Plot empirical response bias for one participant
#'
#' A two-bar summary of [grin_empirical_bias()]: how far each dimension's
#' "respond level 2" rate sits from the unbiased 0.5, averaged across the
#' four stimuli. Works directly from a confusion matrix -- no [grin_infer()]
#' call needed. For the model-based decision-criterion alternative, see
#' [grin_plot_bias()].
#'
#' @param counts,trials As in [grin_empirical_bias()].
#' @param palette Colour palette. `NULL` (default) defers to
#'   `options(grin.palette)`. See [grin_plot_space()].
#' @param title Plot title. `NULL` (default) uses `"Empirical response bias"`.
#' @param dim_labels Length-2 character vector naming the two dimensions on
#'   the x-axis. Defaults to `c("A", "B")`.
#' @param base_size Base font size in points (default 12).
#' @return A ggplot object.
#' @examples
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_plot_empirical_bias(M)
#' @export
grin_plot_empirical_bias <- function(counts, trials = NULL, palette = NULL, title = NULL,
                                     dim_labels = c("A", "B"), base_size = 12) {
  stopifnot(length(dim_labels) == 2)
  b <- grin_empirical_bias(counts, trials)
  df <- data.frame(dimension = factor(dim_labels, levels = dim_labels),
                   bias = c(b$x_bias, b$y_bias))
  col <- .grin_group_colors(1, palette)

  ggplot2::ggplot(df, ggplot2::aes(x = .data$dimension, y = .data$bias)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_col(fill = col, width = 0.5) +
    ggplot2::ylim(-0.5, 0.5) +
    ggplot2::labs(x = NULL, y = "response bias  (P(respond level 2) - 0.5)",
                 title = .grin_title(title, "Empirical response bias"),
                 subtitle = "0 = unbiased; positive favours level 2, negative favours level 1") +
    theme_grin(base_size)
}

#' Plot decision-criterion response bias across many participants
#'
#' Group-level companion to [grin_plot_bias()]: one boxplot per dimension of
#' [grin_response_bias()], computed per participant from their fitted result.
#' For the model-free alternative, see [grin_plot_empirical_bias_group()].
#'
#' @inheritParams grin_plot_space_group
#' @return A ggplot object.
#' @export
grin_plot_bias_group <- function(results, ids = NULL, palette = NULL, title = NULL, base_size = 12) {
  td <- grin_tidy(results, ids)
  df <- data.frame(dimension = rep(c("A", "B"), each = nrow(td)),
                   bias = c(td$x_bias, td$y_bias))
  col <- .grin_group_colors(1, palette)

  ggplot2::ggplot(df, ggplot2::aes(x = .data$dimension, y = .data$bias)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_boxplot(fill = col, alpha = 0.5, outlier.alpha = 0.4, width = 0.5) +
    ggplot2::labs(x = NULL, y = "decision-criterion bias (mean z-score)",
                 title = .grin_title(title, "Response bias across participants"),
                 subtitle = sprintf("n = %d participants", nrow(td))) +
    theme_grin(base_size)
}

#' Plot empirical response bias across many participants
#'
#' Group-level companion to [grin_plot_empirical_bias()]: one boxplot per
#' dimension of [grin_empirical_bias()] computed on each participant's own
#' confusion matrix. Works directly from confusion matrices -- no
#' [grin_infer()] call needed.
#'
#' @param counts_list A list of confusion matrices, one per participant (each
#'   as accepted by [grin_empirical_bias()]).
#' @param trials_list Optional list of per-stimulus trial totals, same length
#'   as `counts_list`; defaults to row sums for each.
#' @param palette,title,base_size As in [grin_plot_space_group()].
#' @return A ggplot object.
#' @export
grin_plot_empirical_bias_group <- function(counts_list, trials_list = NULL, palette = NULL,
                                           title = NULL, base_size = 12) {
  if (is.null(trials_list)) trials_list <- vector("list", length(counts_list))
  stopifnot(length(trials_list) == length(counts_list))
  b <- Map(grin_empirical_bias, counts_list, trials_list)
  df <- data.frame(dimension = rep(c("A", "B"), each = length(b)),
                   bias = c(vapply(b, `[[`, numeric(1), "x_bias"),
                           vapply(b, `[[`, numeric(1), "y_bias")))
  col <- .grin_group_colors(1, palette)

  ggplot2::ggplot(df, ggplot2::aes(x = .data$dimension, y = .data$bias)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_boxplot(fill = col, alpha = 0.5, outlier.alpha = 0.4, width = 0.5) +
    ggplot2::ylim(-0.5, 0.5) +
    ggplot2::labs(x = NULL, y = "response bias  (P(respond level 2) - 0.5)",
                 title = .grin_title(title, "Empirical response bias across participants"),
                 subtitle = sprintf("n = %d participants", length(counts_list))) +
    theme_grin(base_size)
}
