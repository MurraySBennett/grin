# bias.R: response bias, computed directly from a confusion matrix -- no model
# fit required. GRT's separability/independence machinery lives entirely in
# grin_infer()'s 12 identified parameters, but the raw tendency to over- or
# under-report one level of a dimension (independent of how well the observer
# discriminates it) is a property of the data alone, and worth a name.

#' Response bias from a raw confusion matrix
#'
#' The signed tendency to report level 2 of a dimension more or less often
#' than level 1, averaged across the four stimuli: 0 is unbiased, positive
#' means the observer favours the "2" response on that dimension more than a
#' fair coin would, negative means they favour "1". This is a description of
#' the data, independent of [grin_infer()]'s model fit -- it needs no trained
#' network and works even on a matrix GRIN can't otherwise fit.
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
#' grin_response_bias(M)
#' @export
grin_response_bias <- function(counts, trials = NULL) {
  cm <- .to_counts_matrix(counts)
  if (is.null(trials)) trials <- rowSums(cm)
  trials <- as.numeric(trials)[1:4]
  props <- cm / trials
  p_x2 <- props[, 3] + props[, 4]   # respond "A2" (canonical cols a2b1, a2b2)
  p_y2 <- props[, 2] + props[, 4]   # respond "B2" (canonical cols a1b2, a2b2)
  list(x_bias = mean(p_x2) - 0.5, y_bias = mean(p_y2) - 0.5,
      p_resp2 = cbind(x = p_x2, y = p_y2))
}

#' Plot response bias for one participant
#'
#' A two-bar summary of [grin_response_bias()]: how far each dimension's
#' "respond level 2" rate sits from the unbiased 0.5, averaged across the four
#' stimuli. Works directly from a confusion matrix -- no [grin_infer()] call
#' needed.
#'
#' @param counts,trials As in [grin_response_bias()].
#' @param palette Colour palette. `NULL` (default) defers to
#'   `options(grin.palette)`. See [grin_plot_space()].
#' @param title Plot title. `NULL` (default) uses `"Response bias"`.
#' @param dim_labels Length-2 character vector naming the two dimensions on
#'   the x-axis. Defaults to `c("A", "B")`.
#' @param base_size Base font size in points (default 12).
#' @return A ggplot object.
#' @examples
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' grin_plot_bias(M)
#' @export
grin_plot_bias <- function(counts, trials = NULL, palette = NULL, title = NULL,
                           dim_labels = c("A", "B"), base_size = 12) {
  stopifnot(length(dim_labels) == 2)
  b <- grin_response_bias(counts, trials)
  df <- data.frame(dimension = factor(dim_labels, levels = dim_labels),
                   bias = c(b$x_bias, b$y_bias))
  col <- .grin_group_colors(1, palette)

  ggplot2::ggplot(df, ggplot2::aes(x = .data$dimension, y = .data$bias)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_col(fill = col, width = 0.5) +
    ggplot2::ylim(-0.5, 0.5) +
    ggplot2::labs(x = NULL, y = "response bias  (P(respond level 2) - 0.5)",
                 title = .grin_title(title, "Response bias"),
                 subtitle = "0 = unbiased; positive favours level 2, negative favours level 1") +
    theme_grin(base_size)
}
