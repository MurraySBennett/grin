# plot-diagnostics.R: predicted-vs-observed reconstruction reporting for one
# participant. Unlike grin_plot_space()/_params()/_constructs(), these need the
# ORIGINAL confusion matrix as well as the fitted result, because they compare
# what was observed against what the fitted parameters predict -- grin_infer()'s
# return value alone doesn't carry the input matrix back out.
#
# Deliberately NOT called "goodness-of-fit": the identified 12-parameter model is
# saturated (see the manuscript's Introduction/identifiability-frontier study), so
# a single confusion matrix's response proportions cannot, in principle, be used
# to test whether Gaussian perceptual effects or decisional separability hold --
# essentially any proportion table has SOME fitting parameter vector. What this
# view actually shows is whether GRIN's OWN fitted parameters reconstruct the
# matrix, which is informative in one direction only: a poor reconstruction is a
# real signal (network approximation error, or a matrix outside the trained
# envelope -- worth a second look, possibly a fresh maximum-likelihood fit), but
# a good reconstruction does not validate the underlying GRT assumptions, because
# the saturated model was essentially guaranteed to reconstruct it regardless.

#' Predicted-vs-observed reconstruction and marginal-distribution diagnostics for one participant
#'
#' Two independent views of how well GRIN's own fitted parameters reconstruct the
#' data actually observed, either or both of which can be switched off. Neither is
#' a goodness-of-fit test of the GRT model itself -- see the note above the source
#' of this function for why not.
#'
#' \itemize{
#'   \item \strong{Predicted vs. observed}: the forward model's predicted
#'     response probability for each of the 16 stimulus/response cells,
#'     plotted against the cell's observed proportion. Points near the
#'     diagonal indicate a good reconstruction; systematic departure for one
#'     stimulus (told apart by point shape, not colour) says where the fit is
#'     struggling and is worth a second look -- but points near the diagonal
#'     are not themselves evidence that GRT describes this participant, only
#'     that GRIN's fit reconstructs the matrix it was given.
#'   \item \strong{Marginal distributions}: the predicted `Normal(mean, 1)`
#'     density on each dimension for each of the four stimuli (the model's own
#'     unit-variance convention), which is what [grin_plot_space()]'s
#'     `show_marginals = TRUE` also draws alongside the space plot itself --
#'     this function is the same marginals without the perceptual-space panel,
#'     paired instead with the reconstruction check above.
#' }
#'
#' @param result A `grin_result` (e.g. `grin_infer(M)$result`).
#' @param counts The same canonical-order confusion matrix passed to
#'   [grin_infer()] for this participant (a 4x4 matrix or length-16 vector).
#' @param trials Optional per-stimulus trial totals; defaults to row sums of
#'   `counts`.
#' @param show_predicted_observed,show_marginals Toggle each panel (default
#'   `TRUE` for both). At least one must stay `TRUE`.
#' @param palette Colour palette. `NULL` (default) defers to
#'   `options(grin.palette)`. See [grin_plot_space()].
#' @param title Overall title when both panels are shown. `NULL` (default)
#'   adds no overall title (each panel keeps its own).
#' @param base_size Base font size in points (default 12).
#' @return A single ggplot object if only one panel is requested; otherwise a
#'   `patchwork` object combining both (requires the \pkg{patchwork} package
#'   -- if it isn't installed, a named list of the individual ggplot objects
#'   is returned instead, with a warning).
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' out <- grin_infer(M)
#' grin_plot_diagnostics(out$result, M)
#' }
#' @export
grin_plot_diagnostics <- function(result, counts, trials = NULL,
                                  show_predicted_observed = TRUE, show_marginals = TRUE,
                                  palette = NULL, title = NULL, base_size = 12) {
  stopifnot(inherits(result, "grin_result"))
  if (!show_predicted_observed && !show_marginals) {
    stop("nothing to plot: set show_predicted_observed and/or show_marginals to TRUE", call. = FALSE)
  }
  cm <- .to_counts_matrix(counts)
  if (is.null(trials)) trials <- rowSums(cm)
  trials <- as.numeric(trials)[1:4]
  observed <- cm / trials

  p <- stats::setNames(as.numeric(result$params), result$names)
  zx <- p[paste0("zx_", 0:3)]; zy <- p[paste0("zy_", 0:3)]; rho <- p[paste0("rho_", 0:3)]
  col <- .grin_group_colors(1, palette)

  panels <- list()

  if (show_predicted_observed) {
    predicted <- .grin_forward_probabilities(zx, zy, rho)
    df <- data.frame(observed = as.vector(t(observed)), predicted = as.vector(t(predicted)),
                     stimulus = rep(CANON_STIM, each = 4))
    panels$predicted_observed <- ggplot2::ggplot(df,
        ggplot2::aes(.data$observed, .data$predicted, shape = .data$stimulus)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = .grin_colors$mute) +
      ggplot2::geom_point(color = col, size = 2.5) +
      ggplot2::scale_shape_manual(values = c(16, 17, 15, 18), name = "stimulus") +
      ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
      ggplot2::labs(x = "observed proportion", y = "predicted probability",
                   title = "Predicted vs. observed") +
      theme_grin(base_size)
  }

  if (show_marginals) {
    x_rng <- c(min(zx) - 3, max(zx) + 3); y_rng <- c(min(zy) - 3, max(zy) + 3)
    panels$marginal_x <- .grin_marginal_strip(zx, x_rng, "x", col, base_size) +
      ggplot2::labs(title = "dimension A marginals", x = "zx")
    panels$marginal_y <- .grin_marginal_strip(zy, y_rng, "x", col, base_size) +
      ggplot2::labs(title = "dimension B marginals", x = "zy")
  }

  if (length(panels) == 1) {
    out <- panels[[1]]
    if (!is.null(title)) out <- out + ggplot2::labs(title = title)
    return(out)
  }
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    warning("multiple panels requested but the 'patchwork' package is not installed; ",
           "returning a named list of ggplot objects instead of a combined figure",
           call. = FALSE)
    return(panels)
  }
  out <- patchwork::wrap_plots(panels, ncol = 2)
  if (!is.null(title)) out <- out + patchwork::plot_annotation(title = title)
  out
}
