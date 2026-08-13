# plot-group.R: multi-participant plots. All take a list of grin_inference
# objects (as returned by looping grin_infer() over a sample), same input
# grin_tidy() expects. Black-on-white by default; pass palette = "name" (see
# grin_palette_names()) or your own vector of hex colours, or
# options(grin.palette = "name") for the session, to use colour instead.

#' Plot many participants' perceptual spaces
#'
#' @param results A list of `grin_inference` objects (see [grin_tidy()]).
#' @param ids Optional participant IDs; see [grin_tidy()].
#' @param facet If `TRUE` (default), one panel per participant, each stimulus
#'   labelled directly rather than coloured (see [grin_plot_space()]) -- this
#'   is the individual-level analysis GRT actually licenses: every
#'   participant's own estimated space, side by side.
#'
#'   If `FALSE`, overlay every participant's stimulus means with a single
#'   across-participant mean ellipse per stimulus. **This is an exploratory
#'   inspection view only, not a reporting figure**: GRT's perceptual space is
#'   defined per observer, and there is no sense in which a "grand mean"
#'   ellipse over several independently-fitted spaces is itself a fitted GRT
#'   model, or a quantity with the calibrated-uncertainty guarantees
#'   [grin_infer()] gives an individual estimate. Use it to eyeball whether a
#'   sample looks roughly homogeneous or visibly splits into subgroups before
#'   deciding how to report it properly (e.g. per participant, or via
#'   [grin_plot_params_group()] / [grin_plot_model_classes()]) -- not as the
#'   figure itself. A message is printed each time this mode is used, as a
#'   standing reminder rather than a one-off warning you might miss.
#' @param ci Confidence level for ellipses (facet mode) / the mean ellipse
#'   (overlay mode).
#' @param palette Colour palette. `NULL` (default) defers to
#'   `options(grin.palette)`. See [grin_plot_space()].
#' @param title Plot title. `NULL` (default) uses a built-in title.
#' @param base_size Base font size in points (default 12).
#' @return A ggplot object.
#' @export
grin_plot_space_group <- function(results, ids = NULL, facet = TRUE, ci = 0.90,
                                  palette = NULL, title = NULL, base_size = 12) {
  td <- grin_tidy(results, ids)
  stim <- CANON_STIM
  k <- stats::qnorm(0.5 + ci / 2)
  col <- .grin_group_colors(1, palette)

  centers <- do.call(rbind, lapply(1:4, function(i) {
    data.frame(id = td$id, stimulus = stim[i],
              zx = td[[paste0("zx_", i - 1)]], zy = td[[paste0("zy_", i - 1)]],
              rho = td[[paste0("rho_", i - 1)]])
  }))

  if (facet) {
    ellipses <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i) {
      e <- .grin_ellipse_pts(centers$zx[i], centers$zy[i], centers$rho[i], k, n = 60)
      e$id <- centers$id[i]; e$stimulus <- centers$stimulus[i]
      e
    }))
    ggplot2::ggplot() +
      ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
      ggplot2::geom_path(data = ellipses,
                         ggplot2::aes(.data$x, .data$y, group = .data$stimulus), color = col) +
      ggplot2::geom_point(data = centers, ggplot2::aes(.data$zx, .data$zy), color = col, size = 1.5) +
      ggplot2::geom_text(data = centers, ggplot2::aes(.data$zx, .data$zy, label = .data$stimulus),
                         color = col, size = 2, vjust = -0.9) +
      ggplot2::coord_equal() +
      ggplot2::facet_wrap(~id) +
      ggplot2::labs(x = "zx", y = "zy",
                   title = .grin_title(title, "Perceptual spaces by participant")) +
      theme_grin(base_size)
  } else {
    message("grin_plot_space_group(facet = FALSE): exploratory inspection view only -- ",
           "the overlaid mean ellipse is not a fitted GRT model or a reporting figure. ",
           "See ?grin_plot_space_group.")
    means <- do.call(rbind, lapply(stim, function(s) {
      sub <- centers[centers$stimulus == s, ]
      data.frame(stimulus = s, zx = mean(sub$zx), zy = mean(sub$zy), rho = mean(sub$rho))
    }))
    mean_ellipses <- do.call(rbind, lapply(seq_len(nrow(means)), function(i) {
      e <- .grin_ellipse_pts(means$zx[i], means$zy[i], means$rho[i], k)
      e$stimulus <- means$stimulus[i]
      e
    }))
    ggplot2::ggplot() +
      ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
      ggplot2::geom_point(data = centers, ggplot2::aes(.data$zx, .data$zy),
                          color = col, alpha = 0.3, size = 1.5) +
      ggplot2::geom_path(data = mean_ellipses,
                         ggplot2::aes(.data$x, .data$y, group = .data$stimulus),
                         color = col, linewidth = 1) +
      ggplot2::geom_point(data = means, ggplot2::aes(.data$zx, .data$zy),
                          color = col, size = 3, shape = 18) +
      ggplot2::geom_text(data = means, ggplot2::aes(.data$zx, .data$zy, label = .data$stimulus),
                         color = col, size = 3.2, vjust = -1.5) +
      ggplot2::coord_equal() +
      ggplot2::labs(x = "zx", y = "zy",
                   title = .grin_title(title, "Perceptual space, group overlay (exploratory only)"),
                   subtitle = sprintf(
                     "faint points = individuals; diamonds + %.0f%% ellipse = across-participant mean -- inspection view, not a fitted model",
                     100 * ci)) +
      theme_grin(base_size)
  }
}

#' Plot the distribution of each parameter across many participants
#'
#' @inheritParams grin_plot_space_group
#' @return A ggplot object.
#' @export
grin_plot_params_group <- function(results, ids = NULL, palette = NULL, title = NULL, base_size = 12) {
  td <- grin_tidy(results, ids)
  long <- .grin_long_params(td)
  long$param <- factor(long$param, levels = c(paste0("zx_", 0:3), paste0("zy_", 0:3),
                                              paste0("rho_", 0:3)))
  fills <- stats::setNames(.grin_group_colors(3, palette), c("zx", "zy", "rho"))

  ggplot2::ggplot(long, ggplot2::aes(x = .data$param, y = .data$estimate, fill = .data$group)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_boxplot(outlier.alpha = 0.4) +
    ggplot2::scale_fill_manual(values = fills, guide = "none") +
    ggplot2::facet_wrap(~group, scales = "free", nrow = 1) +
    ggplot2::labs(x = NULL, y = "estimate",
                 title = .grin_title(title, "Parameter distributions across participants"),
                 subtitle = sprintf("n = %d participants", nrow(td))) +
    theme_grin(base_size) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}

#' Plot the distribution of GRT model classes across many participants
#'
#' @inheritParams grin_plot_space_group
#' @return A ggplot object.
#' @export
grin_plot_model_classes <- function(results, ids = NULL, palette = NULL, title = NULL, base_size = 12) {
  td <- grin_tidy(results, ids)
  counts <- as.data.frame(table(model_class = td$model_class), stringsAsFactors = FALSE)
  counts <- counts[order(-counts$Freq), ]
  counts$model_class <- factor(counts$model_class, levels = counts$model_class)
  col <- .grin_group_colors(1, palette)

  ggplot2::ggplot(counts, ggplot2::aes(x = .data$model_class, y = .data$Freq)) +
    ggplot2::geom_col(fill = col) +
    ggplot2::geom_text(ggplot2::aes(label = .data$Freq), vjust = -0.4, color = .grin_colors$ink) +
    ggplot2::labs(x = NULL, y = "participants",
                 title = .grin_title(title, "Inferred model class"),
                 subtitle = sprintf("n = %d participants", nrow(td))) +
    theme_grin(base_size) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))
}

#' Plot posterior precision (SD) across many participants
#'
#' Distribution of each parameter's posterior SD across the sample -- how
#' precisely each parameter is being pinned down given the data collected.
#' Useful alongside [grin_target_precision()] for planning adaptive-stopping
#' thresholds from pilot data.
#'
#' @inheritParams grin_plot_space_group
#' @return A ggplot object.
#' @export
grin_plot_precision_group <- function(results, ids = NULL, palette = NULL, title = NULL, base_size = 12) {
  td <- grin_tidy(results, ids)
  long <- .grin_long_params(td)
  fills <- stats::setNames(.grin_group_colors(3, palette), c("zx", "zy", "rho"))

  ggplot2::ggplot(long, ggplot2::aes(x = .data$group, y = .data$sd, fill = .data$group)) +
    ggplot2::geom_boxplot(outlier.alpha = 0.4) +
    ggplot2::scale_fill_manual(values = fills, guide = "none") +
    ggplot2::labs(x = NULL, y = "posterior SD",
                 title = .grin_title(title, "Precision across participants"),
                 subtitle = sprintf("n = %d participants", nrow(td))) +
    theme_grin(base_size)
}
