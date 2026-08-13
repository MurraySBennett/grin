# plot-group.R: multi-participant plots. All take a list of grin_inference
# objects (as returned by looping grin_infer() over a sample), same input
# grin_tidy() expects.

#' Plot many participants' perceptual spaces
#'
#' @param results A list of `grin_inference` objects (see [grin_tidy()]).
#' @param ids Optional participant IDs; see [grin_tidy()].
#' @param facet If `TRUE` (default), one panel per participant. If `FALSE`,
#'   overlay every participant's stimulus means (ellipses omitted -- with many
#'   participants, overlaid ellipses are unreadable; use `facet = TRUE` to see
#'   individual uncertainty) with a group-mean ellipse per stimulus.
#' @param ci Confidence level for ellipses (facet mode) / the group-mean
#'   ellipse (overlay mode).
#' @return A ggplot object.
#' @export
grin_plot_space_group <- function(results, ids = NULL, facet = TRUE, ci = 0.90) {
  td <- grin_tidy(results, ids)
  stim <- c("A1B1", "A1B2", "A2B1", "A2B2")
  k <- stats::qnorm(0.5 + ci / 2)

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
      ggplot2::geom_path(data = ellipses, ggplot2::aes(.data$x, .data$y, color = .data$stimulus)) +
      ggplot2::geom_point(data = centers, ggplot2::aes(.data$zx, .data$zy, color = .data$stimulus),
                          size = 1.5) +
      ggplot2::scale_color_manual(values = .grin_stim_colors(), name = "stimulus") +
      ggplot2::coord_equal() +
      ggplot2::facet_wrap(~id) +
      ggplot2::labs(x = "zx", y = "zy", title = "Perceptual spaces by participant") +
      theme_grin()
  } else {
    means <- do.call(rbind, lapply(stim, function(s) {
      sub <- centers[centers$stimulus == s, ]
      data.frame(stimulus = s, zx = mean(sub$zx), zy = mean(sub$zy),
                rho = mean(sub$rho))
    }))
    mean_ellipses <- do.call(rbind, lapply(seq_len(nrow(means)), function(i) {
      e <- .grin_ellipse_pts(means$zx[i], means$zy[i], means$rho[i], k)
      e$stimulus <- means$stimulus[i]
      e
    }))
    ggplot2::ggplot() +
      ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
      ggplot2::geom_point(data = centers, ggplot2::aes(.data$zx, .data$zy, color = .data$stimulus),
                          alpha = 0.35, size = 1.5) +
      ggplot2::geom_path(data = mean_ellipses, ggplot2::aes(.data$x, .data$y, color = .data$stimulus),
                         linewidth = 1) +
      ggplot2::geom_point(data = means, ggplot2::aes(.data$zx, .data$zy, color = .data$stimulus),
                          size = 3, shape = 18) +
      ggplot2::scale_color_manual(values = .grin_stim_colors(), name = "stimulus") +
      ggplot2::coord_equal() +
      ggplot2::labs(x = "zx", y = "zy", title = "Perceptual space, group overlay",
                    subtitle = sprintf("faint points = individuals; diamonds + %.0f%% ellipse = group mean",
                                       100 * ci)) +
      theme_grin()
  }
}

#' Plot the distribution of each parameter across many participants
#'
#' @inheritParams grin_plot_space_group
#' @return A ggplot object.
#' @export
grin_plot_params_group <- function(results, ids = NULL) {
  td <- grin_tidy(results, ids)
  long <- .grin_long_params(td)
  long$param <- factor(long$param, levels = c(paste0("zx_", 0:3), paste0("zy_", 0:3),
                                              paste0("rho_", 0:3)))
  ggplot2::ggplot(long, ggplot2::aes(x = .data$param, y = .data$estimate, fill = .data$group)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = .grin_colors$mute) +
    ggplot2::geom_boxplot(outlier.alpha = 0.4) +
    ggplot2::scale_fill_manual(values = stats::setNames(.grin_palette[1:3], c("zx", "zy", "rho")),
                               guide = "none") +
    ggplot2::facet_wrap(~group, scales = "free", nrow = 1) +
    ggplot2::labs(x = NULL, y = "estimate", title = "Parameter distributions across participants",
                  subtitle = sprintf("n = %d participants", nrow(td))) +
    theme_grin() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}

#' Plot the distribution of GRT model classes across many participants
#'
#' @inheritParams grin_plot_space_group
#' @return A ggplot object.
#' @export
grin_plot_model_classes <- function(results, ids = NULL) {
  td <- grin_tidy(results, ids)
  counts <- as.data.frame(table(model_class = td$model_class), stringsAsFactors = FALSE)
  counts <- counts[order(-counts$Freq), ]
  counts$model_class <- factor(counts$model_class, levels = counts$model_class)

  ggplot2::ggplot(counts, ggplot2::aes(x = .data$model_class, y = .data$Freq)) +
    ggplot2::geom_col(fill = .grin_colors$blue) +
    ggplot2::geom_text(ggplot2::aes(label = .data$Freq), vjust = -0.4, color = .grin_colors$ink) +
    ggplot2::labs(x = NULL, y = "participants",
                  title = "Inferred model class",
                  subtitle = sprintf("n = %d participants", nrow(td))) +
    theme_grin() +
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
grin_plot_precision_group <- function(results, ids = NULL) {
  td <- grin_tidy(results, ids)
  long <- .grin_long_params(td)

  ggplot2::ggplot(long, ggplot2::aes(x = .data$group, y = .data$sd, fill = .data$group)) +
    ggplot2::geom_boxplot(outlier.alpha = 0.4) +
    ggplot2::scale_fill_manual(values = stats::setNames(.grin_palette[1:3], c("zx", "zy", "rho")),
                               guide = "none") +
    ggplot2::labs(x = NULL, y = "posterior SD", title = "Precision across participants",
                  subtitle = sprintf("n = %d participants", nrow(td))) +
    theme_grin()
}
