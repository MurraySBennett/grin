# style.R: the GRIN figure identity. Default is black-on-white -- publication
# safe, photocopy/greyscale safe, no colour vision assumptions -- with the
# house blue/rose (ported from src/viz/style.py) available as an opt-in via
# the `color` argument every grin_plot_*() takes, or globally via
# options(grin.color = TRUE). Either way: clean axes, only left+bottom
# spines, no gridlines.

#' @keywords internal
.grin_colors <- list(
  blue = "#5AA9E6", blue_deep = "#2E6CA4",
  red  = "#F2A5C0", red_deep  = "#C86A93",
  ink  = "#2B2B2E", mute = "#9AA0A6", paper = "#FFFFFF"
)

#' @keywords internal
.grin_palette <- unlist(.grin_colors[c("blue", "red", "blue_deep", "red_deep", "mute")],
                        use.names = FALSE)

#' @keywords internal
.grin_diverging <- c(.grin_colors$blue_deep, "#FFFFFF", .grin_colors$red_deep)

#' Should plots use colour by default?
#'
#' Controls every `grin_plot_*()` function's default for its `color` argument.
#' Set once with `options(grin.color = TRUE)` rather than passing `color = TRUE`
#' to every call; an explicit `color = ` argument always overrides this.
#' @keywords internal
.grin_use_color <- function(color = NULL) {
  if (!is.null(color)) return(isTRUE(color))
  isTRUE(getOption("grin.color", FALSE))
}

#' Resolve a set of n category colours: the house palette if colour is on,
#' the single ink colour repeated n times if not (so callers can use the same
#' `scale_*_manual()` machinery either way, and a bw plot never carries a
#' legend that only distinguishes "black" from "black").
#' @keywords internal
.grin_group_colors <- function(n, color = NULL) {
  if (.grin_use_color(color)) {
    if (n <= length(.grin_palette)) return(.grin_palette[seq_len(n)])
    return(grDevices::colorRampPalette(.grin_palette)(n))
  }
  rep(.grin_colors$ink, n)
}

#' GRIN's house ggplot2 theme
#'
#' Used internally by every `grin_plot_*()` function; exported so a
#' plot built with one can be further customised the normal ggplot2 way
#' (`grin_plot_space(x) + ggplot2::labs(title = "...")`).
#' @export
theme_grin <- function() {
  ink <- .grin_colors$ink
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(color = ink),
      axis.line.y = ggplot2::element_line(color = ink),
      axis.ticks = ggplot2::element_line(color = ink),
      axis.text = ggplot2::element_text(color = ink),
      axis.title = ggplot2::element_text(color = ink),
      plot.title = ggplot2::element_text(color = ink, face = "bold"),
      legend.position = "right",
      plot.background = ggplot2::element_rect(fill = .grin_colors$paper, color = NA),
      panel.background = ggplot2::element_rect(fill = .grin_colors$paper, color = NA)
    )
}
