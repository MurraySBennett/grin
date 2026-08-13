# style.R: the GRIN figure identity, ported from src/viz/style.py so package
# plots read as the same family as the paper's own figures. Restrained
# blue/rose on clean axes: only left+bottom spines, no gridlines.

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

#' @keywords internal
.grin_stim_colors <- function() {
  stats::setNames(.grin_palette[1:4], c("A1B1", "A1B2", "A2B1", "A2B2"))
}
