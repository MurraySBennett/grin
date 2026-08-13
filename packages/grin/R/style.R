# style.R: the GRIN figure identity. Default is black-on-white -- publication
# safe, photocopy/greyscale safe, no colour vision assumptions -- with a small
# set of named colour palettes available as an opt-in via the `palette` argument
# every grin_plot_*() takes (or globally via options(grin.palette = "...")), plus
# a caller can always supply their own vector of hex colours instead of a preset
# name. Either way: clean axes, only left+bottom spines, no gridlines.

#' @keywords internal
.grin_colors <- list(
  ink = "#2B2B2E", mute = "#9AA0A6", paper = "#FFFFFF"
)

#' Named colour palettes available to every \code{grin_plot_*()}
#'
#' \code{"mono"} (the default) is black-on-white and needs no entry here. The
#' named palettes below are opt-in via \code{palette = "name"}; a caller can
#' also pass their own character vector of hex colours instead of a name.
#' \code{"contrast"} is the colour-vision-deficiency-safe categorical palette
#' of Okabe & Ito (2008), useful when a figure needs several clearly distinct
#' categories rather than a single accent colour.
#' @keywords internal
.grin_palettes <- list(
  contrast = c("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"),
  dusk     = c("#0B3954", "#12678A", "#1C9FC9", "#6FD6E8"),
  ember    = c("#4A0E0E", "#9E2B25", "#D9622B", "#F2A65A")
)

#' List the built-in palette names
#'
#' \code{"mono"} (default, black-on-white) plus the named palettes in
#' \code{\link{.grin_palettes}}. Pass any of these to a \code{palette}
#' argument, or pass your own character vector of hex colours instead.
#' @return A character vector of palette names.
#' @export
grin_palette_names <- function() c("mono", names(.grin_palettes))

#' Resolve a `palette` argument to a vector of hex colours
#'
#' `palette` is one of: `NULL` (defer to `options(grin.palette)`, itself
#' default `"mono"`), the name of a built-in palette (see
#' [grin_palette_names()]), or a character vector of hex colours supplied
#' directly by the caller.
#' @keywords internal
.grin_resolve_palette <- function(palette = NULL) {
  if (is.null(palette)) palette <- getOption("grin.palette", "mono")
  if (length(palette) > 1) return(palette)                    # caller-supplied vector
  if (grepl("^#[0-9A-Fa-f]{3,8}$", palette[1])) return(palette[1])  # caller-supplied single hex
  if (identical(palette, "mono")) return(.grin_colors$ink)
  if (palette %in% names(.grin_palettes)) return(.grin_palettes[[palette]])
  stop(sprintf("unknown palette '%s'; use one of {%s}, or pass your own vector of hex colours",
               palette, paste(grin_palette_names(), collapse = ", ")), call. = FALSE)
}

#' Resolve a set of n category colours
#'
#' The requested palette repeated/interpolated to length `n` (mono repeats a
#' single colour, so callers can use the same `scale_*_manual()` machinery
#' either way and a mono plot never carries a legend that only distinguishes
#' "black" from "black").
#' @keywords internal
.grin_group_colors <- function(n, palette = NULL) {
  cols <- .grin_resolve_palette(palette)
  if (length(cols) == 1) return(rep(cols, n))
  if (n <= length(cols)) return(cols[seq_len(n)])
  grDevices::colorRampPalette(cols)(n)
}

#' GRIN's house ggplot2 theme
#'
#' Used internally by every `grin_plot_*()` function; exported so a
#' plot built with one can be further customised the normal ggplot2 way
#' (`grin_plot_space(x) + ggplot2::labs(title = "...")`).
#'
#' @param base_size Base font size in points, passed to
#'   `ggplot2::theme_minimal()`. Every `grin_plot_*()` function also takes a
#'   `base_size` argument that forwards here.
#' @export
theme_grin <- function(base_size = 12) {
  ink <- .grin_colors$ink
  ggplot2::theme_minimal(base_size = base_size) +
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
