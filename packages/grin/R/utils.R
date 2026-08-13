.new_warn_collector <- function() {
  e <- new.env(parent = emptyenv())
  e$msgs <- character(0)
  e
}

.warn <- function(collector, msg) {
  collector$msgs <- c(collector$msgs, msg)
  invisible(NULL)
}

#' @keywords internal
.grin_ellipse_pts <- function(zx, zy, rho, k, n = 100) {
  theta <- seq(0, 2 * pi, length.out = n)
  L <- matrix(c(1, rho, 0, sqrt(max(1 - rho^2, 0))), nrow = 2)
  xy <- L %*% rbind(cos(theta), sin(theta)) * k
  data.frame(x = zx + xy[1, ], y = zy + xy[2, ])
}
