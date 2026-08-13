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

#' @keywords internal
.grin_title <- function(title, default) if (is.null(title)) default else title

# --------------------------------------------------------------------------- #
# Forward model: identified GRT parameters -> predicted response probabilities.
# Used only for reporting/diagnostics (e.g. grin_plot_diagnostics()'s predicted-
# vs-observed panel) -- inference itself never runs this, the trained network
# does. Ported from the same Sheppard r-integration used to build the training
# data (see the "Software description" section of the manuscript and
# src/grt_model.py in the main GRIN repo), so a diagnostic plot's "predicted"
# values are computed the identical way the network was taught to invert.
# --------------------------------------------------------------------------- #

.grin_gl_cache <- new.env(parent = emptyenv())

#' Gauss-Legendre quadrature nodes/weights on the interval -1 to 1, via the
#' Golub-Welsch eigendecomposition of the Jacobi matrix. Cached per session.
#' @keywords internal
.grin_gauss_legendre <- function(n = 48) {
  key <- as.character(n)
  cached <- .grin_gl_cache[[key]]
  if (!is.null(cached)) return(cached)
  k <- seq_len(n - 1)
  beta <- k / sqrt(4 * k^2 - 1)
  J <- matrix(0, n, n)
  J[cbind(k, k + 1)] <- beta
  J[cbind(k + 1, k)] <- beta
  eig <- eigen(J, symmetric = TRUE)
  ord <- order(eig$values)
  out <- list(nodes = eig$values[ord], weights = 2 * eig$vectors[1, ord]^2)
  .grin_gl_cache[[key]] <- out
  out
}

#' Bivariate normal CDF Phi2(h, k; rho) = P(X <= h, Y <= k), standard normal
#' margins, correlation rho. Sheppard's r-integration: Phi2 = Phi(h)Phi(k) +
#' int_0^rho phi2(h,k;t) dt, evaluated by 48-point Gauss-Legendre quadrature.
#' @keywords internal
.grin_bvn_cdf <- function(h, k, rho) {
  base <- stats::pnorm(h) * stats::pnorm(k)
  if (abs(rho) < 1e-12) return(base)
  gl <- .grin_gauss_legendre(48)
  t <- rho * (gl$nodes + 1) / 2
  omt2 <- 1 - t^2
  dens <- exp(-(h^2 - 2 * t * h * k + k^2) / (2 * omt2)) / (2 * pi * sqrt(omt2))
  base + sum(gl$weights * dens) * (rho / 2)
}

#' Forward model: per-stimulus (zx, zy, rho) -> 4x4 predicted response
#' probabilities (rows = stimuli, cols = responses, canonical order).
#' @keywords internal
.grin_forward_probabilities <- function(zx, zy, rho) {
  p_x1 <- stats::pnorm(-zx)
  p_y1 <- stats::pnorm(-zy)
  p11 <- mapply(.grin_bvn_cdf, -zx, -zy, rho)
  p12 <- p_x1 - p11
  p21 <- p_y1 - p11
  p22 <- 1 - p_x1 - p_y1 + p11
  probs <- cbind(p11, p12, p21, p22)
  pmin(pmax(probs, 0), 1)
}
