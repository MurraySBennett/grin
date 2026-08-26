# model.R: native GRIN inference via the R `torch` package (libtorch bindings).
#
# No Python, no reticulate, no onnxruntime -- this loads the same trained network
# as grintools (the Python package), exported as TorchScript instead of ONNX, and
# numerically verified to match it (see scripts/export_torchscript.py in the main
# grin repo, and tests/testthat/test-parity.R here). All featurisation, link
# functions, and construct heads are inside the traced graph: this wrapper does no
# maths beyond reshaping inputs and reading outputs, mirroring grintools/onnx.py.

.grin_cache <- new.env(parent = emptyenv())

.corr_label <- c("PI", "RHO1", "free")

.class_label <- function(p_corr, p_sep_a, p_sep_b) {
  corr <- .corr_label[which.max(p_corr)]
  parts <- c(corr,
            if (p_sep_a >= 0.5) "PS(A)" else "!PS(A)",
            if (p_sep_b >= 0.5) "PS(B)" else "!PS(B)")
  paste(parts, collapse = " + ")
}

#' Path to the TorchScript model bundled with this package version
#'
#' The package version pins the model: `packageVersion("grin")` identifies exactly
#' which trained weights produced a given inference (mirrors grintools' Python-side
#' `default_model_path()` / version-pinning contract).
#' @export
grin_default_model_path <- function() {
  p <- system.file("models", "npe_model_ts.pt", package = "grin")
  if (!nzchar(p)) stop("bundled model not found -- reinstall the grin package", call. = FALSE)
  p
}

#' Load (and cache) a GRIN TorchScript model
#'
#' Requires the \pkg{torch} package (CRAN; downloads libtorch on first use via
#' `torch::install_torch()`). No Python is used anywhere in this path.
#'
#' @param path Path to a `.pt` TorchScript export. Defaults to the model bundled
#'   with this package.
#' @return A `grin_model` object, ready to pass to [grin_infer()]. Models are
#'   cached per session by path, so repeated calls are free.
#' @export
grin_model <- function(path = NULL) {
  if (is.null(path)) path <- grin_default_model_path()
  cached <- .grin_cache[[path]]
  if (!is.null(cached)) return(cached)
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("the 'torch' package is required for inference. Install it with:\n",
        "  install.packages('torch'); torch::install_torch()", call. = FALSE)
  }
  # No m$eval() here: the graph was already traced in eval mode on the Python side
  # (dropout etc. baked in as no-ops), and toggling train/eval on a jit-loaded
  # module hits a dispatch bug in some versions of the R torch package.
  m <- torch::jit_load(path)
  obj <- structure(list(module = m, path = path), class = "grin_model")
  .grin_cache[[path]] <- obj
  obj
}

#' @export
print.grin_model <- function(x, ...) {
  cat(sprintf("<grin_model> %s\n", x$path))
  invisible(x)
}

#' @keywords internal
.to_counts_matrix <- function(counts) {
  if (!is.null(dim(counts)) && length(dim(counts)) == 2) {
    return(matrix(as.numeric(counts), nrow = 4, ncol = 4))
  }
  v <- as.numeric(counts)
  if (length(v) != 16) {
    stop(sprintf("counts must be a 4x4 matrix or a length-16 vector (row-major); got length %d",
                 length(v)), call. = FALSE)
  }
  matrix(v, nrow = 4, ncol = 4, byrow = TRUE)
}

#' Run GRIN inference on a canonical-order confusion matrix
#'
#' The fast path: `counts` is trusted to already be in canonical order (rows/cols
#' A1B1, A1B2, A2B1, A2B2). If your data isn't already canonical, resolve it first
#' with [grin_to_confusion()] and pass its `$counts`/`$trials`.
#'
#' @param counts A canonical-order 4x4 matrix, or a length-16 vector read row-major
#'   (stimulus varies slower than response).
#' @param trials Optional per-stimulus trial totals (length 4); defaults to row sums.
#' @param model A `grin_model` (see [grin_model()]); the bundled model is loaded and
#'   cached automatically if omitted.
#' @param evidence_tol Width of the "undecided" band around p = 0.5 for the
#'   `evidence_*` construct flags (default 0.5).
#' @return A `grin_inference` object: `$result` (class `grin_result`: `$params`,
#'   `$std`, `$ci_low`, `$ci_high`, `$names`, `$model_class`) and `$constructs`
#'   (`p_PI`, `p_sep_A`, `p_sep_B`, `p_corr`, `evidence_PI`, `evidence_sep_A`,
#'   `evidence_sep_B`). Unlike Python's `(result, constructs)` tuple, R returns one
#'   object with both as named elements -- there is no tuple-unpacking idiom to
#'   mirror here.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5,
#'               20, 67,  5,  9,
#'               13,  6, 63, 20,
#'                5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' out <- grin_infer(M)
#' print(out$result)
#' out$constructs$p_PI
#' }
#' @export
#' Per-family posterior scale factors, fitted on held-out simulations by
#' scripts/fit_recalibration.py and validated on a further held-out set. Applied only
#' when the caller passes calibrated = TRUE. See the package documentation for why this
#' is opt-in: the correction is estimated under the training prior and may not transfer
#' to observers far outside it, and a rescaled interval is a calibrated interval derived
#' from the posterior rather than the posterior itself.
.grin_recalibration <- local({
  cache <- NULL
  function() {
    if (!is.null(cache)) return(cache)
    f <- system.file("extdata", "recalibration.json", package = "grin")
    cache <<- if (nzchar(f) && requireNamespace("jsonlite", quietly = TRUE)) {
      jsonlite::fromJSON(f)
    } else NULL
    cache
  }
})

.grin_scales <- function(calibrated) {
  s <- rep(1, 12)
  if (isTRUE(calibrated)) {
    spec <- .grin_recalibration()
    if (is.null(spec)) {
      warning("no recalibration data shipped with this build; returning raw intervals",
              call. = FALSE)
    } else {
      s[1:8] <- spec$global_scale$z
      s[9:12] <- spec$global_scale$rho
    }
  }
  s
}

grin_infer <- function(counts, trials = NULL, model = NULL, evidence_tol = 0.5,
                       calibrated = FALSE) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("the 'torch' package is required for inference. Install it with:\n",
        "  install.packages('torch'); torch::install_torch()", call. = FALSE)
  }
  if (is.null(model)) model <- grin_model()
  cm <- .to_counts_matrix(counts)
  counts_flat <- as.vector(t(cm))               # row-major flatten: matches the trained graph's contract
  if (is.null(trials)) trials <- rowSums(cm)
  trials <- as.numeric(trials)[1:4]

  counts_t <- torch::torch_tensor(matrix(counts_flat, nrow = 1), dtype = torch::torch_float32())
  trials_t <- torch::torch_tensor(matrix(trials, nrow = 1), dtype = torch::torch_float32())

  out <- torch::with_no_grad(model$module(counts_t, trials_t))
  mean <- as.numeric(out[[1]][1, ]); std <- as.numeric(out[[2]][1, ])
  p_corr <- as.numeric(out[[3]][1, ]); p_sep <- as.numeric(out[[4]][1, ])

  p_pi <- p_corr[1]; p_a <- p_sep[1]; p_b <- p_sep[2]
  band <- 0.5 - evidence_tol / 2.0
  constructs <- list(p_PI = p_pi, p_sep_A = p_a, p_sep_B = p_b, p_corr = p_corr,
                     evidence_PI = abs(p_pi - 0.5) > band,
                     evidence_sep_A = abs(p_a - 0.5) > band,
                     evidence_sep_B = abs(p_b - 0.5) > band)

  scale <- .grin_scales(calibrated)
  std_c <- std * scale
  result <- structure(
    list(params = mean, std = std_c, std_raw = std, scale = scale,
        calibrated = isTRUE(calibrated),
        ci_low = mean - 1.645 * std_c, ci_high = mean + 1.645 * std_c,
        names = PARAM_NAMES, model_class = .class_label(p_corr, p_a, p_b)),
    class = "grin_result")

  structure(list(result = result, constructs = constructs), class = "grin_inference")
}

#' @export
print.grin_result <- function(x, ...) {
  cat("GRIN inference\n"); cat(strrep("-", 46), "\n")
  for (i in seq_along(x$names)) {
    cat(sprintf("  %-7s = %+.2f  +/- %.2f   [90%% %+.2f, %+.2f]\n",
               x$names[i], x$params[i], x$std[i], x$ci_low[i], x$ci_high[i]))
  }
  cat(strrep("-", 46), "\n")
  cat(sprintf("  most likely structure : %s\n", x$model_class))
  invisible(x)
}

#' @export
print.grin_inference <- function(x, ...) {
  print(x$result)
  cat(sprintf("\nconstructs: PI=%.2f  PS_A=%.2f  PS_B=%.2f\n",
             x$constructs$p_PI, x$constructs$p_sep_A, x$constructs$p_sep_B))
  invisible(x)
}
