#' grin: Amortised Inference for General Recognition Theory
#'
#' Feed a 2x2 identification confusion matrix, get a calibrated posterior over the
#' 12 GRT parameters plus construct probabilities (perceptual independence,
#' separability), and an optional stopping decision for adaptive designs. Runs
#' natively via \pkg{torch} (libtorch) -- no Python required.
#'
#' @section Quick start:
#' ```r
#' M <- matrix(c(71, 17,  9,  5,
#'               20, 67,  5,  9,
#'               13,  6, 63, 20,
#'                5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' out <- grin_infer(M)
#' print(out$result)
#' out$constructs$p_PI
#' ```
#'
#' If your matrix is not already in canonical order (rows/cols A1B1, A1B2, A2B1,
#' A2B2), normalise it first with [grin_to_confusion()] -- see its documentation
#' for the labelled/long-format input forms it accepts.
#'
#' @section The ordering contract:
#' [grin_infer()] trusts `counts` to already be canonical. A wrong guess about
#' stimulus/response order returns a confident wrong posterior, so it is never
#' guessed: [grin_to_confusion()] either resolves the order from labels you supply,
#' or requires you to assert `order = "canonical"`.
#'
#' @keywords internal
#' @importFrom stats setNames
"_PACKAGE"
