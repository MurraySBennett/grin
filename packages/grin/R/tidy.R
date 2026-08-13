# tidy.R: turn one or many grin_inference objects into a plain data.frame, the
# shared foundation every plotting function builds on. One row per participant.

#' Tidy one or many GRIN inferences into a data.frame
#'
#' @param results A single `grin_inference` (from [grin_infer()]), or an
#'   (optionally named) list of them -- e.g. from looping `grin_infer()` over
#'   a sample of participants.
#' @param ids Optional character vector of participant IDs, same length as
#'   `results`. Defaults to the list's names if present, otherwise `p1, p2, ...`.
#' @return A data.frame, one row per participant: `id`, `model_class`, the 12
#'   parameter estimates (`zx_0`...`rho_3`), their SDs (`zx_0_sd`...`rho_3_sd`),
#'   `p_PI`/`p_sep_A`/`p_sep_B`, and `evidence_PI`/`evidence_sep_A`/`evidence_sep_B`.
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5, 20, 67,  5,  9,
#'               13,  6, 63, 20,  5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' out <- list(p1 = grin_infer(M), p2 = grin_infer(M))
#' grin_tidy(out)
#' }
#' @export
grin_tidy <- function(results, ids = NULL) {
  if (inherits(results, "grin_inference")) results <- list(results)
  stopifnot(all(vapply(results, inherits, logical(1), "grin_inference")))

  if (is.null(ids)) {
    ids <- if (!is.null(names(results)) && all(nzchar(names(results)))) {
      names(results)
    } else {
      paste0("p", seq_along(results))
    }
  }
  stopifnot(length(ids) == length(results))

  rows <- Map(function(r, id) {
    res <- r$result; con <- r$constructs
    est <- stats::setNames(as.list(res$params), res$names)
    sd_ <- stats::setNames(as.list(res$std), paste0(res$names, "_sd"))
    c(list(id = id, model_class = res$model_class), est, sd_,
      list(p_PI = con$p_PI, p_sep_A = con$p_sep_A, p_sep_B = con$p_sep_B,
          evidence_PI = con$evidence_PI, evidence_sep_A = con$evidence_sep_A,
          evidence_sep_B = con$evidence_sep_B))
  }, results, ids)

  do.call(rbind.data.frame, c(lapply(rows, as.data.frame, stringsAsFactors = FALSE),
                              list(stringsAsFactors = FALSE)))
}

#' @keywords internal
.grin_long_params <- function(tidy_df) {
  groups <- list(zx = paste0("zx_", 0:3), zy = paste0("zy_", 0:3), rho = paste0("rho_", 0:3))
  out <- list()
  for (grp in names(groups)) {
    for (nm in groups[[grp]]) {
      out[[length(out) + 1]] <- data.frame(
        id = tidy_df$id, group = grp, param = nm,
        estimate = tidy_df[[nm]], sd = tidy_df[[paste0(nm, "_sd")]],
        stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, out)
}
