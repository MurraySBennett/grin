# criterion.R: stopping decisions for adaptive designs.
#
# Port of grintools' Criterion/Target/Decision (Python). The EXPERIMENTER declares
# a Criterion from Targets; GRIN evaluates it. Three ways to use it:
#
#   1. precision    grin_target_precision(params=..., sd_max=... | ci_width_max=...)
#                   stop when the parameter posterior is tight enough. This is the
#                   "I want the space measured, verdict aside" target.
#   2. probability  grin_target_probability("PI"|"PS_A"|"PS_B"|"*_violated", at_least=0.9)
#                   stop when a construct probability crosses a threshold. This is
#                   the "I want the verdict" target.
#   3. combine      grin_criterion(list(...targets...), combine = "all"|"any")
#
# Probability targets read the construct list's evidence_* flags. When an evidence
# flag is FALSE, the target is reported as unreachable (Decision$blocked_by), which
# is how the PI identifiability limit surfaces: a threshold on a construct the data
# cannot decide will never be met, and we say so rather than looping forever.

.CONSTRUCT_MAP <- list(
  PI            = list(key = "p_PI",    evkey = "evidence_PI",    violated = FALSE),
  PS_A          = list(key = "p_sep_A", evkey = "evidence_sep_A", violated = FALSE),
  PS_B          = list(key = "p_sep_B", evkey = "evidence_sep_B", violated = FALSE),
  PI_violated   = list(key = "p_PI",    evkey = "evidence_PI",    violated = TRUE),
  PS_A_violated = list(key = "p_sep_A", evkey = "evidence_sep_A", violated = TRUE),
  PS_B_violated = list(key = "p_sep_B", evkey = "evidence_sep_B", violated = TRUE)
)

#' @keywords internal
.select_indices <- function(params) {
  if (is.null(params) || identical(params, "all")) return(seq_along(PARAM_NAMES))
  idx <- integer(0)
  for (p in params) {
    if (p %in% names(PARAM_GROUPS)) {
      idx <- c(idx, PARAM_GROUPS[[p]])
    } else if (p %in% PARAM_NAMES) {
      idx <- c(idx, match(p, PARAM_NAMES))
    } else {
      stop(sprintf("unknown parameter selector '%s'; use a name in {%s} or a group in {%s}",
                   p, paste(PARAM_NAMES, collapse = ", "), paste(names(PARAM_GROUPS), collapse = ", ")),
          call. = FALSE)
    }
  }
  sort(unique(idx))
}

#' Precision stopping target
#'
#' Stop when the posterior for the selected parameters is tight enough. Exactly one
#' of `sd_max`/`ci_width_max` must be given.
#'
#' @param params Character vector of parameter names ([PARAM_NAMES]) and/or groups
#'   (`"zx"`, `"zy"`, `"rho"`), or `NULL`/`"all"` for every parameter.
#' @param sd_max Stop when every selected parameter's posterior SD is at most this.
#' @param ci_width_max Stop when every selected parameter's 90% CI width is at most this.
#' @export
grin_target_precision <- function(params = NULL, sd_max = NULL, ci_width_max = NULL) {
  if (is.null(sd_max) == is.null(ci_width_max)) {
    stop("precision target needs exactly one of sd_max or ci_width_max", call. = FALSE)
  }
  structure(list(kind = "precision",
                cfg = list(params = params, sd_max = sd_max, ci_width_max = ci_width_max)),
           class = "grin_target")
}

#' Construct-probability stopping target
#'
#' Stop when a construct probability crosses a threshold.
#'
#' @param construct One of `"PI"`, `"PS_A"`, `"PS_B"`, or a `"*_violated"` complement.
#' @param at_least Probability threshold.
#' @export
grin_target_probability <- function(construct, at_least) {
  if (!(construct %in% names(.CONSTRUCT_MAP))) {
    stop(sprintf("unknown construct '%s'; choose from {%s}", construct,
                paste(names(.CONSTRUCT_MAP), collapse = ", ")), call. = FALSE)
  }
  structure(list(kind = "probability",
                cfg = list(construct = construct, at_least = as.numeric(at_least))),
           class = "grin_target")
}

#' @keywords internal
.target_check <- function(target, result, constructs) {
  cfg <- target$cfg
  if (identical(target$kind, "precision")) {
    idx <- .select_indices(cfg$params)
    names_ <- if (!is.null(result$names)) result$names else PARAM_NAMES
    if (!is.null(cfg$sd_max)) {
      vals <- as.numeric(result$std)[idx]; thr <- cfg$sd_max; q <- "sd"
    } else {
      vals <- (as.numeric(result$ci_high) - as.numeric(result$ci_low))[idx]
      thr <- cfg$ci_width_max; q <- "ci_width"
    }
    worst <- which.max(vals)
    return(list(met = all(vals <= thr), value = vals[worst],
               name = sprintf("%s:%s", q, names_[idx[worst]]), threshold = thr,
               reachable = TRUE, note = ""))
  }
  # probability target
  cname <- cfg$construct; thr <- cfg$at_least
  if (is.null(constructs)) {
    return(list(met = FALSE, value = NaN, name = cname, threshold = thr,
               reachable = NA, note = "no constructs supplied; cannot evaluate"))
  }
  spec <- .CONSTRUCT_MAP[[cname]]
  p <- as.numeric(constructs[[spec$key]])
  if (spec$violated) p <- 1.0 - p
  evkey_val <- constructs[[spec$evkey]]
  ev <- if (is.null(evkey_val)) TRUE else isTRUE(evkey_val)
  note <- if (ev) "" else paste("evidence flag is False: the data may not be able to decide",
                                "this construct in the current regime (a property of the",
                                "data, not of GRIN)")
  list(met = (p >= thr) && ev, value = p, name = cname, threshold = thr,
      reachable = ev, note = note)
}

#' Combine stopping targets into a criterion
#'
#' @param targets A list of `grin_target` objects (see [grin_target_precision()],
#'   [grin_target_probability()]).
#' @param combine `"all"` (default, stop only once every target is met) or `"any"`.
#' @export
grin_criterion <- function(targets, combine = "all") {
  if (!(combine %in% c("all", "any"))) stop("combine must be 'all' or 'any'", call. = FALSE)
  structure(list(targets = targets, combine = combine), class = "grin_criterion")
}

#' @keywords internal
.criterion_needs_constructs <- function(criterion) {
  any(vapply(criterion$targets, function(t) identical(t$kind, "probability"), logical(1)))
}

#' Evaluate a stopping criterion against an inference result
#'
#' @param criterion A `grin_criterion` (see [grin_criterion()]).
#' @param result A `grin_result` (e.g. `grin_infer(...)$result`).
#' @param constructs Optional constructs list (e.g. `grin_infer(...)$constructs`);
#'   required if any target in `criterion` is a probability target.
#' @return A `grin_decision`: `$stop` (logical), `$checks` (list of per-target detail),
#'   `$combine`, `$blocked_by` (names of targets whose evidence flag says the data
#'   cannot decide them -- these will never be met, no matter how much data you add).
#' @examples
#' \donttest{
#' M <- matrix(c(71, 17,  9,  5,
#'               20, 67,  5,  9,
#'               13,  6, 63, 20,
#'                5, 10, 15, 71), nrow = 4, byrow = TRUE)
#' out <- grin_infer(M)
#' crit <- grin_criterion(list(
#'   grin_target_precision(params = c("zx", "zy"), sd_max = 0.10),
#'   grin_target_probability("PS_A", at_least = 0.90)
#' ), combine = "any")
#' decision <- grin_evaluate(crit, out$result, out$constructs)
#' print(decision)
#' }
#' @export
grin_evaluate <- function(criterion, result, constructs = NULL) {
  checks <- lapply(criterion$targets, .target_check, result = result, constructs = constructs)
  mets <- vapply(checks, function(c) isTRUE(c$met), logical(1))
  stop_ <- if (identical(criterion$combine, "all")) all(mets) else any(mets)
  blocked <- vapply(checks, function(c) identical(c$reachable, FALSE), logical(1))
  blocked_names <- vapply(checks[blocked], function(c) c$name, character(1))
  structure(list(stop = stop_, checks = checks, combine = criterion$combine,
                blocked_by = blocked_names),
           class = "grin_decision")
}

#' @export
print.grin_decision <- function(x, ...) {
  cat(sprintf("stop = %s  (combine = '%s')\n", x$stop, x$combine))
  for (c in x$checks) {
    mark <- if (isTRUE(c$met)) "met" else "not met"
    val <- if (is.nan(c$value)) "nan" else sprintf("%.3f", c$value)
    cat(sprintf("    %-16s %s vs %.3f  [%s]\n", c$name, val, c$threshold, mark))
    if (nzchar(c$note)) cat(sprintf("        note: %s\n", c$note))
  }
  if (length(x$blocked_by) > 0) {
    cat(sprintf("    unreachable target(s): %s (threshold may never be met with current data)\n",
               paste(x$blocked_by, collapse = ", ")))
  }
  invisible(x)
}

#' Convenience: evaluate a single precision target
#' @inheritParams grin_target_precision
#' @param result A `grin_result` (e.g. `grin_infer(...)$result`).
#' @export
grin_stop_on_precision <- function(result, sd_max = NULL, ci_width_max = NULL, params = NULL) {
  grin_evaluate(grin_criterion(list(grin_target_precision(params = params, sd_max = sd_max,
                                                          ci_width_max = ci_width_max))),
               result)
}
