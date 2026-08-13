# io.R: the input contract for GRIN.
#
# Port of grintools' grin_io.py (Python). Same job, same principle: liberal about
# shape, strict about meaning. Any container is coerced, but two things are never
# guessed because guessing returns a confident wrong answer: (1) stimulus/response
# ORDER for a bare unlabelled matrix, and (2) counts vs proportions (the network
# reads trial totals as a second input, so proportions silently wreck the
# posterior's width). Both are refused unless the caller resolves them explicitly.
#
# Canonical layout: rows = stimuli, cols = responses, both in order A1B1, A1B2,
# A2B1, A2B2 (dimension A varies slowest, B fastest). A bare length-16 vector is
# read row-major (stimulus-major): position (0-based) k = stim_index*4 + resp_index.

#' Parse one cell label into 0-based (a_level, b_level)
#' @keywords internal
.parse_cell_label <- function(label, factor_a, factor_b, sep) {
  if (length(label) == 2) {
    a_name <- label[[1]]; b_name <- label[[2]]
  } else if (length(label) == 1) {
    parts <- strsplit(as.character(label), sep, fixed = TRUE)[[1]]
    if (length(parts) != 2) {
      stop(sprintf(
        "cannot split label '%s' on sep '%s' into two factor levels; pass a length-2 ",
        "c(a_name, b_name) or use a matching sep", label, sep), call. = FALSE)
    }
    a_name <- trimws(parts[1]); b_name <- trimws(parts[2])
  } else {
    stop(sprintf("label of length %d is not a 2-element pair or a single '%s'-separated string",
                 length(label), sep), call. = FALSE)
  }
  if (!(a_name %in% factor_a)) {
    stop(sprintf("'%s' is not a level of factor_a = c(%s)", a_name,
                 paste(factor_a, collapse = ", ")), call. = FALSE)
  }
  if (!(b_name %in% factor_b)) {
    stop(sprintf("'%s' is not a level of factor_b = c(%s)", b_name,
                 paste(factor_b, collapse = ", ")), call. = FALSE)
  }
  c(a = match(a_name, factor_a) - 1L, b = match(b_name, factor_b) - 1L)
}

#' Map 4 labels onto canonical order, or fail loudly
#'
#' Returns list(perm, placement). `perm` is a length-4 index vector into `labels`
#' such that `labels[perm[c]]` belongs at canonical position c. Raises unless the 4
#' labels form a complete 2x2 factorial (or are already the canonical tokens).
#' @keywords internal
.permutation_to_canonical <- function(labels, factor_a, factor_b, sep) {
  tok <- function(x) tolower(paste(x, collapse = sep))
  low <- vapply(labels, tok, character(1))
  if (identical(low, tolower(CANON_STIM)) || identical(low, tolower(CANON_RESP))) {
    placement <- setNames(labels, CANON_STIM)
    return(list(perm = 1:4, placement = placement))
  }
  if (is.null(factor_a) || is.null(factor_b)) {
    stop(paste("labels are not canonical A1B1.. tokens, so I need factor_a=c(A1,A2) and",
               "factor_b=c(B1,B2) to place them. Refusing to assume an order."), call. = FALSE)
  }
  factor_a <- as.character(factor_a); factor_b <- as.character(factor_b)
  source_of_canon <- rep(NA_integer_, 4)
  seen <- logical(4)
  placement <- vector("list", 4)
  for (pos in seq_along(labels)) {
    ab <- .parse_cell_label(labels[[pos]], factor_a, factor_b, sep)
    c_idx <- 2L * ab[["a"]] + ab[["b"]] + 1L
    if (seen[c_idx]) {
      stop(sprintf("label duplicates canonical cell %s; the 4 labels must be a complete 2x2 factorial",
                   CANON_STIM[c_idx]), call. = FALSE)
    }
    seen[c_idx] <- TRUE
    source_of_canon[c_idx] <- pos
    placement[[c_idx]] <- labels[[pos]]
  }
  if (!all(seen)) {
    stop(sprintf("labels do not cover the full factorial; missing: %s",
                 paste(CANON_STIM[!seen], collapse = ", ")), call. = FALSE)
  }
  names(placement) <- CANON_STIM
  list(perm = source_of_canon, placement = placement)
}

#' @keywords internal
.counts_guard <- function(counts, trials, warn) {
  counts <- matrix(as.numeric(counts), nrow = 4, ncol = 4)
  row_sums <- rowSums(counts)
  looks_like_props <- all(abs(row_sums - 1.0) < 1e-3) && all(counts <= 1.0 + .PROP_TOL)
  non_integer <- any(abs(counts - round(counts)) > .INT_TOL)
  if (looks_like_props || non_integer) {
    if (is.null(trials)) {
      stop(paste("input looks like PROPORTIONS, not counts (rows sum to ~1 or contain",
                 "non-integers). The network reads per-stimulus trial totals as a",
                 "separate input, so proportions would silently wreck the posterior",
                 "uncertainty. Pass counts, or pass trials=[...] to rescale."), call. = FALSE)
    }
    trials_vec <- as.numeric(trials)[1:4]
    counts <- round(sweep(counts, 1, trials_vec, `*`))   # row i scaled by trials[i]
    .warn(warn, paste("input treated as proportions and rescaled to counts using the",
                      "supplied trials; verify this is what you intended"))
  }
  matrix(as.integer(round(counts)), nrow = 4, ncol = 4)
}

#' @keywords internal
.row_warnings <- function(counts, trials, warn) {
  for (i in 1:4) {
    if (trials[i] < .SPARSE_TRIALS) {
      .warn(warn, sprintf("stimulus %s has only %d trials (< %d); posterior will be wide here",
                          CANON_STIM[i], as.integer(trials[i]), .SPARSE_TRIALS))
    }
  }
  if (any(counts == 0)) {
    zeros <- which(counts == 0, arr.ind = TRUE)
    zeros <- zeros[order(zeros[, "row"], zeros[, "col"]), , drop = FALSE]
    z_desc <- apply(zeros, 1, function(rc) sprintf("(%s, %s)", CANON_STIM[rc["row"]], CANON_RESP[rc["col"]]))
    .warn(warn, sprintf("%d empty cell(s) [%s]; fine for GRIN, but this is the cell-separation regime where MLE baselines diverge",
                        length(z_desc), paste(z_desc, collapse = ", ")))
  }
}

#' @keywords internal
.from_long <- function(data, factor_a, factor_b, sep, warn) {
  if (is.data.frame(data)) {
    cols <- tolower(names(data))
    if (!("stimulus" %in% cols) || !("response" %in% cols)) {
      stop("long data.frame needs 'stimulus' and 'response' columns (a 'count' column is optional)",
           call. = FALSE)
    }
    stim <- as.character(data[[which(cols == "stimulus")[1]]])
    resp <- as.character(data[[which(cols == "response")[1]]])
    cnt <- if ("count" %in% cols) as.numeric(data[[which(cols == "count")[1]]]) else rep(1, nrow(data))
  } else {
    stim <- vapply(data, function(r) as.character(r[[1]]), character(1))
    resp <- vapply(data, function(r) as.character(r[[2]]), character(1))
    cnt <- vapply(data, function(r) if (length(r) >= 3) as.numeric(r[[3]]) else 1, numeric(1))
  }
  stim_labels <- unique(stim); resp_labels <- unique(resp)
  if (length(stim_labels) != 4 || length(resp_labels) != 4) {
    stop(sprintf("expected 4 distinct stimuli and 4 responses; got %d stimuli, %d responses",
                 length(stim_labels), length(resp_labels)), call. = FALSE)
  }
  sp <- .permutation_to_canonical(as.list(stim_labels), factor_a, factor_b, sep)
  rp <- .permutation_to_canonical(as.list(resp_labels), factor_a, factor_b, sep)
  s_to_c <- setNames(integer(4), stim_labels[sp$perm]); s_to_c[stim_labels[sp$perm]] <- 1:4
  r_to_c <- setNames(integer(4), resp_labels[rp$perm]); r_to_c[resp_labels[rp$perm]] <- 1:4
  counts <- matrix(0, nrow = 4, ncol = 4)
  for (k in seq_along(stim)) {
    si <- s_to_c[[stim[k]]]; ri <- r_to_c[[resp[k]]]
    counts[si, ri] <- counts[si, ri] + cnt[k]
  }
  list(counts = counts, placement = sp$placement)
}

#' Normalise any supported input into a canonical-order confusion matrix
#'
#' Coerces a confusion matrix in (almost) any shape and label scheme into the
#' canonical-order (A1B1, A1B2, A2B1, A2B2 x A1B1, A1B2, A2B1, A2B2) form GRIN's
#' models expect, and never guesses the two things a wrong guess would silently
#' corrupt: stimulus/response order, and counts-vs-proportions.
#'
#' @param data A 4x4 matrix/data.frame, a length-16 vector (read row-major: stimulus
#'   varies slower than response), or (with `long = TRUE`) a long-format data.frame
#'   with `stimulus` and `response` columns (plus an optional `count` column).
#' @param stim_labels,resp_labels Character vectors of the 4 row/column labels, in
#'   the order `data` actually has them. If labels are canonical A1B1-style tokens
#'   they are matched directly; otherwise supply `factor_a`/`factor_b` too.
#' @param factor_a,factor_b Length-2 character vectors giving the two levels of each
#'   dimension, e.g. `c("Old","Young")`, `c("Neg","Pos")`, needed to place non-canonical
#'   labels like `"Old/Neg"`.
#' @param order Pass `"canonical"` to assert `data` is already in canonical order
#'   (skips label resolution entirely). A bare, unlabelled 4x4 with neither `order`
#'   nor labels is refused rather than guessed.
#' @param trials Optional per-stimulus trial totals (length 4); required if `data`
#'   looks like proportions rather than counts. Defaults to row sums of the counts.
#' @param sep Separator used to split single-string labels like `"Old/Neg"`.
#' @param long If `TRUE` (or `data` is a data.frame with a `stimulus` column),
#'   treat `data` as long-format trial-level or aggregated data.
#' @return A `grin_confusion_input` object: `$counts` (4x4 integer matrix), `$trials`
#'   (length-4 integer), `$placement`, `$warnings`, `$asserted_order`.
#' @export
grin_to_confusion <- function(data, stim_labels = NULL, resp_labels = NULL,
                              factor_a = NULL, factor_b = NULL, order = NULL,
                              trials = NULL, sep = "/", long = FALSE) {
  warn <- .new_warn_collector()
  asserted <- FALSE

  is_long <- isTRUE(long) || (is.data.frame(data) && "stimulus" %in% tolower(names(data)))
  if (is_long) {
    parsed <- .from_long(data, factor_a, factor_b, sep, warn)
    counts <- parsed$counts; placement <- parsed$placement
  } else {
    if (is.data.frame(data)) {
      if (is.null(resp_labels)) resp_labels <- names(data)
      idx <- rownames(data)
      if (is.null(stim_labels) && !all(grepl("^[0-9]+$", idx))) stim_labels <- idx
      mat <- matrix(as.numeric(as.matrix(data)), nrow = 4, ncol = 4)
    } else if (!is.null(dim(data)) && length(dim(data)) == 2) {
      mat <- matrix(as.numeric(data), nrow = 4, ncol = 4)
    } else {
      v <- as.numeric(data)
      if (length(v) != 16) {
        stop(sprintf(paste("expected a 4x4 matrix/data.frame or a length-16 vector",
                           "(read row-major: stimulus-major); got length %d"), length(v)),
             call. = FALSE)
      }
      mat <- matrix(v, nrow = 4, ncol = 4, byrow = TRUE)
    }

    if (identical(order, "canonical")) {
      counts <- mat
      placement <- setNames(as.list(CANON_STIM), CANON_STIM)
      asserted <- TRUE
    } else if (!is.null(stim_labels) || !is.null(resp_labels)) {
      if (!is.null(resp_labels) && is.null(stim_labels)) {
        stim_labels <- resp_labels
        .warn(warn, "rows were unlabelled; assumed to follow the same category order as the column labels")
      }
      if (is.null(stim_labels) || is.null(resp_labels)) {
        stop("need both stim_labels and resp_labels (or order='canonical') to place a 4x4 matrix",
             call. = FALSE)
      }
      rp <- .permutation_to_canonical(as.list(stim_labels), factor_a, factor_b, sep)
      cp <- .permutation_to_canonical(as.list(resp_labels), factor_a, factor_b, sep)
      counts <- mat[rp$perm, cp$perm, drop = FALSE]
      placement <- rp$placement
    } else {
      stop(paste("a bare 4x4 with no labels and no order assertion is refused: I will",
                 "not guess the stimulus/response order and hand back a confident wrong",
                 "posterior. Either pass order='canonical' to assert your matrix is",
                 "already A1B1,A1B2,A2B1,A2B2, or pass stim_labels/resp_labels with",
                 "factor_a and factor_b."), call. = FALSE)
    }
  }

  counts <- .counts_guard(counts, trials, warn)
  resolved_trials <- if (!is.null(trials)) {
    as.integer(round(as.numeric(trials)))[1:4]
  } else {
    as.integer(rowSums(counts))
  }
  if (!is.null(trials) && !identical(resolved_trials, as.integer(rowSums(counts)))) {
    .warn(warn, "supplied trials disagree with row sums of the counts; using the supplied trials")
  }
  .row_warnings(counts, resolved_trials, warn)

  structure(list(counts = counts, trials = resolved_trials, placement = placement,
                warnings = warn$msgs, asserted_order = asserted),
           class = "grin_confusion_input")
}

#' @export
print.grin_confusion_input <- function(x, ...) {
  cat(sprintf("<grin_confusion_input> trials=[%s] warnings=%d\n",
             paste(x$trials, collapse = ", "), length(x$warnings)))
  invisible(x)
}

#' Describe how input would be parsed, without running the network
#'
#' Setup guide / dev mode / exception surface in one. Never raises: parse errors are
#' captured and printed, not thrown, so this is safe to run while wiring up an
#' experiment. Pass the same arguments you would pass to [grin_to_confusion()].
#'
#' @inheritParams grin_to_confusion
#' @param printout Print a human-readable report (default `TRUE`).
#' @param ... Passed on to [grin_to_confusion()] (`stim_labels`, `factor_a`, etc.).
#' @return Invisibly, a list report (`$ready`, `$errors`, `$warnings`, `$counts`,
#'   `$trials`, `$placement`, `$asserted_order`).
#' @export
grin_describe <- function(data, printout = TRUE, ...) {
  report <- list(ready = FALSE, errors = character(0), warnings = character(0),
                 counts = NULL, trials = NULL, placement = NULL, asserted_order = NULL)
  ci <- tryCatch(grin_to_confusion(data, ...), error = function(e) e)
  if (inherits(ci, "grin_confusion_input")) {
    report$ready <- TRUE
    report$counts <- ci$counts; report$trials <- ci$trials
    report$placement <- ci$placement; report$warnings <- ci$warnings
    report$asserted_order <- ci$asserted_order
  } else {
    report$errors <- c(report$errors, sprintf("%s: %s", class(ci)[1], conditionMessage(ci)))
  }
  if (isTRUE(printout)) {
    lines <- c("GRIN input check", strrep("-", 52))
    if (report$ready) {
      lines <- c(lines, paste0("parsed OK, ready for inference",
                if (isTRUE(report$asserted_order)) "  (order asserted by caller)"
                else "  (order resolved from labels)"))
      lines <- c(lines, "canonical placement (canonical cell <- your label):")
      for (cnm in CANON_STIM) {
        lines <- c(lines, sprintf("    %s <- %s", cnm, paste(report$placement[[cnm]], collapse = "/")))
      }
      lines <- c(lines, paste("trials per stimulus:", paste(report$trials, collapse = ", ")))
      lines <- c(lines, "counts (canonical order):")
      for (i in 1:4) {
        lines <- c(lines, sprintf("    %-5s | %s", CANON_STIM[i],
                                  paste(sprintf("%4d", report$counts[i, ]), collapse = " ")))
      }
    } else {
      lines <- c(lines, "NOT ready, could not parse:")
    }
    for (w in report$warnings) lines <- c(lines, sprintf("  warning: %s", w))
    for (e in report$errors) lines <- c(lines, sprintf("  ERROR:   %s", e))
    cat(paste(lines, collapse = "\n"), "\n")
  }
  invisible(report)
}
