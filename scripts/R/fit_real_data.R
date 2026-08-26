# fit_real_data.R — fit the REAL 2x2 identification matrices shipped with mdsdt using both
# R baselines, extracting model selection AND parameters so GRIN can be compared against
# them on data with no ground truth.
#
#   Rscript scripts/R/fit_real_data.R      then:   python scripts/compare_real_data.py
#
# Writes: data/real/real_matrices.csv          (the matrices, for GRIN to read)
#         results/mle_fits/real_data_fits.csv  (mdsdt + grtools selection and parameters)
#         results/mle_fits/real_subsample_fits.csv  (the thinning analysis, see below)
#
# Real 2x2 identification confusion matrices in mdsdt:
#   thomas01a, thomas01b   — face identification, observers A and B (Thomas, 2001)
#   silbert09a, silbert09b — Silbert, Townsend & Lentz (2009)
#   silbert12              — Silbert (2012)
# thomas15a/b are 3x3 and are skipped automatically (GRIN is 2x2 only).
#
# THINNING. Each real matrix is also resampled down to a range of smaller trial counts
# (multinomial resampling of each stimulus row at its own observed response proportions,
# N_THIN independent draws per level), and every method is refit to each thinned matrix.
# This is the real-data analogue of the simulated sparse-data comparison: it asks how each
# method behaves as an observer's data thin, using the full-data fit as the reference,
# without needing ground truth.
#
# ORDERING: mdsdt::fit.grt() and GRIN both use a_1b_1, a_1b_2, a_2b_1, a_2b_2 (B fastest).
# grtools documents a_1b_1, a_2b_1, a_1b_2, a_2b_2 (A fastest) — positions 2 and 3 swapped.
# Matrices are permuted before every grt_hm_fit() call and the parameters permuted back.

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tibble)
})

have_mdsdt   <- requireNamespace("mdsdt", quietly = TRUE)
have_grtools <- requireNamespace("grtools", quietly = TRUE)
if (!have_mdsdt) stop("mdsdt is not installed.  install.packages('mdsdt')")
if (!have_grtools) message("(!) grtools not installed — mdsdt-only run")

grtools_perm <- c(1, 3, 2, 4)
N_THIN <- 10
THIN_LEVELS <- c(200, 100, 50, 25, 12)   # trials per stimulus

extract_grtools_params <- function(hm) {
  bm <- hm$best_model
  # SIGN: PLUS. grtools' $a1/$a2 are the NEGATED bound positions; see the long note in
  # fit_baselines.R, which this mirrors exactly.
  list(zx  = as.numeric(bm$means[, 1]) + bm$a1,
       zy  = as.numeric(bm$means[, 2]) + bm$a2,
       rho = vapply(bm$covmat, function(cm) cm[1, 2], numeric(1)))
}

# ---- mdsdt over its full 12-model factorial hierarchy, selected by AIC ------------
fit_mdsdt <- function(cmat) {
  t0 <- Sys.time()
  grid <- expand.grid(psx = c(TRUE, FALSE), psy = c(TRUE, FALSE),
                      pi = c("all", "same_rho", "none"), stringsAsFactors = FALSE)
  best <- NULL; best_aic <- Inf; best_lab <- NA_character_
  for (k in seq_len(nrow(grid))) {
    g <- grid[k, ]
    f <- tryCatch({
      if (g$pi == "none") mdsdt::fit.grt(cmat, PS_x = g$psx, PS_y = g$psy)
      else mdsdt::fit.grt(cmat, PS_x = g$psx, PS_y = g$psy, PI = g$pi)
    }, error = function(e) NULL)
    a <- if (is.null(f)) NA_real_ else
      tryCatch(as.numeric(mdsdt::GOF(f, teststat = "AIC")), error = function(e) NA_real_)
    if (!is.na(a) && a < best_aic) {
      best_aic <- a; best <- f
      best_lab <- paste0(if (g$psx) "PS(A)" else "-", "_",
                         if (g$psy) "PS(B)" else "-", "_", g$pi)
    }
  }
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (is.null(best)) return(list(ok = FALSE, secs = secs, model = NA_character_,
                                 zx = rep(NA_real_, 4), zy = rep(NA_real_, 4),
                                 rho = rep(NA_real_, 4)))
  d <- best$dists   # 4x5: mu, sigma, nu, tau, rho — rows already in GRIN order
  list(ok = TRUE, secs = secs, model = best_lab, aic = best_aic,
       zx = as.numeric(d[, "mu"]), zy = as.numeric(d[, "nu"]), rho = as.numeric(d[, "rho"]))
}

# ---- grtools' own hierarchy fit ---------------------------------------------------
fit_grtools <- function(cmat, n_reps = 10) {
  if (!have_grtools) return(list(ok = NA, secs = NA_real_, model = NA_character_,
                                 zx = rep(NA_real_, 4), zy = rep(NA_real_, 4),
                                 conv = NA_integer_, rho = rep(NA_real_, 4)))
  cg <- cmat[grtools_perm, grtools_perm]
  t0 <- Sys.time()
  hm <- tryCatch(grtools::grt_hm_fit(cg, n_reps = n_reps), error = function(e) NULL)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (is.null(hm)) return(list(ok = FALSE, secs = secs, model = NA_character_,
                               conv = NA_integer_,
                               zx = rep(NA_real_, 4), zy = rep(NA_real_, 4),
                               rho = rep(NA_real_, 4)))
  conv <- tryCatch(hm$best_model$convergence, error = function(e) 1)
  p <- tryCatch(extract_grtools_params(hm), error = function(e) NULL)
  if (is.null(p)) return(list(ok = FALSE, secs = secs, model = NA_character_,
                              conv = conv,
                              zx = rep(NA_real_, 4), zy = rep(NA_real_, 4),
                              rho = rep(NA_real_, 4)))
  # The model LABEL is best_model$model (e.g. "GRT-{PI, PS, DS}"), matching
  # fit_baselines.R. rownames(hm$table)[1] is the winning row's INDEX into grtools'
  # own 12-model hierarchy, which is not a label and is not comparable across fits.
  list(ok = isTRUE(conv == 0), secs = secs,
       model = paste(hm$best_model$model, collapse = "/"),
       conv = conv,
       zx = p$zx[grtools_perm], zy = p$zy[grtools_perm], rho = p$rho[grtools_perm])
}

row_of <- function(name, n_trials, rep_id, tps, m, g, cmat = NULL) {
  cm_cols <- if (is.null(cmat)) rep(NA_real_, 16) else as.numeric(t(cmat))
  tibble(dataset = name, n_trials = n_trials, rep = rep_id, tps_target = tps,
         !!!setNames(as.list(cm_cols),
                     paste0("cm_", rep(0:3, each = 4), rep(0:3, times = 4))),
         mdsdt_ok = m$ok, mdsdt_secs = m$secs, best_model = m$model,
         !!!setNames(as.list(m$zx),  paste0("mdsdt_zx_", 0:3)),
         !!!setNames(as.list(m$zy),  paste0("mdsdt_zy_", 0:3)),
         !!!setNames(as.list(m$rho), paste0("mdsdt_rho_", 0:3)),
         grtools_ok = g$ok, grtools_secs = g$secs, grtools_model = g$model,
         grtools_conv = if (is.null(g$conv)) NA_integer_ else g$conv,
         !!!setNames(as.list(g$zx),  paste0("grtools_zx_", 0:3)),
         !!!setNames(as.list(g$zy),  paste0("grtools_zy_", 0:3)),
         !!!setNames(as.list(g$rho), paste0("grtools_rho_", 0:3)))
}

candidates <- c("thomas01a", "thomas01b", "silbert09a", "silbert09b", "silbert12",
                "thomas15a", "thomas15b")
rows <- list(); fits <- list(); subs <- list()
set.seed(20260826)

for (nm in candidates) {
  ok <- tryCatch({ data(list = nm, package = "mdsdt", envir = environment()); TRUE },
                 error = function(e) FALSE)
  if (!ok) { message("skip (not found): ", nm); next }
  cmat <- tryCatch(as.matrix(get(nm)), error = function(e) NULL)
  if (is.null(cmat) || !all(dim(cmat) == c(4, 4))) {
    message("skip (not a 2x2 design): ", nm); next
  }
  rows[[nm]] <- c(dataset = nm, as.numeric(t(cmat)))

  m <- fit_mdsdt(cmat); g <- fit_grtools(cmat)
  fits[[nm]] <- row_of(nm, sum(cmat), 0L, NA_real_, m, g, cmat)
  message(sprintf("%-11s n=%5d  mdsdt %-22s (%.1fs)  grtools %-22s (%.1fs, ok=%s)",
                  nm, sum(cmat), m$model, m$secs, g$model, g$secs,
                  paste0(g$ok, " conv=", g$conv)))

  # ---- thinning ------------------------------------------------------------
  props <- sweep(cmat, 1, rowSums(cmat), "/")
  for (tps in THIN_LEVELS) {
    for (r in seq_len(N_THIN)) {
      thin <- t(vapply(1:4, function(s) as.numeric(rmultinom(1, tps, props[s, ])),
                       numeric(4)))
      # n_reps=3 in the thinning arm only: the full-data fits above use grtools' default
      # of 10. Restarts guard against local optima, they are not the object of study here,
      # and 500 hierarchy fits at the default would dominate the runtime of this script.
      mm <- fit_mdsdt(thin); gg <- fit_grtools(thin, n_reps = 3)
      subs[[length(subs) + 1]] <- row_of(nm, sum(thin), r, tps, mm, gg, thin)
    }
    message(sprintf("   thinned to %3d/stimulus x %d reps", tps, N_THIN))
  }
}

if (length(rows) == 0) stop("No usable 2x2 datasets found in mdsdt.")
dir.create("data/real", recursive = TRUE, showWarnings = FALSE)
dir.create("results/mle_fits", recursive = TRUE, showWarnings = FALSE)

mat <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
colnames(mat) <- c("dataset", paste0("cm_", rep(0:3, each = 4), rep(0:3, times = 4)))
write_csv(mat, "data/real/real_matrices.csv")
write_csv(bind_rows(fits), "results/mle_fits/real_data_fits.csv")
write_csv(bind_rows(subs), "results/mle_fits/real_subsample_fits.csv")

message("\nwrote data/real/real_matrices.csv (", nrow(mat), " observers)")
message("wrote results/mle_fits/real_data_fits.csv")
message("wrote results/mle_fits/real_subsample_fits.csv (",
        length(subs), " thinned fits)")
message("next:  python scripts/compare_real_data.py")
