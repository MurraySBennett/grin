# fit_baselines.R — fit grtools and mdsdt to the SAME matrices GRIN was tested on.
#
#   Rscript scripts/R/fit_baselines.R
#
# Reads : data/simulated/test_set_for_R.csv   (from scripts/export_for_r.py)
# Writes: results/mle_fits/baseline_fits.csv  (exactly ONE row per matrix)
#
# Baselines are the SINGLE-PARTICIPANT fits:
#   * mdsdt   : fit.grt() over the FULL 12-model factorial hierarchy (PS on {both, A only,
#               B only, neither} x rho {independent/PI, same, free}), selected by AIC.
#               This is the SAME 12-node hierarchy grtools' grt_hm_fit() tests -- earlier
#               versions of this script only tested 5 of the 12 nodes (never the asymmetric
#               PS(A)-only / PS(B)-only cases), which made timing and agreement numbers
#               against grtools an apples-to-oranges comparison. Fixed below; see notes.
#   * grtools : grt_hm_fit() — fits the hierarchy of traditional GRT models to ONE matrix.
#               (NOT grt_wind_fit, which is the multi-participant GRT-wIND model.)
#
# We record the selected model, wall-clock time, and whether the fit CONVERGED. grtools'
# best_model$convergence is optim()'s own convergence code (0 = success); anything else
# (or a thrown R error) is counted as a failure.
#
# ORDERING (confirmed, not assumed): mdsdt::fit.grt()'s own docs require row/col order
# a_1b_1, a_1b_2, a_2b_1, a_2b_2 (B varies fastest) — this IS GRIN's canonical order, so
# cmat is passed to mdsdt unmodified. grtools' own source (grt_hm_fit.R @details) documents
# a DIFFERENT order, a_1b_1, a_2b_1, a_1b_2, a_2b_2 (A varies fastest) — positions 2 and 3
# swapped relative to mdsdt/GRIN. cmat is therefore permuted (rows AND cols) before every
# grt_hm_fit() call below; do not "simplify" this away, the two packages really do disagree.
#
# SPEED, two numbers per matrix for grtools: grt_hm_fit()'s default (10 random restarts per
# model in its hierarchy, to avoid local optima) and a forced single-restart run
# (n_reps = 1), so the "grtools is slow" comparison isn't conflating "bigger search" with
# "different package." mdsdt's calls below are also single-shot (no restarts) AND now fit
# the same 12 models as grtools, so grtools_1rep vs mdsdt_secs is a genuine apples-to-apples
# per-hierarchy comparison; grtools_secs is the honest cost of grtools' actual default
# (10x the restarts, on the same 12 models).
#
# PARAMETERS, so we can compare recovery (not just model class) against ground truth. BOTH
# now CONFIRMED against the actual package source (mdsdt's grt_base.R; grtools' grt_hm_fit.R
# and grt_hm_neglogliks.R), AND against a live run (column names below were corrected after
# a real fit showed dists' actual colnames are mu/sigma/nu/tau/rho, not mu_r/sd_r/mu_c/sd_c
# as the constructor's fallback default would suggest -- two_by_two_fit.grt() always sets
# colnames explicitly via cbind() before the grt() constructor's is.null() check ever fires):
#   mdsdt   -- winning fit's $dists is a 4x5 matrix (mu, sigma, nu, tau, rho), one row per
#              stimulus, rows already in GRIN's canonical order (no permutation needed).
#              mdsdt's 2x2 model hard-codes sigma=tau=1 for every stimulus (same as GRIN),
#              so mu/nu/rho map directly onto zx_i/zy_i/rho_i.
#   grtools -- best_model$means is a 4x2 matrix in grtools' own row order, with row 1
#              (A1B1) FIXED at (0,0) as the reference stimulus; best_model$a1/$a2 are the
#              two decision bounds (grtools estimates the bound and anchors stimulus 1,
#              the mirror image of mdsdt/GRIN which anchor the bound at 0 and float every
#              stimulus mean -- a pure choice of origin, not a different model: the
#              identification likelihood is shift-invariant, so this is an exact
#              re-expression, not an approximation). Unit variances are hard-coded
#              throughout, so converting is a clean shift: zx_i = means[i,1] + a1,
#              zy_i = means[i,2] + a2 (PLUS -- see the sign note in extract_grtools_params;
#              grtools' a1/a2 are the NEGATED bound positions, confirmed in matrixloglikC.cpp).
#              rho_i comes from best_model$covmat[[i]][1,2].
#              See extract_grtools_params() below. The raw (unshifted) bounds a1/a2 are
#              ALSO kept as grtools_bound_a1/a2 columns -- not needed for the comparison
#              itself, but useful for sanity-checking the shift after the fact rather than
#              trusting it silently (e.g. wildly unstable bounds across otherwise-similar
#              matrices would be a red flag worth investigating).

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(purrr)
  library(tibble)
})

have_mdsdt <- requireNamespace("mdsdt", quietly = TRUE)
have_grtools <- requireNamespace("grtools", quietly = TRUE)
if (!have_mdsdt) message("!! mdsdt is NOT installed  ->  install.packages('mdsdt')")
# install.packages("remotes")
# remotes::install_github("hawkrobe/mdsdt")
if (!have_grtools) message("!! grtools is NOT installed -> devtools::install_github('fsotoc/grtools')")
if (!have_mdsdt && !have_grtools) stop("Neither baseline package is installed.")

dat <- read_csv("data/simulated/test_set_for_R.csv", show_col_types = FALSE)
cm_cols <- paste0("cm_", rep(0:3, each = 4), rep(0:3, times = 4))

# GRIN/mdsdt order is a1b1,a1b2,a2b1,a2b2 (B fastest). grtools wants a1b1,a2b1,a1b2,a2b2
# (A fastest) -- i.e. positions 2 and 3 (1-indexed) swapped. Apply to BOTH rows and cols
# (stimuli and responses use the same convention) before any grtools call.
grtools_perm <- c(1, 3, 2, 4)

# RNG seed for the random-restart searches; see fit_one(). Change only deliberately,
# and report the value alongside any convergence rate derived from this script.
BASELINE_SEED <- 20260826L

# --- helper: collapse whatever grtools/mdsdt hands back into ONE scalar string ---
as_scalar <- function(x) {
  if (is.null(x) || length(x) == 0) {
    return(NA_character_)
  }
  x <- unlist(x, use.names = FALSE)
  paste(as.character(x), collapse = "|")
}

# --- CONFIRMED grtools parameter extraction (from the package's actual R source) -------
# grt_hm_fit() returns list(table=<AIC-ranked data.frame>, best_model=<list>). best_model$means
# is a 4x2 matrix (rows in grtools order A1B1,A2B1,A1B2,A2B2; row 1 fixed at (0,0) as the
# reference stimulus; col 1 = dimension A, col 2 = dimension B). best_model$a1/$a2 are the
# two decision bounds (ESTIMATED -- grtools floats the bound and anchors stimulus 1 at the
# origin, whereas GRIN/mdsdt fix the bound at the origin and float all 4 stimulus means).
# best_model$covmat[[i]][1,2] is rho for stimulus i; variances are hard-coded to 1 throughout
# (matches GRIN's fixed-unit-variance convention), so this is a clean shift, not an
# approximation: GRIN_zx_i = means[i,1] + a1, GRIN_zy_i = means[i,2] + a2.
# (PLUS: grtools' a1/a2 are negated bound positions -- see the sign note in the function.)
# Also returns the raw (unshifted) bounds a1/a2 -- useful as a diagnostic: they should be
# stable/sane-looking within a model class, and let you re-derive the shift by hand if the
# zx/zy columns ever look wrong, rather than trusting the shift silently.
extract_grtools_params <- function(hm) {
  bm <- hm$best_model
  # SIGN: PLUS, not minus. Verified in grtools' own src/matrixloglikC.cpp, which builds the
  # decision statistic as
  #     cons[s] = c(s,0) + sum_d b(s,d) * means(i,d)
  #     h[g]    = cons[g] + (b %*% P %*% z)[g]
  # and calls response a2 when h[0] >= 0, b2 when h[1] >= 0. With b = diag(2) the signed
  # distance from the bound is therefore means[i,d] + c[d]. grtools' c (surfaced as $a1/$a2)
  # is the NEGATED bound position -- the bound sits at x = -a1 -- which is the opposite of
  # what the name suggests. Subtracting instead of adding offsets every zx by exactly 2*a1
  # (and every zy by 2*a2), i.e. roughly 2-3 z-units: the recovery scatter keeps slope ~1 but
  # is bodily shifted, and the A1/B1 estimates come out POSITIVE when GRIN's sign convention
  # requires them to be negative. That is the check to run if this ever looks wrong again.
  zx <- as.numeric(bm$means[, 1]) + bm$a1
  zy <- as.numeric(bm$means[, 2]) + bm$a2
  rho <- vapply(bm$covmat, function(cm) cm[1, 2], numeric(1))
  list(zx = zx, zy = zy, rho = rho, a1 = as.numeric(bm$a1), a2 = as.numeric(bm$a2))
}

fit_one <- function(i) {
  row <- dat[i, ]
  cmat <- matrix(as.numeric(row[cm_cols]), nrow = 4, byrow = TRUE)

  # REPRODUCIBILITY. grt_hm_fit() searches from n_reps RANDOM starting points and reports
  # the winning fit's optim() convergence code, so the same matrix refitted under a
  # different RNG state can return a different code -- verified directly: one real matrix
  # refitted under 12 seeds returned code 0 on 10 of them and 52 on the other 2, while
  # selecting the identical model every time. Without a seed the reported convergence RATE
  # is therefore not reproducible, and re-running this script would legitimately produce a
  # different number. Seed per matrix (not once per run) so the result is invariant to the
  # order matrices are processed in and to whether the run is resumed partway.
  # See scripts/R/grtools_seed_stability.R for the quantification.
  set.seed(BASELINE_SEED + i)
  
  # ---------------- mdsdt ----------------
  md_ok <- FALSE
  md_secs <- NA_real_
  md_model <- NA_character_
  md_zx <- rep(NA_real_, 4); md_zy <- rep(NA_real_, 4); md_rho <- rep(NA_real_, 4)
  md_fzx <- rep(NA_real_, 4); md_fzy <- rep(NA_real_, 4); md_frho <- rep(NA_real_, 4)
  
  #' https://www.sciencedirect.com/science/article/pii/S0022249616300219?casa_token=Eu-Eh46aU60AAAAA:r7j2tavugWVnvqt_JqktUJUqPGwMtmrGz9xCkdwRmjZPBp0RPb9mbKf_OerYOT9JiOoinxNiRw#s000100
  #' STILL OPEN: what's the expectation for "proper testing"? AIC/BIC (below) answers
  #' "which model class", but mdsdt::anova.grt() / grtools::lr_test() answer a different,
  #' per-hypothesis question ("is PS/PI rejected") via nested likelihood-ratio tests. Decide
  #' whether we also want to report LRT rejection rates, not just AIC-selected class.
  #' RESOLVED: parameter extraction, for both packages, confirmed against actual source AND
  #' a live run (mdsdt's fit$dists columns are mu/sigma/nu/tau/rho; grtools'
  #' best_model$means/$a1/$a2/$covmat). See header + extract_grtools_params() above.
  #' RESOLVED: data ordering. mdsdt's docs require a1b1,a1b2,a2b1,a2b2 (matches GRIN's
  #' canonical order, no change needed); grtools' actual source (grt_hm_fit.R @details)
  #' requires a1b1,a2b1,a1b2,a2b2 -- different from mdsdt. See grtools_perm above; applied
  #' to every grt_hm_fit() call below.
  #' RESOLVED: mdsdt now fits the SAME 12-node hierarchy as grtools (PS on both/A-only/
  #' B-only/neither x PI/same-rho/free-rho), not the 5-node subset used previously, so
  #' mdsdt_secs vs grtools_1rep_secs and model-agreement checks are now comparing the same
  #' search space, not different ones.
  if (have_mdsdt) {
    t0 <- Sys.time()
    r <- tryCatch(
      {
        # Full 4x3 factorial: PS on {both, A only, B only, neither} x PI/rho {independence,
        # same-rho, free-rho}. This is the SAME 12-node hierarchy grt_hm_fit() tests in
        # grtools (see model_names in the grtools source) -- not a subset. Labels below
        # match grtools' own naming exactly, so mdsdt_model and grtools_model are directly
        # comparable strings, not just "does it contain PI".
        hierarchy <- list(
          list(PS_x = TRUE,  PS_y = TRUE,  PI = "all",       name = "{PI, PS, DS}"),
          list(PS_x = TRUE,  PS_y = FALSE, PI = "all",       name = "{PI, PS(A), DS}"),
          list(PS_x = FALSE, PS_y = TRUE,  PI = "all",       name = "{PI, PS(B), DS}"),
          list(PS_x = TRUE,  PS_y = TRUE,  PI = "same_rho",  name = "{1_RHO, PS, DS}"),
          list(PS_x = TRUE,  PS_y = FALSE, PI = "same_rho",  name = "{1_RHO, PS(A), DS}"),
          list(PS_x = FALSE, PS_y = TRUE,  PI = "same_rho",  name = "{1_RHO, PS(B), DS}"),
          list(PS_x = TRUE,  PS_y = FALSE, PI = "none",      name = "{PS(A), DS}"),
          list(PS_x = FALSE, PS_y = TRUE,  PI = "none",      name = "{PS(B), DS}"),
          list(PS_x = TRUE,  PS_y = TRUE,  PI = "none",      name = "{PS, DS}"),
          list(PS_x = FALSE, PS_y = FALSE, PI = "same_rho",  name = "{1_RHO, DS}"),
          list(PS_x = FALSE, PS_y = FALSE, PI = "all",       name = "{PI, DS}"),
          list(PS_x = FALSE, PS_y = FALSE, PI = "none",      name = "{DS}")
        )
        fits <- lapply(hierarchy, function(h) {
          mdsdt::fit.grt(cmat, PS_x = h$PS_x, PS_y = h$PS_y, PI = h$PI)
        })
        names(fits) <- vapply(hierarchy, function(h) h$name, character(1))
        aics <- vapply(fits, function(f) {
          tryCatch(as.numeric(mdsdt::GOF(f, teststat = "AIC")),
                   error = function(e) NA_real_
          )
        }, numeric(1))
        if (all(is.na(aics))) stop("all AICs NA")
        best_name <- names(which.min(aics))
        # $dists: 4x5 matrix (mu, sigma, nu, tau, rho), one row per stimulus, rows already
        # in GRIN's a1b1,a1b2,a2b1,a2b2 order -- direct mapping, no permutation. mu=x/A
        # mean, nu=y/B mean (confirmed against a live fit; see conversation notes).
        d <- fits[[best_name]]$dists
        # ALSO keep the UNCONSTRAINED {DS} fit. It is already in the hierarchy above, so
        # this costs nothing. Why bother: the selected model's parameters confound estimator
        # error with SELECTION error -- when a matrix is truly free-rho but {PI, PS, DS} wins
        # on AIC, the reported rho are exactly 0 and the MAE records a selection failure
        # dressed up as an estimation failure. The unconstrained fit separates the two.
        dfull <- fits[["{DS}"]]$dists
        list(ok = TRUE, model = best_name,
             zx = as.numeric(d[, "mu"]), zy = as.numeric(d[, "nu"]),
             rho = as.numeric(d[, "rho"]),
             full_zx = as.numeric(dfull[, "mu"]), full_zy = as.numeric(dfull[, "nu"]),
             full_rho = as.numeric(dfull[, "rho"]))
      },
      error = function(e) list(ok = FALSE, model = paste0("ERROR: ", conditionMessage(e)),
                               zx = rep(NA_real_, 4), zy = rep(NA_real_, 4), rho = rep(NA_real_, 4),
                               full_zx = rep(NA_real_, 4), full_zy = rep(NA_real_, 4),
                               full_rho = rep(NA_real_, 4))
    )
    md_secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    md_ok <- isTRUE(r$ok)
    md_model <- as_scalar(r$model)
    md_zx <- r$zx; md_zy <- r$zy; md_rho <- r$rho
    md_fzx <- r$full_zx; md_fzy <- r$full_zy; md_frho <- r$full_rho
  }
  
  # ---------------- grtools ----------------
  # IMPORTANT: grtools expects a1b1,a2b1,a1b2,a2b2, not GRIN/mdsdt's a1b1,a1b2,a2b1,a2b2.
  # Permute rows AND cols (stimuli and responses share the convention) before every call.
  cmat_gt <- cmat[grtools_perm, grtools_perm]
  
  # small helper so the default run and the n_reps=1 control run share one code path
  run_grtools <- function(...) {
    t0 <- Sys.time()
    r <- tryCatch(
      {
        hm <- grtools::grt_hm_fit(cmat_gt, ...)
        bm <- hm$best_model
        m <- as_scalar(bm$model)                       # e.g. "GRT-{PI, PS, DS}"
        conv_ok <- isTRUE(bm$convergence == 0)          # optim()'s own convergence code
        p <- extract_grtools_params(hm)
        # p's rows are in cmat_gt's (grtools) stimulus order; un-permute back to GRIN order.
        # grtools_perm is its own inverse (swapping positions 2,3 twice = identity), so the
        # same vector inverts it. a1/a2 are scalars (one shared bound per dimension, not
        # per-stimulus), so no permutation applies to them.
        list(ok = conv_ok, model = m,
             zx = p$zx[grtools_perm], zy = p$zy[grtools_perm], rho = p$rho[grtools_perm],
             a1 = p$a1, a2 = p$a2)
      },
      error = function(e) list(ok = FALSE, model = paste0("ERROR: ", conditionMessage(e)),
                               zx = rep(NA_real_, 4), zy = rep(NA_real_, 4), rho = rep(NA_real_, 4),
                               a1 = NA_real_, a2 = NA_real_)
    )
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    list(ok = isTRUE(r$ok), secs = secs, model = as_scalar(r$model),
         zx = r$zx, zy = r$zy, rho = r$rho, a1 = r$a1, a2 = r$a2)
  }
  
  gt <- list(ok = FALSE, secs = NA_real_, model = NA_character_,
             zx = rep(NA_real_, 4), zy = rep(NA_real_, 4), rho = rep(NA_real_, 4),
             a1 = NA_real_, a2 = NA_real_)
  gt1 <- gt   # n_reps = 1 control
  if (have_grtools) {
    gt <- run_grtools()             # grtools' actual default: full hierarchy x 10 restarts
    gt1 <- run_grtools(n_reps = 1)  # apples-to-apples vs mdsdt's single-shot calls above
  }
  
  # EXACTLY one row per matrix (all fields are guaranteed length-1 scalars, except the
  # per-stimulus parameter columns which are exactly length 4 each)
  tibble(
    row_id = as.integer(row[["row_id"]]),
    mdsdt_ok = md_ok, mdsdt_secs = md_secs, mdsdt_model = md_model,
    mdsdt_zx_0 = md_zx[1], mdsdt_zx_1 = md_zx[2], mdsdt_zx_2 = md_zx[3], mdsdt_zx_3 = md_zx[4],
    mdsdt_zy_0 = md_zy[1], mdsdt_zy_1 = md_zy[2], mdsdt_zy_2 = md_zy[3], mdsdt_zy_3 = md_zy[4],
    mdsdt_rho_0 = md_rho[1], mdsdt_rho_1 = md_rho[2], mdsdt_rho_2 = md_rho[3], mdsdt_rho_3 = md_rho[4],
    mdsdt_full_zx_0 = md_fzx[1], mdsdt_full_zx_1 = md_fzx[2], mdsdt_full_zx_2 = md_fzx[3], mdsdt_full_zx_3 = md_fzx[4],
    mdsdt_full_zy_0 = md_fzy[1], mdsdt_full_zy_1 = md_fzy[2], mdsdt_full_zy_2 = md_fzy[3], mdsdt_full_zy_3 = md_fzy[4],
    mdsdt_full_rho_0 = md_frho[1], mdsdt_full_rho_1 = md_frho[2], mdsdt_full_rho_2 = md_frho[3], mdsdt_full_rho_3 = md_frho[4],
    grtools_ok = gt$ok, grtools_secs = gt$secs, grtools_model = gt$model,
    grtools_zx_0 = gt$zx[1], grtools_zx_1 = gt$zx[2], grtools_zx_2 = gt$zx[3], grtools_zx_3 = gt$zx[4],
    grtools_zy_0 = gt$zy[1], grtools_zy_1 = gt$zy[2], grtools_zy_2 = gt$zy[3], grtools_zy_3 = gt$zy[4],
    grtools_rho_0 = gt$rho[1], grtools_rho_1 = gt$rho[2], grtools_rho_2 = gt$rho[3], grtools_rho_3 = gt$rho[4],
    grtools_bound_a1 = gt$a1, grtools_bound_a2 = gt$a2,
    grtools_1rep_ok = gt1$ok, grtools_1rep_secs = gt1$secs, grtools_1rep_model = gt1$model
  )
}

message("fitting ", nrow(dat), " matrices ...")
res <- map_dfr(seq_len(nrow(dat)), function(i) {
  if (i %% 10 == 0) message("  ", i, "/", nrow(dat))
  fit_one(i)
})
stopifnot(nrow(res) == nrow(dat)) # guard against the row-explosion bug

dir.create("results/mle_fits", recursive = TRUE, showWarnings = FALSE)
write_csv(res, "results/mle_fits/baseline_fits.csv")

message("\n--- summary (", nrow(res), " matrices) ---")
if (have_mdsdt) {
  message(
    "mdsdt   converged ", sum(res$mdsdt_ok), "/", nrow(res),
    "  (", round(100 * mean(!res$mdsdt_ok)), "% failure)  ",
    round(mean(res$mdsdt_secs, na.rm = TRUE), 2), " s/matrix"
  )
}
if (have_grtools) {
  message(
    "grtools          converged ", sum(res$grtools_ok), "/", nrow(res),
    "  (", round(100 * mean(!res$grtools_ok)), "% failure)  ",
    round(mean(res$grtools_secs, na.rm = TRUE), 2), " s/matrix",
    "  [default: full hierarchy x 10 restarts]"
  )
  message(
    "grtools (1 rep)  converged ", sum(res$grtools_1rep_ok), "/", nrow(res),
    "  (", round(100 * mean(!res$grtools_1rep_ok)), "% failure)  ",
    round(mean(res$grtools_1rep_secs, na.rm = TRUE), 2), " s/matrix",
    "  [n_reps=1, fair vs mdsdt's single-shot calls]"
  )
}
message("wrote results/mle_fits/baseline_fits.csv")
