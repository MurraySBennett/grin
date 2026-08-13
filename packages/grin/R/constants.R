CANON_STIM <- c("A1B1", "A1B2", "A2B1", "A2B2")
CANON_RESP <- c("a1b1", "a1b2", "a2b1", "a2b2")

#' GRT parameter names, in canonical model order
#'
#' The 12 identified parameters GRIN reports: four per-stimulus x-sensitivities
#' (`zx_0..zx_3`), four y-sensitivities (`zy_0..zy_3`), four within-stimulus
#' correlations (`rho_0..rho_3`), one per canonical stimulus (A1B1, A1B2, A2B1,
#' A2B2, in that order).
#' @export
PARAM_NAMES <- c(paste0("zx_", 0:3), paste0("zy_", 0:3), paste0("rho_", 0:3))

#' Named parameter groups (1-indexed positions into PARAM_NAMES)
#' @export
PARAM_GROUPS <- list(zx = 1:4, zy = 5:8, rho = 9:12)

.SPARSE_TRIALS <- 20
.PROP_TOL <- 1e-6
.INT_TOL <- 1e-8
