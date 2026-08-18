# Export the bundled Thomas et al. (2015) 3-way xtabs to GRIN's canonical 9x9 order.
#
# Source arrays are [nose response, eyes response, stimulus], but the stimulus
# labels are stored as 1,4,7,2,5,8,3,6,9.  We reorder by the labels themselves;
# never reshape the third dimension positionally.  Within each response slice,
# as.numeric(t(slice)) gives nose-major/eyes-minor row-major response order.

inputs <- c(
  thomas15a = "data/mdsdt_data/review_format/thomas15a.rda",
  thomas15b = "data/mdsdt_data/review_format/thomas15b.rda"
)
expected_source_labels <- c("1", "4", "7", "2", "5", "8", "3", "6", "9")
canonical_labels <- as.character(1:9)
rows <- list()

for (dataset in names(inputs)) {
  env <- new.env(parent = emptyenv())
  load(inputs[[dataset]], envir = env)
  x <- env[[dataset]]
  if (!identical(dim(x), c(3L, 3L, 9L))) {
    stop(dataset, " is not a [3,3,9] response-by-stimulus xtabs")
  }
  source_labels <- dimnames(x)[["stim"]]
  if (!identical(source_labels, expected_source_labels)) {
    stop(dataset, " has unexpected stimulus labels/order: ",
         paste(source_labels, collapse = ","))
  }
  if (!identical(dimnames(x)[["nose"]], c("1", "2", "3")) ||
      !identical(dimnames(x)[["eyes"]], c("1", "2", "3"))) {
    stop(dataset, " has unexpected response labels")
  }

  source_index <- match(canonical_labels, source_labels)
  for (stimulus in seq_along(canonical_labels)) {
    response_counts <- as.numeric(t(x[, , source_index[[stimulus]]]))
    rows[[length(rows) + 1L]] <- data.frame(
      dataset = dataset,
      stimulus = stimulus,
      source_position = source_index[[stimulus]],
      source_label = source_labels[[source_index[[stimulus]]]],
      matrix(response_counts, nrow = 1L),
      check.names = FALSE
    )
  }
}

out <- do.call(rbind, rows)
names(out)[5:13] <- paste0("r", rep(1:3, each = 3), rep(1:3, times = 3))
if (!all(rowSums(out[, 5:13]) == 80)) stop("Thomas row totals are not all 80")
dir.create("data/real", recursive = TRUE, showWarnings = FALSE)
write.csv(out, "data/real/thomas15_3x3.csv", row.names = FALSE, quote = FALSE)
message("wrote data/real/thomas15_3x3.csv (", nrow(out), " rows; canonical stimulus order 1..9)")
