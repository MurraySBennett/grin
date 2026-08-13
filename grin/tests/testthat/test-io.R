# Port of grintools' test_grin_io.py, against the same real confusion matrix.

M <- matrix(c(71, 17,  9,  5,
              20, 67,  5,  9,
              13,  6, 63, 20,
               5, 10, 15, 71), nrow = 4, byrow = TRUE)
FA <- c("Old", "Young")     # dimension A: A1=Old, A2=Young
FB <- c("Neg", "Pos")       # dimension B: B1=Neg, B2=Pos
LABELS <- c("Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos")

test_that("canonical assert leaves counts unchanged", {
  ci <- grin_to_confusion(M, order = "canonical")
  expect_true(all(ci$counts == M))
  expect_equal(as.integer(ci$trials), c(102, 101, 102, 101))
})

test_that("ordering guard repairs a scrambled, labelled matrix", {
  perm <- c(3, 1, 4, 2)
  Ms <- M[perm, perm]
  labs <- LABELS[perm]
  ci <- grin_to_confusion(Ms, stim_labels = labs, resp_labels = labs, factor_a = FA, factor_b = FB)
  expect_true(all(ci$counts == M))
  expect_equal(ci$placement[["A1B1"]], "Old/Neg")
})

test_that("a bare unlabelled, non-canonical matrix is refused", {
  perm <- c(3, 1, 4, 2)
  Ms <- M[perm, perm]
  expect_error(grin_to_confusion(Ms), "refus")
})

test_that("a labelled data.frame resolves to canonical", {
  perm <- c(3, 1, 4, 2)
  Ms <- M[perm, perm]
  labs <- LABELS[perm]
  df <- as.data.frame(Ms)
  names(df) <- labs
  ci <- grin_to_confusion(df, stim_labels = labs, factor_a = FA, factor_b = FB)
  expect_true(all(ci$counts == M))
})

test_that("aggregated long-format data resolves to canonical", {
  rows <- list()
  k <- 1
  for (i in 1:4) for (j in 1:4) {
    rows[[k]] <- list(LABELS[i], LABELS[j], M[i, j]); k <- k + 1
  }
  ci <- grin_to_confusion(rows, long = TRUE, factor_a = FA, factor_b = FB)
  expect_true(all(ci$counts == M))
})

test_that("trial-level long-format data resolves to canonical", {
  rows <- list()
  for (i in 1:4) for (j in 1:4) {
    n <- M[i, j]
    if (n > 0) for (t in seq_len(n)) rows[[length(rows) + 1]] <- list(LABELS[i], LABELS[j])
  }
  ci <- grin_to_confusion(rows, long = TRUE, factor_a = FA, factor_b = FB)
  expect_true(all(ci$counts == M))
})

test_that("proportions without trials are refused; with trials, rescaled to counts", {
  props <- M / rowSums(M)
  expect_error(grin_to_confusion(props, order = "canonical"), "PROPORTIONS")
  ci <- grin_to_confusion(props, order = "canonical", trials = c(102, 101, 102, 101))
  expect_true(all(ci$counts == M))
})

test_that("sparse and empty-cell warnings fire", {
  M_thin <- matrix(c(15, 0, 0, 0,  0, 14, 0, 1,  1, 0, 13, 0,  0, 0, 2, 12), nrow = 4, byrow = TRUE)
  ci <- grin_to_confusion(M_thin, order = "canonical")
  expect_true(any(grepl("only", ci$warnings) & grepl("trials", ci$warnings)))
  expect_true(any(grepl("empty cell", ci$warnings)))
})

test_that("describe reports readiness without raising", {
  perm <- c(3, 1, 4, 2)
  Ms <- M[perm, perm]
  labs <- LABELS[perm]
  rep <- grin_describe(Ms, printout = FALSE, stim_labels = labs, resp_labels = labs,
                       factor_a = FA, factor_b = FB)
  expect_true(rep$ready)

  rep2 <- grin_describe(Ms, printout = FALSE)
  expect_false(rep2$ready)
  expect_true(length(rep2$errors) > 0)
})
