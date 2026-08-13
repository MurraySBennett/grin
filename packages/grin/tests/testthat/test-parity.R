# Numerical parity between the R (TorchScript) and Python (ONNX) runtimes.
#
# The two ship the same trained weights in different export formats
# (scripts/export_torchscript.py verifies the trace against the eager PyTorch
# wrapper at export time; this test pins the R side against reference values
# independently obtained from the Python grintools package on the same matrix,
# so a divergence here means the two runtimes have drifted apart, not that either
# is "correct" in isolation).
skip_if_not_installed("torch")
skip_if_not(isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE)),
           "libtorch is not installed (torch::install_torch())")

test_that("R inference matches the Python grintools reference to 1e-3", {
  M <- matrix(c(71, 17,  9,  5,
                20, 67,  5,  9,
                13,  6, 63, 20,
                 5, 10, 15, 71), nrow = 4, byrow = TRUE)
  out <- grin_infer(M)

  # From: python -c "import grintools as gt; gt.infer(M)" on the same matrix,
  # same released model version (see grintools/ in the main grin repo).
  ref_mean <- c(-1.1174, -1.1139, 0.9649, 0.9894, -0.6954, 0.7010, -0.6700, 0.7569,
                0.1236, -0.0126, 0.0140, 0.1247)
  ref_std  <- c(0.1617, 0.1576, 0.1608, 0.1673, 0.1638, 0.1603, 0.1598, 0.1605,
                0.1878, 0.1752, 0.1689, 0.1677)
  ref_p_corr <- c(0.8419, 0.1318, 0.0263)
  ref_p_sep  <- c(0.9481, 0.9597)

  expect_equal(out$result$params, ref_mean, tolerance = 1e-3)
  expect_equal(out$result$std, ref_std, tolerance = 1e-3)
  expect_equal(out$constructs$p_corr, ref_p_corr, tolerance = 1e-3)
  expect_equal(c(out$constructs$p_sep_A, out$constructs$p_sep_B), ref_p_sep, tolerance = 1e-3)
})
