# Numerical parity between the R (TorchScript) and Python (ONNX) runtimes.
#
# The two ship the same trained weights in different export formats
# (scripts/export_torchscript.py verifies the trace against the eager PyTorch
# wrapper at export time; this test pins the R side against reference values
# independently obtained from the Python grintools package on the same matrix,
# so a divergence here means the two runtimes have drifted apart, not that either
# is "correct" in isolation).
#
# 2026-09-29: this test previously carried reference constants taken from the
# 2026-08-12 PRE-release checkpoint. Both packages shipped that checkpoint while
# the web app served v1.0.0, so the constants and the bundled model agreed with
# each other and the test passed for six weeks while the packages were wrong. Two
# changes follow from that. The constants below now record WHICH weights produced
# them, and the hash check runs first and without libtorch -- a wrong model is a
# packaging fault and must not be reported as a skip just because libtorch is
# absent on the machine.

test_that("the bundled model matches its provenance record", {
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("digest")
  path <- grin_default_model_path()
  sidecar <- file.path(dirname(path), "model_provenance.json")
  expect_true(file.exists(sidecar))

  rec <- jsonlite::fromJSON(sidecar)
  expect_identical(rec$model_file, basename(path))
  expect_identical(digest::digest(path, algo = "sha256", file = TRUE), rec$sha256)

  # Both packages must be built from the same training checkpoint. This is the
  # specific comparison whose absence let the drift ship.
  expect_true(nzchar(rec$checkpoint_sha256))
  expect_identical(rec$checkpoint_sha256,
                   "0f9deae1c5623b2403dc30241b09108368422ef1b3f444e92626eababc1f8fc0")
})

test_that("R inference matches the Python grintools reference to 1e-3", {
  skip_if_not_installed("torch")
  skip_if_not(isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE)),
             "libtorch is not installed (torch::install_torch())")

  M <- matrix(c(71, 17,  9,  5,
                20, 67,  5,  9,
                13,  6, 63, 20,
                 5, 10, 15, 71), nrow = 4, byrow = TRUE)
  out <- grin_infer(M)

  # Generated 2026-09-29 from grintools with the v1.0.0 ONNX,
  # sha256 d9ce8f1f5632b8aef5b47c5f37b01f75002384eb7389815280c2d2aa731da4b2,
  # itself exported from checkpoint npe_model.pt sha256 0f9deae1...fc0.
  # Regenerate with tests/gen_parity_reference.py whenever the weights change --
  # do not hand-edit, and do not copy values out of a package whose provenance
  # you have not checked.
  ref_mean <- c(-1.1052, -1.1379, 0.9450, 0.9921,
                -0.6745, 0.7170, -0.6622, 0.7400,
                0.1157, -0.0429, -0.0329, 0.1224)
  ref_std  <- c(0.1589, 0.1551, 0.1575, 0.1575,
                0.1560, 0.1530, 0.1494, 0.1578,
                0.1826, 0.1704, 0.1708, 0.1650)
  ref_p_corr <- c(0.8465, 0.1182, 0.0353)
  ref_p_sep  <- c(0.9625, 0.9600)

  expect_equal(out$result$params, ref_mean, tolerance = 1e-3)
  expect_equal(out$result$std, ref_std, tolerance = 1e-3)
  expect_equal(out$constructs$p_corr, ref_p_corr, tolerance = 1e-3)
  expect_equal(c(out$constructs$p_sep_A, out$constructs$p_sep_B), ref_p_sep, tolerance = 1e-3)
})
