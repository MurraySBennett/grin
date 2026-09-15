## Test environments

* local: Ubuntu 22.04 (WSL2), R 4.5.3
* GitHub Actions: ubuntu-latest, R release

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## libtorch and conditional evaluation

Inference is performed by the 'torch' package, which downloads the libtorch
runtime on first use (`torch::install_torch()`) rather than at install time.
libtorch is therefore not present on your check machines.

The package is written so that this degrades cleanly rather than failing:

* Every example that performs inference is wrapped in `\donttest{}` **and**
  guarded by `if (torch::torch_is_installed())`, so the example code is parsed
  and run but performs no work when libtorch is absent.
* The vignette sets `knitr::opts_chunk$set(eval = <libtorch present>)`, so it
  builds either way and prints a short note explaining the omitted output when
  libtorch is unavailable.
* Tests that require inference call `skip_if_not()` on
  `torch::torch_is_installed()`.

Examples, vignette and tests all run in full on a machine where libtorch has
been installed; this is exercised in CI.

## Bundled model weights

`inst/models/npe_model_ts.pt` (332 kB) is the trained TorchScript network the
package exists to serve. The package version pins the weights, so a reported
inference can be traced to exactly the model that produced it. The installed
package is well under 5 MB.
