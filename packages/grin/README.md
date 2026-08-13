# grin

Amortised, uncertainty-calibrated inference for General Recognition Theory (GRT)
from 2x2 identification confusion matrices. Feed a confusion matrix, get a
calibrated posterior over the 12 GRT parameters plus construct probabilities
(perceptual independence, separability), and an optional stopping decision for
adaptive designs.

Runs natively via the [torch](https://torch.mlverse.org/) package (libtorch
bindings) — **no Python required**. This is the R companion to
[`grintools`](https://pypi.org/project/grintools/) (Python): both wrap the same
trained weights, in different export formats (TorchScript here, ONNX there), and
are numerically verified to agree (`tests/testthat/test-parity.R`).

## Install

```r
install.packages("torch")          # one-time; downloads libtorch
torch::install_torch()

# from GitHub until this is on CRAN:
# install.packages("remotes")
remotes::install_github("MurraySBennett/grin", subdir = "packages/grin")
```

## Use

```r
library(grin)

M <- matrix(c(71, 17,  9,  5,
              20, 67,  5,  9,
              13,  6, 63, 20,
               5, 10, 15, 71), nrow = 4, byrow = TRUE)   # rows = stimuli, cols = responses

out <- grin_infer(M)              # trials default to row sums
print(out$result)
out$constructs$p_PI
```

## The ordering contract

The matrix must be in canonical order: rows and columns `A1B1, A1B2, A2B1, A2B2`
(dimension A varies slowest, B fastest). A bare matrix in unknown order is refused
rather than guessed, because a silent row/column swap returns a confident wrong
posterior. Resolve it explicitly with [`grin_to_confusion()`]:

```r
grin_to_confusion(M, order = "canonical")     # you assert it is already canonical

grin_to_confusion(M, stim_labels = c("Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"),
                  resp_labels = c("Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"),
                  factor_a = c("Old", "Young"), factor_b = c("Neg", "Pos"))  # let it permute

grin_describe(M, order = "canonical")         # print exactly what the model will see
```

Proportions (rows summing to 1) are refused unless you also pass `trials = ...`,
because the model reads per-stimulus trial totals as a second input and proportions
would destroy the posterior's uncertainty.

## Stopping rules for adaptive designs

The rule is the experimenter's, declared as a `grin_criterion()` of targets:

```r
crit <- grin_criterion(list(
  grin_target_precision(params = c("zx", "zy"), sd_max = 0.10),   # want the space measured
  grin_target_probability("PS_A", at_least = 0.90)                # want the verdict
), combine = "any")

decision <- grin_evaluate(crit, out$result, out$constructs)
if (decision$stop) { ... }
```

`grin_target_precision()` stops when the parameter posterior is tight enough;
`grin_target_probability()` stops when a construct probability (`PI`, `PS_A`,
`PS_B`, or a `*_violated` complement) crosses a threshold; `combine` is `"all"` or
`"any"`.

Perceptual-independence questions are information-limited from a single confusion
matrix. Probability targets carry the model's evidence flags, so a threshold on a
construct the data cannot decide is reported in `decision$blocked_by` and does not
stop the loop. The limit is in the data, not the tool.

## Model provenance

The bundled TorchScript model is pinned to the package version: `packageVersion("grin")`
identifies exactly which trained weights produced a given inference, and matches
the model bundled with the same-versioned `grintools` Python release.

## Differences from the Python `grintools` API

R has no tuple-unpacking idiom, so `grin_infer()` returns one object with
`$result`/`$constructs` fields rather than a `(result, constructs)` pair. Function
names are `grin_`-prefixed (`grin_infer`, `grin_criterion`, `grin_target_precision`,
...) rather than bare (`infer`, `Criterion`, `Target.precision`, ...), following R
package convention and avoiding collisions with other packages' `infer()`/`Target()`.
Otherwise the API surface, argument names, and behaviour are a direct port.
