# grin

Amortised, uncertainty-calibrated inference for General Recognition Theory (GRT)
from 2x2 identification confusion matrices. Feed a confusion matrix, get a
calibrated approximate posterior over the 12 GRT parameters plus construct
probabilities (perceptual independence, separability), and an optional stopping
decision for adaptive designs. "Calibrated" means checked against simulated
ground truth over the trained model's prior envelope (trial counts, parameter
ranges) -- see the validation studies in the accompanying paper for what that
covers and does not.

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

`grin_infer()` needs to know which row/column of your matrix is which
stimulus/response. **Canonical order** means: number the four stimuli 1-4 by
crossing dimension A's two levels with dimension B's two levels, A changing
slower than B:

| position | A level | B level | canonical label |
|:--:|:--:|:--:|:--:|
| 1 | 1 | 1 | `A1B1` |
| 2 | 1 | 2 | `A1B2` |
| 3 | 2 | 1 | `A2B1` |
| 4 | 2 | 2 | `A2B2` |

Row *i* and column *i* of your matrix must be stimulus/response *i* in this
table. A bare matrix in unknown order is refused rather than guessed, because a
silent row/column swap returns a confident wrong posterior. Resolve it
explicitly with [`grin_to_confusion()`]:

```r
grin_to_confusion(M, order = "canonical")     # you assert it is already canonical

# or let it permute a different order for you -- e.g. dimension A = Age (Old/Young),
# dimension B = Emotion (Neg/Pos): position 1 = Old+Neg, ..., position 4 = Young+Pos
grin_to_confusion(M, stim_labels = c("Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"),
                  resp_labels = c("Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"),
                  factor_a = c("Old", "Young"), factor_b = c("Neg", "Pos"))

grin_describe(M, order = "canonical")         # print exactly what the model will see
```

Proportions (rows summing to 1) are refused unless you also pass `trials = ...`,
because the model reads per-stimulus trial totals as a second input and proportions
would destroy the posterior's uncertainty.

`grin_to_confusion()` also accepts trial-level ("long format") data directly —
one row per trial with `stimulus`/`response` columns, no manual tallying —
which is the shape a PsychoPy or jsPsych/Pavlovia/Gorilla export already comes
in:

```r
grin_to_confusion(trial_log, factor_a = c("Old", "Young"), factor_b = c("Neg", "Pos"), long = TRUE)
```

See [`docs/data_collection.md`](../../docs/data_collection.md) for the full
walkthrough by platform, and a note on where `grin` (post-hoc analysis) fits
against `grintools`/in-browser inference (live, adaptive use).

## Scope: native 2×2 designs only

`grin` is built for a design that actually crosses two binary dimensions --
four stimuli, four responses. If your experiment has a third dimension (a
2×2×2 or larger identification design) and you're tempted to hand `grin` a
pairwise margin -- summing over the third dimension to get a 2×2 table for
each pair -- know that this is **not** equivalent to fitting the full
higher-dimensional GRT model. Marginalising over a dimension the observer
was actually attending to folds a mixture into what `grin` will read as a
single bivariate-normal representation, and the perceptual-independence and
separability conclusions it reports are about that pairwise projection, not
about the design as a whole. Treat a pairwise margin as a descriptive summary
if you use one at all, not as a native `grin` analysis of a 2×2×2 experiment.

## Response bias

Separate from separability/independence, and GRT (as a multidimensional
extension of signal detection theory) gives two ways to ask about it.
`grin_empirical_bias()` reads the raw tendency to favour one response over
another straight off the matrix, no model fit required:

```r
grin_empirical_bias(M)   # $x_bias, $y_bias in [-0.5, 0.5]; 0 = unbiased
```

`grin_response_bias()` is the SDT-native version: the decision bound sits at
0 by convention, so an unbiased observer's two levels on a dimension are
mirror images about it and their identified z-scores average to zero -- a
nonzero average is a shifted decision criterion, read directly off a fit you
already have, uncertainty included:

```r
grin_response_bias(out$result)   # $x_bias, $y_bias, plus $x_bias_se/$y_bias_se
```

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

## Plotting and reporting

See `vignette("grin")` for a full walkthrough (one participant through a small
sample) with rendered figures. Summary of the API — individual level (one
participant):

```r
grin_plot_space(out$result)                     # perceptual space: means, correlation ellipses, error bars
grin_plot_params(out$result)                     # all 12 estimates, dot-and-whisker with 90% CIs
grin_plot_constructs(out$result, out$constructs) # P(PI)/P(RHO1)/P(free), P(separable A/B)
grin_plot_bias(out$result)                       # decision-criterion response bias per dimension
grin_plot_diagnostics(out$result, M)             # predicted-vs-observed + marginal distributions
```

`grin_plot_space()` also takes `title`, `xlab`/`ylab`, `stim_labels`,
`show_labels`, `show_uncertainty`, `show_marginals`, and `base_size` — see
`?grin_plot_space` for the full set. `grin_plot_diagnostics()` needs the
original matrix alongside the fitted result, since `grin_infer()`'s return
value doesn't carry the input back out. `grin_plot_empirical_bias()` is the
matrix-only, no-fit-required counterpart to `grin_plot_bias()` (see
"Response bias" above).

Group level (many participants — loop `grin_infer()` over a sample and pass the
list straight in; `grin_tidy()` is the shared foundation if you want the raw
data.frame instead of a plot):

```r
sample <- list(p01 = grin_infer(M1), p02 = grin_infer(M2), p03 = grin_infer(M3))

grin_tidy(sample)                   # one row per participant: estimates, SDs, constructs
grin_plot_space_group(sample)       # one panel per participant -- the individual-level analysis GRT licenses
grin_plot_params_group(sample)      # per-parameter distribution across the sample
grin_plot_model_classes(sample)     # how many participants landed in each GRT model class
grin_plot_precision_group(sample)   # posterior SD distribution -- data quality across the sample
grin_plot_bias_group(sample)        # response bias distribution across the sample
```

`grin_plot_space_group(sample, facet = FALSE)` overlays all participants with a
single across-participant mean ellipse instead of one panel each.
**This is an exploratory inspection view only, not a reporting figure**: GRT's
perceptual space is defined per observer, and a "grand mean" ellipse over
several independently-fitted spaces is not itself a fitted GRT model. Use it to
eyeball whether a sample looks homogeneous, not to report it; `grin` prints a
reminder every time this mode runs.

Every plot is a `ggplot` object (`theme_grin()` is exported), so it composes the
normal way. Bars/labels for a construct the data can't decide (`evidence_* ==
FALSE`) are visually flagged rather than plotted as if informative — same
principle as the stopping-rule API's `blocked_by`.

**Style**: black-on-white by default (publication- and greyscale-safe). Pass
`palette = "<name>"` to any `grin_plot_*()` for one of a few built-in colour
sets (`grin_palette_names()` lists them, including `"contrast"`, the
colour-vision-deficiency-safe categorical palette of Okabe & Ito, 2008), your
own character vector of hex colours, or set `options(grin.palette = "<name>")`
once for the session. Either way, `grin_plot_space()` never splits one
participant's four stimuli into separate colours — stimuli are told apart by a
label instead.

**Editing a figure further**: every `grin_plot_*()` returns a plain `ggplot`
(or, for the multi-panel functions, a `patchwork`) object, so ordinary ggplot2
calls apply directly rather than needing a package argument for every possible
customisation:

```r
grin_plot_space(out$result) +
  ggplot2::labs(title = "Participant 7, session 2", subtitle = NULL) +
  ggplot2::annotate("text", x = 0, y = 2.3, label = "post-training session")
```

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
Plotting functions take `palette = "<name>"` in both languages; otherwise the API
surface, argument names, and behaviour are a direct port.
