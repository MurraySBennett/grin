# grintools

Amortised Bayesian inference for General Recognition Theory (GRT) from 2x2
identification confusion matrices. Feed a confusion matrix, get a calibrated
posterior over the 12 GRT parameters plus construct probabilities (perceptual
independence, separability), and an optional stopping decision for adaptive designs.

Run time is torch-free: the trained network ships as an ONNX graph and runs under
`onnxruntime`. Retraining the pipeline re-exports the `.onnx`; this package is
otherwise unchanged.

## Install

```
pip install grintools
```

Dependencies are numpy and onnxruntime only. Re-exporting the model needs torch,
available as an extra: `pip install grintools[train]`.

## Use

```python
import grintools as gt

M = [[71, 17, 9, 5],
     [20, 67, 5, 9],
     [13, 6, 63, 20],
     [5, 10, 15, 71]]                      # rows = stimuli, cols = responses

result, constructs = gt.infer(M)           # trials default to row sums
print(result.summary())
print(constructs["p_PI"], constructs["p_sep_A"], constructs["p_sep_B"])
```

Or from the shell:

```
grin-fit --csv mydata.csv --construct PS_A --at-least 0.9 --sd 0.15 --combine any
```

## The ordering contract

`gt.infer()` needs to know which row/column of your matrix is which
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
explicitly:

```python
gt.to_confusion(M, order="canonical")      # you assert it is already canonical

# or let it permute a different order for you -- e.g. dimension A = Age (Old/Young),
# dimension B = Emotion (Neg/Pos): position 1 = Old+Neg, ..., position 4 = Young+Pos
gt.to_confusion(M, stim_labels=["Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"],
               resp_labels=["Old/Neg", "Old/Pos", "Young/Neg", "Young/Pos"],
               factor_a=("Old", "Young"), factor_b=("Neg", "Pos"))

gt.describe(M, order="canonical")          # print exactly what the model will see
```

Proportions (rows summing to 1) are refused unless you also pass `trials=[...]`,
because the model reads per-stimulus trial totals as a second input and proportions
would destroy the posterior's uncertainty.

`gt.to_confusion()` also accepts trial-level ("long format") data directly —
one row per trial with `stimulus`/`response` columns, no manual tallying — which
is the shape a PsychoPy or jsPsych/Pavlovia/Gorilla export already comes in:

```python
gt.to_confusion(trial_log, factor_a=("Old", "Young"), factor_b=("Neg", "Pos"), long=True)
```

See [`docs/data_collection.md`](../../docs/data_collection.md) for the full
walkthrough by platform (including running `grintools` live inside a PsychoPy
trial loop), and a note on where `grin` (R) fits in against this package for
post-hoc analysis.

## Response bias

Separate from separability/independence, and GRT (as a multidimensional
extension of signal detection theory) gives two ways to ask about it.
`gt.empirical_bias()` reads the raw tendency to favour one response over
another straight off the matrix, no model fit required:

```python
gt.empirical_bias(M)   # {'x_bias': ..., 'y_bias': ...} in [-0.5, 0.5]; 0 = unbiased
```

`gt.response_bias()` is the SDT-native version: the decision bound sits at 0
by convention, so an unbiased observer's two levels on a dimension are
mirror images about it and their identified z-scores average to zero -- a
nonzero average is a shifted decision criterion, read directly off a fit you
already have, uncertainty included:

```python
result, constructs = gt.infer(M)
gt.response_bias(result)   # {'x_bias', 'y_bias', 'x_bias_se', 'y_bias_se'}
```

## Stopping rules for adaptive designs

The rule is the experimenter's, declared as a `Criterion` of `Target`s:

```python
crit = gt.Criterion([
    gt.Target.precision(params=["zx", "zy"], sd_max=0.10),   # want the space measured
    gt.Target.probability("PS_A", at_least=0.90),            # want the verdict
], combine="any")
decision = crit.evaluate(result, constructs)
if decision.stop:
    ...
```

`Target.precision` stops when the parameter posterior is tight enough;
`Target.probability` stops when a construct probability (`PI`, `PS_A`, `PS_B`, or a
`_violated` complement) crosses a threshold; `combine` is `"all"` or `"any"`.

Perceptual-independence questions are information-limited from a single confusion
matrix. Construct targets carry the model's evidence flags, so a threshold on a
construct the data cannot decide is reported in `decision.blocked_by` and does not
stop the loop. The limit is in the data, not the tool.

## Plotting and reporting

See [`docs/quickstart.md`](docs/quickstart.md) for a full walkthrough (one
participant through a small sample) with rendered figures. Summary of the
API: needs the `[plot]` extra (`pip install grintools[plot]`, adds
matplotlib/pandas/scipy -- the core package stays torch-free and
dependency-light without it):

```python
import grintools.plot as gtplot

gtplot.plot_space(result)                   # perceptual space: means, correlation ellipses, error bars
gtplot.plot_params(result)                  # all 12 estimates, dot-and-whisker with 90% CIs
gtplot.plot_constructs(result, constructs)  # P(PI)/P(RHO1)/P(free), P(separable A/B)
gtplot.plot_bias(result)                    # decision-criterion response bias per dimension
gtplot.plot_diagnostics(result, M)          # predicted-vs-observed + marginal distributions
```

`plot_space()` also takes `title`, `xlabel`/`ylabel`, `stim_labels`,
`show_labels`, `show_uncertainty`, `show_marginals`, and `base_size`; see its
docstring for the full set. `plot_diagnostics()` needs the original matrix
alongside the fitted result, since `gt.infer()`'s return value doesn't carry
the input back out. `gtplot.plot_empirical_bias()` is the matrix-only,
no-fit-required counterpart to `plot_bias()` (see "Response bias" above).

Group level (many participants — collect `gt.infer()` results into a list;
`gtplot.tidy()` is the shared foundation if you want the raw DataFrame instead
of a plot):

```python
sample = [gt.infer(M1), gt.infer(M2), gt.infer(M3)]

gtplot.tidy(sample)                    # one row per participant: estimates, SDs, constructs
gtplot.plot_space_group(sample)        # one panel per participant -- the individual-level analysis GRT licenses
gtplot.plot_params_group(sample)       # per-parameter distribution across the sample
gtplot.plot_model_classes(sample)      # how many participants landed in each GRT model class
gtplot.plot_precision_group(sample)    # posterior SD distribution -- data quality across the sample
gtplot.plot_bias_group(sample)         # response bias distribution across the sample
```

`gtplot.plot_space_group(sample, facet=False)` overlays all participants with
a single across-participant mean ellipse instead of one panel each.
**This is an exploratory inspection view only, not a reporting figure**: GRT's
perceptual space is defined per observer, and a "grand mean" ellipse over
several independently-fitted spaces is not itself a fitted GRT model. Use it
to eyeball whether a sample looks homogeneous, not to report it; a warning is
raised every time this mode runs.

Same visual identity as `grin` (the R package) -- a figure made with either
package reads as the same family. Bars/labels for a construct the data can't
decide (`evidence_* == False`) are visually flagged rather than plotted as if
informative, same principle as the stopping-rule API's `blocked_by`.

**Style**: black-on-white by default (publication- and greyscale-safe). Pass
`palette="<name>"` to any plot function for one of a few built-in colour sets
(`gtplot.palette_names()` lists them, including `"contrast"`, the
colour-vision-deficiency-safe categorical palette of Okabe & Ito, 2008), your
own list of hex colours, or set `gtplot.DEFAULT_PALETTE = "<name>"` once for
the session. Either way, `plot_space()` never splits one participant's four
stimuli into separate colours -- stimuli are told apart by a label instead.

**Editing a figure further**: every plot function returns a plain matplotlib
`Axes` (or `Figure`, for the multi-panel ones), so ordinary matplotlib calls
apply directly rather than needing a package argument for every possible
customisation:

```python
ax = gtplot.plot_space(result)
ax.set_title("Participant 7, session 2")   # replaces the whole default title, subtitle included
ax.annotate("post-training session", (0, 2.3), ha="center", color="grey")
```

## Model provenance

Each release bundles one specific trained `.onnx`. The package version pins the
model, so `grintools==X.Y.Z` corresponds to a fixed set of weights with known
recovery and calibration behaviour (see the project's validation suite).
