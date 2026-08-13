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

The matrix must be in canonical order: rows and columns `A1B1, A1B2, A2B1, A2B2`
(dimension A varies slowest, B fastest). A bare matrix in unknown order is refused
rather than guessed, because a silent row/column swap returns a confident wrong
posterior. Resolve it explicitly:

```python
gt.to_confusion(M, order="canonical")      # you assert it is already canonical
gt.to_confusion(M, stim_labels=[...], resp_labels=[...],
                factor_a=("Old", "Young"), factor_b=("Neg", "Pos"))   # let it permute
gt.describe(M, order="canonical")          # print exactly what the model will see
```

Proportions (rows summing to 1) are refused unless you also pass `trials=[...]`,
because the model reads per-stimulus trial totals as a second input and proportions
would destroy the posterior's uncertainty.

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

gtplot.plot_space(result)                # perceptual space: means + correlation ellipses
gtplot.plot_params(result)                # all 12 estimates, dot-and-whisker with 90% CIs
gtplot.plot_constructs(result, constructs)  # P(PI)/P(RHO1)/P(free), P(separable A/B)
```

Group level (many participants — collect `gt.infer()` results into a list;
`gtplot.tidy()` is the shared foundation if you want the raw DataFrame instead
of a plot):

```python
sample = [gt.infer(M1), gt.infer(M2), gt.infer(M3)]

gtplot.tidy(sample)                   # one row per participant: estimates, SDs, constructs
gtplot.plot_space_group(sample)       # one panel per participant (facet=False overlays instead)
gtplot.plot_params_group(sample)      # per-parameter distribution across the sample
gtplot.plot_model_classes(sample)     # how many participants landed in each GRT model class
gtplot.plot_precision_group(sample)   # posterior SD distribution -- data quality across the sample
```

Same visual identity as `grin` (the R package) and the paper's own figures --
a figure made with either package reads as the same family. Bars/labels for a
construct the data can't decide (`evidence_* == False`) are visually flagged
rather than plotted as if informative, same principle as the stopping-rule
API's `blocked_by`.

**Style**: black-on-white by default (publication- and greyscale-safe). Pass
`color=True` to any plot function for the house blue/rose style instead, or
set it once for the session with `grintools.plot.DEFAULT_COLOR = True`. Either
way, `plot_space()` never splits one participant's four stimuli into four
colours -- with everything else in the plot fixed, that's decoration, not
information, so stimuli are told apart by a label instead.

## Model provenance

Each release bundles one specific trained `.onnx`. The package version pins the
model, so `grintools==X.Y.Z` corresponds to a fixed set of weights with known
recovery and calibration behaviour (see the project's validation suite).
