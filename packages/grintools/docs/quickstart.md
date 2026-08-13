# Getting started with grintools

`grintools` turns a 2x2 identification confusion matrix into a calibrated
posterior over the 12 parameters of a General Recognition Theory (GRT)
perceptual model -- in milliseconds, via a neural network trained once and
shipped inside the package, rather than a per-participant maximum-likelihood
fit. This walks through a single participant end to end, then a small sample.

```
pip install grintools          # core: numpy + onnxruntime only
pip install grintools[plot]    # + matplotlib, pandas, scipy, for everything below
```

## One participant

A confusion matrix in **canonical order** -- rows and columns
`A1B1, A1B2, A2B1, A2B2` (dimension A varies slower than B):

```python
import grintools as gt

M = [[71, 17,  9,  5],
     [20, 67,  5,  9],
     [13,  6, 63, 20],
     [ 5, 10, 15, 71]]

result, constructs = gt.infer(M)
print(result.summary())
```

```
GRIN inference (onnx)
----------------------------------------------
  zx_0    = -1.12  +/- 0.16   [90% -1.38, -0.85]
  zx_1    = -1.11  +/- 0.16   [90% -1.37, -0.85]
  zx_2    = +0.96  +/- 0.16   [90% +0.70, +1.23]
  zx_3    = +0.99  +/- 0.17   [90% +0.71, +1.26]
  zy_0    = -0.70  +/- 0.16   [90% -0.96, -0.43]
  zy_1    = +0.70  +/- 0.16   [90% +0.44, +0.96]
  zy_2    = -0.67  +/- 0.16   [90% -0.93, -0.41]
  zy_3    = +0.76  +/- 0.16   [90% +0.49, +1.02]
  rho_0   = +0.12  +/- 0.19   [90% -0.19, +0.43]
  rho_1   = -0.01  +/- 0.18   [90% -0.30, +0.28]
  rho_2   = +0.01  +/- 0.17   [90% -0.26, +0.29]
  rho_3   = +0.12  +/- 0.17   [90% -0.15, +0.40]
----------------------------------------------
  most likely structure : PI + PS(A) + PS(B)
```

If your data isn't already in canonical order -- different stimulus/response
labelling, factors in a different sequence -- resolve it explicitly with
`gt.to_confusion()` rather than guessing; see the README. A silent
row/column swap would return a confident wrong posterior, so `gt.infer()`
trusts you've already done this.

Each of the 12 parameters comes back as an estimate with a posterior SD and a
90% interval: `zx_0`..`zx_3` and `zy_0`..`zy_3` are each stimulus's position
on dimensions A and B (sensitivity, in the identified/standardised space --
0 is chance, ~2.5+ is excellent discrimination), and `rho_0`..`rho_3` are
each stimulus's within-trial perceptual correlation between the two
dimensions (0 = independent, away from 0 = linked or traded off).
`result.model_class` is grintools' best guess at the underlying GRT
structure -- here, `"PI + PS(A) + PS(B)"`.

`constructs` carries the probabilities behind that label -- how likely the
correlation structure is perceptual independence (PI) / one shared
correlation (RHO1) / a free correlation per stimulus, and how likely each
dimension is perceptually separable:

```python
constructs["p_PI"]      # 0.842
constructs["p_sep_A"]   # 0.948
constructs["p_sep_B"]   # 0.960
```

`evidence_PI`/`evidence_sep_A`/`evidence_sep_B` flag whether this matrix
actually carries enough information to decide that construct at all --
perceptual independence in particular is information-limited from a single
matrix. A probability near 0.5 with `evidence_* = False` means "undecided,"
not "roughly 50/50 chance."

## Plotting one participant

Needs the `[plot]` extra.

```python
import grintools.plot as gtplot

gtplot.plot_space(result)
```

![Perceptual space plot for one participant: four stimulus points in the (zx, zy) plane with correlation ellipses](img/space_p1.png)

Four stimulus means in the (zx, zy) plane, one ellipse per stimulus showing
its correlation, dashed lines at the decision bounds (always at 0 in this
identified space -- that's the point of working in it). Stimuli are told
apart by a label, not a colour: with everything else about the plot fixed,
four colours for four points is decoration, not information. See "Style"
below for switching this on anyway.

```python
gtplot.plot_params(result)
```

![Forest plot of all 12 parameter estimates with 90 percent credible intervals](img/params_p1.png)

```python
gtplot.plot_constructs(result, constructs)
```

![Bar chart of construct probabilities: correlation structure and separability](img/constructs_p1.png)

Bars for a construct the data can't decide are visually flagged
("insufficient evidence") rather than drawn as if they were informative --
same principle as `evidence_*` above.

Every plot function returns a matplotlib `Axes` (`.figure` for the parent
figure), so it composes the normal way:
`gtplot.plot_space(result); ax.set_title("Participant 7")`.

### Style

Every plot function defaults to black-on-white (publication- and
greyscale-safe). Pass `color=True` for the house blue/rose style instead, or
set it once for the session with `grintools.plot.DEFAULT_COLOR = True`; an
explicit `color=` argument always overrides the module default.

```python
gtplot.plot_params(result, color=True)
```

## Stopping rules for adaptive designs

If you're collecting data adaptively -- more trials until you know enough --
declare what "enough" means as a `Criterion` of `Target`s, then check it
after each `gt.infer()`:

```python
from grintools.criterion import Criterion, Target

crit = Criterion([
    Target.precision(params=["zx", "zy"], sd_max=0.10),   # want the space measured
    Target.probability("PS_A", at_least=0.90),            # want the verdict
], combine="any")

decision = crit.evaluate(result, constructs)
print(decision.stop)         # True
for c in decision.checks:
    print(c)
```

```
True
{'met': False, 'value': 0.167, 'name': 'sd:zx_3', 'threshold': 0.1, 'reachable': True, 'note': ''}
{'met': True, 'value': 0.948, 'name': 'PS_A', 'threshold': 0.9, 'reachable': True, 'note': ''}
```

Here the precision target isn't met yet (`zx_3`'s SD is still above 0.10),
but the probability target is, so with `combine="any"` the loop stops. A
threshold on a construct the data cannot decide (an `evidence_* = False`
construct) is reported in `decision.blocked_by` and never stops the loop --
the limit is in the data, not the tool.

## Many participants

Collect `gt.infer()` results into a list and pass it straight to the
group-level tools:

```python
mats = {
    "p1": [[71, 17,  9,  5], [20, 67,  5,  9], [13,  6, 63, 20], [ 5, 10, 15, 71]],
    "p2": [[50, 10, 15, 25], [12, 55, 20, 13], [18, 22, 48, 12], [ 8, 14, 18, 60]],
    "p3": [[65, 20, 10,  5], [15, 70,  3, 12], [ 8,  4, 68, 20], [ 3, 10, 18, 69]],
    "p4": [[45, 40, 10,  5], [38, 47,  7,  8], [ 6,  9, 48, 37], [ 8,  7, 42, 43]],
    "p5": [[30, 28, 22, 20], [26, 29, 24, 21], [22, 25, 27, 26], [19, 23, 26, 32]],
    "p6": [[88,  5,  5,  2], [ 4, 90,  2,  4], [ 5,  3, 87,  5], [ 2,  4,  6, 88]],
}
sample = [gt.infer(M) for M in mats.values()]
```

`gtplot.tidy()` is the shared foundation every group plot builds on -- one
row per participant, in case you want the raw numbers instead of a plot:

```python
df = gtplot.tidy(sample, ids=list(mats.keys()))
df[["id", "model_class", "p_PI", "p_sep_A", "p_sep_B"]]
```

```
 id          model_class     p_PI  p_sep_A  p_sep_B
 p1   PI + PS(A) + PS(B) 0.841902 0.948095 0.959657
 p2 free + PS(A) + PS(B) 0.002157 0.632464 0.970686
 p3   PI + PS(A) + PS(B) 0.859414 0.944745 0.957630
 p4   PI + PS(A) + PS(B) 0.822008 0.941755 0.985177
 p5   PI + PS(A) + PS(B) 0.867445 0.971427 0.985978
 p6 free + PS(A) + PS(B) 0.118910 0.926840 0.925832
```

This toy sample is illustrative, not simulated from known ground truth, but
it's a real, if small, spread: two participants (`p2`, `p6`) come back with a
`free`-correlation structure rather than PI, and precision (the width of
each interval) varies with how the counts happen to fall. That variation is
exactly what the group plots below are for.

```python
gtplot.plot_space_group(sample, ids=list(mats.keys()))   # facet=True by default
```

![One perceptual-space panel per participant, faceted](img/space_group_facet.png)

One panel per participant (`facet=True`, the default) so individual
uncertainty stays visible. Set `facet=False` to overlay everyone instead,
with faint individual points and a labelled group-mean ellipse per stimulus
-- readable with more participants than a facet grid can hold, at the cost
of hiding each person's own uncertainty:

```python
gtplot.plot_space_group(sample, ids=list(mats.keys()), facet=False)
```

![All participants' stimulus points overlaid with group-mean ellipses](img/space_group_overlay.png)

```python
gtplot.plot_params_group(sample, ids=list(mats.keys()))
```

![Boxplots of each of the 12 parameters' estimates across participants](img/params_group.png)

```python
gtplot.plot_model_classes(sample, ids=list(mats.keys()))
```

![Bar chart of how many participants fall in each inferred GRT model class](img/model_classes.png)

```python
gtplot.plot_precision_group(sample, ids=list(mats.keys()))
```

![Boxplots of posterior SD per parameter group across participants -- data quality across the sample](img/precision_group.png)

`plot_precision_group()` is useful for planning: run it on pilot data to see
what posterior SD your trial counts are actually buying you, before picking
a `sd_max` for `Target.precision()`.

## Where next

- The package [README](../README.md) for the full API reference, and the R
  (`grin`) equivalent of every function here.
- [Interpreting GRIN output](https://github.com/MurraySBennett/grin/blob/main/docs/interpreting.md)
  -- a plain-language guide to reading the numbers above.
- [The GRT model specification](https://github.com/MurraySBennett/grin/blob/main/docs/GRT_model_spec.md)
  -- the underlying math: the 12 parameters, the model classes, the sign
  convention, decisional separability.
