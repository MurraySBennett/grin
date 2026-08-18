# Interpreting GRIN output (a plain-language guide)

GRIN takes a confusion matrix from a 2×2 identification task and tells you how a
participant _perceives_ the stimuli. Here's how to read what it gives back.

## The numbers

**Sensitivity (per dimension), ~0 to ~3.** How well the participant tells the two
levels of a dimension apart — a d′-like quantity.

- **~0** — at chance; the levels look the same to them.
- **~1** — moderate; they get it right most of the time.
- **~2.5+** — excellent; near-perfect discrimination.
  If sensitivity on A is much higher than on B, they "see" A more clearly than B.

**Perceptual correlation ρ, −1 to +1.** Whether the two features are perceived
_independently_.

- **≈ 0** — independent (perceptual independence holds).
- **> 0** — the features are perceived as linked.
- **< 0** — the features trade off.

**Uncertainty (± and the shaded interval).** Every estimate comes with an error bar.
Wide bars mean the data are too sparse to pin the value down — collect more trials.
This is the whole point of GRIN: it tells you _how much to trust_ each number.

## The model label (e.g. `PI · PS · DS`)

The simplest GRT model consistent with the data:

- **PI** = perceptual independence (ρ = 0); **RHO1** = one shared correlation; **free** = correlations differ.
- **PS** = both dimensions separable; **PS(A)/PS(B)** = only that dimension; **none** = neither.
- **DS** = decisional separability (assumed throughout).

Two cautions: model identification needs **enough trials** to be reliable — with few
trials the label is a best guess. And the perceptual-vs-decisional distinction is
resolved here by convention (decisional separability), not measured — pinning it down
requires extra experimental conditions.

## The envelope check (`envelope_deviance`)

This is **not** a test of whether GRT describes the participant — a single confusion
matrix almost always has *some* GRT-Gaussian parameters that reproduce it exactly, so
that question can't be answered from a matrix alone (lapses, a strategy shift, and
non-standard responding all still just produce some ordinary-looking matrix, fittable
like any other). What this actually checks is narrower and more mechanical: does
GRIN's own fitted answer reproduce this specific matrix well.

**Good (low deviance)** — GRIN's fit reproduces the matrix; nothing more to check.
**Flagged (high deviance)** — GRIN's fit does *not* reproduce the matrix well, most
often because the matrix falls outside the range of data the network was trained on
(very extreme sensitivity, an unusual trial-count regime, or similar). Treat the
estimate with more caution and, if it matters, cross-check with a direct
maximum-likelihood fit — not because GRT has been shown not to apply, but because
GRIN's amortised answer may be extrapolating.

## Rule of thumb

Read it as: _"They discriminate A [well/poorly] and B [well/poorly], perceive the
features [independently/together], the data fit a [model] model, and the envelope
check is [clear/flagged] — with [tight/loose] certainty given the trial count."_
