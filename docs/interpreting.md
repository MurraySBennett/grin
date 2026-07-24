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

## The fit / OOD flag

**Good fit** — GRT describes this participant; trust the estimates.
**Questionable** — the matrix has structure no GRT-Gaussian model can produce (lapses,
a strategy shift, non-standard responding). Interpret with caution, or fall back to a
careful maximum-likelihood fit and inspect residuals.

## Rule of thumb

Read it as: _"They discriminate A [well/poorly] and B [well/poorly], perceive the
features [independently/together], the data fit a [model] model, and GRT [does/doesn't]
describe them — with [tight/loose] certainty given the trial count."_
