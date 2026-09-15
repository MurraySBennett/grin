# Figure sizing convention

Every figure script in `scripts/`/`src/viz/` used by the manuscript follows this
convention, adopted 2026-08-31 after finding that every multi-panel figure was reading
several points smaller than intended once actually printed.

## The problem it fixes

Figures were originally authored at "poster/slide" width (11-16in figsize, since the
same plotting code is reused for posters and talks via `src/viz/style.py`'s `scale`
parameter) with 8-13pt fonts sized for that width. `GRIN_combined_edited.tex` embeds
every figure at `\includegraphics[width=\textwidth]`, and APA7's `man` class textwidth
on US letter (1in margins) is ~6.5in. A 12.8in-wide figure placed at 6.5in shrinks by
~0.5x, so a "10pt" tick label actually prints at ~5pt -- legible on a screen at 100%
zoom if you lean in, not at arm's length or on paper.

## The fix

**Native figsize width = intended print width, always.** Don't author a figure wide and
rely on LaTeX to shrink it down "close enough" -- match the figsize to whatever width it
will actually be embedded at, so the coded font sizes (already tuned for readability by
`src/viz/style.py::set_style`) come out at their true point size on the page.

```python
PRINT_W = 6.5   # \textwidth in GRIN_combined_edited.tex (APA7 man, US letter, 1in margins)
```

- A figure meant to run the full manuscript column width: `figsize=(PRINT_W, ...)`.
- A figure meant to sit two-up (half column): `figsize=(PRINT_W * 0.48, ...)`.
- Never save the "poster" size and let the manuscript build shrink it.

## Multi-panel figures: also save individually, and in more than one grid

For every figure with more than one thematically-separable panel, refactor the panel
drawing into standalone `_panel_x(ax, ...)` functions and call them into:

1. the combined manuscript figure, at whatever row/column layout reads best at `PRINT_W`
2. **the same panels individually**, each its own file at `PRINT_W` -- gives a journal's
   production team (or you, at revision time) the freedom to reposition/resize panels
   without regenerating anything or fighting a fixed composite image
3. **at least one alternative grid** (e.g. a 4-panel figure as 2x2 *and* 1x4, or a
   2-panel figure as 1x2 *and* 2x1) for whichever layout a target venue prefers

See `scripts/make_adaptive_figure.py` for the reference implementation.

## Narrow panels need their own text, not just smaller text

Shrinking a combined figure's total width means each panel's *column* is narrower even
though the coded font size didn't change -- a title or label that fit fine at 12in wide
can now be longer than the ~3in column it sits in, and will visibly run into the next
panel. `constrained_layout` does not reliably fix this for `axes.titlelocation="left"`
titles (the left-anchored title's true rendered width isn't well accounted for). Two
things actually fix it:

- **Shorter panel titles/axis labels in cramped layouts.** The full description belongs
  in the LaTeX caption anyway (every figure here already has one) -- a panel title only
  needs to be a short pointer ("A  Cost of reaching a target"), not a repeat of the
  caption's sentence.
- **Generous explicit padding**: `fig.tight_layout(w_pad=6.0)` (or `h_pad=` for stacked
  rows) rather than relying on the default spacing or on `constrained_layout`.

## Verifying it actually worked

Don't trust the raw PNG viewed on a screen -- an image viewer renders it at whatever
size fills the window, which hides exactly the shrink problem this convention exists to
fix. Simulate the actual printed size instead: resize to the print width at a normal
screen resolution (96dpi) and *then* look at it.

```bash
# PRINT_W inches at 96dpi = PRINT_W * 96 px
magick some_figure.png -resize $(python3 -c 'print(int(6.5*96))')x sim_some_figure.png
```

If text is cramped or colliding at that size, it will be cramped or colliding on the
actual page -- fix it before considering the figure done, not after a co-author or
reviewer notices in the PDF.
