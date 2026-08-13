# SBI workshop poster — 48 × 36 in landscape

Source for the GRIN poster. Everything is plain LaTeX; no poster class, because
`tikzposter` / `beamerposter` block grids fight the full-width hero figure.

```
presentations/sbi_workshop/
├── SBI_workshop_poster.tex              the document — edit the copy here
├── grinposter.sty          palette, type scale, panel commands
├── pipeline.tex            the hero figure (TikZ)
├── pipeline_preview.tex    compile this to iterate on the hero alone
├── make_poster_figures.py  re-renders the four figures at poster scale
├── latexmkrc
└── figures/                poster-scale figures land here (gitignored)
```

## Build

```bat
cd presentations\sbi_workshop
latexmk -pdf SBI_workshop_poster.tex
```

MiKTeX will offer to install `sourcesanspro`, `qrcode`, and `tcolorbox` on the
first run — say yes. `pdflatex SBI_workshop_poster.tex` works too if you'd rather skip
latexmk. There are no cross-references, so **one pass is enough**.

While drafting the hero figure, compile it on its own — it is far faster and the
page is cropped to the figure's real footprint, so what you see is 1:1 with what
lands on the poster:

```bat
latexmk -pdf pipeline_preview.tex
```

## Figures

`SBI_workshop_poster.tex` searches `figures/` first, then `..\..\results\figures\`. So the
poster picks up the standard suite automatically, and anything you drop in
`figures/` overrides it.

**Run this once before you print** (from the project root, not this folder):

```bat
python presentations\sbi_workshop\make_poster_figures.py
```

An 8.6 in column is unforgiving. The manuscript figures are drawn at
`scale=1.0`; squeezed into a column and read from three feet, that type is far
too small. The script re-renders at `scale=1.8` and replaces the 12-panel
`recovery.png` with a legible **four-panel** variant (`poster_recovery.png`).
It also prints the recovery correlations and timings so you can confirm the
`\chk{}` numbers below.

If a figure is missing, the poster still compiles — you get a dashed placeholder
box on the page naming the file it wanted. That is deliberate: a missing figure
should be visible, not buried in the log.

## Layout

| band          | height   | contents                                             |
| ------------- | -------- | ---------------------------------------------------- |
| title         | ~6.4 in  | title, subtitle, authors, OSU logo, two QR codes      |
| **hero**      | ~10.1 in | the five-stage pipeline, full 45 in width             |
| five columns  | ~13.0 in | motivation · recovery · speed · limits · payoff       |
| footer        | ~1.3 in  | three references, contact, reproducibility line       |

It fits on one page with roughly 30 pt to spare, so **if you add a paragraph,
something else has to go**. Cheapest places to claw back space, in order:

1. `\bodyfont` in `grinposter.sty`: `{30}{40}` → `{28}{38}`. Still comfortably
   readable at three feet, and buys about half an inch per column.
2. `\parskip` in `SBI_workshop_poster.tex` (currently `0.11in`).
3. The hero's stage captions — they are already terse, and the columns carry the
   detail, so they can lose a line each.

Do **not** wrap `pipeline.tex` in `\resizebox`. Its coordinates are in cm at 1:1
so every font size in it is a real point size on the printed poster; scaling it
would silently break the type scale.

## Colour

The palette in `grinposter.sty` is copied from `src/viz/style.py`. **Those are
the only two places it lives — change both or neither.**

The board is cream (`#FBF7F0`), not white, because `style.py` sets
`savefig.facecolor = PAPER`. Every figure PNG carries a cream background; on a
white board each one would sit in a visible rectangle.

## Before you print

- [ ] **Confirm every `\chk{}` number.** They render highlighted in yellow until
      you set `\highlightchecksfalse` in `SBI_workshop_poster.tex`. Flip it *last*.
- [ ] **Confirm Kvam's affiliation** — I assumed Ohio State. It is inside a
      `\chk{}` for exactly this reason.
- [ ] **No `figure not found` placeholders** left on the page.
- [ ] `osu-logo.png` (or `.pdf`) dropped in this folder — otherwise you get a
      grey `[ osu-logo.png goes here ]` note.
- [ ] The QR code points at `grin.murraysbennett.com`. **If the site is not live,
      change it or remove it** — a dead QR on a poster is worse than none.
- [ ] Check the PDF at 100 % zoom, not fit-to-window. Fit-to-window hides
      exactly the type-too-small problems this poster is trying to avoid.
- [ ] Fonts embedded: `pdffonts SBI_workshop_poster.pdf` — every row should say `yes` under
      `emb`. pdflatex embeds by default, so this is a formality.

### Numbers currently marked `\chk{}`

| value                | where            | source                                  |
| -------------------- | ---------------- | --------------------------------------- |
| ~3 µs / ~90 ms       | hero stage 4     | `make_poster_figures.py` prints both     |
| r ≈ 0.98 / r ≈ 0.75  | column 2         | `make_poster_figures.py` prints both     |
| ~10⁴× , ~100 trials  | column 3         | `make_poster_figures.py` prints speed-up |
| ≈0.94 separability   | column 4         | validation v11                           |
| 0.88 / 0.97 / .60→.65| column 4         | validation v13, v14                      |
| 55 % / 46 %          | column 5         | the adaptive runs                        |
| Kvam's affiliation   | title band       | ask Peter                                |

## Print

No bleed is set — the cream `\pagecolor` covers the full 48 × 36 sheet, so the
board can be trimmed anywhere without a white edge appearing. If your printer
asks for bleed, add it in `geometry` rather than scaling the PDF: scaling would
change the physical type sizes, which are the one thing on this poster that has
been tuned deliberately.
