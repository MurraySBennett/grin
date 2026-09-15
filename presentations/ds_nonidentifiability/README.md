# Decisional Separability Non-identifiability Deck

Teaching slide deck for the DS non-identifiability issue discussed during the
GRIN manuscript revision.

Open `index.html` in a browser. The deck is self-contained and uses local assets
copied from the current analysis outputs:

- `assets/vignette.png`
- `assets/robustness_addendum.png`
- `assets/robustness_addendum.json`

The central demonstration is that a tilted A decision bound can be rewritten as
an ordinary vertical decision bound after transforming the perceptual coordinate:

```text
U = X - sY
V = Y
```

The plotted DS robustness numbers are from:

```text
python scripts/robustness_addendum.py --n-per-class 150 --seed 20260828
```

Recommended use: classroom, lab meeting, or reviewer response. The manuscript
should continue to cite the known identifiability result and leave this material
out of the main text unless requested.
