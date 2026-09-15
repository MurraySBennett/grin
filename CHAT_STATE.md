# GRIN Manuscript Working State

Last updated: 2026-08-28.

## Current Goal / State

Manuscript edits from the external review are mostly incorporated. Revised figures have been regenerated in `results/figures/` and copied to `/home/msb/projects/manuscripts/grin/figures/`.

The decisional-separability robustness run is complete, but should be treated as a supplemental/reviewer-response or teaching artifact rather than a main-text result.

## Decisions Made

- Earlier context asked to set up, execute, and report the robustness addendum in the paper. The later/current decision supersedes that: keep the output available, but do not spend main-text space on DS as though it were a new finding.
- Do not present the DS-violation analysis as a novel robustness discovery.
- The manuscript should instead cite the known identifiability result, already in the bibliography as `silbert2013decisional`, and state that GRIN inherits this GRT-level limitation.
- Keep `results/figures/robustness_addendum.png` and `results/validation/robustness_addendum.json` available in case a reviewer asks.
- The software-comparison figure should use the revised 2x2 layout.
- The vignette figure should use one contour per stimulus, with no unexplained outer rings.
- DS non-identifiability teaching material belongs in `presentations/ds_nonidentifiability/`, separate from the journal manuscript unless a reviewer asks for it.
- The rejected first full project deck was deleted. The fresh content-first presentation kit is in `presentations/grin_project_talks/`.

## Files Changed

- `scripts/make_vignette_figure.py`: adds repo-root import path handling; simplifies stimulus contours to one ellipse per stimulus.
- `scripts/compare_to_r.py`: adds repo-root import path handling; changes comparison figure from 1x4 to 2x2.
- `scripts/robustness_addendum.py`: new reproducible robustness addendum for DS-bound tilt, lapse/contamination, and overdispersion.
- `results/validation/robustness_addendum.json`: generated addendum metrics.
- `results/figures/robustness_addendum.png`: generated addendum figure.
- `results/figures/vignette.png`: regenerated revised Figure 1.
- `results/figures/comparison_to_r.png`: regenerated revised software-comparison figure.
- `/home/msb/projects/manuscripts/grin/GRIN_combined_edited.tex`: removed stale note that the comparison figure still needed a 2x2 layout.
- `/home/msb/projects/manuscripts/grin/GRIN_combined_edited.tex`: added an APA author note and BRM-style declarations scaffold.
- `/home/msb/projects/manuscripts/grin/GRIN_combined_edited.tex`, `/home/msb/projects/manuscripts/grin/GRIN_combined_references.bib`, `/home/msb/projects/manuscripts/grin/references.bib`: audited checked bibliography entries and added confirmed DOIs.
- `/home/msb/projects/manuscripts/grin/REVIEWER_FEEDBACK_TRIAGE.md`: updated DS addendum status and next-step language.
- `/home/msb/projects/manuscripts/grin/MANUSCRIPT_REVIEW.md`: updated figure/addendum status.
- `/home/msb/projects/manuscripts/grin/GRIN_combined_edited.pdf`: rebuilt through Windows MiKTeX from a staged copy.
- `presentations/ds_nonidentifiability/`: added a self-contained HTML teaching slide deck with copied vignette/addendum assets and embedded DS robustness metrics.
- `presentations/grin_project_talks/`: added planning docs only: `README.md`, `MODULE_SPINE.md`, `TALK_PROFILES.md`, and `BUILD_PLAN.md`. Do not build the HTML deck until these are reviewed.

## Commands Run / Key Results

- `python scripts/make_vignette_figure.py`
  - Regenerated `results/figures/vignette.png`.
  - Matched overall accuracies remain around 48%.
- `python scripts/compare_to_r.py`
  - Regenerated `results/figures/comparison_to_r.png` as a 2x2 panel.
- `python scripts/robustness_addendum.py --n-per-class 150 --seed 20260828`
  - Wrote `results/validation/robustness_addendum.json`.
  - Wrote `results/figures/robustness_addendum.png`.
  - DS-bound tilt drives false construct evidence upward, as expected from known non-identifiability.
- Copied current `vignette.png`, `comparison_to_r.png`, and `robustness_addendum.png` into the manuscript `figures/` directory.
- `cmd.exe /c pdflatex ...`, `cmd.exe /c biber ...`, `cmd.exe /c pdflatex ...`
  - Built successfully in `/mnt/c/Users/bennett.1755/AppData/Local/Temp/grin-tex-build`.
  - Copied the fresh 46-page PDF and build sidecars back to `/home/msb/projects/manuscripts/grin/`.
  - Remaining warnings are MiKTeX update state, missing `australian-apa.lbx`, harmless hyperref empty-link warnings, and underfull boxes in compact method-comparison tables.

## Blockers

- WSL TeX still lacks `logreq.def`; use the Windows MiKTeX staging route for builds unless WSL TeX is fixed.
- Funding, Competing interests, and Preregistration statements need author-supplied final wording.
- Corresponding-author postal address should be checked before journal submission.
- Availability section still needs final DOI/release/package-version details.

## Exact Next Step

Continue with one of these manuscript-closeout paths:

1. fill final author/submission metadata: funding, competing interests, preregistration, archive DOI, and corresponding-author postal address, or
2. inspect the rebuilt PDF visually and decide whether to promote the robustness addendum to supplement/reviewer response only.

For the DS teaching deck, open `presentations/ds_nonidentifiability/index.html`.
For the full project presentation reset, review `presentations/grin_project_talks/MODULE_SPINE.md` and `presentations/grin_project_talks/TALK_PROFILES.md` before building HTML.
