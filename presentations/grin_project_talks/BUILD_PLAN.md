# Build Plan For The Fresh HTML Deck

Do not implement this until `MODULE_SPINE.md` and `TALK_PROFILES.md` are
reviewed.

## Proposed Artifact

Build one local HTML presentation at:

```text
presentations/grin_project_talks/index.html
```

The deck should be a route-aware static app, not a generic slide dump.

## Required Capabilities

- Route presets:
  - Math Psych: 5, 10, 15 minutes.
  - Psychonomics: 5, 10, 15 minutes.
  - University-wide: 5, 10, 15 minutes.
  - Job talk segment: 8-12 minutes.
  - Postdoc/lab: 20 and 45 minutes.
  - Casual lab meeting: modular discussion route.
- Optional modules:
  - DS non-identifiability.
  - Technical simulator/prior details.
  - Software/demo.
  - Real-data examples.
  - Future research agenda.
- Presenter controls:
  - next/previous.
  - route selector.
  - optional module toggles.
  - presenter notes.
  - print/export-friendly mode.

## Design Direction

The deck should feel like a serious scientific talk, not a SaaS pitch and not a
generic AI-themed slideshow.

Visual rules:

- Use the actual figures where they carry the argument.
- Use diagrams only when they clarify a transformation or workflow.
- Avoid decorative gradients, stock imagery, and generic "AI" visuals.
- Put fewer words on slides than in the planning documents.
- Keep speaker notes substantive; keep visible text spare.

## Likely Asset Set

Copy local assets into `assets/` only after the final slide plan is approved.

Likely assets:

- `results/figures/vignette.png`
- `results/figures/recovery/recovery_grin.png`
- `results/figures/calibration_breakdown.png`
- `results/figures/comparison_to_r.png`
- `results/figures/accuracy_stratified.png`
- `results/figures/adaptive_stopping.png`
- `results/figures/real_data_spaces.png`
- `results/figures/real_data_params.png`
- `results/figures/robustness_addendum.png`

## Build Sequence

1. Murray reviews `MODULE_SPINE.md` and `TALK_PROFILES.md`.
2. Revise the narrative and route profiles.
3. Create a slide manifest as structured data: slide id, module, route
   membership, audience/venue variants, visible copy, notes, figure.
4. Build HTML/CSS/JS around the manifest.
5. Validate route counts and asset references.
6. Open in a browser or run screenshot checks if local browser tooling is
   available.

## Quality Bar

Every slide must answer these questions:

1. Why is this slide here?
2. What does the audience need to understand before the next slide?
3. What figure or visual evidence carries the point?
4. What should be said aloud that should not be printed on the slide?

If a slide cannot answer those, remove it.
