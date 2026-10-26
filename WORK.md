---
project: grin
---

# Work — grin

## Now

Figure 1's four stimulus images and provenance manifest are tracked locally in
commit `7941c7c` (2026-09-29), ready to include in the eventual public release.
The manuscript claim/citation fixes and full editorial review are recorded in
`../manuscripts/grin/WORK.md` and `docs/GRIN_editorial_review.md` there.
Release work remains in `docs/RELEASE.md`; no push, tag, release, or submission
was performed in this editorial pass. The existing vignette-script edit remains.

**2026-10-16, uncommitted — Figure 1 revision.** One stimulus swapped and six
presentation changes, all on the author's instruction.

- `A2B2` is now `ISIC_0046449` (was `ISIC_0022661`). Candidates were drawn by
  the rule already at `scripts/make_vignette_figure.py:26-29`, with ranking
  changed to the weaker of the two z-scores — a sum let an extreme border
  compensate for a middling colour. `A2B1` was swapped to `ISIC_5951430` and
  then **reverted** to `ISIC_0000022`: the replacement rendered too faint to
  carry "irregular border" at panel size. Do not re-propose it.
- `ISIC_0046449` is **CC-BY, not CC-0** — the first non-CC-0 stimulus in the
  set. The figure note at `../manuscripts/grin/GRIN_manuscript.tex:281` no
  longer claims the set is uniformly CC-0.
- Panel titles are now Identification Design / All constructs hold /
  Separability fails / Independence fails / Both fail / Decisional separability
  fails. The baseline panel's subtitle is gone, so `sub` is `None` there and
  `_draw_case` guards the text call.
- The "How to read each observer" key was **removed at the author's
  instruction** — scaffolding aimed at a reader assumed not to know GRT, in a
  specialist methods journal. The lower block of the design slot is now empty;
  the 1.05/1.0 split stays because it is what aligns the stimuli with the
  perceptual spaces beside them.
- Confusion matrices now carry two ramps: correct cells on `CMAP_SEQ`, errors on
  the new `CMAP_ERR` in `src/viz/style.py`. Both share vmin/vmax so hue encodes
  correct-vs-error and lightness encodes magnitude independently.
- `BOTH_PS_SHIFT = 1.3` makes separability fail harder in the Both-fail panel
  only. **This has a cost the author accepted:** that panel's A1B2 row is now
  22% correct against a 60% error cell, which is self-diagnosing by eye — the
  exact effect the `PS_SHIFT = 0.7` comment argues against for the
  separability-only panel. If it reads as too much, 1.0 is the middle option.

Figures are copied to the manuscript by hand; `build.sh` has no sync step. Note
`python3 scripts/<x>.py` from the repo root fails on `import src` — it needs
`PYTHONPATH=.`.

## Streams

### writing
- [x] 2026-10-02 **the within-session learning/dynamics direction is now
  `~/projects/latent-perceptual-grt`'s**, decided after a status audit, and is the
  designated next paper ahead of this repo's two other extension tracks (the RT
  model in `docs/dynamic_grt_rt_design.md`, blocked at gate 2 on a threshold
  decision; the 3×3 track in `docs/grin_3x3_development.md`, experimental). That
  project adopts `src/grt_model.py`'s parameterisation, `src/data/rt_dynamic_grt.py`
  and `bayesflow_port/`; its own `WORK.md` carries the detail. Prompted by Joe
  Houpt asking whether GRT can track perceptual learning rather than assume a
  static representation. Nothing to change in this repo or the manuscript.
- [x] 2026-09-29 tracked the four Figure 1 stimulus JPEGs and
  `data/vignette_stimuli/MANIFEST.json` in local commit `7941c7c`; the private
  manuscript now cites the exact path. Include these assets in the final release.
- [ ] next — Reconcile the figure/checkpoint provenance questions in
  `../manuscripts/grin/docs/GRIN_editorial_review.md` before the co-author draft;
  this requires existing records, not new analyses.
- [x] 2026-09-15 fill the final submission metadata: competing interests, preregistration, corresponding-author postal address, archive DOI
- [ ] now (S) **email Peter and Joe for co-author feedback — ask for their grant numbers** for the Funding declaration, which is still a placeholder
- [ ] now (S) inspect the rebuilt 50-page PDF visually, then decide whether the DS robustness addendum is supplement / reviewer-response only
- [ ] now (M) full pipeline/tooling push before the co-author send: tagged release, CRAN and PyPI distribution, then close out the Code availability declaration
  - [x] 2026-09-15 release-readiness audit, fixes and verification (see build stream)
  - [ ] now (S) register `grintools` on PyPI + add the repo as a trusted publisher
  - [ ] now (S) enable Zenodo on the repo, then tag `v0.1.0` and cut the GitHub release
  - [ ] now (S) submit `grin` to CRAN with `packages/grin/cran-comments.md`
  - [ ] next (S) once both land, rewrite Code availability (`GRIN_manuscript.tex:1176`) with the tag, the Zenodo DOI and the real install lines, and drop the bracketed placeholder
- [ ] next (S) add a "Murray S. Bennett is now at ..." present-address note to the author footnote once the next post is confirmed
- [ ] later review `presentations/grin_project_talks/MODULE_SPINE.md` and `TALK_PROFILES.md` before building any HTML deck

### build
- [x] 2026-09-15 readied the `decision_*` API in both packages (`for`/`against`/`undecided`, `evidence_tol`-width band, 0.25–0.75 undecided by default, R and Python verified identical at every boundary) plus the "componentwise modal structure" relabelling
- [x] 2026-09-15 grintools: deleted 7 pre-port flat-layout duplicates that were being hand-synced against `grintools/` and shadowed the installed package; LICENSE now ships in sdist and wheel; added CHANGELOG.md and Documentation/Issues URLs
- [x] 2026-09-15 fixed `publish.yml`: moved to `.github/workflows/`, scoped every step to `packages/grintools` (a bare `python -m build` there builds the ROOT package, since Actions runs from the repo root), added a tag-vs-version gate
- [x] 2026-09-15 renamed the root distribution `grin` → `grin-pipeline` with `Private :: Do Not Upload`; `grin` on PyPI belongs to an unrelated project
- [x] 2026-09-15 R package to CRAN standard: all 9 libtorch-dependent examples guarded (CRAN *runs* `\donttest{}`), vignette eval made conditional, `test-recalibration.R` was missing its libtorch guard, added `\value` ×6 and `\examples` ×16, `.Rbuildignore`, `cran-comments.md`, trimmed DESCRIPTION
- [x] 2026-09-15 fixed a latent roxygen bug: `model.R:109` had `#' @export` followed by an `#'`-prefixed *comment*, so `grin_infer`'s docs attached to `.grin_recalibration` and `man/grin_infer.Rd` was surviving only by hand-editing
- [x] 2026-09-15 added a deterministic `r-package-cran-conditions` CI job (libtorch deliberately absent) so the CRAN path is not left to whether a 500MB download happened to succeed. Took two fixes: the job must not override `args` (that drops `--no-manual` and fails on LaTeX the runner lacks), and it must *remove* libtorch rather than assert its absence -- `setup-r-dependencies` caches the R library and the sibling `r-package` job installs libtorch into it
- [ ] next (M) record what the web workstream delivered — live task, session persistence, tutorials and the stopping-savings report all landed unrecorded
- [x] 2026-10-26 **web: five correctness fixes behind the P1 copy pass** (`30d9106`
  rewrote the prose but left the claims underneath it unchecked).
  - `validate.html` headlined **92.8%** interval coverage. That figure is in
    `results/validation/manuscript_recovery.json:92` only, from `scripts/evaluate.py`,
    which pools all 12 parameters including the ~32,000 structural-zero correlations
    that `scripts/calibration_breakdown.py` excludes. It appears nowhere in the
    manuscript. Replaced with the paper's separate figures, 94.5% sensitivities /
    84.1% correlations (`GRIN_manuscript.tex:50-51`), and a note on why no pooled
    number is quoted — which is what `validate.js:258-278` already said.
  - `grt-io.js` training-envelope warning was dead on the low side: `n < tMin` with
    `tMin = 1` only fires at zero trials, so a 2-trial matrix passed silently while
    Home and Learn advertised the check. Added `validatedTrials: [5, 500]` (the
    `generator.py` default that `evaluate.py` and `calibration_breakdown.py` inherit)
    and split the warning into outside-the-prior vs outside-what-was-validated.
  - `checkpointProvenanceVerified` was `false` and the warning text called the web
    weights "legacy". Wrong: `web/assets/models/cm/npe_model.v1.0.0.onnx` is
    byte-identical (`d9ce8f1f…`) to the one `grintools` ships. Only the *package*
    copies were ever stale. Flag flipped, comment rewritten; nothing reads it.
  - `tutorial-existing-data.html` claimed "every output below is the real thing" but
    its fitted output predated the 2026-09-29 checkpoint correction. Re-ran the
    shipped package: counts reproduce exactly, fits do not (`zx_0` −0.85 → −0.77).
    `p_sep_A` moved 0.292 "undecided" → 0.184 "against", which **falsified the prose
    beneath it**; both updated.
  - Self-hosted Inter, Press Start 2P and the six decorative figure families to
    `web/assets/fonts/` (332 KB, OFL, LICENSE.txt beside them). Every page was
    fetching CSS and fonts from `fonts.googleapis.com`/`fonts.gstatic.com`, which
    disclosed visitor IP and page to a third party and contradicted "everything on
    this site runs in your browser". No third-party subresource remains anywhere.
- [ ] now (S) **web: the manuscript says the 24,000 test observers were "generated
  from the training prior" (`GRIN_manuscript.tex:490`). On trial counts they were
  not** — the prior is 1–1000 (`src/config.py:37`), the generator default used by
  both evaluation scripts is 5–500 (`src/data/generator.py:40`). Fix the manuscript
  sentence, not the code.
- [ ] next (S) web: `validate.html` is still absent from `components/nav.html` and is
  called "Validate" in its own `h1`, `learn.html` ×2 and `independence.html`, but
  "Evidence" on `index.html` ×2, `live.html` and `tutorial-install.html`. Pick one.
- [ ] next (M) web accessibility, untouched by the copy review because its extractor
  strips tags: `learn.html` goes h1→h3→h4 with no h2; `validate.html` has one h1 and
  no subheadings; `index.html`, `space-builder.html`, `validate.html`,
  `independence.html` and the tutorials have no `<main>` or skip link.

### infra
- [x] 2026-09-15 diagnosed the TeX build: WSL cannot build the manuscript at all (nix texlive-medium has no biblatex/apa7; the ~/texmf biblatex 3.22 fills the gap but crashes). Builds route through Windows MiKTeX via `manuscripts/grin/build.sh`
- [ ] later (M) if WSL builds are wanted, install a full texlive scheme and remove the shadowing `~/texmf` packages — the current half-install is the actual fault, not the `logreq.def` noted earlier

### admin
- [ ] now (M) work out what the remaining modified/untracked files are — `packages/` is now committed and accounted for; `src/`, `scripts/`, `results/`, `web/` still carry the same `decision_*`/componentwise reframing uncommitted, plus a stray `NUL` at the repo root (a Windows redirect artifact, safe to delete)
- [ ] later chase the missing `australian-apa.lbx` and the underfull boxes in the compact method-comparison tables
