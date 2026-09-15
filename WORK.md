---
project: grin
---

# Work — grin

## Now

The tool is being made releasable ahead of the co-author send. Audit found that
three public claims were false: `pip install grintools` 404s, there are no git
tags at all, and `publish.yml` sat outside `.github/workflows/` so it had never
run -- and would have built the *root* package rather than grintools if it had.
The `decision_*` API the manuscript documents (`GRIN_manuscript.tex:440,1002`)
existed only in the working tree. All of that is now fixed and committed. Both
packages build and test clean; `R CMD check --as-cran` is clean both with libtorch
absent (CRAN's environment) and present (125 tests, 0 fail, 0 skip), and the
grintools wheel and sdist both install and infer correctly. Names verified free:
`grintools` on PyPI, `grin` on CRAN.
Remaining work is the part that needs credentials: register on PyPI, enable
Zenodo, tag, and submit to CRAN. Runbook is `docs/RELEASE.md` -> "Shipping the
packages".

## Streams

### writing
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
- [x] 2026-09-15 added a deterministic `r-package-cran-conditions` CI job (libtorch deliberately absent) so the CRAN path is not left to whether a 500MB download happened to succeed
- [ ] next (M) record what the web workstream delivered — live task, session persistence, tutorials and the stopping-savings report all landed unrecorded

### infra
- [x] 2026-09-15 diagnosed the TeX build: WSL cannot build the manuscript at all (nix texlive-medium has no biblatex/apa7; the ~/texmf biblatex 3.22 fills the gap but crashes). Builds route through Windows MiKTeX via `manuscripts/grin/build.sh`
- [ ] later (M) if WSL builds are wanted, install a full texlive scheme and remove the shadowing `~/texmf` packages — the current half-install is the actual fault, not the `logreq.def` noted earlier

### admin
- [ ] now (M) work out what the remaining modified/untracked files are — `packages/` is now committed and accounted for; `src/`, `scripts/`, `results/`, `web/` still carry the same `decision_*`/componentwise reframing uncommitted, plus a stray `NUL` at the repo root (a Windows redirect artifact, safe to delete)
- [ ] later chase the missing `australian-apa.lbx` and the underfull boxes in the compact method-comparison tables
