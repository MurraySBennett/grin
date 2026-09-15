---
project: grin
---

# Work — grin

## Now

Two workstreams, only one of them recorded. The manuscript is at closeout:
review edits incorporated, figures regenerated (vignette as one contour per
stimulus, software comparison as a 2×2), a 46-page PDF rebuilt through
Windows MiKTeX. The settled position is *not* to present the DS-violation
analysis as a novel finding — cite `silbert2013decisional` and state that
GRIN inherits the GRT-level identifiability limit, keeping the addendum
available in case a reviewer asks. Meanwhile an entire web workstream landed
after `CHAT_STATE.md`'s last update on 2026-08-28 and was never recorded
there: the live task now runs as a real full-screen experiment with a
persisted session and a fade trail. 48 modified and 24 untracked files are
sitting in the tree with no record of what they are.

## Streams

### writing
- [x] 2026-09-15 fill the final submission metadata: competing interests, preregistration, corresponding-author postal address, archive DOI
- [ ] now (S) **email Peter and Joe for co-author feedback — ask for their grant numbers** for the Funding declaration, which is still a placeholder
- [ ] now (S) inspect the rebuilt 50-page PDF visually, then decide whether the DS robustness addendum is supplement / reviewer-response only
- [ ] next (M) full pipeline/tooling push before the co-author send: tagged release, CRAN and PyPI distribution, then close out the Code availability declaration
- [ ] next (S) add a "Murray S. Bennett is now at ..." present-address note to the author footnote once the next post is confirmed
- [ ] later review `presentations/grin_project_talks/MODULE_SPINE.md` and `TALK_PROFILES.md` before building any HTML deck

### build
- [ ] next (M) record what the web workstream delivered — live task, session persistence, tutorials and the stopping-savings report all landed unrecorded

### infra
- [x] 2026-09-15 diagnosed the TeX build: WSL cannot build the manuscript at all (nix texlive-medium has no biblatex/apa7; the ~/texmf biblatex 3.22 fills the gap but crashes). Builds route through Windows MiKTeX via `manuscripts/grin/build.sh`
- [ ] later (M) if WSL builds are wanted, install a full texlive scheme and remove the shadowing `~/texmf` packages — the current half-install is the actual fault, not the `logreq.def` noted earlier

### admin
- [ ] now (M) work out what the 48 modified and 24 untracked files are — the R package, grintools, results and scripts all carry uncommitted changes with no record
- [ ] later chase the missing `australian-apa.lbx` and the underfull boxes in the compact method-comparison tables
