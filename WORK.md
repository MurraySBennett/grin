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
- [ ] now (M) fill the final submission metadata: funding, competing interests, preregistration, archive DOI, corresponding-author postal address
- [ ] now (S) inspect the rebuilt 46-page PDF visually, then decide whether the DS robustness addendum is supplement / reviewer-response only
- [ ] next (S) complete the Availability section — final DOI, release, and package-version details
- [ ] later review `presentations/grin_project_talks/MODULE_SPINE.md` and `TALK_PROFILES.md` before building any HTML deck

### build
- [ ] next (M) record what the web workstream delivered — live task, session persistence, tutorials and the stopping-savings report all landed unrecorded

### infra
- [ ] next (S) fix WSL TeX's missing `logreq.def`, or keep routing builds through the Windows MiKTeX staging path

### admin
- [ ] now (M) work out what the 48 modified and 24 untracked files are — the R package, grintools, results and scripts all carry uncommitted changes with no record
- [ ] later chase the missing `australian-apa.lbx` and the underfull boxes in the compact method-comparison tables
