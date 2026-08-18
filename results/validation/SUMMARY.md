# GRIN validation summary

**This is the development/CI smoke-check ledger, not production-checkpoint
validation.** Every row below trains its own small, fresh network for that run
(2,500 examples/class, 20 epochs by default; v06 uses 1,200/12) -- none load the
released production checkpoint (`results/models/npe_model.pt`). See
`validation/README_validation.md` for the full scope note. The manuscript's
headline numbers come from separate, larger, production-checkpoint evaluations
(`results/validation/manuscript_recovery.json`, `results/rt_metrics.json`,
`results/mle_fits/baseline_fits.csv`), not from this table.

`GATE` below means the check has a stated quantitative pass/fail threshold and the
network was required to clear it. `REPORT` means there is no threshold -- the check
runs unconditionally and its "PASS" records only that it produced output, not that
anything was met.

| id | claim | status | type | sec |
|---|---|---|---|---|
| v01 | forward model is exact | PASS | GATE | 0.1 |
| v02 | prior coverage is broad and verified | PASS | GATE | 0.0 |
| v03 | parameter recovery | PASS | GATE | 7.3 |
| v04 | posterior is calibrated | PASS | GATE | 6.5 |
| v05 | ERROR | FAIL | GATE | 11.1 |
| v06 | stable across training seeds | PASS | GATE | 5.3 |
| v07 | reliability across trial counts | ran, reported | REPORT | 6.3 |
| v08 | PI identifiability frontier (the honest limit) | ran, reported | REPORT | 6.0 |
| v09 | data outside the trained envelope (reversed response mapping) is flagged, gradedly with severity, at a controlled false-alarm rate against in-envelope data | PASS | GATE | 6.0 |
| v10 | degrades gracefully under lapses | ran, reported | REPORT | 6.0 |
| v11 | ERROR | FAIL | REPORT | 35.5 |
| v12 | RT and accuracy are collinear w.r.t. rho (why RTs can't fix PI) | ran, reported | REPORT | 0.0 |
| v13 | RT gain (real but modest) | ran, reported | REPORT | 5.3 |
| v14 | architecture recovery (5-way SFT) | PASS | GATE | 3.4 |
| v15 | architecture is read from DISTRIBUTION SHAPE, not speed | PASS | GATE | 6.9 |
| v16 | LBA recovery; extra params cost no GRT identifiability | PASS | GATE | 3.4 |
