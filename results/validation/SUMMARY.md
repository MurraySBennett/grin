# GRIN validation summary

| id | claim | status | sec |
|---|---|---|---|
| v01 | forward model is exact | PASS | 0.1 |
| v02 | prior coverage is broad and verified | PASS | 0.0 |
| v03 | parameter recovery | PASS | 14.6 |
| v04 | posterior is calibrated | PASS | 15.1 |
| v05 | amortized speedup vs MLE | PASS | 19.2 |
| v06 | stable across training seeds | PASS | 13.3 |
| v07 | reliability across trial counts | PASS | 16.3 |
| v08 | PI identifiability frontier (the honest limit) | PASS | 15.4 |
| v09 | out-of-family data is flagged | FAIL | 15.6 |
| v10 | degrades gracefully under lapses | PASS | 15.6 |
| v11 | amortized comparison vs AIC/BIC | PASS | 43.2 |
| v12 | RT and accuracy are collinear w.r.t. rho (why RTs can't fix PI) | PASS | 0.0 |
| v13 | RT gain (real but modest) | PASS | 11.3 |
| v14 | architecture recovery (5-way SFT); dimension-neglect detection | PASS | 7.5 |
| v15 | architecture is read from DISTRIBUTION SHAPE, not speed | PASS | 14.0 |
| v16 | LBA recovery; extra params cost no GRT identifiability | PASS | 7.1 |
