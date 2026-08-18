# GRIN 3x3 development track

Status: experimental; no public API or validated checkpoint.

This track extends GRIN to two perceptual dimensions with three levels each. It
does not alter the released 2x2 model, packages, or checkpoint contract.

## Identified models

Two complementary models are implemented.

### Unit marginal variance

The response space is divided by two axis-aligned decision bounds per dimension.
Under decisional separability, the first bound is fixed at zero, the second is an
estimated positive spacing, and each stimulus distribution has unit marginal
variances. The canonical 29-vector contains 18 stimulus-specific locations, nine
within-stimulus correlations, and two boundary spacings.

A 9x9 confusion matrix has 72 independent probabilities. The 29-parameter full
Gaussian model is therefore not saturated. Unlike the current 2x2 envelope
deviance, 3x3 posterior-predictive or likelihood discrepancy can contain genuine
evidence against this Gaussian/DS family, once calibrated for finite trials and
network approximation.

### Free marginal variance

The heteroscedastic model instead fixes the two decision bounds to 0 and 1, which
sets latent location and scale, and estimates two means, two positive marginal
standard deviations, and one correlation for each stimulus. Its canonical vector
has 45 parameters. Under PS, both the corresponding means and variances are tied
across levels of the irrelevant dimension.

These are substantively different model families, not merely two coordinate systems:
unit variance asserts homoscedastic perceptual distributions, whereas the free model
allows stimulus-specific scale.

Both models assume decisional separability through globally shared, axis-aligned
decision bounds. Estimating departures from DS is outside this development track.

## Thomas (2015) empirical matrices

The bundled `thomas15a` and `thomas15b` objects are three-way xtabs with dimensions
`[nose response (3), eyes response (3), stimulus (9)]`, not stored 9x9 matrices. The
stimulus labels occur in source positions `1,4,7,2,5,8,3,6,9`. The export script
reorders by label into canonical stimulus order `1..9` and flattens each response
slice in nose-major, eyes-minor order. Each observer has 80 trials per stimulus and
720 trials total. The CSV retains each canonical row's source position and label.

## Experimental commands

Generate a small corpus:

```powershell
python scripts/generate_data_3x3.py --variance unit --n-per-class 100 --output data/simulated/grt_3x3_unit_smoke.npz
python scripts/generate_data_3x3.py --variance free --n-per-class 100 --output data/simulated/grt_3x3_free_smoke.npz
```

Train a smoke checkpoint:

```powershell
python scripts/train_3x3.py --data data/simulated/grt_3x3_unit_smoke.npz --output results/models/npe_3x3_unit_smoke.pt --epochs 2
python scripts/train_3x3.py --data data/simulated/grt_3x3_free_smoke.npz --output results/models/npe_3x3_free_smoke.pt --epochs 2
```

Production generation and training should run only after the prior and model-class
design pass simulation-based prior predictive review.

## Decisions required before a public model

- Validate the boundary-spacing prior against the Thomas et al. 3x3 matrices, with
  particular attention to empty/near-empty cells conditional on simulated SD magnitude.
- Compare the unit- and free-variance models through cross-model simulation: fit each
  checkpoint to data generated under both assumptions and quantify construct bias,
  interval coverage, and posterior-predictive detection of heteroscedasticity.
- Calibrate parameter and construct recovery by trial count, imbalance, accuracy,
  boundary proximity, and correlation magnitude.
- Add posterior predictive checks, since this model is not saturated.
- Maintain exact multinomial-likelihood reference fits for both 3x3 models. `mdsdt`
  stores the Thomas objects but its fitting and test functions are natively 2x2; it
  does not provide a 3x3 reference fitter. Our exact-likelihood implementation is
  therefore both the neural-posterior benchmark and the only fitting implementation
  currently available in this project for these matrices.
- Use MLE before MAP. The current priors are known to underrepresent Thomas-like
  low-accuracy/high-entropy data; adding their penalty now would silently turn that
  mismatch into regularization. Implement MAP only after the prior-predictive audit
  locks the priors.
- Limit the eventual exact-posterior benchmark to 14 matrices: six unit-variance
  simulations, six free-variance simulations, and the two Thomas observers. Within
  each variance model, select two simulations from each of three preregistered
  information strata: low (Thomas-B-like), moderate (Thomas-A-like), and high.
  Expanding that set requires an explicit compute decision.
- Report recovery and posterior-approximation diagnostics by variance model and
  information stratum. Thomas A and Thomas B must remain separate rows/anchors;
  a pooled statistic may be supplementary but can never be the sole headline.
- Treat the low-information stratum as an identifiability stress test. In particular,
  report SD and correlation recovery separately there: Thomas B's accuracy (.272) and
  mean row entropy (1.827; maximum ln(9)=2.197) place it close to the regime where
  spread parameters are expected to carry the least Fisher information.
- Restrict local-C2ST to that same benchmark set initially; it is a diagnostic with
  its own classifier-training cost, not a default full-grid metric.
- Add checkpoint/data hashes, export formats, cross-language parity, inference API,
  plotting, and documentation only after the statistical gates pass.
