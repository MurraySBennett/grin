# Missing model modules referenced by scripts

These modules are imported by scripts but were not in the original directory:

- customised_gate_control_model.py  (exposes `Expert`, `Gate` custom Keras layers)
    referenced by: scripts/model_comparison.py, src/utils/model_comparison_funcs.py,
                   scripts/visualise_pretraining.py
- independent_separate_param_loss.py
    (exposes `convert_targets_to_scales_corr`, `scale_corr_loss`,
     `convert_scales_corr_to_cov`, `make_regression_losses`)
    referenced by: scripts/pretrain_parameters.py, scripts/visualise_pretraining.py

`model_comparison*.py` guard this import with try/except (they degrade gracefully),
but `pretrain_parameters.py` and `visualise_pretraining.py` import it directly and
will crash until it is restored.
