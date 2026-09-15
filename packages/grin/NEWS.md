# grin 0.1.0

* Initial release: native R inference for GRIN via the `torch` package
  (TorchScript), numerically verified against the Python `grintools` package.
* `grin_infer()`, `grin_to_confusion()`, `grin_describe()`.
* Construct outputs report direction explicitly: `decision_PI`,
  `decision_sep_A` and `decision_sep_B` each take the value `"for"`,
  `"against"` or `"undecided"`, with the width of the undecided band set by
  `evidence_tol`. The older `evidence_*` flags are retained, but they encode
  only whether the evidence was decisive, not its direction.
* Structure labels are reported as the *componentwise modal structure*: the
  mode of each head taken separately, not the mode of a jointly normalised
  twelve-class posterior.
* Stopping-rule API: `grin_criterion()`, `grin_target_precision()`,
  `grin_target_probability()`, `grin_evaluate()`, `grin_stop_on_precision()`.
* Plotting: individual (`grin_plot_space()`, `grin_plot_params()`,
  `grin_plot_constructs()`, `grin_plot_bias()`, `grin_plot_diagnostics()`) and
  group-level (`grin_plot_space_group()`, `grin_plot_params_group()`,
  `grin_plot_model_classes()`, `grin_plot_precision_group()`,
  `grin_plot_bias_group()`) reporting, plus `grin_tidy()`.
