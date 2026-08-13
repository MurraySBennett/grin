# grin 0.1.0

* Initial release: native R inference for GRIN via the `torch` package
  (TorchScript), numerically verified against the Python `grintools` package.
* `grin_infer()`, `grin_to_confusion()`, `grin_describe()`.
* Stopping-rule API: `grin_criterion()`, `grin_target_precision()`,
  `grin_target_probability()`, `grin_evaluate()`, `grin_stop_on_precision()`.
