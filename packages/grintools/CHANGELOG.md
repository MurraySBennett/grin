# Changelog

## 0.1.0

Initial release. Amortised, uncertainty-calibrated GRT inference from a 2x2
identification confusion matrix, wrapping a trained network shipped as an ONNX
graph. Runtime dependencies are numpy and onnxruntime only; torch is not
required to use the package.

* `infer()`, `to_confusion()`, `describe()`, `default_model_path()`.
* Construct outputs report direction explicitly: `decision_PI`,
  `decision_sep_A` and `decision_sep_B` each take the value `"for"`,
  `"against"` or `"undecided"`, with the width of the undecided band set by
  `evidence_tol`. The older `evidence_*` flags are retained, but they encode
  only whether the evidence was decisive, not its direction.
* Structure labels are reported as the *componentwise modal structure*: the
  mode of each head taken separately, not the mode of a jointly normalised
  twelve-class posterior.
* Stopping-rule API: `Target`, `Criterion`, `Decision`.
* `grin-fit` command-line entry point.
* Optional extras: `[plot]` for the individual and group reporting figures,
  `[train]` to re-export the ONNX graph from the trained weights.
