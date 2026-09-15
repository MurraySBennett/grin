# grintools: porting and publishing status

This records what the package contains and how it works -- principally the ONNX
contract and the input/stopping layers, which are the parts worth reading before
changing anything.

**Publishing is not documented here.** The live runbook for cutting a release --
PyPI, CRAN, the git tag and the Zenodo DOI -- is `docs/RELEASE.md`, under
"Shipping the packages". Keeping a second copy here is how the two drifted apart.

## What is done

The distributable is a thin, torch-free inference client wrapping the exported
ONNX model. It has been built into a wheel, installed into a clean virtual
environment (pulling only numpy and onnxruntime), and exercised end to end: the
bundled model loads, inference returns a sane 12-parameter posterior with construct
probabilities, and the installed `grin-fit` console command runs from a directory
containing none of the source. The packaged smoke test passes on that clean install.

The ONNX contract is confirmed against the shipped `cm` model: inputs are raw
`counts` (B, 16) and `trials` (B, 4); outputs are parameter-space `mean` and `std`
(12 each), `p_corr` (PI, RHO1, free) and `p_sep` (separable A, separable B). All
featurisation, link functions, and construct heads are inside the graph, so the
wrapper does no maths beyond reshaping and reading outputs.

The input layer enforces the two failure modes that silently return wrong answers:
stimulus/response ordering (a bare unlabelled matrix is refused) and counts versus
proportions (proportions are refused unless trials are supplied). The stopping layer
lets the experimenter declare a Criterion of precision and/or construct-probability
Targets, and surfaces the perceptual-independence identifiability limit through the
model's directional decision_* outputs rather than hiding it.

## What the package contains

    grintools/
      __init__.py        public API: infer(), to_confusion(), describe(),
                         Criterion/Target/Decision, GrinOnnx, default_model_path()
      io.py              normalisation + describe(), the ordering/counts guards
      criterion.py       Target / Criterion / Decision, the stopping API
      onnx.py            GrinOnnx: torch-free inference from the .onnx
      cli.py             the `grin-fit` console command
      models/            npe_model.onnx (bundled as package data, ~300 KB)
      data/              example_cm.csv
    tests/               test_packaged.py (installed-wheel smoke test), test_plot.py
    pyproject.toml       PEP 621 metadata, deps, package-data, entry point, extras
    MANIFEST.in          sdist inclusion of the licence, model and data
    CHANGELOG.md         per-release notes

The trusted-publishing workflow lives at the repository root, in
`.github/workflows/publish.yml`; it only fires on a published GitHub release.

The package is independent of the research code: it does not import the torch `src`
package. This means the existing `src`-layout of the main repo does NOT need to be
renamed for this to publish; grintools is a self-contained subtree. Retraining
regenerates the `.onnx`, which is then copied into `grintools/models/`.

## Tie to the paper

Cite the exact released version in the manuscript so a reader can reproduce the
reported inferences with the same weights. The package version pins the weights,
and the GitHub release is archived to Zenodo for a citable DOI.
