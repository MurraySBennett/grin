# grintools: porting and publishing status

This records what has been built, what the package contains and how it works, and
what remains before publishing to PyPI and conda-forge. The intent is to ship the
package alongside the paper, so the release is deliberately staged: everything is
built and validated locally, but nothing is uploaded.

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
model's evidence flags rather than hiding it.

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
    tests/test_packaged.py   smoke test against the installed wheel
    pyproject.toml       PEP 621 metadata, deps, package-data, entry point, [train] extra
    MANIFEST.in          sdist inclusion of the model and data
    .github/workflows/publish.yml   trusted-publishing workflow (inactive until release)

The package is independent of the research code: it does not import the torch `src`
package. This means the existing `src`-layout of the main repo does NOT need to be
renamed for this to publish; grintools is a self-contained subtree. Retraining
regenerates the `.onnx`, which is then copied into `grintools/models/`.

## What is left to do before publishing

- [ ] Confirm the name `grintools` is free on PyPI (visit pypi.org/project/grintools).
- [ ] Choose and set the licence (pyproject currently has a placeholder MIT).
- [ ] Fill author email, classifiers, and a longer project description in pyproject.
- [ ] Decide whether to bundle the RT model (npe_rt_model.onnx, ~700 KB). It needs a
      second wrapper class handling the extra `rtq` input and `p_arch`/`lba` outputs.
- [ ] Pin version to model: document which trained weights are in each release and
      point at the calibration evidence from the validation suite.
- [ ] Add a CI test-matrix workflow (Python 3.9-3.12, Linux/macOS/Windows) alongside
      publish.yml, running the packaged smoke test on each combination.
- [ ] Decide the home: keep grintools as a subtree of the main repo, or split it into
      its own repo. Either works; the subtree keeps model and code in one place.
- [ ] Optional: a make/script target that re-exports the ONNX and copies it into
      grintools/models/ so releases can't drift from the trained model.

## How to publish, when the paper is ready

1. Bump `version` in pyproject.toml; refresh `grintools/models/npe_model.onnx`.
2. `python -m build` -> sdist + wheel in dist/.
3. `twine check dist/*`.
4. Install the wheel in a fresh venv and run `pytest tests/test_packaged.py`.
5. Dry run on TestPyPI: `twine upload -r testpypi dist/*`, then `pip install` from
   TestPyPI into a clean env and confirm the model loads.
6. Real PyPI: either `twine upload dist/*` with a PyPI API token, or configure
   trusted publishing (register the project, add this repo as a trusted publisher)
   and cut a GitHub release, which fires publish.yml. The authenticated upload is a
   manual step performed by the maintainer.
7. conda-forge (later, optional): `grayskull pypi grintools` generates a recipe;
   submit to conda-forge/staged-recipes. Once merged, the feedstock auto-updates on
   each PyPI release. numpy and onnxruntime are already on conda-forge.

## Tie to the paper

Cite the exact released version in the manuscript so a reader can reproduce the
reported inferences with the same weights. A GitHub release can be archived to
Zenodo for a citable DOI if the journal wants one.
