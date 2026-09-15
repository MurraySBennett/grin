import setuptools

with open("README.md") as f:
    long_description = f.read()
with open("requirements.txt") as f:
    requirements = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

setuptools.setup(
    # NOT the user-facing tool. The distributable packages are grintools (PyPI) and
    # grin (CRAN), under packages/. This is the training/validation pipeline, installed
    # only as `pip install -e .` for CI and local work. The name "grin" is taken on PyPI
    # by an unrelated project, and the classifier below makes PyPI refuse an upload.
    name="grin-pipeline",
    version="0.1.0",
    description="GRIN training and validation pipeline (not the distributable tool; see grintools on PyPI).",
    classifiers=["Private :: Do Not Upload"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Only src.* is packaged — scripts/ and tests/ deliberately have no __init__.py
    # so they are not installed as importable packages.
    packages=setuptools.find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
)