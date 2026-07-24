import setuptools

with open("README.md") as f:
    long_description = f.read()
with open("requirements.txt") as f:
    requirements = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

setuptools.setup(
    name="grin",
    version="0.1.0",
    description="GRIN: simulation-based inference for General Recognition Theory.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Only src.* is packaged — scripts/ and tests/ deliberately have no __init__.py
    # so they are not installed as importable packages.
    packages=setuptools.find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
)