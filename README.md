**_This is still a work in progress. That is, none of it will work if you try running it in it's current form. Everything below is a hope, not a promise, but feel free to follow along with the mayhem._**

# GRIN: General Recognition Inference Network

General Recognition Inference Network (GRIN) is a project designed to rapidly fit data for General Recognition Theory (GRT) models.
The core idea is to train a neural network to predict GRT parameters and model classes to then bypass the relatively slow process of traditional GRT model fitting via MLE.

Because it's so fast and efficient (and definitely absolutely most certainly works\*), thereby facilitating a wealth of practical applications and potential, it's sure to make you grin!

# Table of Contents

[Introduction](#grin-general-recognition-inference-network)

[Features](#features)

[Getting Started](#getting-started)

[Project Structure](#project-structurej)

[Contributing](#contributing)

[Contact & Acknowledgments](#contact--acknowledgments)

# Features

**Rapid Data Generation:** Quickly generate large-scale synthetic GRT datasets.

**Efficient Model Fitting:** Use a neural network to "fit" GRT parameters in a fraction of the time required by traditional methods.

**Multi-Task Learning:** Models can be trained to perform both classification (identifying model types) and regression (estimating parameters).

**Modularity:** A clean and organized codebase with separated modules for data generation, model architecture, and evaluation.

**Reproducibility:** A clear project structure and requirements.txt file ensure that others can easily replicate your work.

# Getting Started

## Model training

These instructions will get you a copy of the project up and running on your local machine for reproducing the model training. For using GRIN to fit data, see the following (NOT YET CREATED) section.

Prerequisites
You will probably need Python 3.9 or newer and probably have `git` installed.

Installation
Clone the repository:

```
git clone https://github.com/your-username/GRIN.git
cd GRIN
```

Install dependencies:

```
pip install -r requirements.txt
```

**Usage**

All configuration content is housed in `src/utils/config.py`.
The simulated data produces confusion matrices populated with response counts, the generating parameters and model class, and the number of trials (although this can be pulled from the matrices).

<!-- You can try updating things like training hyperparameters, and whether to use pretrained weights or not, as well as others.
But be careful because this file (and project) still needs a proper review to make clear what content can be or should best not be tampered with. Below is the workflow I followed: -->

Simulate data for the full set of models `--full`, pretraining confusion matrices that only vary particular parameters `--pretraining`, trial-by-trial response data `--tbt`, or all of them `--all`:

```
python -m src.utils.GRT_data_generator --all
```

Pretrain weights on the means, covariance matrices, and decision bounds, then visualise the pretrained state of parameter predictions.
Figures should save to `results/figures/pretraining` and weights save to `src/models/pretrained`.

```
python -m scripts.pretrain_parameters
python -m scripts.visualise_pretraining
```

Training the models on a curriculum schedule that progressively includes the models with greater complexity (i.e., more free parameters) over 4 stages. This script also saves the train/validation/test splits for the simulated matrices. We need this to run before the next component so that the test data can be identified and exported for model fitting in R (or other software). Models should save to `results/models/`, and training figures should save to `results/figures`.

**TODO**: update npz_to_csv to something informative

```
python -m scripts.train_models
python -m src.utils.npz_to_csv
```

At this point, you now have trained networks capable of making predictions on model classes and their parameters. From here, I shift into an R workspace to fit the simulated test data using [`grtools`](https://github.com/fsotoc/grtools/). This requires the [`grtools`](https://github.com/fsotoc/grtools/), [`here`](https://here.r-lib.org/) and [`tidyverse`](https://www.tidyverse.org/) libraries.

# Contributing

Fork the Project

Create your Feature Branch
`git checkout -b feature/AmazingFeature`

Commit your Changes `git commit -m 'Add some AmazingFeature'`

Push to the Branch `git push origin feature/AmazingFeature`

Open a Pull Request

# Contact & Acknowledgments

Project Link: https://github.com/murraysbennett/grin
