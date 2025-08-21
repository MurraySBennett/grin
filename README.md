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

something like this:

```

cd grin
python -m src.train_models

```

- generate_data.py
- train_models.py
- evalute_models.py

# Contributing

Fork the Project

Create your Feature Branch
`git checkout -b feature/AmazingFeature`

Commit your Changes `git commit -m 'Add some AmazingFeature'`

Push to the Branch `git push origin feature/AmazingFeature`

Open a Pull Request

# Contact & Acknowledgments

Project Link: https://github.com/murraysbennett/grin
