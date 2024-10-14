---
title: Pneumonia Predictor
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: gpl-3.0
---

<div align="center">


<img src="./images/icon.png" alt="App icon" style="width: 100px;" />


# Pneumonia Predictor

[![License](https://img.shields.io/badge/License-GPL%20v3-5b940a?style=flat)](./LICENSE)
[![Spaces](https://img.shields.io/badge/Spaces-Online-5b940a?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fdatsudo%2Fpneumonia-predictor)](https://huggingface.co/spaces/datsudo/pneumonia-predictor)
[![python](https://img.shields.io/badge/Python-3.12-5b940a.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

</div>

Documentation: <https://datsudo.github.io/pneumonia-predictor/>

## Setup

Install the following prerequisites:

1. [Python 3.12+](https://www.python.org/downloads/)
2. [Poetry (for dependency management)](https://python-poetry.org/docs/#installation)

Make sure that poetry is installed by running:

```bash
poetry --version

# Result must be:
# Poetry (version X.X.X)
```

Clone this repository:

```bash
git clone https://github.com/datsudo/pneumonia-predictor
```

Then install the project dependencies when you're inside the directory:

```bash
poetry install
```

Before working on this project, **ALWAYS** make sure that everytime you're working on this project,
**you're in the proper virtual environment** created by `poetry`:

```bash
poetry shell
```

After that, setup the provided pre-commit hooks (to fix proper code styling for staged files before commiting) after cloning.

```bash
pre-commit install
```

## Running

### Sample usage

``` python
from pneumonia_predictor.backend.rf_active_smote import RfActiveSMOTE()

# Create an RFActiveSMOTE instance
rf_as = RfActiveSMOTE(X_train, y_train, X_test, y_test, "target_column_name")

# Training
rf_as.smote.train(n_iterations=5)

# Other options
rf_as.save("model_name")  # will save the model at ./saved_models/model_name.pkl

rf_as.display_results("acc")  # Print results [acc (Accuracy), min (Minority), maj (Majority), avg (Weighted Average)]
```

### Web app

Open the app by running

```bash
streamlit run app.py
```

The web app will become accessible at http://localhost:8502.

## License

This project is licensed under GNU General Public License v3.0.
