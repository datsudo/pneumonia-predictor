---
title: Pneumonia Predictor
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: gpl-3.0
---

<div align="center">

# Pneumonia Predictor

[![License](https://img.shields.io/badge/License-GPL%20v3-5b940a?style=flat)](./LICENSE)
[![Spaces](https://img.shields.io/badge/Spaces-Online-5b940a?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fdatsudo%2Fpneumonia-predictor)](https://huggingface.co/spaces/datsudo/pneumonia-predictor)
[![python](https://img.shields.io/badge/Python-3.12-5b940a.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

</div>

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

### Training

> *TODO: training instructions*

### Web app

Open the app by running

```bash
python app.py

# or this command for hot-reloading
streamlit run app.py
```

The web app will become accessible at http://127.0.0.1:7860 (http://127.0.0.1:7860?__theme=light for light theme)

## License

This project is licensed under GNU General Public License v3.0.
