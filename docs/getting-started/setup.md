# Local Setup

---

## Prerequisites

To setup the project, you must have the following prerequisites installed on your machine:

- Python (3.12+)
- Poetry (1.8.3+) - for managing dependencies and virtual environment

## Setting up the environment

Clone first the repository:

```bash
git clone https://github.com/datsudo/pneumonia-predictor.git
```

Then install the dependencies see [pyproject.toml](../../pyproject.toml):

```bash
cd pneumonia-predictor
poetry install
```

You can now start the virtual environment
```bash
poetry shell
```

> ℹ️ [**Optional**] If you want to contribute:
>
> Run `pre-commit install` to setup [pre-commit](https://pre-commit.com/) hooks
> that performs automatic code formatting and linting with [Ruff](https://docs.astral.sh/ruff/).

## Running the app

The interface for this program uses [Streamlit](https://streamlit.io/). Assuming that you're
in the root folder in your terminal, you can try it out by running

```bash
streamlit run app.py
```

This should automatically visit the website. If not, go to <http://127.0.0.1:8501>. Tweak `app.py` to edit the page.
