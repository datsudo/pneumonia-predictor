## Local setup
First, make sure you have the following prerequisites installed on your machine:

- Python (3.12+)
- Poetry (1.8.3+) - for virtual environment setup

Clone this repository by running:
```bash
git clone https://huggingface.co/spaces/datsudo/pneumonia-predictor
```

Setup and enter the virtual environment using `poetry`:
```bash
poetry install  # installs all the dependencies
poetry shell
```

Start the web app using `streamlit`
```bash
streamlit run app.py
```

For more information about this project, check out the
[documentation](https://datsudo.github.io/pneumonia-predictor).
