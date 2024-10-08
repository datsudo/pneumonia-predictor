RUFF_PATH=/home/datsudo/.cache/pypoetry/virtualenvs/pneumonia-predictor-oYuqsjsZ-py3.12/bin/ruff

format:
	$(RUFF_PATH) check --select I --fix $(git diff --name-only --cached -- '*.py')

format_all:
	$(RUFF_PATH) check --select I --fix .
	$(RUFF_PATH) format .
