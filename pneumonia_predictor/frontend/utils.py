from pathlib import Path

from pneumonia_predictor.config import HTML_FILES_DIR


def load_component(component_name: str) -> str:
    return Path(f"{HTML_FILES_DIR}/{component_name}.html").read_text()
