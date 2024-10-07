from gradio.themes import Soft

from pneumonia_predictor.config import APP_FONTS
from pneumonia_predictor.frontend.utils import load_component

DefaultTheme = Soft(radius_size="sm", font=APP_FONTS).set()
Header = load_component("header")
Description = load_component("description")
