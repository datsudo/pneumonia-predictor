from gradio.themes import GoogleFont

HTML_FILES_DIR = "pneumonia_predictor/frontend/html"
DATASET_DIR = "datasets"
LOGS_ENABLED = True
LOGFILE_ENABLED = True
LOGFILE_LOC = "logs.txt"
N_ITERATIONS = 5  # For model retraining
N_ESTIMATORS = 100  # For random forest
SAVED_MODELS_PATH = "saved_models"

APP_TITLE = "Pneumonia Predictor"
APP_FONTS = [
    # Font inside this must be available at https://fonts.google.com
    GoogleFont("Inter Tight"),
    "ui-sans-serif",
    "system-ui",
    "sans-serif",
]
