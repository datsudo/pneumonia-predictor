import urllib.request
from pathlib import Path

import pandas as pd
import patoolib

from pneumonia_predictor.backend.logger import Logger
from pneumonia_predictor.config import DATASET_DIR

SUPPORTED_DS_TYPES = {"csv", "parquet"}
LOGGER = Logger()


def download_data(url: str, filename: str, location: str = DATASET_DIR) -> None:
    download_path = f"{location}/{filename}"

    LOGGER.log("op", f"Preparing to download: {url}")
    if not Path(download_path).is_file():
        LOGGER.log("op", f"Downloading dataset from: {url}")
        urllib.request.urlretrieve(url, Path(download_path))
        if patoolib.is_archive(Path(download_path)):
            LOGGER.log("op", f"Extracting archive: ./{download_path}")
            patoolib.extract_archive(
                archive=Path(download_path), verbosity=1, outdir=location
            )
    else:
        LOGGER.log(
            "inf", f"File already exists: ./{download_path}. Downloading skipped"
        )


def load_data(
    dataset_name: str, dataset_type: str = "csv", location: str = DATASET_DIR
) -> pd.DataFrame:
    dataset_readers = {
        "csv": lambda csv: pd.read_csv(csv),
        "parquet": lambda parquet: pd.read_parquet(parquet, engine="pyarrow"),
    }

    full_path = f"{location}/{dataset_name}.{dataset_type}"
    if not Path(full_path).is_file():
        LOGGER.log("err", f"Dataset not found: ./{full_path}")
    if dataset_type not in SUPPORTED_DS_TYPES:
        LOGGER.log("err", f"File type not supported: {dataset_type}")

    LOGGER.log("inf", f"Dataset found: ./{full_path}. Loaded.")
    return dataset_readers[dataset_type](Path(full_path))
