from pathlib import Path
from typing import Any, Callable

import pandas as pd

from pneumonia_predictor.backend.logger import Logger
from pneumonia_predictor.config import DATASET_DIR


class DataTransformer(Logger):
    def __init__(self) -> None:
        super().__init__()

    def remove_columns(self, dataset: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.log("op", f"Removing columns from dataset: {columns}")
        return dataset.drop(columns, axis=1)

    def transform_columns(
        self, dataset: pd.DataFrame, columns: dict[str, Callable[..., Any]]
    ) -> pd.DataFrame:
        new_dataset = dataset.copy()
        for col in columns:
            self.log("op", f"Transforming all values of column '{col}'")
            new_dataset[col] = new_dataset[col].apply(columns[col])
        return new_dataset

    def map_col_values(
        self, dataset: pd.DataFrame, columns_w_mapper: dict[str, dict[str, int]]
    ) -> pd.DataFrame:
        new_dataset = dataset.copy()
        for col in columns_w_mapper:
            self.log("op", f"Mapping values: Column {col} -> {columns_w_mapper[col]}")
            new_dataset[col] = new_dataset[col].map(columns_w_mapper[col])
        return new_dataset

    def change_col_type(
        self, dataset: pd.DataFrame, columns: list[str], to_type: str
    ) -> pd.DataFrame:
        type_opts = {"str": str, "int": int, "float": float}

        new_dataset = dataset.copy()
        for col in columns:
            self.log("op", f"Changing datatype of column {col} -> {to_type}")
            new_dataset[col] = new_dataset[col].astype(type_opts[to_type])
        return new_dataset

    def save(
        self,
        dataset: pd.DataFrame,
        filename: str,
        location: str = DATASET_DIR,
        filetype: str = "csv",
    ) -> None:
        full_path = f"{location}/{filename}.{filetype}"
        self.log(
            "op",
            f"Saving dataframe as {filetype}: ./{full_path}",
        )

        Path(location).mkdir(parents=True, exist_ok=True)

        type_opts = {
            "csv": lambda df, path: df.to_csv(path, index=False),
            "parquet": lambda df, path: df.to_parquet(
                path, engine="pyarrow", index=False
            ),
        }

        type_opts[filetype](dataset, Path(full_path))
