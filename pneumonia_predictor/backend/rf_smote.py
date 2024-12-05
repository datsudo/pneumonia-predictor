from collections import Counter
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from pneumonia_predictor.backend.logger import Logger
from pneumonia_predictor.config import N_ESTIMATORS, SAVED_MODELS_PATH


class RfSMOTE(Logger):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        target_name: str,
        num_est: int = N_ESTIMATORS,
    ) -> None:
        super().__init__()

        self.target_name = target_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.X_train_resampled = X_train.copy()
        self.y_train_resampled = y_train.copy()

        self.maj_class_val = y_train.value_counts().idxmax()[0]
        self.min_class_val = y_train.value_counts().idxmin()[0]

        self.classifier = RandomForestClassifier(n_estimators=num_est, random_state=42)
        self.smote = SMOTE(sampling_strategy="not majority")

    def train(self) -> None:
        self.log("sep", "=")
        self.log("op", "Training starts")
        self.create_synthetic_samples()
        self.fit_classifier()

    def fit_classifier(self) -> None:
        self.log("op", "Process classifier.fit started")
        self.classifier.fit(
            self.X_train_resampled,
            self.y_train_resampled[self.target_name].values.ravel(),
        )
        self.log("op", "Process classifier.predict started")
        self.y_pred = self.classifier.predict(self.X_test)
        self.report = classification_report(self.y_test, self.y_pred, output_dict=True)
        self.overall_accuracy = self.report["accuracy"]
        self.overall_weighted_avg = self.report["weighted avg"]
        self.log("op", "Classification report generated")

    def create_synthetic_samples(self) -> None:
        self.min_maj_count = Counter(
            self.y_train_resampled[self.target_name].to_numpy()
        )
        self.log(
            "inf",
            f"Minority/Majority count: {self.min_maj_count[self.min_class_val]} / "
            + f"{self.min_maj_count[self.maj_class_val]}",
        )
        self.log("op", "SMOTE process started")

        y_train_arr = self.y_train[self.target_name].to_numpy().astype(int)

        self.log("op", "Running SMOTE.fit_resample")
        self.X_smote, self.y_smote = self.smote.fit_resample(self.X_train, y_train_arr)
        self.y_smote = DataFrame(self.y_smote, columns=[self.target_name])

        self.log("op", "Creating sets: X_synthetic, y_synthetic")
        self.X_synthetic = self.X_smote.iloc[len(self.X_train) :]
        self.y_synthetic = self.y_smote.iloc[len(self.y_train) :]

        self.log("op", "Creating sets: synthetic_samples")
        self.synthetic_samples = concat([self.X_synthetic, self.y_synthetic], axis=1)

        self.log("op", "Applying synthetic_samples to: X_train, y_train")
        self.X_train_resampled = concat(
            [self.X_train_resampled, self.X_synthetic], ignore_index=1
        )
        self.y_train_resampled = concat(
            [self.y_train_resampled, self.y_synthetic], ignore_index=1
        )

        # Recalculate
        self.min_maj_count = Counter(
            self.y_train_resampled[self.target_name].to_numpy()
        )

    def init_stats(self) -> None:
        self.X_train_resampled = self.X_train.copy()
        self.y_train_resampled = self.y_train.copy()

    def save(self, model_name: str) -> None:
        models_path = Path(SAVED_MODELS_PATH)
        models_path.mkdir(exist_ok=True)
        joblib.dump(self.classifier, f"{SAVED_MODELS_PATH}/{model_name}.pkl")
        self.log("inf", f"Pickle {model_name}.pkl saved at ./{SAVED_MODELS_PATH}")

    def __str__(self) -> str:
        return "Random Forest + SMOTE Model"
