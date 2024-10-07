from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from pneumonia_predictor.backend.active_smote import ActiveSMOTE
from pneumonia_predictor.config import N_ESTIMATORS, N_ITERATIONS, SAVED_MODELS_PATH


class RfActiveSMOTE(ActiveSMOTE):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_test: DataFrame,
        target_name: str,
        num_clusters: int = 4,
        sampling_ratio: float = 0.25,
        update_sampratio_per_iter: bool = False,
    ) -> None:
        self.probabilities = []

        super().__init__(
            X_train, y_train, self.probabilities, target_name, num_clusters
        )

        self.X_test = X_test
        self.y_test = y_test

        self.classifier = RandomForestClassifier()
        # stores all synthetic samples throughout the iteration
        self.total_synthetic_samples = DataFrame()

        # FIXME: Temporary
        self.current_ratio = sampling_ratio
        self.update_sampratio_per_iter = update_sampratio_per_iter

        # For results
        self.init_stats()

    def train(self, n_iterations: int = N_ITERATIONS) -> None:
        self.init_stats()
        self.n_iterations = n_iterations

        self.log("sep", "=")
        self.log("op", "Initial training starts")
        self.fit_classifier()
        self.probabilities = self.classifier.predict_proba(self.X_train)

        self.log("sep", "=")
        self.log("op", "Model resampling starts")
        for i in range(n_iterations):
            self.log("sep", "=")
            self.log("inf", f"ITERATION {i + 1}")

            self.uncertainty_sampling()
            self.diversity_sampling()

            # FIXME: Temporary
            if self.update_sampratio_per_iter:
                self.current_ratio += self.current_ratio * 0.024625692695214109

            self.create_synthetic_samples(self.current_ratio)
            self.total_synthetic_samples = concat(
                [self.total_synthetic_samples, self.current_synthetic_samples], axis=1
            )
            self.fit_classifier()

            self.record_curr_iteration()
        self.log("inf", "Retraining done")

    def fit_classifier(self, num_est: int = N_ESTIMATORS):
        self.log("op", "Process classifier.fit started")
        self.classifier.fit(
            self.X_train_resampled,
            self.y_train_resampled[self.target_name].values.ravel(),
        )
        self.log("op", "Process classifier.predict started")
        self.y_pred = self.classifier.predict(self.X_test)
        self.current_report = classification_report(
            self.y_test, self.y_pred, output_dict=True
        )

    def record_curr_iteration(self) -> None:
        self.accuracy_stats.append(self.current_report["accuracy"])
        for metric in ["precision", "recall", "f1-score"]:
            self.min_class_stats[metric].append(
                self.current_report[str(self.min_class_val)][metric]
            )
            self.maj_class_stats[metric].append(
                self.current_report[str(self.maj_class_val)][metric]
            )
            self.weighted_avg[metric].append(
                self.current_report["weighted avg"][metric]
            )

    def display_results(self, opt: str) -> None:
        if opt not in {"acc", "min", "maj", "avg"}:
            self.log("err", f"Unknown option: {opt}. Allowed: acc, min, maj, avg")

        self.x_ax = [str(i) for i in range(1, self.n_iterations + 1)]

        if opt == "acc":
            self.display_accuracy()
        else:
            self.display_stats(opt)

        plt.xlabel("Iteration")
        plt.legend()
        plt.show()

    def display_accuracy(self) -> None:
        plt.plot(self.x_ax, self.accuracy_stats, label="Accuracy", marker="o")

    def display_stats(self, opt: str) -> None:
        stats = {
            "min": ["Minority Class", self.min_class_stats],
            "maj": ["Majority Class", self.maj_class_stats],
            "avg": ["Weighted Average", self.weighted_avg],
        }
        plt.title(stats[opt][0])
        for metric in stats[opt][1]:
            plt.plot(
                self.x_ax, stats[opt][1][metric], label=metric.capitalize(), marker="o"
            )

    def init_stats(self) -> None:
        self.min_class_stats = defaultdict(list)
        self.maj_class_stats = defaultdict(list)
        self.weighted_avg = defaultdict(list)
        self.accuracy_stats = []

    def save(self, model_name: str) -> None:
        models_path = Path(SAVED_MODELS_PATH)
        models_path.mkdir(exist_ok=True)
        joblib.dump(self.classifier, f"{SAVED_MODELS_PATH}/{model_name}.pkl")
        self.log("inf", f"Pickle {model_name}.pkl saved at ./{SAVED_MODELS_PATH}")
