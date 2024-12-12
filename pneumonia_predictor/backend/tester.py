from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

from pneumonia_predictor.backend.logger import Logger


class ModelTester(Logger):
    def __init__(self, model_a, model_b) -> None:
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.metrics = ["accuracy", "precision", "recall", "f1-score"]

    def run_tests(self, num_tests: int) -> None:
        self.a_tests_arr = []  # acc, prec, rec, f1
        self.b_tests_arr = []
        self.a_tests_per_metric = defaultdict(list)
        self.b_tests_per_metric = defaultdict(list)

        for t in range(num_tests):
            # Reset stats of both models
            self.model_b.X_train_resampled = self.model_b.X_train.copy()
            self.model_b.y_train_resampled = self.model_b.y_train.copy()

            self.log("sep", "=")
            self.log("inf", f"STARTING TEST {t + 1}")

            self.log("op", f"Training Started: {str(self.model_a)}")
            self.model_a.train()
            self.a_tests_arr.append(self.parse_res_per_test(self.model_a, t, "a"))

            self.log("inf", f"Training Started: {str(self.model_b)}")
            self.model_b.train()
            self.b_tests_arr.append(self.parse_res_per_test(self.model_b, t, "b"))

            self.store_res_per_metric()

        self.generate_final_res()

    def parse_res_per_test(self, model_class, n_test: int, model: str) -> list[float]:
        avg = (
            model_class.overall_weighted_avg
            if model == "a"
            else model_class.overall_macro_avg
        )
        test_res_arr = [
            n_test + 1,
            model_class.overall_accuracy,
            avg["precision"],
            avg["recall"],
            avg["f1-score"],
        ]
        return test_res_arr

    def store_res_per_metric(self) -> None:
        for m in self.metrics:
            if m == "accuracy":
                self.a_tests_per_metric[m].append(self.model_a.overall_accuracy)
                self.b_tests_per_metric[m].append(self.model_b.overall_accuracy)
            else:
                self.a_tests_per_metric[m].append(self.model_a.overall_weighted_avg[m])
                self.b_tests_per_metric[m].append(self.model_b.overall_macro_avg[m])

    def generate_final_res(self) -> None:
        self.model_a_res = pd.DataFrame(
            self.a_tests_arr, columns=["Test", *self.metrics]
        )
        self.model_a_res["Average"] = self.model_a_res[self.metrics].mean(axis=1)

        self.model_b_res = pd.DataFrame(
            self.b_tests_arr, columns=["Test", *self.metrics]
        )
        self.model_b_res["Average"] = self.model_b_res[self.metrics].mean(axis=1)

        self.compare_res = pd.DataFrame({"Metrics": self.metrics})
        self.t_values, self.p_values = self.get_ttest_res()
        self.compare_res["t-value"] = self.t_values
        self.compare_res["p-value"] = self.p_values

    def get_ttest_res(self) -> tuple[list[float], list[float]]:
        t_values = []
        p_values = []
        for m in self.metrics:
            self.log("op", "Computing t-values and p-values")
            res = ttest_rel(self.a_tests_per_metric[m], self.b_tests_per_metric[m])
            t_values.append(res.statistic)
            p_values.append(res.pvalue)
        return t_values, p_values

    def save_result(self, location: str = "results") -> None:
        result_path = Path(f"{location}")
        result_path.parent.mkdir(exist_ok=True, parents=True)
        self.model_a_res.to_csv(Path(f"{location}/model_a.csv"), index=False)
        self.model_b_res.to_csv(Path(f"{location}/model_b.csv"), index=False)
