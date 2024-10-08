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
        self.model_a_tests = []  # acc, prec, rec, f1
        self.model_b_tests = []

        for t in range(num_tests):
            self.log("sep", "=")
            self.log("inf", f"STARTING TEST {t + 1}")

            self.log("op", f"Training Started: {str(self.model_a)}")
            self.model_a.train()
            self.model_a_tests.append(self.parse_test_res(self.model_a, t))

            self.log("inf", f"Training Started: {str(self.model_b)}")
            self.model_b.train()
            self.model_b_tests.append(self.parse_test_res(self.model_b, t))

        self.generate_final_res()

    def parse_test_res(self, model, n_test: int) -> list[float]:
        test_result = [
            n_test + 1,
            model.overall_accuracy,
            model.overall_weighted_avg["precision"],
            model.overall_weighted_avg["recall"],
            model.overall_weighted_avg["f1-score"],
        ]
        return test_result

    def generate_final_res(self) -> None:
        self.model_a_res = pd.DataFrame(
            self.model_a_tests, columns=["Test", *self.metrics]
        )
        self.model_a_res["Average"] = self.model_a_res[self.metrics].mean(axis=1)

        self.model_b_res = pd.DataFrame(
            self.model_b_tests, columns=["Test", *self.metrics]
        )
        self.model_b_res["Average"] = self.model_b_res[self.metrics].mean(axis=1)

        res = {
            "Metrics": self.metrics,
            str(self.model_a): [
                self.model_a_res["accuracy"][0],
                self.model_a_res["precision"][0],
                self.model_a_res["recall"][0],
                self.model_a_res["f1-score"][0],
            ],
            str(self.model_b): [
                self.model_b_res["accuracy"][0],
                self.model_b_res["precision"][0],
                self.model_b_res["recall"][0],
                self.model_b_res["f1-score"][0],
            ],
        }

        self.compare_res = pd.DataFrame(res)
        t_values, p_values = ttest_rel(
            self.compare_res[str(self.model_a)].values,
            self.compare_res[str(self.model_b)].values,
        )
        self.compare_res["t-value"] = t_values
        self.compare_res["p-value"] = p_values

    def save_result(self, location: str = "results") -> None:
        result_path = Path(f"{location}")
        result_path.parent.mkdir(exist_ok=True, parents=True)
        self.model_a_res.to_csv(Path(f"{location}/model_a.csv"), index=False)
        self.model_b_res.to_csv(Path(f"{location}/model_b.csv"), index=False)
