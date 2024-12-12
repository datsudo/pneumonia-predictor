from collections import Counter

from imblearn.over_sampling import SMOTENC
from numpy import ndarray
from pandas import DataFrame, concat
from sklearn.cluster import KMeans

import pneumonia_predictor.backend.logger as logger


class ActiveSMOTE(logger.Logger):
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame,
        probabilities: ndarray,
        target_name: str,
        categ_features: list[int],
        num_clusters: int = 4,
    ) -> None:
        super().__init__()

        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.target_name = target_name
        self.categ_features = categ_features
        self.probabilities = probabilities
        self.maj_class_val = y_train.value_counts().idxmax()[0]
        self.min_class_val = y_train.value_counts().idxmin()[0]
        self.num_clusters = num_clusters

        # To be used during SMOTE process
        self.X_train_resampled = X_train.copy()
        self.y_train_resampled = y_train.copy()

        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        self.create_min_maj_sets()

    def uncertainty_sampling(self, min_sample_frac: float = 0.25) -> None:
        """This creates uncertainty set, as well as a separate set for data under
        minority class and minority samples
        """
        self.log("op", "Creating uncertainty set")
        self.uncertainty_set = self.X_train.copy()
        self.uncertainty_set[self.target_name] = self.y_train.copy()
        self.compute_class_probability()

        # Create minority set and samples
        self.uncertainty_min_set = self.uncertainty_set[
            self.uncertainty_set[self.target_name] == self.min_class_val
        ]
        self.uncertainty_min_samples = self.uncertainty_min_set.sample(
            frac=min_sample_frac, random_state=42
        )

    def diversity_sampling(self) -> None:
        self.create_cluster_set()
        self.stratified_sampling()

        self.X_diverse_min = self.diverse_min_set.drop(
            columns=[self.target_name, "cluster"]
        )
        self.y_diverse_min = self.diverse_min_set[[self.target_name]]

        self.X_diverse = concat(
            [self.X_diverse_min, self.X_train_set_maj], ignore_index=True
        )
        self.y_diverse = concat([self.y_diverse_min, self.y_train_set_maj])

    def create_synthetic_samples(self, sampling_ratio: float) -> None:
        self.log("op", "SMOTE process started")
        self.min_maj_count, self.min_maj_ratio = self.calculate_ratio()
        self.log(
            "inf",
            f"Majority/Minority count: {self.min_maj_count[self.maj_class_val]} "
            + f"{self.min_maj_count[self.min_class_val]}",
        )
        self.log("inf", f"SMOTE sampling ratio: {sampling_ratio}")

        self.smote = SMOTENC(
            sampling_strategy=sampling_ratio, categorical_features=self.categ_features
        )

        y_diverse_arr = self.y_diverse[self.target_name].to_numpy().astype(int)
        self.log("op", "Running SMOTE.fit_resample")
        self.X_smote, self.y_smote = self.smote.fit_resample(
            self.X_diverse, y_diverse_arr
        )
        self.y_smote = DataFrame(self.y_smote, columns=[self.target_name])

        self.log("op", "Creating sets: X_synthetic, y_synthetic")
        self.X_synthetic = self.X_smote.iloc[len(self.X_diverse) :]
        self.y_synthetic = self.y_smote.iloc[len(self.y_diverse) :]

        self.log("op", "Creating sets: synthetic_samples")
        self.current_synthetic_samples = concat(
            [self.X_synthetic, self.y_synthetic], axis=1
        )

        self.log("op", "Applying synthetic_samples to: X_train, y_train")
        self.X_train_resampled = concat(
            [self.X_train_resampled, self.X_synthetic], ignore_index=True
        )
        self.y_train_resampled = concat(
            [self.y_train_resampled, self.y_synthetic], ignore_index=True
        )

        # Recalculate
        self.min_maj_count, self.min_maj_ratio = self.calculate_ratio()
        self.log(
            "inf",
            f"Majority/Minority count: {self.min_maj_count[self.maj_class_val]} "
            + f"{self.min_maj_count[self.min_class_val]}",
        )
        self.log("op", "SMOTE process done")

    def create_min_maj_sets(self) -> None:
        self.train_set = concat([self.X_train, self.y_train], axis=1)

        self.log(
            "op", "Creating set: train_set_min/maj, y_train_min/maj, X_train_min/maj"
        )

        self.train_set_maj = self.train_set[
            self.train_set[self.target_name] == self.maj_class_val
        ]
        self.train_set_min = self.train_set[
            self.train_set[self.target_name] == self.min_class_val
        ]
        self.y_train_set_min = self.train_set_min[
            self.train_set_min[self.target_name] == self.min_class_val
        ][[self.target_name]]
        self.y_train_set_maj = self.train_set_maj[
            self.train_set_maj[self.target_name] == self.maj_class_val
        ][[self.target_name]]
        self.X_train_set_maj = self.train_set_maj.drop(columns=[self.target_name])
        self.X_train_set_min = self.train_set_min.drop(columns=[self.target_name])

    def compute_class_probability(self) -> None:
        # There are other ways of calculating class probability
        self.log("op", "Computing 'class_probability'")
        self.uncertainty_set["class_probability"] = 1 * self.probabilities.max(axis=1)
        self.uncertainty_set.sort_values(by="class_probability", ascending=False)

    def create_cluster_set(self) -> None:
        uncertainty_min_samples = self.uncertainty_min_samples.drop(
            columns=["class_probability"]
        ).copy()
        self.kmeans.fit(uncertainty_min_samples)
        self.log("op", "Creating clustered_set")
        self.cluster_labels = self.kmeans.labels_
        self.clustered_set = uncertainty_min_samples.copy()
        self.clustered_set["cluster"] = self.cluster_labels

    def stratified_sampling(self) -> None:
        self.log("op", "Creating diverse_min_set")
        self.diverse_min_set = DataFrame()
        for n_cluster in range(self.num_clusters):
            target_n_samples = int(
                0.20
                * len(self.clustered_set[self.clustered_set["cluster"] == n_cluster])
            )
            sampled_data = self.clustered_set[
                self.clustered_set["cluster"] == n_cluster
            ].sample(n=target_n_samples, replace=False)
            self.log(
                "op",
                f"Adding sample to diverse_min_set from cluster {n_cluster}: "
                + f"target num samples: {target_n_samples}",
            )
            self.diverse_min_set = concat(
                [self.diverse_min_set, sampled_data], ignore_index=True
            )

    def calculate_ratio(self) -> tuple[dict, float]:
        div_count = Counter(self.y_diverse[self.target_name].to_numpy())
        min_maj_count = Counter(self.y_train_resampled[self.target_name].to_numpy())
        ratio = div_count[self.min_class_val] / div_count[self.maj_class_val] + 0.001

        return min_maj_count, ratio
