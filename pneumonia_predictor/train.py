from sklearn.mode_selection import train_test_split

from pneumonia_predictor.backend.data_fetcher import load_data
from pneumonia_predictor.backend.utils import get_feat_target


def process_age(age):
    parts = age.split()
    x = int(parts[0])

    if parts[2].isdigit():
        y = int(parts[2])
        return int(round((x + y) / 2))
    else:
        return x


if __name__ == "__main__":
    dataset = load_data("transformed_set", "csv")

    X, y = get_feat_target(dataset, "Diabetes_binary")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
