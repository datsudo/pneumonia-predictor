from pathlib import Path

from matplotlib.pyplot import savefig
from pandas import DataFrame


def get_feature_target_set(
    dataset: DataFrame, target_name: str
) -> tuple[DataFrame, DataFrame]:
    X = dataset.drop(columns=[target_name], axis=1)
    y = dataset[[target_name]]

    return X, y


def save_figure(fig_id: str, fig_ext: str = "png", resolution: int = 300) -> None:
    images_dir = Path() / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fig_path = images_dir / f"{fig_id}.{fig_ext}"
    savefig(fig_path, format=fig_ext, dpi=resolution)


def format_input(fu, session, columns: list[str]) -> tuple[list[int], DataFrame]:
    user_input = [
        session.age + 1 if fu else session.age - 1,
        0 if session.sex == "Male" else 1,
    ]
    user_input.extend(
        list(
            map(
                lambda x: 1 if x else 0,
                [
                    session.crd,
                    session.dm,
                    session.hf,
                    session.cn,
                    session.ckd,
                ],
            )
        )
    )
    user_input_df = DataFrame(
        [user_input],
        columns=columns,
    )
    return user_input, user_input_df


def features_updated(features):
    for f in features:
        if type(f) is str:
            if f == "Yes":
                return True
            else:
                return False
        if f > 1:
            return True
    return True


def flip(features):
    new = []
    for f in features:
        if new:
            new.append(0)
        else:
            new.append(1)

    return new
