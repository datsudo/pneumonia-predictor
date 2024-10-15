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


def format_input(session, columns: list[str]) -> None:
    user_input = [
        session.age,
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
    user_input = DataFrame(
        [user_input],
        columns=columns,
    )
    return user_input
