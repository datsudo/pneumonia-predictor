from pathlib import Path

import matplotlib.pyplot as plt
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
    plt.savefig(fig_path, format=fig_ext, dpi=resolution)


def format_input(session, columns: list[str]) -> tuple[list[int], DataFrame]:
    bool_feat = [
        session.crd,
        session.ckd,
        session.hf,
        session.cn,
    ]

    user_input = [
        session.age,
        1 if session.sex == "Male" else 0,
        1 if session.ftg else 0,
        1 if session.phlm else 0,
    ]
    user_input.extend(list(map(lambda x: 1 if x else 0, bool_feat)))
    user_input.extend(
        [
            session.sys_bp,
            session.dias_bp,
            session.pulse_rate,
            session.resp_rate,
            1 if session.dm else 0,
            session.hgb,
            session.platelet_count,
            1 if session.cgh else 0,
            session.temp,
            session.ht,
            session.rbc,
            session.wbc,
        ]
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
    for _ in features:
        if new:
            new.append(0)
        else:
            new.append(1)

    return new


def check_distribution(
    df: DataFrame, target: str, title: str, class_a: str, class_b: str
) -> None:
    pneumonia_counts = df[target].value_counts()

    plt.figure(figsize=(6, 6))
    pneumonia_counts.plot.pie(
        autopct="%1.1f%%",
        labels=[f"{class_a} (0)", f"{class_b} (1)"],
        colors=["skyblue", "salmon"],
        startangle=90,
        explode=[0, 0.1],
    )
    plt.title(title)
    plt.ylabel("")
    plt.show()
