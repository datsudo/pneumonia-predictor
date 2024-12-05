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


def format_input(session, columns: list[str]) -> tuple[list[int], DataFrame]:
    bool_feat = [
        session.crd,
        session.ckd,
        session.dm,
        session.hf,
        session.cn,
    ]
    cough_phlegm = {
        "No": 0,
        "Yes, dry cough": 1,
        "Yes, with phlegm": 2,
    }

    user_input = [
        session.age,
        1 if session.sex == "Male" else 0,
        1 if session.ftg else 0,
        cough_phlegm[session.cough],
    ]
    user_input.extend(list(map(lambda x: 1 if x else 0, bool_feat)))
    user_input.extend(
        [
            session.sys_bp, session.dias_bp, session.pulse_rate, session.resp_rate,
            session.temp, session.hgb, session.ht, session.rbc, session.wbc, session.platelet_count
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
