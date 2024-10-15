import joblib
import pandas as pd
import streamlit as st

from pneumonia_predictor.frontend.components import (
    display_page,
    display_banner,
    display_result,
)
from pneumonia_predictor.backend.utils import format_input

PREDICTOR_RF_SMOTE = joblib.load("saved_models/pneumonia_predictor_rfsmote.pkl")
PREDICTOR_RF_ACTIVE_SMOTE = joblib.load("saved_models/pneumonia_predictor.pkl")


def predict(model, new_data: pd.DataFrame) -> None:
    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    with st.container(border=True):
        class_0_proba = round(probabilities[0] * 100, 2)
        class_1_proba = round(probabilities[1] * 100, 2)
        result = pd.DataFrame(
            {
                "Prediction": ["Low", "High"],
                "Percentage (%)": [class_0_proba, class_1_proba],
            }
        ).set_index("Prediction")

        display_result(prediction, result, class_0_proba, class_1_proba)


def main() -> None:
    display_banner()

    tab1, tab2, tab3 = st.tabs(["About", "Local Setup", "Predict"])

    with tab1:
        display_page("about")

    with tab2:
        display_page("local_setup")

    with tab3:
        with st.container(border=True):
            st.subheader("Choose Model", divider=True)
            st.radio(
                "**Model**",
                key="chosen_model",
                options=["RfSMOTE", "RfActiveSMOTE"],
                horizontal=True,
            )

        with st.container(border=True):
            col1, col2 = st.columns(2)

            with col1:
                with st.container(border=True):
                    st.subheader("Input Fields", divider=True)

                    with st.container(border=True):
                        st.number_input(
                            "**Age** (must be between 18-65 years)",
                            key="age",
                            min_value=18,
                            max_value=65,
                        )
                    with st.container(border=True):
                        st.radio(
                            "**Sex**",
                            key="sex",
                            options=["Male", "Female"],
                            captions=["0", "1"],
                            horizontal=True,
                        )
                    with st.container(border=True):
                        st.write(
                            "**Indicate whether the patient has any of the following conditions:**"
                        )
                        conditions = {
                            "crd": "Chronic respiratory disease",
                            "dm": "Diabetes mellitus",
                            "hf": "Heart failure",
                            "cn": "Cancer",
                            "ckd": "Chronic kidney disease",
                        }
                        for cond in conditions:
                            st.checkbox(label=conditions[cond], key=cond)

            with col2:
                X_input = format_input(
                    st.session_state,
                    [
                        "age",
                        "sex",
                        "chronic_respiratory_disease",
                        "diabetes_mellitus",
                        "heart_failure",
                        "cancer",
                        "chronic_kidney_disease",
                    ],
                )

                if st.session_state.chosen_model == "Rf-SMOTE":
                    model = PREDICTOR_RF_SMOTE
                else:
                    model = PREDICTOR_RF_ACTIVE_SMOTE

                predict(model, X_input)


if __name__ == "__main__":
    main()
