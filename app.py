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
                "Percentage": [class_0_proba, class_1_proba],
            }
        )

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
            with st.container(border=True):
                st.subheader("Input Fields", divider=True)

                row1col1, row1col2 = st.columns(2)
                with row1col1:
                    with st.container(border=True):
                        st.number_input(
                            "**Age** (must be 18 years old and above)",
                            key="age",
                            min_value=18,
                        )
                with row1col2:
                    with st.container(border=True):
                        st.radio(
                            "**Sex**",
                            key="sex",
                            options=["Male", "Female"],
                            captions=["1", "0"],
                            horizontal=True,
                        )

                row2col1, row2col2, row2col3 = st.columns(3)
                with row2col1:
                    with st.container(border=True):
                        st.number_input("**Respiratory rate**", key="resp_rate")
                with row2col2:
                    with st.container(border=True):
                        st.number_input(
                            "**Pulse rate**",
                            key="pulse_rate",
                        )
                with row2col3:
                    with st.container(border=True):
                        st.number_input(
                            "**Temperature (Celsius)**",
                            key="temp",
                        )

                row3col1, row3col2 = st.columns(2)
                with row3col1:
                    with st.container(border=True):
                        st.number_input(
                            "**Systolic blood pressure (mm Hg)**",
                            key="sys_bp",
                        )
                with row3col2:
                    with st.container(border=True):
                        st.number_input(
                            "**Diastolic blood pressure (mm Hg)**",
                            key="dias_bp",
                        )

                with st.container(border=True):
                    st.markdown("**Complete Blood Count (CBC)**")

                    row4col1, row4col2, row4col3 = st.columns(3)
                    with row4col1:
                        st.number_input("**White Blood Cells (g/L)**", key="wbc")
                    with row4col2:
                        st.number_input("**Red Blood Cells (g/L)**", key="rbc")
                    with row4col3:
                        st.number_input("**Hemoglobin (HGB) (g/L)**", key="hgb")

                    row5col1, row5col2 = st.columns(2)
                    with row5col1:
                        st.number_input("**Hematocrit (HT) (%)**", key="ht")
                    with row5col2:
                        st.number_input(
                            "**Platelet count (g/L)**", key="platelet_count"
                        )

                with st.container(border=True):
                    st.radio(
                        "**Do you have a cough**",
                        key="cough",
                        options=["No", "Yes, dry cough", "Yes, with phlegm"],
                        captions=["0", "1", "2"],
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
                        "ftg": "Fatigue",
                    }
                    for cond in conditions:
                        st.checkbox(label=conditions[cond], key=cond)

            _, X_input = format_input(
                st.session_state,
                [
                    "age",
                    "sex",
                    "fatigue",
                    "cough_phlegm",
                    "chronic_resp_disease",
                    "chronic_kidney_disease",
                    "diabetes_mellitus",
                    "heart_failure",
                    "cancer",
                    "systoic_bp",
                    "dias_bp",
                    "pulse_rate",
                    "resp_rate",
                    "temp",
                    "hemoglobin",
                    "hematocrit",
                    "rbc",
                    "wbc",
                    "platelets",
                ],
            )

            if st.session_state.chosen_model == "RfSMOTE":
                model = PREDICTOR_RF_SMOTE
            else:
                model = PREDICTOR_RF_ACTIVE_SMOTE

            predict(model, X_input)


if __name__ == "__main__":
    main()
