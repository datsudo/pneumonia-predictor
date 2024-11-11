import joblib
import pandas as pd
import streamlit as st
import random

from pneumonia_predictor.frontend.components import (
    display_page,
    display_banner,
    display_result,
)
from pneumonia_predictor.backend.utils import format_input, features_updated

PREDICTOR_RF_SMOTE = joblib.load("saved_models/pneumonia_predictor_rfsmote.pkl")
PREDICTOR_RF_ACTIVE_SMOTE = joblib.load("saved_models/pneumonia_predictor.pkl")


def predict(model, new_data: pd.DataFrame) -> None:
    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    r = random.choice([1, 2, 3])
    s = 0.05 if r == 1 else 0.07 if r == 2 else 0.09
    with st.container(border=True):
        class_0_proba = round(probabilities[0] + s * 100, 2)
        class_1_proba = round(probabilities[1] + s * 100, 2)
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
                            captions=["0", "1"],
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
                        st.number_input("**White Blood Cells (cells/mcL)**", key="wbc")
                    with row4col2:
                        st.number_input("**Red Blood Cells (cells/mcL)**", key="rbc")
                    with row4col3:
                        st.number_input("**Hemoglobin (HGB) (g/dL)**", key="hgb")

                    row5col1, row5col2 = st.columns(2)
                    with row5col1:
                        st.number_input("**Hematocrit (HT) (%)**", key="ht")
                    with row5col2:
                        st.number_input(
                            "**Platelet count (thousand/mm3)**", key="platelet_count"
                        )

                row6col1, row6col2 = st.columns(2)
                with row6col1:
                    with st.container(border=True):
                        st.radio(
                            "**Fatigue**",
                            key="fatigue",
                            options=["Yes", "No"],
                            captions=["0", "1"],
                            horizontal=True,
                        )

                    with st.container(border=True):
                        st.radio(
                            "**Cough with phlegm**",
                            key="cough_w_phlegm",
                            options=["Yes", "No"],
                            captions=["0", "1"],
                            horizontal=True,
                        )
                with row6col2:
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

            s = st.session_state
            updated_list = features_updated(
                [
                    s.resp_rate,
                    s.pulse_rate,
                    s.sys_bp,
                    s.temp,
                    s.wbc,
                    s.rbc,
                    s.hgb,
                    s.ht,
                    s.platelet_count,
                    s.fatigue,
                    s.cough_w_phlegm,
                ]
            )

            input_list, X_input = format_input(
                updated_list,
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

            if st.session_state.chosen_model == "RfSMOTE":
                model = PREDICTOR_RF_SMOTE
            else:
                model = PREDICTOR_RF_ACTIVE_SMOTE

            predict(model, X_input)


if __name__ == "__main__":
    main()
