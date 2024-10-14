import joblib
import pandas as pd
import streamlit as st

PREDICTOR_RF_SMOTE = joblib.load("saved_models/pneumonia_predictor_rfsmote.pkl")
PREDICTOR_RF_ACTIVE_SMOTE = joblib.load("saved_models/pneumonia_predictor.pkl")


def predict(model, new_data: pd.DataFrame) -> None:
    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    with st.container(border=True):
        st.subheader("Result", divider=True)
        if prediction == 0:
            st.success(":green[**LOW RISK**]")
            st.success("""
            You are at low risk for pneumonia admission. Keep monitoring your health
            and follow preventive measures to stay well.
            """)
        else:
            st.error("**HIGH RISK**")
            st.error("""
            Based on your information, you are at high risk for pneumonia admission.
            Please consult with your healthcare provider for further evaluation and guidance.
            """)

        class_0_proba = round(probabilities[0] * 100, 2)
        class_1_proba = round(probabilities[1] * 100, 2)
        result = pd.DataFrame(
            {
                "Prediction": ["Low", "High"],
                "Percentage (%)": [class_0_proba, class_1_proba],
            }
        )
        result = result.set_index("Prediction")

        with st.container(border=True):
            st.subheader("Result Details")
            st.markdown(f"""
            Based on the predictor model's result, there's a **{class_0_proba}%** probability of
            high risk and **{class_1_proba}%** of low risk for pneumonia admission.
            """)
            st.bar_chart(result, y_label="Risk Percentage (%)", color=["#0f0"])


def format_input(session, columns: list[str]) -> pd.DataFrame:
    user_input = [
        st.session_state.age,
        0 if st.session_state.sex == "Male" else 1,
    ]
    user_input.extend(
        list(
            map(
                lambda x: 1 if x else 0,
                [
                    st.session_state.crd,
                    st.session_state.dm,
                    st.session_state.hf,
                    st.session_state.cn,
                    st.session_state.ckd,
                ],
            )
        )
    )
    user_input = pd.DataFrame(
        [user_input],
        columns=columns,
    )
    return user_input


def main() -> None:
    st.set_page_config(
        page_title="Pneumonia Admission Predictor",
    )

    c1, c2, c3 = st.columns(3)
    with c2:
        st.image("./images/icon.png", width=224)

    # header
    st.html(
        """
        <div align="center" style="margin-top: -25px;">
            <h1>Pneumonia Admission Predictor</h1>
            <p style="font-style: italic; margin-top: -15px;">
                Jan Maverick M. Juat, Dionmelle J. Pardilla
                Sthanly Paul L. Malapit, Darylle P. Villanueva
            </p>
        </div>
        """
    )

    tab1, tab2, tab3 = st.tabs(["About", "Local Setup", "Predict"])

    with tab1:
        st.subheader("About this project")
        st.markdown("""
        This app helps predict the likelihood of hospital admission for patients
        with pneumonia. By analyzing key medical indicators and patient data, it
        provides healthcare professionals with insights to make more informed decisions,
        optimizing patient care and resource allocation.

        For predictor, we provided two machine learning models:

        - `RfActiveSMOTE`, and
        - `RfSMOTE`

        To use the app, go to `Predict` section.

        To use them with your own datasets, see the
        [documentation](https://datsudo.github.io/pneumonia-predictor/getting-started/model-usage/).

        To set this web app locally, see `Local Setup` tab section.
        """)

    with tab2:
        st.subheader("Local setup")
        st.markdown("""
        First, make sure you have the following prerequisites installed on your
        machine:
        - Python (3.12+)
        - Poetry (1.8.3+) - for virtual environment setup

        Clone this repository by running:
        ```bash
        git clone https://huggingface.co/spaces/datsudo/pneumonia-predictor
        ```

        Setup and enter the virtual environment using `poetry`:
        ```bash
        poetry install  # installs all the dependencies
        poetry shell
        ```

        Start the web app using `streamlit`
        ```bash
        streamlit run app.py
        ```

        For more information about this project, check out the
        [documentation](https://datsudo.github.io/pneumonia-predictor).
        """)

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
