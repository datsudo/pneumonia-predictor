import streamlit as st
import pandas as pd
import joblib


PREDICTOR = joblib.load("saved_models/pneumonia_predictor.pkl")


def predict(new_data: pd.DataFrame) -> None:
    prediction = PREDICTOR.predict(new_data)[0]
    probabilities = PREDICTOR.predict_proba(new_data)[0]

    with st.container(border=True):
        st.subheader("Result", divider=True)
        if prediction == 0:
            st.success(":green[**LOW RISK**]")
        else:
            st.error("**HIGH RISK**")

        result = pd.DataFrame(
            {
                "Prediction": ["Low", "High"],
                "Percentage (%)": [
                    round(probabilities[0] * 100, 2),
                    round(probabilities[1] * 100, 2),
                ],
            }
        )
        result = result.set_index("Prediction")

        with st.container(border=True):
            st.bar_chart(result, y_label="Risk Percentage (%)")


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
        columns,
    )
    return user_input


def main() -> None:
    st.set_page_config(
        page_title="Pneumonia Admission Predictor",
    )

    st.title("Pneumonia Admission Predictor")

    st.html(
        "<p style='font-style: italic; margin-top: -15px;'>"
        + "Jan Maverick M. Juat, Dionmelle J. Pardilla,"
        + "Sthanly Paul L. Malapit, Darylle P. Villanueva</p>"
    )

    st.markdown("""
    Hybrid approach predictor model using Random Forest Integrated with Active SMOTE
    to predict pneumonia admissions
    """)

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

        predict(X_input)


if __name__ == "__main__":
    main()
