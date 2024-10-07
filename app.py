import streamlit as st
import pandas as pd
import joblib


PREDICTOR = joblib.load("saved_models/pneumonia_predictor.pkl")


def predict(new_data: pd.DataFrame) -> None:
    prediction = PREDICTOR.predict(new_data)
    probabilities = PREDICTOR.predict_proba(new_data)

    if prediction[0] == 0:
        st.markdown("**LOW RISK**:")
    else:
        st.write("HIGH RISK")

    st.markdown(f"""
    - Prediction: {prediction[0]}
    - Probabilities:
        - Low risk: {probabilities[0][0]}
        - High risk: {probabilities[0][1]}
    """)


def main() -> None:
    st.set_page_config(
        page_title="Pneumonia Admission Predictor",
    )

    st.title("Pneumonia Admission Predictor")

    # TODO: Author names
    st.html(
        "<p style='font-style: italic; margin-top: -15px;'>"
        + "Jan Maverick M. Juat, Dionmelle J. Pardilla,"
        + "Sthanly Paul L. Malapit, Darylle P. Villanueva</p>"
    )
    st.markdown("---")

    # TODO: Description
    st.write("""

    """)

    st.number_input(
        "Age (must be between 18-65 years)",
        key="age",
        min_value=18,
        max_value=65,
    )

    st.markdown("---")

    st.radio("Sex", key="sex", options=["Male", "Female"], captions=["0", "1"])

    st.markdown("---")

    st.write("Indicate whether the patient has any of the following conditions:")
    conditions = {
        "crd": "Chronic respiratory disease",
        "dm": "Diabetes mellitus",
        "hf": "Heart failure",
        "cn": "Cancer",
        "ckd": "Chronic kidney disease",
    }
    for cond in conditions:
        st.checkbox(label=conditions[cond], key=cond)

    # You can access the value at any point with:
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

    X_new = pd.DataFrame(
        [user_input],
        columns=[
            "age",
            "sex",
            "chronic_respiratory_disease",
            "diabetes_mellitus",
            "heart_failure",
            "cancer",
            "chronic_kidney_disease",
        ],
    )

    st.markdown("---")
    st.write("Your input:")

    X_new

    st.markdown("---")

    st.subheader("Assessment:")

    predict(X_new)


if __name__ == "__main__":
    main()
