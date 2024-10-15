import pandas as pd
import streamlit as st

PAGES_DIR = "pneumonia_predictor/frontend/pages"


def display_banner() -> None:
    st.set_page_config(page_title="Pneumonia Admission Predictor")
    st.image("images/banner.png")
    st.html("""
        <div align="center" style="margin-top: -25px;">
            <h1>Pneumonia Admission Predictor</h1>
            <p style="font-style: italic; margin-top: -15px;">
                Jan Maverick M. Juat, Dionmelle J. Pardilla
                Sthanly Paul L. Malapit, Darylle P. Villanueva
            </p>
        </div>
    """)


def display_page(page_name: str) -> None:
    with open(f"{PAGES_DIR}/{page_name}.md") as f:
        contents = f.read()
    st.markdown(contents)


def display_result(
    prediction: int, result: pd.DataFrame, class_0_proba: float, class_1_proba: float
) -> None:
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
        Please consult with your healthcare provider for further evaluation and
        guidance.
        """)

    with st.container(border=True):
        st.markdown("#### Result Details")
        st.markdown(f"""
        Based on the predictor model's result, there's a **{class_1_proba}%** probability
        of high risk and **{class_0_proba}%** of low risk for pneumonia admission.
        """)
        st.bar_chart(result, y_label="Risk Percentage (%)", color=["#0f0"])
