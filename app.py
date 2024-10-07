import gradio as gr

import pneumonia_predictor.config as config
from pneumonia_predictor.frontend import components


# FIXME: Temporary
def get_inputs(age: int, sex: int, conditions: list[int]) -> str:
    return f"{age=}\n{sex=}\n{conditions=}"


if __name__ == "__main__":
    with gr.Blocks(theme=components.DefaultTheme, title=config.APP_TITLE) as app:
        gr.HTML(components.Header)
        gr.HTML(components.Description)

        with gr.Row():
            with gr.Column():
                age = gr.Number(
                    label="Age",
                    info="Patient's age (must be between 18-65 years)",
                    minimum=18,
                    maximum=65,
                )
                sex = gr.Radio(
                    label="Sex",
                    info="Patient's sex",
                    choices=["Male", "Female"],
                    type="index",
                )

                conditions = gr.CheckboxGroup(
                    label="Indicate whether the patient has any of the following conditions:",
                    choices=[
                        "Chronic respiratory disease",
                        "Diabetes mellitus",
                        "Heart failure",
                        "Cancer",
                        "Chronic kidney disease",
                    ],
                )

                predict_btn = gr.Button(value="Predict", variant="primary")

            out = gr.Label(
                label="Result",
            )

            # FIXME: Temporary
            predict_btn.click(
                fn=get_inputs,
                inputs=[age, sex, conditions],
                outputs=out,
                api_name="outputs",
            )

    app.launch()
