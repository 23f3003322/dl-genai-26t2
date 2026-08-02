import gradio as gr

from src.inference import predict


MIN_QUESTION_WORDS = 3


def _validate_and_predict(
    question_text: str,
    option_a_text: str,
    option_b_text: str,
    option_c_text: str,
    option_d_text: str,
    option_e_text: str,
) -> tuple[str, str, dict[str, float]]:
    question_clean = question_text.strip()
    options = [
        option_a_text.strip(),
        option_b_text.strip(),
        option_c_text.strip(),
        option_d_text.strip(),
        option_e_text.strip(),
    ]

    question_word_count = len(question_clean.split())
    if question_word_count < MIN_QUESTION_WORDS:
        raise gr.Error(
            f"Question should contain at least {MIN_QUESTION_WORDS} words. "
            f"Current count: {question_word_count}."
        )

    for idx, option_text in enumerate(options):
        if not option_text:
            option_label = chr(ord("A") + idx)
            raise gr.Error(f"Option {option_label} should not be empty.")

    return predict(
        question_clean,
        options[0],
        options[1],
        options[2],
        options[3],
        options[4],
    )


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(
    title="Smart MCQ Solver"
) as demo:

    gr.Markdown(
        """
        # 🧠 Smart MCQ Solver

        Predict the correct answer for a Multiple Choice Question using a
        **BiLSTM + Attention** model trained from scratch.

        Enter a question and its five answer choices below.
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            question = gr.Textbox(
                label="Question",
                lines=5,
                placeholder="Enter the question..."
            )

            option_a = gr.Textbox(
                label="Option A"
            )

            option_b = gr.Textbox(
                label="Option B"
            )

            option_c = gr.Textbox(
                label="Option C"
            )

            option_d = gr.Textbox(
                label="Option D"
            )

            option_e = gr.Textbox(
                label="Option E"
            )

            predict_button = gr.Button(
                "🚀 Predict",
                variant="primary"
            )

            clear_button = gr.Button(
                "🗑 Clear"
            )

        with gr.Column(scale=1):

            predicted_answer = gr.Textbox(
                label="Predicted Answer"
            )

            top3_predictions = gr.Textbox(
                label="Top-3 Ranking"
            )

            confidence_scores = gr.Label(
                label="Confidence Scores"
            )

    predict_button.click(
        fn=_validate_and_predict,
        inputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        ],
        outputs=[
            predicted_answer,
            top3_predictions,
            confidence_scores
        ]
    )

    clear_button.click(
        lambda: ("", "", "", "", "", "", "", "", {}),
        outputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e,
            predicted_answer,
            top3_predictions,
            confidence_scores
        ]
    )

    gr.Markdown(
        """
        ---
        **Model:** BiLSTM + Attention (PyTorch)

        **Task:** Multiple Choice Question Answering (5 options)

        """
    )

demo.launch(
    theme=gr.themes.Soft(),
    server_name="0.0.0.0",
    server_port=7860
)
