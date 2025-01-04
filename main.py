from app.translator import Translator
import gradio as gr

# Initialize the translator
translator = Translator(
    model_path="models/model2.h5",
    tokenizer_path="models/tokenizers.pkl",
    max_eng=15,  # Encoder max length
    max_fr=15,   # Decoder max length
)

def translate(input_text):
    return translator.translate(input_text)

def gradio_interface():
    with gr.Blocks() as interface:
        gr.Markdown("### English to French Translator")
        input_text = gr.Textbox(label="Enter text in English")
        output_text = gr.Textbox(label="Translation in French")
        translate_button = gr.Button("Translate")
        translate_button.click(translate, inputs=input_text, outputs=output_text)
    return interface

if __name__ == "__main__":
    gradio_interface().launch(
        server_name="127.0.0.1",
        server_port=8000,
        share=True
    )

