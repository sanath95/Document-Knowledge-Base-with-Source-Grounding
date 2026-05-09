from __future__ import annotations

import gradio as gr

from app.answer import ABSTENTION
from app.config import load_settings
from app.pipeline import ask
from app.vector_store import collection_exists


EXAMPLES = [
    "What is Test1.pdf about?",
    "What does the document say about reporting?",
    "What is the German dissertation in Test2.pdf about?",
    "Which document mentions layout optimization?",
    "What does the material say about password rotation?",
]


def _status_message() -> str:
    settings = load_settings()
    if not settings.openai_api_key:
        return "OPENAI_API_KEY is missing. Add it to `.env` before asking questions."
    if not collection_exists(settings):
        return "No document index found. Run `uv run python -m app.ingest` before launching the demo."
    return "Ready. Ask a question about the indexed PDFs."


def respond(message: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    settings = load_settings()
    history = history or []
    if not message.strip():
        return history, ""
    history.append({"role": "user", "content": message})

    if not settings.openai_api_key:
        reply = "OPENAI_API_KEY is missing. Add it to `.env` and restart the app."
    elif not collection_exists(settings):
        reply = "No document index found. Run `uv run python -m app.ingest`, then restart the app."
    else:
        try:
            answer = ask(settings, message)
            reply = answer.text if answer.text.strip() else ABSTENTION
        except Exception as exc:
            reply = f"Sorry, something went wrong while answering: `{exc}`"
    history.append({"role": "assistant", "content": reply})
    return history, ""


def fill_example(example: str) -> str:
    return example


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Document Knowledge Base") as demo:
        gr.Markdown("# Document Knowledge Base with Source Grounding")
        status = gr.Markdown(_status_message())
        chatbot = gr.Chatbot(height=560)
        with gr.Row():
            question = gr.Textbox(
                placeholder="Ask a natural-language question about the PDFs...",
                show_label=False,
                scale=8,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)
        clear = gr.Button("Clear chat")
        gr.Examples(examples=EXAMPLES, inputs=question, label="Example questions")

        submit.click(respond, inputs=[question, chatbot], outputs=[chatbot, question])
        question.submit(respond, inputs=[question, chatbot], outputs=[chatbot, question])
        clear.click(lambda: ([], _status_message()), outputs=[chatbot, status])
    return demo


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
