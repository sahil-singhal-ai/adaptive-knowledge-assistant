

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from pipeline import run_knowledge_assistant


# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(title="Adaptive Knowledge Assistant")


class AskRequest(BaseModel):
    file_url: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: AskRequest):
    return run_knowledge_assistant(
        file_url=request.file_url,
        question=request.question
    )


# ----------------------------
# Gradio UI (URL Only)
# ----------------------------

def gradio_handler(file_url, question):
    try:
        result = run_knowledge_assistant(
            file_url=file_url,
            question=question
        )

        answer = result.get("answer", "")
        chunks = "\n\n".join(result.get("retrieved_chunks", []))
        evaluation = str(result.get("evaluation", ""))

        return answer, chunks, evaluation

    except Exception as e:
        return f"Error: {str(e)}", "", ""


with gr.Blocks(title="Adaptive Knowledge Assistant") as demo:

    gr.Markdown("# 🤖 Adaptive Knowledge Assistant")
    gr.Markdown("Provide a public document URL and ask a question.")

    url_input = gr.Textbox(label="Public File URL")
    question_input = gr.Textbox(label="Question")

    submit_btn = gr.Button("Run")

    answer_output = gr.Textbox(label="Answer")
    chunks_output = gr.Textbox(label="Retrieved Chunks")
    eval_output = gr.Textbox(label="Evaluation")

    submit_btn.click(
        gradio_handler,
        inputs=[url_input, question_input],
        outputs=[answer_output, chunks_output, eval_output],
    )


# Mount Gradio inside FastAPI at root path
app = gr.mount_gradio_app(app, demo, path="/")
