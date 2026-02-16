

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import uvicorn

from pipeline import run_knowledge_assistant

# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(title="Adaptive Knowledge Assistant")


class AskRequest(BaseModel):
    query: str
    file_path: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: AskRequest):
    result = run_knowledge_assistant(request.query, request.file_path)
    return result


# ----------------------------
# Gradio UI
# ----------------------------

def gradio_handler(query, file):
    try:
        file_path = file if file else None

        result = run_knowledge_assistant(query, file_path)

        answer = result.get("answer", "")
        chunks = "\n\n".join(result.get("retrieved_chunks", []))
        evaluation = "\n".join(
            [f"{k}: {v}" for k, v in result.get("evaluation", {}).items()]
        )

        return answer, chunks, evaluation

    except Exception as e:
        return f"Error: {str(e)}", "", ""


with gr.Blocks(title="Adaptive Knowledge Assistant") as demo:
    gr.Markdown("# 🤖 Adaptive Knowledge Assistant")

    file_input = gr.File(label="Upload Document", type="filepath")
    query_input = gr.Textbox(label="Ask a Question")
    submit = gr.Button("Run")

    answer_output = gr.Textbox(label="Answer")
    chunks_output = gr.Textbox(label="Retrieved Chunks")
    eval_output = gr.Textbox(label="Evaluation")

    submit.click(
        gradio_handler,
        inputs=[query_input, file_input],
        outputs=[answer_output, chunks_output, eval_output],
    )


# Mount Gradio inside FastAPI
app = gr.mount_gradio_app(app, demo, path="/")


# Run via uvicorn if local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
