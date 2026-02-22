
import uuid
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
    conversation_id: str = "api_session"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: AskRequest):
    return run_knowledge_assistant(
        file_url=request.file_url,
        question=request.question,
        conversation_id=request.conversation_id
    )


# ----------------------------
# Gradio UI (Multi-Turn + Logs + Evaluation)
# ----------------------------

def gradio_handler(file_url, question, chat_history, logs_state, session_id):
    try:
        result = run_knowledge_assistant(
            file_url=file_url,
            question=question,
            conversation_id=session_id
        )

        answer = result.get("answer", "")
        evaluation = result.get("evaluation", {})
        logs = result.get("logs", {})
        retrieved_chunks = result.get("retrieved_chunks", [])

        # Update chat history
        chat_history.append((question, answer))

        # Accumulate logs
        logs_state.append(logs)

        # Format evaluation nicely
        if isinstance(evaluation, dict):
            evaluation_str = "\n".join(
                f"{k}: {v}" for k, v in evaluation.items()
            )
        else:
            evaluation_str = str(evaluation)

        logs_str = "\n\n".join(str(l) for l in logs_state)
        chunks_str = "\n\n".join(retrieved_chunks)

        return (
            chat_history,
            evaluation_str,
            logs_str,
            chunks_str,
            chat_history,
            logs_state,
        )

    except Exception as e:
        return (
            chat_history,
            "",
            f"Error: {str(e)}",
            "",
            chat_history,
            logs_state,
        )


with gr.Blocks(title="Adaptive Knowledge Assistant") as demo:

    gr.Markdown("# 🤖 Adaptive Knowledge Assistant")
    gr.Markdown("Multi-turn Document Q&A with Evaluation and Structured Logs")

    url_input = gr.Textbox(label="Public File URL")

    chatbot = gr.Chatbot(label="Conversation")

    question_input = gr.Textbox(label="Ask a Question")

    submit_btn = gr.Button("Ask")

    # 🔥 Session ID (one per user session)
    session_state = gr.State(str(uuid.uuid4()))

    # 🔥 Chat + Logs state storage
    chat_state = gr.State([])
    logs_state = gr.State([])

    with gr.Accordion("📊 Evaluation", open=False):
        eval_output = gr.Textbox(label="Evaluation", lines=10)

    with gr.Accordion("🧠 Retrieved Chunks", open=False):
        chunks_output = gr.Textbox(label="Retrieved Chunks", lines=10)

    with gr.Accordion("📁 Logs", open=False):
        logs_output = gr.Textbox(label="Logs", lines=12)

    submit_btn.click(
        gradio_handler,
        inputs=[
            url_input,
            question_input,
            chat_state,
            logs_state,
            session_state,
        ],
        outputs=[
            chatbot,
            eval_output,
            logs_output,
            chunks_output,
            chat_state,
            logs_state,
        ],
    )


# Mount Gradio inside FastAPI
app = gr.mount_gradio_app(app, demo, path="/")
