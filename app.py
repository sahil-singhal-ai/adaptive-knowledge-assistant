

import gradio as gr
from pipeline import run_knowledge_assistant
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from models.llm_loader import get_llm_model

import gradio as gr
from pipeline import run_knowledge_assistant


def handle_query(query, file):
    try:
        # file will be a file path (string) if uploaded
        file_path = file if file else None

        result = run_knowledge_assistant(query, file_path)

        answer = result.get("answer", "")
        chunks = "\n\n".join(result.get("retrieved_chunks", []))

        evaluation_dict = result.get("evaluation", {})
        evaluation = "\n".join(
            [f"{k}: {v}" for k, v in evaluation_dict.items()]
        )

        return answer, chunks, evaluation

    except Exception as e:
        return f"Error: {str(e)}", "", ""


with gr.Blocks() as demo:

    gr.Markdown("# Simple Knowledge Assistant (Test Deployment)")

    file_input = gr.File(label="Upload Document", type="filepath")
    query_input = gr.Textbox(label="Ask a Question")

    submit_btn = gr.Button("Run")

    answer_output = gr.Textbox(label="Answer")
    chunks_output = gr.Textbox(label="Retrieved Chunks")
    eval_output = gr.Textbox(label="Evaluation Metrics")

    submit_btn.click(
        handle_query,
        inputs=[query_input, file_input],
        outputs=[answer_output, chunks_output, eval_output],
    )

demo.launch()
