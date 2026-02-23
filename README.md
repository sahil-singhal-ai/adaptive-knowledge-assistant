
---
title: Adaptive Knowledge Assistant
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🤖 Adaptive Knowledge Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system with multi-turn memory, dynamic context management, automated evaluation, and structured logging — deployed via Docker on Hugging Face Spaces.

[![Live Demo - slow due to free tier CPU](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face%20Spaces-orange)](https://huggingface.co/spaces/sahilsinghal/adaptive-knowledge-assistant)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

---


## 📌 What This Project Demonstrates

This project showcases end-to-end applied ML engineering for a document-grounded QA system. It goes beyond a basic RAG tutorial by solving real production challenges:

- **Dynamic context window management** — intelligent budget allocation between retrieved chunks and conversation history, with unused tokens reallocated automatically
- **LLM-summarized memory** — instead of naively truncating chat history, the system uses the LLM itself to summarize past turns before injecting them into new prompts
- **Automated answer evaluation** — a second LLM pass scores every answer on groundedness, relevance, completeness, and hallucination risk using a structured JSON rubric
- **Session-aware multi-turn conversations** — isolated conversation state per user via UUID session management
- **Structured JSONL logging** — every request is logged with token counts, evaluation scores, and metadata for observability
- **REST API + Gradio UI** — dual interfaces served from a single FastAPI application

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
│          Gradio UI (chat interface)  │  FastAPI REST /ask endpoint   │
└─────────────────────────────┬───────────────────────┬───────────────┘
                              │                       │
                              ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ORCHESTRATOR (pipeline.py)           │
│  Session Manager (UUID-based)  →  ChatMemory  →  RequestLogger       │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │                                  │
    ┌──────▼──────────┐              ┌─────────▼────────────┐
    │  INGESTION       │              │  RETRIEVAL            │
    │  ─────────────  │              │  ───────────────────  │
    │  download_file   │              │  embed_text (query)   │
    │  read_file       │              │  FAISS vector search  │
    │  clean_document  │              │  top-k chunk recall   │
    │  chunk_text      │              └─────────┬────────────┘
    │  embed_text      │                        │
    │  store_embeddings│                        │ retrieved_chunks
    └──────────────────┘                        │
                                                ▼
                              ┌─────────────────────────────────────┐
                              │       ANSWER GENERATOR               │
                              │  ─────────────────────────────────  │
                              │  1. Summarize chat memory (LLM)      │
                              │  2. Allocate token budget            │
                              │     ├── 70% → context chunks         │
                              │     └── 30% → conversation summary   │
                              │  3. Trim chunks + conversation       │
                              │  4. Build prompt                     │
                              │  5. Generate (Qwen2.5-1.5B-Instruct) │
                              └─────────────┬───────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────────────┐
                              │       EVALUATION LAYER               │
                              │  ─────────────────────────────────  │
                              │  Second LLM pass scores answer on:   │
                              │  • Groundedness (1–5)                │
                              │  • Relevance (1–5)                   │
                              │  • Completeness (1–5)                │
                              │  • Hallucination Risk (Low/Med/High) │
                              │  • Overall Score + Feedback          │
                              └─────────────┬───────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────────────┐
                              │       STRUCTURED LOGGER              │
                              │  JSONL log: tokens, scores, chunks   │
                              └─────────────────────────────────────┘
```

---

## 🔄 Data Flow — Request Lifecycle

```
User Question
     │
     ▼
[1] Download & Cache Document (skip if already exists)
     │
     ▼
[2] Clean → Chunk (500 chars, 50 overlap) → Embed → Store in FAISS
     │
     ▼
[3] Embed Query → FAISS Top-10 Recall
     │
     ▼
[4] LLM Summarizes Chat History → Trimmed Summary
     │
     ▼
[5] Dynamic Token Budget Allocation
     ├── context_budget = (max_input_tokens - base_tokens) × 0.7
     └── conversation_budget = remaining × 0.3
              └── unused budget redistributed to context
     │
     ▼
[6] Build Prompt: [System | Question | Context Chunks | Conv Summary]
     │
     ▼
[7] LLM Generate Answer (Qwen2.5-1.5B-Instruct, 4-bit quantized)
     │
     ▼
[8] Evaluation Pass → JSON scores
     │
     ▼
[9] Update Chat Memory + Log to JSONL
     │
     ▼
[10] Return: { answer, evaluation, retrieved_chunks, conversation, logs }
```

---

## 📁 Project Structure

```
adaptive-knowledge-assistant/
│
├── app.py                        # FastAPI app + Gradio UI (dual interface)
├── pipeline.py                   # Main orchestrator — wires all modules
├── requirements.txt
├── Dockerfile
│
├── ingestion/
│   ├── download_file.py          # Downloads + caches PDFs/TXTs from URL
│   ├── read_file.py              # Parses PDF (pypdf) and plain text
│   ├── cleaner.py                # Boilerplate removal, deduplication, whitespace normalization
│   └── chunker.py                # Character-level chunking with overlap
│
├── models/
│   ├── embedder.py               # Sentence Transformer (all-MiniLM-L6-v2), singleton
│   └── llm_loader.py             # Qwen2.5-1.5B-Instruct, 4-bit quantized (bitsandbytes)
│
├── retrieval/
│   ├── vectorstore.py            # FAISS IndexFlatL2 — build & persist index
│   ├── retriever.py              # Query embedding → FAISS top-k search
│   ├── trim_chunks.py            # Token-aware chunk selection
│   └── trim_conversation.py      # Token-aware conversation trimming
│
├── memory/
│   ├── chat_memory.py            # Session-isolated message store (system + N turns)
│   └── summarize_memory.py       # LLM-powered conversation summarizer
│
├── answer_generator/
│   └── answer_generator.py       # Full generation pipeline with budget management
│
├── prompt/
│   └── build_prompt.py           # Prompt assembly: system + question + context + memory
│
└── evaluation/
    ├── evaluation_prompt.py      # Evaluator prompt builder (structured JSON rubric)
    ├── evaluation_answer.py      # Runs second LLM pass for scoring
    └── logger.py                 # JSONL request/response logger
```

---

## ⚙️ Key Technical Design Decisions

### 1. Dynamic Token Budget Allocation
Rather than using fixed-size context windows, the system computes available tokens at runtime:

```python
remaining_tokens = max_input_tokens - system_tokens - question_tokens
context_budget   = int(remaining_tokens * 0.7)   # retrieval-first priority
conv_budget      = remaining_tokens - context_budget

# Unused conversation budget flows back to context
unused = conv_budget - actual_conv_tokens
context_budget += max(unused, 0)
```

This means longer system prompts or longer questions automatically reduce the retrieval context — no silent truncation or crashes.

### 2. LLM-Based Memory Summarization
Instead of truncating raw chat messages, the model first summarizes the conversation into 2–3 lines of key facts and intent. This preserves semantic continuity across turns while consuming far fewer tokens.

### 3. 4-Bit Quantized Inference
The LLM runs under `BitsAndBytesConfig` with `nf4` 4-bit quantization and double quantization enabled. This allows running a 1.5B parameter model efficiently on a single GPU, while keeping response quality acceptable for document QA.

### 4. Automated Evaluation Loop
Every answer is scored by the same LLM using a separate evaluation prompt. The rubric covers:
- **Groundedness**: Is the answer supported by the retrieved context?
- **Relevance**: Does it directly address the question?
- **Completeness**: Is it thorough given the available context?
- **Hallucination Risk**: Does it introduce information not in the documents?

### 5. Session Isolation
Each Gradio session or API client gets a UUID-keyed `ChatMemory` object stored in `conversation_store`. System instructions are preserved across turns; user/assistant messages roll off after a configurable window (default: 8 messages).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA GPU recommended (CPU inference is slow for generation)

### Local Setup - Option 1
```bash
git clone https://github.com/yourusername/adaptive-knowledge-assistant.git
cd adaptive-knowledge-assistant

pip install -r requirements.txt
import pipeline
import json
file_url = "https://sherlock-holm.es/stories/plain-text/stud.txt"
question1 = "What is Holme's detective style. Be very detailed and cover all aspects of his detective style"
result=pipeline.run_knowledge_assistant(file_url,question1)
print (result['answer'])
print ("\n\n")
question2 = "Who are the other main characters described in the story and what is their character persona"
result=pipeline.run_knowledge_assistant(file_url,question2)
print (result['answer'])
```



## 📓 Demo Notebook

A clean, production-style demo notebook is included:

**`Adaptive Knowledge Assistant_demo notebook.ipynb`**

The notebook serves as a **thin client** – all business logic lives in Python modules, demonstrating proper separation of concerns.


### Local Setup - Option 2

```bash
git clone https://github.com/yourusername/adaptive-knowledge-assistant.git
cd adaptive-knowledge-assistant

pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860` for the Gradio UI, or use the REST API at `http://localhost:7860/ask`.

### Docker

```bash
docker build -t knowledge-assistant .
docker run -p 7860:7860 knowledge-assistant
```

---

## 🔌 API Reference

### Health Check
```
GET /health
→ { "status": "ok" }
```

### Ask a Question
```
POST /ask
Content-Type: application/json

{
  "file_url": "https://example.com/document.pdf",
  "question": "What are the main conclusions of this report?",
  "conversation_id": "session-abc123"
}
```

**Response:**
```json
{
  "answer": "The report concludes that...",
  "evaluation": {
    "groundedness": 4,
    "relevance": 5,
    "completeness": 3,
    "hallucination_risk": "Low",
    "overall_score": 4.0,
    "feedback": "Answer is well-grounded in the provided context..."
  },
  "retrieved_chunks": ["...chunk 1...", "...chunk 2..."],
  "conversation": [...],
  "logs": { "prompt_tokens": 412, "context_tokens": 287, ... }
}
```

---

## 📊 Observability — Structured Logging

Every request produces a JSONL log entry:

```json
{
  "timestamp": "2025-01-15T10:32:11.123456",
  "conversation_id": "uuid-session-xyz",
  "question": "What does section 3 say about risk?",
  "retrieved_chunk_count": 10,
  "answer": "Section 3 discusses...",
  "prompt_tokens": 512,
  "context_tokens": 320,
  "response_tokens": 95,
  "evaluation": {
    "groundedness": 5,
    "relevance": 4,
    "completeness": 4,
    "hallucination_risk": "Low",
    "overall_score": 4.3,
    "feedback": "..."
  }
}
```

Logs are appended to `logs.jsonl` and can be streamed into any observability stack (Datadog, Elastic, BigQuery, etc.).

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Qwen2.5-1.5B-Instruct (HuggingFace Transformers) |
| Quantization | BitsAndBytes (NF4, 4-bit, double quant) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (IndexFlatL2) |
| API Framework | FastAPI + Pydantic |
| UI | Gradio (mounted inside FastAPI) |
| PDF Parsing | pypdf |
| Deployment | Docker → Hugging Face Spaces |
| Logging | JSONL structured logs |

---

## 🗺️ Roadmap / Future Improvements

- [ ] **Hybrid retrieval** — BM25 + dense vector re-ranking for better recall on lexically precise queries
- [ ] **Multi-document support** — per-document FAISS indices with document routing
- [ ] **Streaming responses** — token-by-token output via SSE for lower perceived latency
- [ ] **Evaluation dashboard** — Gradio tab with score trends across sessions
- [ ] **Fine-tuned evaluator** — replace self-evaluation with a dedicated reward model
- [ ] **Metadata filtering** — chunk-level page/section metadata for citation support
- [ ] **OpenAI-compatible endpoint** — drop-in compatibility for existing tooling

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.


---

## 👤 Author

Built by **Sahil Singhal** — connecting AI product thinking with applied ML engineering.
