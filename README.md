
---
title: Adaptive Knowledge Assistant
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Adaptive Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system combining:

- FastAPI backend (REST API)
- Gradio UI (interactive interface)
- Docker deployment on Hugging Face Spaces

---

## 🏗 Architecture

Browser  
→ Gradio UI  
→ FastAPI Endpoints  
→ Pipeline  
→ Embeddings + Vector Store  
→ LLM  
→ Evaluation Layer  

---

## 📌 API Endpoints

- `GET /health` → Health check  
- `POST /ask` → Query the knowledge assistant  

---

## 🚀 Deployment

This Space runs using a custom Docker container and exposes port 7860 via Uvicorn.

---

## 🔐 Security

- API keys stored using Hugging Face Secrets
- No credentials hardcoded
