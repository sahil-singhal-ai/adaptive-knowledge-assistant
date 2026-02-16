
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

A FastAPI-based Retrieval-Augmented Generation (RAG) system with:

- Document ingestion
- Chunking
- Embeddings
- Vector retrieval
- LLM-based answer generation
- Evaluation layer

## 🏗 Architecture

FastAPI → Pipeline → Embeddings → Vector Store → LLM → Evaluation

## 🚀 Deployment

This Space uses a custom Dockerfile and runs on port 7860 using Uvicorn.

## 🔒 Security Notes

- API keys are stored using Hugging Face Secrets
- No credentials are hardcoded
- CORS will be restricted in production

## 📌 Endpoints

- `GET /` → Health check
- `POST /ask` → Ask questions over uploaded document URL
