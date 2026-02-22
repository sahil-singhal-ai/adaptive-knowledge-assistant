
FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install numpy first to prevent version conflicts with faiss
RUN pip install --no-cache-dir numpy==1.26.4

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time (internet is available during build, not runtime)

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
