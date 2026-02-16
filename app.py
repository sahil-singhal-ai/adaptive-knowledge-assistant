
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


from pipeline import run_knowledge_assistant

app = FastAPI(title="Adaptive Knowledge Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    file_url: str
    question: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Adaptive Knowledge Assistant</h1>
    <p>System is running.</p>
    <a href="/docs">Go to API Docs</a>
    """

@app.post("/ask")
def ask(request: QueryRequest):
    result = run_knowledge_assistant(
        file_url=request.file_url,
        question=request.question
    )
    return result
