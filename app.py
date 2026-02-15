
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    file_url: str
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    result = run_knowledge_assistant(
        file_url=request.file_url,
        question=request.question
    )
    return result
