from ingestion.download_file import download_file
from ingestion.read_file import get_file_content
from ingestion.chunker import chunk_text
from models.embedder import embed_text
from retrieval.vectorstore import store_embeddings
from retrieval.retriever import retrieve
from prompt.build_prompt import build_prompt
from prompt.evaluation_prompt import evaluation_prompt_builder
from models.llm_loader import get_llm_model
from answer_generator.answer_generator import generate_answer


def run_knowledge_assistant(file_url: str, question: str):
    
    
    chunk_size = 500
    chunk_overlap = 50

    local_file_path = download_file(file_url)
    file_content = get_file_content(local_file_path)
    
    text_chunks = chunk_text(file_content, chunk_size, chunk_overlap)
    embedded_text_vector = embed_text(text_chunks)
    store_embeddings(embedded_text_vector, local_file_path)

    distance, indices = retrieve(question, local_file_path, 10)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]

    prompt = build_prompt(retrieved_chunks, question)
    answer = generate_answer(prompt, 300)

    evaluation_prompt = evaluation_prompt_builder(question, retrieved_chunks, answer)
    evaluation_response = generate_answer(evaluation_prompt, 1000)

    return {
        "answer": answer,
        "evaluation": evaluation_response,
        "retrieved_chunks": retrieved_chunks
    }
