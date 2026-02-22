import time
from datetime import datetime
from ingestion.download_file import download_file
from ingestion.read_file import get_file_content
from ingestion.chunker import chunk_text
from ingestion.cleaner import clean_document_text
from models.embedder import embed_text
from retrieval.vectorstore import store_embeddings
from retrieval.retriever import retrieve
from models.llm_loader import get_llm_model
from answer_generator.answer_generator import generate_answer
from evaluation.evaluation_answer import evaluation_answer
from evaluation.logger import RequestLogger
from memory.chat_memory import ChatMemory

chat_memory=ChatMemory(max_messages=8)
logger=RequestLogger("logs.jsonl")

def run_knowledge_assistant(file_url: str, question: str,conversation_id: str = "local_session"):
    start_time=time.time()
    failure_type = None


    
    chunk_size = 500
    chunk_overlap = 50

    #Ingestion
    local_file_path = download_file(file_url)
    file_content = clean_document_text(get_file_content(local_file_path))
    
    text_chunks = chunk_text(file_content, chunk_size, chunk_overlap)
    embedded_text_vector = embed_text(text_chunks)
    store_embeddings(embedded_text_vector, local_file_path)

    #Retrieval
    distance, indices = retrieve(question, local_file_path, 10)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]

    #model load
    model, tokenizer = get_llm_model()
    
    # Add user message to chat memory
    chat_memory.add_user(question)

    answer,generation_meta = generate_answer(model,tokenizer,retrieved_chunks, indices, question, chat_memory.get_messages(),300) #explicitely passing question as would be trimming conversation for token management

    # Add assistant response to memory
    chat_memory.add_assistant(answer)

    #evaluatipon
    recent_history = chat_memory.get_recent_conversation(n_turns=1)

    #evaluation_response = evaluation_answer(model,tokenizer,question,answer,trimmed_chunks,chat_memory.get_messages,500)
    trimmed_chunks=generation_meta['trimmed_chunks']
    evaluation_response = evaluation_answer(model,tokenizer,question, answer, trimmed_chunks,recent_history,500)


    # Detect hallucination
    if evaluation_response.get("faithfulness_score", 1) < 0.5:
        failure_type = "hallucination"

    latency = int((time.time() - start_time) * 1000)

    # ---------------- STRUCTURED LOGGING ----------------
    log_object = {
        "timestamp": datetime.utcnow().isoformat(),
        "conversation_id": conversation_id,
        "question": question,
        "retrieved_chunk_count": len(retrieved_chunks),
        "prompt_tokens": generation_meta.get("prompt_tokens"),
        "context_tokens": generation_meta.get("context_tokens"),
        "response_tokens": generation_meta.get("response_tokens"),
        "latency_ms": latency,
        "evaluation": evaluation_response,
        "failure_type": failure_type
    }

    logger.log(log_object)


    return {
        "answer": answer,
        "evaluation": evaluation_response,
        "retrieved_chunks": retrieved_chunks,
        "conversation": chat_memory.get_messages(),
        "log_object": log_object
    }
