
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
from memory.chat_memory import ChatMemory

chat_memory=ChatMemory(max_messages=8)


def run_knowledge_assistant(file_url: str, question: str):
    
    
    chunk_size = 500
    chunk_overlap = 50

    local_file_path = download_file(file_url)
    file_content = clean_document_text(get_file_content(local_file_path))
    
    text_chunks = chunk_text(file_content, chunk_size, chunk_overlap)
    embedded_text_vector = embed_text(text_chunks)
    store_embeddings(embedded_text_vector, local_file_path)

    distance, indices = retrieve(question, local_file_path, 10)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]

    model, tokenizer = get_llm_model()
    
    # Add user message to chat memory
    chat_memory.add_user(question)

    answer,trimmed_chunks = generate_answer(model,tokenizer,retrieved_chunks, indices, question, chat_memory.get_messages(),300) #explicitely passing question as would be trimming conversation for token management

    # Add assistant response to memory
    chat_memory.add_assistant(answer)

    recent_history = chat_memory.get_recent_conversation(n_turns=1)

    #evaluation_response = evaluation_answer(model,tokenizer,question,answer,trimmed_chunks,chat_memory.get_messages,500)
    evaluation_response = evaluation_answer(model,tokenizer,question, answer, trimmed_chunks,recent_history,500)

    return {
        "answer": answer,
        "evaluation": evaluation_response,
        "retrieved_chunks": retrieved_chunks,
        "conversation": chat_memory.get_messages()
    }
