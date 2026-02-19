
def trim_chunks_to_fit(retrieved_chunks, tokenizer, max_context_tokens):
    total_tokens = 0
    selected_chunks = []

    for chunk in retrieved_chunks:
        chunk_tokens = len(tokenizer(chunk)["input_ids"])
        
        if total_tokens + chunk_tokens <= max_context_tokens:
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            break

    return selected_chunks
