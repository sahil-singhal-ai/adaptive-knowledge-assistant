import torch
from models.llm_loader import get_llm_model
from retrieval.trim_chunks import trim_chunks_to_fit

def generate_answer(prompt,retrieved_chunks,question,max_new_tokens=300):
  model, tokenizer = get_llm_model()

  device = next(model.parameters()).device

  # Get model max context length
  max_context = model.config.max_position_embeddings

  # Tokenize each component separately
  prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=False)
  context_tokens = tokenizer(retrieved_chunks, return_tensors="pt", add_special_tokens=False)

  prompt_ids = prompt_tokens["input_ids"].to(device)
  question_ids = question_tokens["input_ids"].to(device)
  context_ids = context_tokens["input_ids"].to(device)
  
  #Calculate available token budget
  max_input_tokens = max_context - max_new_tokens
  prompt_len = prompt_ids.shape[1]
  question_len = question_ids.shape[1]
  
  available_for_context = max_input_tokens - prompt_len - question_len

  if available_for_context <= 0:
    raise ValueError("System + Question exceed model context window")

  #Trim chunks first and then truncate ONLY context if needed

  if context_ids.shape[1] > available_for_context:
    #Keep most recent context tokens (important for RAG)
    context_ids = context_ids[:, -available_for_context:]

  #Concatenate properly

  input_ids = torch.cat([prompt_ids, context_ids, question_ids], dim=1)

  attention_mask = torch.ones_like(input_ids).to(device)

  with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)

  #Only decode newly generated tokens
  generated_tokens = outputs[0][input_ids.shape[1]:]
  answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

  return answer.strip()
