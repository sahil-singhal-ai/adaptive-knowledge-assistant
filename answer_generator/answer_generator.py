import torch
from models.llm_loader import get_llm_model
from retrieval.trim_chunks import trim_chunks_to_fit

def generate_answer(prompt,retrieved_chunks,question,max_new_tokens=300):
  model, tokenizer = get_llm_model()

  device = next(model.parameters()).device

  # Get model max context length
  max_context = model.config.max_position_embeddings
  max_input_tokens = max_context - max_new_tokens

  # Tokenize prompt + question to measure their length
  prompt_len = len(tokenizer(prompt)["input_ids"])
  question_len = len(tokenizer(question)["input_ids"])

  #Calculate available token budget
  available_for_context = max_input_tokens - prompt_len - question_len

  if available_for_context <= 0:
    raise ValueError("System + Question exceed model context window")

  #Trim chunks first and then truncate ONLY context if needed

  
  #Trim at TEXT level BEFORE tokenization
  trimmed_chunks = trim_chunks_to_fit(retrieved_chunks, tokenizer, available_for_context)
  

  # Join context
  context_text = "\n\n".join(trimmed_chunks)

  # Build final input text
  final_input_text = prompt + context_text + question

  # Tokenize once
  inputs = tokenizer(final_input_text, return_tensors="pt").to(device)
  
  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]

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
