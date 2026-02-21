import torch
from models.llm_loader import get_llm_model
from retrieval.trim_chunks import trim_chunks_to_fit
from prompt.build_prompt import build_prompt

def generate_answer(model,tokenizer,retrieved_chunks,indices,question,max_new_tokens):

  device = next(model.parameters()).device

  # Get model max context length
  max_context = model.config.max_position_embeddings
  max_input_tokens = max_context - max_new_tokens

  # Tokenize prompt + question to measure their length
  base_prompt=build_prompt([], question)
  base_tokens = len(tokenizer(base_prompt)["input_ids"])

  #Calculate available token budget
  available_for_context = max_input_tokens - base_tokens

  if available_for_context <= 0:
    raise ValueError("System + Question exceed model context window")

  #Trim chunks first and then truncate ONLY context if needed

  #Trim at TEXT level BEFORE tokenization
  trimmed_chunks = trim_chunks_to_fit(retrieved_chunks, tokenizer, available_for_context)

  #call prompt function to build prompt basis question and context text
  prompt=build_prompt(trimmed_chunks,question)

  #final token length check 
  final_token_len = len(tokenizer(prompt)["input_ids"])
  if final_token_len > max_input_tokens:
    print ("\nThere was token budgeting error\n")

  # Tokenize once
  inputs = tokenizer(prompt, return_tensors="pt").to(device)
  
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
