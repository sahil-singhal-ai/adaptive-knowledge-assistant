import torch
from models.llm_loader import get_llm_model

def generate_answer(prompt,max_new_tokens=300):
  model, tokenizer = get_llm_model()

  device = next(model.parameters()).device

  # Get model max context length
  max_context = model.config.max_position_embeddings

  # Tokenize without truncation first
  inputs= tokenizer(prompt, return_tensors="pt").to(device)
  input_ids = inputs["input_ids"]

  # Calculate safe input length
  max_input_tokens = max_context - max_new_tokens

  if input_ids.shape[1] > max_input_tokens:
    input_ids = input_ids[:, -max_input_tokens:]  # Keep last tokens only

  input_ids = input_ids.to(device)
  attention_mask = inputs["attention_mask"].to(device)

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
