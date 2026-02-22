

import torch
import json
from evaluation.evaluation_prompt import evaluation_prompt_builder

def evaluation_answer(model,tokenizer,question, answer, trimmed_chunks,recent_chats,max_new_tokens):
  device = next(model.parameters()).device
  
  max_context = model.config.max_position_embeddings
  max_input_tokens = max_context - max_new_tokens

  conversation_text = ""
  for msg in recent_chats:
    role = msg["role"].capitalize()
    conversation_text += f"{role}: {msg['content']}\n"
  
  # -------- Clean Document Context --------
  document_context = "\n".join(trimmed_chunks)
  
  prompt=evaluation_prompt_builder(conversation_text, document_context, question, answer)

  

  # -------- Token Budget Check --------
  inputs = tokenizer(prompt, return_tensors="pt")

  if inputs["input_ids"].shape[1] > max_input_tokens:
        # Hard truncate safely (simple version)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        )

  inputs = inputs.to(device)
  
  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]

  with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
  generated_tokens = outputs[0][input_ids.shape[1]:]
  eval_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

  return eval_text
