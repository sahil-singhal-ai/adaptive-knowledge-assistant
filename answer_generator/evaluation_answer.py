import torch
from prompt.evaluation_prompt import evaluation_prompt_builder

def evaluation_answer(model,tokenizer,question, answer, trimmed_chunks,recent_chats,max_new_tokens):
  
  conversation_text = ""
  for msg in recent_chats:
    role = msg["role"].capitalize()
    conversation_text += f"{role}: {msg['content']}\n"
  
  prompt=evaluation_prompt_builder(conversation_text, trimmed_chunks, question, answer)

  device = next(model.parameters()).device

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



