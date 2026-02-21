import torch

from retrieval.trim_chunks import trim_chunks_to_fit
from retrieval.trim_conversation import trim_conversation_to_fit
from prompt.build_prompt import build_prompt

def generate_answer(model,tokenizer,retrieved_chunks,indices,question,chat_messages,max_new_tokens):

  device = next(model.parameters()).device

  # Get model max context length
  max_context = model.config.max_position_embeddings
  max_input_tokens = max_context - max_new_tokens

  #Build role-based conversation string
  conversation_turns = []

  for message in chat_messages:
    role = message["role"].capitalize()
    content = message["content"]
    conversation_turns.append(f"{role}: {content}")

  system_instruction = "You are a helpful knowledge assistant."

  system_tokens = len(tokenizer(system_instruction)["input_ids"])
  question_tokens = len(tokenizer(question)["input_ids"])

  base_tokens = system_tokens + question_tokens

  if base_tokens >= max_input_tokens:
    raise ValueError("Question exceeds model context window")

  remaining_tokens = max_input_tokens - base_tokens

  #Dynamic allocation
  # Give context priority (60%)
  context_budget = int(remaining_tokens * 0.6)
  conversation_budget = remaining_tokens - context_budget

  #Trim conversation first
  trimmed_conversation = trim_conversation_to_fit(
        conversation_turns,
        tokenizer,
        conversation_budget
    )

  conversation_text = "\n".join(trimmed_conversation)

  # Recalculate actual conversation token usage
  actual_conv_tokens = len(tokenizer(conversation_text)["input_ids"])

  # Give unused conversation budget back to context
  unused_conv_budget = conversation_budget - actual_conv_tokens
  context_budget += max(unused_conv_budget, 0)


  #Trim context chunks
  trimmed_chunks = trim_chunks_to_fit(retrieved_chunks, tokenizer, context_budget)

  #call prompt function to build prompt basis question and context text
  prompt=build_prompt(trimmed_chunks,conversation_text,question,system_instruction)

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

  return answer.strip(),trimmed_chunks
