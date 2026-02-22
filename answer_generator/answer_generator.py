import torch

from memory.summary_memory import summarize_memory
from retrieval.trim_chunks import trim_chunks_to_fit
from retrieval.trim_conversation import trim_conversation_to_fit
from prompt.build_prompt import build_prompt

def generate_answer(model,tokenizer,retrieved_chunks,indices,question,chat_messages,max_new_tokens):

  device = next(model.parameters()).device

  max_context = model.config.max_position_embeddings
  max_input_tokens = max_context - max_new_tokens

  system_instruction = next(
        (msg["content"] for msg in chat_messages if msg["role"] == "system"),
        ""
    )

  system_tokens = len(tokenizer(system_instruction)["input_ids"])
  question_tokens = len(tokenizer(question)["input_ids"])

  base_tokens = system_tokens + question_tokens

  if base_tokens >= max_input_tokens:
    raise ValueError("Question exceeds model context window")

  remaining_tokens = max_input_tokens - base_tokens

  # 🔥 Summarize conversation first
  conversation_summary = summarize_memory(
        model,
        tokenizer,
        chat_messages
    )


  #Dynamic allocation
  # Give context priority (70%)
  context_budget = int(remaining_tokens * 0.7)
  conversation_budget = remaining_tokens - context_budget

  #Trim conversation first
  trimmed_summary_list = trim_conversation_to_fit(
        [conversation_summary],   # wrap as list for reuse
        tokenizer,
        conversation_budget
    )

  trimmed_summary = "\n".join(trimmed_summary_list)
  actual_conv_tokens = len(tokenizer(trimmed_summary)["input_ids"])


  # Give unused conversation budget back to context
  unused_conv_budget = conversation_budget - actual_conv_tokens
  context_budget += max(unused_conv_budget, 0)

  #Trim context chunks
  trimmed_chunks = trim_chunks_to_fit(retrieved_chunks, tokenizer, context_budget)

  context_text = "\n".join(trimmed_chunks)
  context_tokens = len(tokenizer(context_text)["input_ids"])
  
  #call prompt function to build prompt basis question and context text
  prompt = build_prompt(
        context_chunks=trimmed_chunks,
        conversation_text=trimmed_summary,
        question=question,
        system_instruction=system_instruction
    )


  final_token_len = len(tokenizer(prompt)["input_ids"])
  if final_token_len > max_input_tokens:
    raise ValueError("Final prompt exceeds context window")

  # Tokenize once
  inputs = tokenizer(prompt, return_tensors="pt").to(device)
  
  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]

  prompt_token_count = input_ids.shape[1]

  with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
  

  #Only decode newly generated tokens
  generated_tokens = outputs[0][input_ids.shape[1]:]
  answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

  response_token_count = generated_tokens.shape[0]

  # ---------------- METADATA ----------------
  generation_meta = {
        "trimmed_chunks": trimmed_chunks,
        "prompt_tokens": int(prompt_token_count),
        "context_tokens": int(context_tokens),
        "conversation_tokens": int(actual_conv_tokens),
        "response_tokens": int(response_token_count),
        "max_context_window": int(max_context),
        "model_name": getattr(model.config, "_name_or_path", "unknown")
    }

  return answer.strip(),generation_meta
