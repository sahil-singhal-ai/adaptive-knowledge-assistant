import torch

def summarize_memory(model, tokenizer, chat_messages, max_new_tokens=120):

    # Remove system message
    conversation_only = [
        msg for msg in chat_messages
        if msg["role"] != "system"
    ]

    # Build compact conversation text
    conversation_text = ""
    for msg in conversation_only:
        role = msg["role"].capitalize()
        conversation_text += f"{role}: {msg['content']}\n"

    if conversation_text.strip() == "":
        return ""

    prompt = f"""
Summarize the following conversation in 2-3 concise lines.
Focus only on important facts, user intent, and key conclusions.
Do not add new information.

Conversation:
{conversation_text}

Summary:
"""

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return summary.strip()