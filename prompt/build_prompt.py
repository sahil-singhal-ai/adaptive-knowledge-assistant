
def build_prompt(context_chunks:list,conversation_text,question,system_instruction):
  context_text = "\n\n".join(context_chunks)

  final_prompt = f"""

{system_instruction}

User question : {question}

Document Context:
{context_text}

Conversation:
{conversation_text}

Instructions:
- Answer the latest user question using the document context.
- Maintain conversational continuity.
- Suggest one intelligent follow-up question and briefly answer it.

Format:

Answer:
...
Follow-up Question:
...
Follow-up Answer:
...
"""

  return final_prompt
