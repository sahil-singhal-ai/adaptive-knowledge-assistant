
def build_prompt(context_chunks:list,conversation_text,question,system_instruction):
  context_text = "\n\n".join(context_chunks)

  final_prompt = f"""

{system_instruction}

Current user question : {question}

Relevant Document Context which you should definitely for answering user question:
{context_text}

Previous Conversation summary (if relevant):
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
