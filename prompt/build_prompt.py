
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

"""

  return final_prompt
