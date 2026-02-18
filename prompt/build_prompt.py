
def build_prompt(context_chunks:list,question):
  retrieved_context = "\n\n".join(context_chunks)

  final_prompt = f"""
  You are a helpful assistant.

The following text is extracted from a larger document.

Based ONLY on the provided context, answer the question clearly.

If the context does not contain enough information, say so.
  

Context : {retrieved_context}

Question : {question}

"""

  return final_prompt
