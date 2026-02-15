
def build_prompt(context_chunks:list,question):
  retrieved_context = "\n\n".join(context_chunks)

  final_prompt = f"""Answer the question based on the context below.

  Context : {retrieved_context}

  Question : {question}

  """

  return final_prompt
