
def evaluation_prompt_builder(question, context, answer):
  evaluation_prompt = f"""You are an expert evaluator for a retrieval-augmented generation system.
User Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the answer on:

1. Groundedness (1-5)
2. Relevance (1-5)
3. Completeness (1-5)
4. Hallucination Risk (Low/Medium/High)

Provide a short explanation and return JSON:
{{
  "groundedness": int,
  "relevance": int,
  "completeness": int,
  "hallucination_risk": str,
  "overall_score": float,
  "feedback": str
}}
"""
  return evaluation_prompt
