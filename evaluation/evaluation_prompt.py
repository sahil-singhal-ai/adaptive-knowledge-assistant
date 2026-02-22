


def evaluation_prompt_builder(conversation_text, document_context, question, answer):
  evaluation_prompt = f"""
  
You are an expert evaluator for a retrieval-augmented generation system.

Return ONLY valid JSON.
Do not include any explanation outside the JSON.
Do not add markdown.

User question : {question}

Answer:{answer}

Recent Conversation:
{conversation_text}


Document Context:
{document_context}

Evaluate:
1. Is the last assistant response grounded in context?
2. Is it consistent with conversation?
3. Is it logically coherent?

Return scores and justification.

Evaluate the answer on:

1. Groundedness (1-5)
2. Relevance (1-5)
3. Completeness (1-5)
4. Hallucination Risk (Low/Medium/High)


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
