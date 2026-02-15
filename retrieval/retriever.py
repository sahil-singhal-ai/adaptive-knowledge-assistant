
from core.embedder import embed_text

#will have to import embed function in this when modularize

from pathlib import Path

def retrieve(question1:str,original_file_path,top_k:int =5):

  embedded_question_vector=embed_text(question1).astype("float32").reshape(1,-1)

  original_name=Path(original_file_path).stem #removes .pdf or .txt

  #load faiss index
  index = faiss.read_index("vectors/"+original_name+".index")

  distance,indices = index.search(embedded_question_vector,top_k)

  return distance,indices
