
import faiss
import numpy as np
from pathlib import Path

def store_embeddings(embeddings, original_file_path):
    original_name=Path(original_file_path).stem #removes .pdf or .txt
    vector_dir=Path("vectors")

    index_path=vector_dir/f"{original_name}.index"
    vector_dir.mkdir(exist_ok=True, parents=True)


    # Convert embeddings to float32 (FAISS requirement)
    embeddings=np.array(embeddings).astype('float32')

    # Get embedding dimension
    dimension = embeddings.shape[1]

    # Create fresh index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    # Save index (overwrite every time)

    faiss.write_index(index, str(index_path))


    print(f"Stored {index.ntotal} vectors in {index_path}")
