
model = None

def get_embedding_model(): # Renamed function
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def embed_text(texts):
    model = get_embedding_model() # Updated call
    return model.encode(texts)
