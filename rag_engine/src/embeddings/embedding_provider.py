from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    model = _load_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()


def embed_query(text):
    model = _load_model()
    embeddings = model.encode(text)
    return embeddings.tolist()
