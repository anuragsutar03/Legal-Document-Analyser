"""
retrieve.py
Given a natural-language legal question, embeds it and retrieves the most
relevant chunks from the ChromaDB index built by ingest.py.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "legal_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = 4):
    """
    Returns a list of dicts: {text, doc_title, section_label, source_file, distance}
    ordered by relevance (lower distance = more relevant).
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "doc_title": results["metadatas"][0][i]["doc_title"],
            "section_label": results["metadatas"][0][i]["section_label"],
            "source_file": results["metadatas"][0][i]["source_file"],
            "distance": results["distances"][0][i],
        })
    return retrieved


if __name__ == "__main__":
    # quick manual test
    q = "What happens if a party says in advance they won't deliver goods?"
    for r in retrieve(q):
        print(f"[{r['doc_title']} - {r['section_label']}] (distance={r['distance']:.3f})")
        print(r["text"][:200], "...\n")
