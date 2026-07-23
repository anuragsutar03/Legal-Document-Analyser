"""
ingest.py
Loads legal documents from data/documents, splits them into legally meaningful
chunks (by SECTION/CASE boundaries rather than arbitrary character counts),
embeds each chunk, and stores them in a local ChromaDB collection.

Run:
    python src/ingest.py
"""

import os
import re
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "legal_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good default. Swap for a
# legal-domain model (e.g. "nlpaueb/legal-bert-base-uncased" via a different
# loading path) if you want higher domain accuracy.


def split_into_chunks(text: str, doc_title: str):
    """
    Splits legal text on natural boundaries (SECTION / CASE headers) instead
    of fixed character windows, which preserves legal argument structure.
    Falls back to paragraph splitting if no section headers are found.
    """
    section_pattern = re.compile(r"(?=^(?:SECTION \d+|CASE):)", re.MULTILINE)
    parts = [p.strip() for p in section_pattern.split(text) if p.strip()]

    if len(parts) <= 1:
        # fallback: split on blank lines (paragraphs)
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for i, part in enumerate(parts):
        header_match = re.match(r"(SECTION \d+|CASE):\s*(.*)", part)
        label = header_match.group(2).strip() if header_match else f"Part {i+1}"
        chunks.append({
            "text": part,
            "section_label": label,
            "doc_title": doc_title,
        })
    return chunks


def load_documents():
    docs = []
    for filepath in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        title_match = re.search(r"TITLE:\s*(.*)", text)
        doc_title = title_match.group(1).strip() if title_match else os.path.basename(filepath)
        docs.append({"filename": os.path.basename(filepath), "title": doc_title, "text": text})
    return docs


def build_index():
    print("Loading documents...")
    documents = load_documents()
    print(f"Found {len(documents)} documents.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads on first run)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh index each run for reproducibility in a course project
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_ids, all_texts, all_metadatas = [], [], []
    chunk_id = 0
    for doc in documents:
        chunks = split_into_chunks(doc["text"], doc["title"])
        for chunk in chunks:
            all_ids.append(str(chunk_id))
            all_texts.append(chunk["text"])
            all_metadatas.append({
                "doc_title": chunk["doc_title"],
                "section_label": chunk["section_label"],
                "source_file": doc["filename"],
            })
            chunk_id += 1

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True).tolist()

    print("Writing to vector store...")
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
    )

    print(f"Done. Indexed {len(all_texts)} chunks from {len(documents)} documents into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()
