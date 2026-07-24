"""
Build the vector index from data/documents.csv.
Run this once, and again anytime you change documents.csv.

    python build_index.py
"""

from src.config import INDEX_DIR
from src.ingest import load_knowledge_base_documents
from src.vectorstore import build_vectorstore


def main():
    # STEP 1: load and group the raw sentences into full documents
    documents = load_knowledge_base_documents()
    print(f"Loaded {len(documents)} documents from data/documents.csv")

    # STEP 2: split into chunks, embed them, save the FAISS index to disk
    vectorstore = build_vectorstore(documents)
    print(f"Saved FAISS index with {vectorstore.index.ntotal} chunks to {INDEX_DIR}")


if __name__ == "__main__":
    main()
