"""
Build the vector index from data/documents/.
Run this once, and again anytime you add/change a document file.

    python build_index.py
"""

from src.config import INDEX_DIR
from src.ingest import load_knowledge_base_documents
from src.vectorstore import build_vectorstore


def main():
    # STEP 1: load every file in data/documents/ as one document
    documents = load_knowledge_base_documents()
    print(f"Loaded {len(documents)} documents from data/documents/")

    # STEP 2: split into chunks, embed them, save the FAISS index to disk
    vectorstore = build_vectorstore(documents)
    print(f"Saved FAISS index with {vectorstore.index.ntotal} chunks to {INDEX_DIR}")


if __name__ == "__main__":
    main()
