"""
STEP 1 of the RAG pipeline: load the raw data files and turn them into
plain Python objects we can work with in the next steps.
"""

import csv

from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR, QA_FILE


def load_knowledge_base_documents():
    """
    Read every file in data/documents/ and turn it into a "document".

    This is one real text file per document (e.g. data/documents/Physics_1.txt),
    named "<subject>_<document id>.txt" - the same layout a real deployment
    would use: a folder of source documents, not a spreadsheet of loose
    sentences. Each file's lines are joined into one paragraph, which
    becomes one Document that we can later chunk and embed.
    """

    documents = []

    # sorted() so the order is stable/predictable every run
    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        subject, document_id = file_path.stem.rsplit("_", 1)
        lines = file_path.read_text(encoding="utf-8").splitlines()
        paragraph = " ".join(line.strip() for line in lines if line.strip())

        documents.append(
            Document(
                page_content=paragraph,
                metadata={"subject": subject, "document_id": document_id},
            )
        )

    return documents


def load_qa_pairs():
    """
    Read data/QA.txt and return a list of question/answer dictionaries,
    e.g. {"topic": "Statistics", "answer": "Sample", "question": "A ___ is..."}.

    These are only used to test the pipeline (run_eval.py) - they are NOT
    part of the knowledge base that gets embedded/searched.
    """
    qa_pairs = []

    with open(QA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qa_pairs.append(
                {
                    "topic": row["Topic"].strip(),
                    "answer": row["ANSWER"].strip(),
                    "question": row["Questions"].strip(),
                }
            )

    return qa_pairs
