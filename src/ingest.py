"""
STEP 1 of the RAG pipeline: load the raw data files and turn them into
plain Python objects we can work with in the next steps.
"""

import csv

from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR, QA_FILE


def load_knowledge_base_documents():
    """
    Read every CSV file in data/documents/ and turn it into a "document".

    This is one real CSV file per document (e.g. data/documents/Physics_1.csv),
    named "<subject>_<document id>.csv" - the same layout a real deployment
    would use: 16 separate source documents, not one spreadsheet of loose
    sentences. Each file has its own subject/document/sentence rows (same
    columns as the original data/documents.csv, just filtered to that one
    document). All of a file's sentences are joined into one paragraph,
    which becomes one Document that we can later chunk and embed.
    """

    documents = []

    # sorted() so the order is stable/predictable every run
    for file_path in sorted(DOCUMENTS_DIR.glob("*.csv")):
        with open(file_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        sentences = [row["sentence"].strip() for row in rows if row["sentence"].strip()]
        paragraph = " ".join(sentences)

        # subject/document are the same on every row of this file, so the
        # first row tells us both.
        subject = rows[0]["subject"]
        document_id = rows[0]["document"]

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
