"""
STEP 1 of the RAG pipeline: load the raw data files and turn them into
plain Python objects we can work with in the next steps.
"""

import csv

from langchain_core.documents import Document

from src.config import DOCUMENTS_CSV, QA_FILE


def load_knowledge_base_documents():
    """
    Read data/documents.csv and turn it into a list of "documents".

    The CSV has one row per SENTENCE, tagged with which subject and
    document it belongs to, like this:

        subject          document   sentence
        ComputerScience  1          Algorithms are instructions...
        ComputerScience  1          Algorithms outline a beginning...

    We don't want to embed single sentences on their own (too little
    context), so we glue all the sentences of the same (subject, document)
    pair back together into one paragraph. Each paragraph becomes one
    Document that we can later chunk and embed.
    """

    # sentences_by_doc looks like: {("ComputerScience", "1"): ["sentence 1", "sentence 2", ...]}
    sentences_by_doc = {}
    doc_order = []  # keeps track of the order documents first appeared in

    with open(DOCUMENTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            doc_key = (row["subject"], row["document"])

            if doc_key not in sentences_by_doc:
                sentences_by_doc[doc_key] = []
                doc_order.append(doc_key)

            sentences_by_doc[doc_key].append(row["sentence"].strip())

    # Now join each document's sentences into one paragraph.
    documents = []
    for subject, document_id in doc_order:
        sentences = sentences_by_doc[(subject, document_id)]
        paragraph = " ".join(sentences)

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
