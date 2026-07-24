"""
Split data/documents.csv (255 sentences, tagged by subject + document id)
into one CSV file per document - 16 files total.

Each output file keeps the same columns as the original (subject, document,
sentence), just filtered down to that one document's rows, and is named
after its subject and document number, e.g. data/documents_csv/Physics_3.csv

Run:
    python data_analysis.py
"""

import csv
from collections import defaultdict
from pathlib import Path

SOURCE_CSV = Path("data/documents.csv")
OUTPUT_DIR = Path("data/documents_csv")

# Smart/curly quotes found in the source data, mapped to their plain ASCII
# equivalents, so every file uses the same characters consistently.
QUOTE_REPLACEMENTS = {
    "‘": "'",  # ‘
    "’": "'",  # ’
    "“": '"',  # “
    "”": '"',  # ”
}


def clean_sentence(sentence: str) -> str:
    """
    Normalize one sentence:
      - strip stray leading/trailing whitespace (found in several rows)
      - collapse any repeated internal whitespace into single spaces
      - replace smart/curly quotes with plain straight quotes
    """
    for curly, straight in QUOTE_REPLACEMENTS.items():
        sentence = sentence.replace(curly, straight)

    return " ".join(sentence.split())


def main():
    # STEP 1: read the original CSV and group its rows by (subject, document)
    rows_by_doc = defaultdict(list)
    doc_order = []  # keeps track of the order documents first appeared in

    with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames  # ["subject", "document", "sentence"]

        for row in reader:
            row["sentence"] = clean_sentence(row["sentence"])
            if not row["sentence"]:
                continue  # skip any rows that turn out empty after cleaning

            doc_key = (row["subject"], row["document"])
            if doc_key not in rows_by_doc:
                doc_order.append(doc_key)
            rows_by_doc[doc_key].append(row)

    # STEP 2: write each document's rows to its own CSV file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subject, document_id in doc_order:
        out_path = OUTPUT_DIR / f"{subject}_{document_id}.csv"

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows_by_doc[(subject, document_id)])

        print(f"wrote {out_path} ({len(rows_by_doc[(subject, document_id)])} rows)")

    print(f"\nDone: {len(doc_order)} CSV files written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
