"""
Ask a single question against the RAG pipeline.

    python ask.py --question "What is inferential statistics?"
"""

import argparse
import json

from src.rag_pipeline import answer_question, load_llm
from src.vectorstore import load_vectorstore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Question to ask")
    args = parser.parse_args()

    # Load the saved vector index (built earlier by build_index.py)
    # and connect to Claude.
    vectorstore = load_vectorstore()
    llm = load_llm()

    result = answer_question(args.question, vectorstore, llm)

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {json.dumps(result['sources'], indent=2)}")
    if result["usage"]:
        print(f"\nToken usage: {result['usage']}")


if __name__ == "__main__":
    main()
