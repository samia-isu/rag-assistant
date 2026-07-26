import argparse
import json
from datetime import datetime

from src.config import OUTPUT_DIR
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

    # Save this run to its own file, named with the current date/time so
    # every run gets a fresh file instead of overwriting the last answer.
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"answer_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved result to {out_file}")


if __name__ == "__main__":
    main()
