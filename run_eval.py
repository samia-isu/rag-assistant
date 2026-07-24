"""
Run a batch of sample questions from data/QA.txt through the pipeline and
save the results to output/.

This is a small demo/smoke-test, not a real grading mechanism (rough
case-insensitive substring match against the expected answer) - useful for
eyeballing pipeline quality and estimating API cost per question.

Each run saves to its own file, e.g. output/qa_results_20260724_153012.json,
so previous runs are never overwritten.

    python run_eval.py --num 10
"""

import argparse
import json
from datetime import datetime

from src.config import OUTPUT_DIR
from src.ingest import load_qa_pairs
from src.rag_pipeline import answer_question, load_llm
from src.vectorstore import load_vectorstore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="How many QA pairs to run")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs()[: args.num]

    # Load the vector index and connect to Claude ONCE, then reuse them
    # for every question in the loop below (no need to reload each time).
    vectorstore = load_vectorstore()
    llm = load_llm()

    results = []
    correct_count = 0
    total_tokens = 0

    for pair in qa_pairs:
        result = answer_question(pair["question"], vectorstore, llm)

        # Very rough check: does the expected answer appear anywhere in
        # Claude's answer? Good enough for a sanity check, not real grading.
        is_match = pair["answer"].lower() in result["answer"].lower()
        if is_match:
            correct_count += 1

        usage = result["usage"] or {}
        tokens_used = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
        total_tokens += tokens_used

        results.append(
            {
                "topic": pair["topic"],
                "question": pair["question"],
                "expected_answer": pair["answer"],
                "rag_answer": result["answer"],
                "rough_match": is_match,
                "sources": result["sources"],
                "usage": usage,
            }
        )

        status = "match" if is_match else "miss "
        print(f"[{status}] {pair['question']} -> {result['answer']}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"qa_results_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{correct_count}/{len(qa_pairs)} rough matches, ~{total_tokens} tokens used")
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
