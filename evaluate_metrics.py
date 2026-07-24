"""
Compute evaluation metrics on a saved output/qa_results_*.json file
(produced by run_eval.py). Reads a file already on disk, so it doesn't
call the Claude API or cost anything to run.

Saves the summary to its own file in output/, same pattern as ask.py/
run_eval.py, so past evaluation runs are never overwritten.

    python evaluate_metrics.py                                    # most recent qa_results file
    python evaluate_metrics.py output/qa_results_20260724_174344.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import OUTPUT_DIR


def normalize_topic(text):
    """
    data/QA.txt's Topic column says "Computer Science" (with a space);
    the document metadata's subject says "ComputerScience" (no space).
    Same subject, different spelling - normalize both before comparing.
    """
    return text.replace(" ", "").lower()


def retrieval_topic_hit_rate(topic, sources):
    """
    Of the chunks retrieved for this question, what fraction came from a
    document tagged with the question's own topic/subject? A proxy for
    "did retrieval find relevant material," independent of whether
    Claude's final wording was correct.
    """
    if not sources:
        return 0.0

    normalized_topic = normalize_topic(topic)
    hits = sum(
        1 for s in sources if normalize_topic(s.get("subject", "")) == normalized_topic
    )
    return hits / len(sources)


def find_latest_results_file():
    files = sorted(OUTPUT_DIR.glob("qa_results_*.json"))
    if not files:
        raise FileNotFoundError(
            "No output/qa_results_*.json files found. Run run_eval.py first."
        )
    return files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_file",
        nargs="?",
        default=None,
        help="Path to a qa_results_*.json file (default: most recent in output/)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file) if args.results_file else find_latest_results_file()
    print(f"Evaluating: {results_path}\n")

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    per_question = []
    correct_count = 0
    topic_hit_rates = []

    for item in results:
        is_correct = item["rough_match"]
        hit_rate = retrieval_topic_hit_rate(item["topic"], item["sources"])

        if is_correct:
            correct_count += 1
        topic_hit_rates.append(hit_rate)

        per_question.append(
            {
                "question": item["question"],
                "rough_match": is_correct,
                "retrieval_topic_hit_rate": hit_rate,
            }
        )

        status = "correct" if is_correct else "miss   "
        print(f"[{status}] {item['question'][:70]}")

    n = len(results)
    rough_match_rate = correct_count / n
    avg_topic_hit_rate = sum(topic_hit_rates) / n

    print("\n--- Summary ---")
    print(f"Questions evaluated:      {n}")
    print(f"Rough match (correct):    {correct_count}/{n} ({rough_match_rate:.1%})")
    print(f"Retrieval topic hit rate: {avg_topic_hit_rate:.1%}")

    # Save this evaluation to its own file, so past runs are never overwritten.
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"evaluation_{timestamp}.json"

    summary = {
        "source_file": str(results_path),
        "questions_evaluated": n,
        "correct_count": correct_count,
        "rough_match_rate": rough_match_rate,
        "retrieval_topic_hit_rate": avg_topic_hit_rate,
        "per_question": per_question,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved evaluation summary to {out_file}")


if __name__ == "__main__":
    main()
