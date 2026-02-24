# -*- coding: utf-8 -*-
"""
Evaluate the fine-tuned reflection model against a held-out dataset.

Usage:
    python evaluate.py --model-path ../training/output/reflection-lora
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.model import ReflectionModel
from evaluation.metrics import score_single, score_batch


def parse_input_field(input_str):
    """Parse a pipe-separated input string into a dictionary."""
    parts = {}
    for segment in input_str.split(" | "):
        if ": " in segment:
            key, value = segment.split(": ", 1)
            key = key.strip().lower().replace(" ", "_")
            parts[key] = value.strip()
    return parts


def main():
    parser = argparse.ArgumentParser(description="Evaluate reflection model")
    parser.add_argument(
        "--model-path", required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--eval-data", default="../data/eval_set.jsonl",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Max examples to evaluate"
    )
    args = parser.parse_args()

    print("Loading model...")
    model = ReflectionModel(args.model_path)
    model.load()

    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        print(f"Error: {eval_path} not found")
        print("Run: python ../data/build_dataset.py --split-only")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f if line.strip()]

    if args.max_examples:
        eval_data = eval_data[:args.max_examples]

    print(f"Evaluating {len(eval_data)} examples...\n")

    results = []
    for i, example in enumerate(eval_data):
        parsed = parse_input_field(example["input"])

        generated = model.generate(
            title=parsed.get("activity", "Unknown"),
            activity_type=parsed.get("type", "Unknown"),
            duration_min=int(parsed["duration"].replace(" min", ""))
                if "duration" in parsed else None,
            age=int(parsed["age"]) if "age" in parsed else None,
            framework=parsed.get("framework", "what_so_what"),
            stage=parsed.get("stage", "what"),
            history_count=int(parsed.get("history", 0)),
        )

        results.append({
            "input": example["input"],
            "reference": example["output"],
            "generated": generated,
            "title": parsed.get("activity", ""),
            "age": int(parsed["age"]) if "age" in parsed else None,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(eval_data)}...")

    all_scores, summary = score_batch(results)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Examples evaluated: {summary['count']}")
    print()
    print("Metric Scores (0.0 = worst, 1.0 = best):")
    for metric, value in summary.items():
        if metric not in ("count", "avg_total"):
            print(f"  {metric:25s}: {value:.2f}")
    print(f"\n  Average total score:      {summary['avg_total']:.2f}")

    print("\n" + "-" * 60)
    print("SAMPLE OUTPUTS")
    print("-" * 60)
    for r in results[:5]:
        print(f"\nInput:     {r['input']}")
        print(f"Expected:  {r['reference']}")
        print(f"Generated: {r['generated']}")
        scores = score_single(r["generated"], title=r["title"], age=r["age"])
        print(f"Scores:    {scores}")

    results_path = Path(args.model_path) / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()
