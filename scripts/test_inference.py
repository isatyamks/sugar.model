# -*- coding: utf-8 -*-
"""
Smoke test — run examples through the model and save result.json.

Usage:
    python test_inference.py --model-path ../training/output/reflection-lora
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.model import ReflectionModel

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_FILE = SCRIPT_DIR.parent / "result.json"

TEST_CASES = [
    {
        "title": "TurtleArt Activity",
        "duration_min": 15,
        "age": 9,
        "framework": "what_so_what",
        "stage": "what",
        "history_count": 0,
    },
    {
        "title": "Write Activity",
        "mime_type": "text/plain",
        "duration_min": 25,
        "age": 11,
        "framework": "kolb",
        "stage": "reflective_observation",
        "history_count": 2,
    },
    {
        "title": "Pippy Activity",
        "mime_type": "application/json",
        "duration_min": 35,
        "age": 14,
        "framework": "gibbs",
        "stage": "analysis",
        "history_count": 3,
    },
    {
        "title": "Paint Activity",
        "mime_type": "image/png",
        "duration_min": 8,
        "age": 5,
        "framework": "what_so_what",
        "stage": "what",
        "history_count": 0,
    },
    {
        "title": "Music Blocks Activity",
        "duration_min": 20,
        "age": 10,
        "framework": "kolb",
        "stage": "active_experimentation",
        "history_count": 1,
    },
]


def score_output(text):
    """Simple quality scoring."""
    if not text or not text.strip():
        return "poor", "Empty output"
    text = text.strip()
    has_question = "?" in text
    is_short = len(text.split()) <= 40
    echoes_input = "Activity:" in text and "Framework:" in text

    if echoes_input:
        return "poor", "Echoed input prompt instead of generating a question"
    if has_question and is_short:
        return "good", "Well-formed reflection question"
    if has_question and not is_short:
        return "fair", "Contains a question but too verbose"
    if not has_question:
        return "poor", "No question mark — not a reflection question"
    return "fair", "Partially correct output"


def main():
    parser = argparse.ArgumentParser(description="Test model and save results")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output JSON path")
    args = parser.parse_args()

    print("Loading model...")
    model = ReflectionModel(args.model_path)
    model.load()

    print("=" * 60)
    print("INFERENCE SMOKE TEST")
    print("=" * 60)

    results = []
    good_count = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i} ---")
        print(f"  Activity: {tc['title']}")
        print(f"  Age: {tc['age']} | Framework: {tc['framework']}")
        print(f"  Stage: {tc['stage']} | History: {tc['history_count']}")

        output = model.generate(**tc)
        print(f"  Output: {output}")

        quality, issue = score_output(output)
        print(f"  Quality: {quality} — {issue}")

        if quality == "good":
            good_count += 1

        results.append({
            "test": i,
            "input": {
                "activity": tc["title"],
                "mime_type": tc.get("mime_type", ""),
                "age": tc["age"],
                "framework": tc["framework"],
                "stage": tc["stage"],
                "duration_min": tc.get("duration_min"),
                "history": tc.get("history_count", 0),
            },
            "output": output.strip(),
            "quality": quality,
            "issue": issue,
        })

    # Health check
    health = model.health_check()
    print(f"\nHealth check: {'PASS' if health else 'FAIL'}")

    # Build final JSON
    total = len(results)
    result_json = {
        "model": {
            "name": "reflection-lora",
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "method": "LoRA fine-tuning",
            "adapter_path": args.model_path,
        },
        "summary": {
            "total_tests": total,
            "good": sum(1 for r in results if r["quality"] == "good"),
            "fair": sum(1 for r in results if r["quality"] == "fair"),
            "poor": sum(1 for r in results if r["quality"] == "poor"),
            "pass_rate": f"{sum(1 for r in results if r['quality'] in ('good', 'fair')) / total:.0%}",
        },
        "health_check": "PASS" if health else "FAIL",
        "inference_results": results,
    }

    # Save
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"Score: {result_json['summary']['good']}/{total} good, "
          f"{result_json['summary']['fair']}/{total} fair, "
          f"{result_json['summary']['poor']}/{total} poor")


if __name__ == "__main__":
    main()
