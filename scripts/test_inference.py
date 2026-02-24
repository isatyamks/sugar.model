# -*- coding: utf-8 -*-
"""
Smoke test for the reflection model.

Usage:
    python test_inference.py --model-path ../training/output/reflection-lora
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.model import ReflectionModel


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


def main():
    parser = argparse.ArgumentParser(description="Test model inference")
    parser.add_argument("--model-path", required=True, help="Path to model")
    args = parser.parse_args()

    model = ReflectionModel(args.model_path)
    model.load()

    print("=" * 60)
    print("INFERENCE SMOKE TEST")
    print("=" * 60)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i} ---")
        print(f"  Activity: {tc['title']}")
        print(f"  Age: {tc['age']} | Framework: {tc['framework']}")
        print(f"  Stage: {tc['stage']} | History: {tc['history_count']}")

        result = model.generate(**tc)
        print(f"  Output: {result}")

    print(f"\nHealth check: {'PASS' if model.health_check() else 'FAIL'}")


if __name__ == "__main__":
    main()
