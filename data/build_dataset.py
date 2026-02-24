# -*- coding: utf-8 -*-
"""
Dataset builder for reflection prompt training data.

Reads seed examples from reflection_prompts.jsonl, expands the dataset
using a teacher LLM via the OpenAI API, and splits into train/eval sets.

Usage:
    python build_dataset.py --expand --api-key YOUR_KEY
    python build_dataset.py --split-only
"""

import json
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict

SEED_FILE = Path(__file__).parent / "reflection_prompts.jsonl"
EXPANDED_FILE = Path(__file__).parent / "reflection_prompts_expanded.jsonl"
TRAIN_FILE = Path(__file__).parent / "train.jsonl"
EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"

ACTIVITIES = [
    ("Write Activity", "", "text/plain"),
    ("Pippy Activity", "", "application/json"),
    ("TurtleArt Activity", "", ""),
    ("Calculate Activity", "", ""),
    ("Paint Activity", "", "image/png"),
    ("Browse Activity", "", ""),
    ("Scratch Activity", "", ""),
    ("Music Blocks Activity", "", ""),
    ("Terminal Activity", "", ""),
    ("New Source File 1", "", "text/x-python"),
    ("Untitled", "", ""),
    ("Untitled", "", "text/plain"),
    ("Untitled", "", "image/png"),
]

AGES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

FRAMEWORKS_STAGES = {
    "what_so_what": ["what", "so_what", "now_what"],
    "gibbs": ["description", "feelings", "evaluation",
              "analysis", "conclusion", "action_plan"],
    "kolb": ["concrete_experience", "reflective_observation",
             "abstract_conceptualization", "active_experimentation"],
}


def load_seed_examples() -> List[Dict]:
    """Load seed examples from the JSONL file."""
    examples = []
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_expansion_prompt(seeds: List[Dict], activity: str,
                           bundle_id: str, mime_type: str,
                           age: int, framework: str,
                           stage: str, history: int) -> str:
    """Construct a few-shot prompt for the teacher LLM."""
    relevant = [s for s in seeds
                 if framework in s["input"] and stage in s["input"]]
    if len(relevant) < 3:
        relevant = seeds[:5]
    demos = random.sample(relevant, min(3, len(relevant)))

    demo_text = ""
    for d in demos:
        demo_text += f"Input: {d['input']}\nOutput: {d['output']}\n\n"

    duration = random.choice([5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45])
    input_line = (
        f"Activity: {activity} | Bundle: {bundle_id} | "
        f"MIME: {mime_type} | Duration: {duration} min | Age: {age} | "
        f"Framework: {framework} | Stage: {stage} | History: {history}"
    )

    prompt = f"""You generate age-appropriate reflection questions for children using Sugar learning software.

The children use Sugar activities like Write, Paint, TurtleArt, Pippy (Python coding),
Calculate, Browse, Scratch, Music Blocks, and Terminal. Activities often have default
names like "Pippy Activity" or generic names like "Untitled" or "New Source File 1".

Given an activity context, generate a single reflection question. The question should:
- Be appropriate for the child's age ({age} years old)
- Follow the {framework} framework, {stage} stage
- Reference what the child was doing (infer from activity name and MIME type)
- Be warm, encouraging, and open-ended
- Be 1-2 sentences maximum
- Handle generic titles gracefully (don't just repeat "Untitled" awkwardly)

Here are some examples:

{demo_text}
Now generate ONE reflection question for:
Input: {input_line}
Output:"""

    return prompt


def expand_with_openai(seeds: List[Dict], api_key: str,
                       target_count: int = 500) -> List[Dict]:
    """Generate additional training examples using the OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: pip install openai")
        return seeds

    client = OpenAI(api_key=api_key)
    expanded = list(seeds)
    attempts = 0
    max_attempts = target_count * 2

    print(f"Starting expansion from {len(seeds)} seeds to {target_count}...")

    while len(expanded) < target_count and attempts < max_attempts:
        attempts += 1

        activity, bundle_id, mime_type = random.choice(ACTIVITIES)
        age = random.choice(AGES)
        history = random.randint(0, 6)

        if age < 10:
            framework = "what_so_what"
        elif age < 12:
            framework = random.choice(["what_so_what", "kolb"])
        else:
            framework = random.choice(["gibbs", "kolb"])

        stage = random.choice(FRAMEWORKS_STAGES[framework])
        duration = random.choice([5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45])

        prompt = build_expansion_prompt(
            seeds, activity, bundle_id, mime_type,
            age, framework, stage, history
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.8,
            )
            output = response.choices[0].message.content.strip()

            if output.startswith('"') and output.endswith('"'):
                output = output[1:-1]

            input_str = (
                f"Activity: {activity} | Bundle: {bundle_id} | "
                f"MIME: {mime_type} | Duration: {duration} min | "
                f"Age: {age} | Framework: {framework} | "
                f"Stage: {stage} | History: {history}"
            )

            new_example = {"input": input_str, "output": output}
            expanded.append(new_example)

            if len(expanded) % 50 == 0:
                print(f"  Generated {len(expanded)}/{target_count} examples...")

        except Exception as e:
            print(f"  API error (attempt {attempts}): {e}")
            continue

    print(f"Done. Total examples: {len(expanded)}")
    return expanded


def split_dataset(examples: List[Dict], eval_ratio: float = 0.1):
    """Split examples into train and eval sets."""
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * (1 - eval_ratio)))
    train = examples[:split_idx]
    eval_set = examples[split_idx:]
    return train, eval_set


def save_jsonl(examples: List[Dict], filepath: Path):
    """Write examples to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Build reflection prompt dataset"
    )
    parser.add_argument(
        "--expand", action="store_true",
        help="Expand seed dataset using teacher LLM"
    )
    parser.add_argument(
        "--api-key", type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key for expansion"
    )
    parser.add_argument(
        "--target", type=int, default=500,
        help="Target number of examples after expansion"
    )
    parser.add_argument(
        "--split-only", action="store_true",
        help="Split existing data into train/eval without expansion"
    )
    args = parser.parse_args()

    seeds = load_seed_examples()
    print(f"Loaded {len(seeds)} seed examples")

    if args.expand:
        if not args.api_key:
            print("Error: provide --api-key or set OPENAI_API_KEY")
            return
        expanded = expand_with_openai(seeds, args.api_key, args.target)
        save_jsonl(expanded, EXPANDED_FILE)
        examples = expanded
    elif args.split_only:
        source = EXPANDED_FILE if EXPANDED_FILE.exists() else SEED_FILE
        print(f"Reading from {source}")
        with open(source, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
    else:
        print("Use --expand to generate more data or --split-only to split")
        print(f"Current seed count: {len(seeds)}")
        return

    train, eval_set = split_dataset(examples)
    save_jsonl(train, TRAIN_FILE)
    save_jsonl(eval_set, EVAL_FILE)
    print(f"\nSplit: {len(train)} train, {len(eval_set)} eval")


if __name__ == "__main__":
    main()
