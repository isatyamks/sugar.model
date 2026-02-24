# Evaluation

## Purpose

The evaluation module scores generated reflection prompts against a held-out eval set to measure model quality before deployment.

## Metrics

Each generated prompt is scored on four binary metrics (0 or 1):

| Metric | What It Checks |
|--------|---------------|
| **is_question** | Does the output contain a question mark? |
| **is_open_ended** | Does it start with an open-ended word (what, how, why, etc.) rather than being a yes/no question? |
| **appropriate_length** | Is it between 5 and 60 words? |
| **age_appropriate** | Does it avoid overly complex vocabulary for younger learners? |

If the activity title is provided, a fifth metric is added:

| Metric | What It Checks |
|--------|---------------|
| **references_activity** | Does the output mention the activity title? |

These are intentionally simple heuristics. They catch obvious failures (non-questions, excessively long outputs, inappropriate language) without requiring human evaluation for every example.

## Age-Appropriateness Check

For learners under 12, the metric flags prompts that contain words like "conceptualize", "methodology", "synthesize", "paradigm", etc. For learners under 8, it also checks that the average word length does not exceed 7 characters.

## Running Evaluation

```bash
python evaluation/evaluate.py --model-path training/output/reflection-lora
```

Options:
- `--eval-data` — Path to eval dataset (default: `../data/eval_set.jsonl`)
- `--max-examples` — Limit the number of examples for faster runs

## Output

The evaluation prints:
- Per-metric average scores (0.0 to 1.0)
- Overall average total score
- Five sample outputs with their scores
- Full results saved to `eval_results.json` in the model directory
