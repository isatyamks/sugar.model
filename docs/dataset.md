# Dataset

## Format

Training data is stored in JSONL (JSON Lines) format. Each line is a single example with an `input` and `output` field.

```json
{"input": "Activity: Write Activity | Bundle: | MIME: text/plain | Duration: 20 min | Age: 10 | Framework: what_so_what | Stage: what | History: 0", "output": "You wrote something today! What was it about?"}
```

## Input Fields

| Field | Description | Example |
|-------|-------------|---------|
| Activity | Title of the Sugar activity or Journal entry | `Pippy Activity`, `Untitled` |
| Bundle | Sugar bundle ID (often empty) | `org.laptop.Write` |
| MIME | Content MIME type | `text/plain`, `image/png` |
| Duration | Time spent in the session | `15 min` |
| Age | Learner's age | `9` |
| Framework | Reflection framework to use | `what_so_what`, `gibbs`, `kolb` |
| Stage | Current stage within the framework | `what`, `description`, `concrete_experience` |
| History | Number of past sessions with this activity type | `3` |

## Files

| File | Purpose |
|------|---------|
| `data/reflection_prompts.jsonl` | Seed examples (hand-written + teacher LLM generated) |
| `data/train.jsonl` | Training split (90% of data) |
| `data/eval_set.jsonl` | Evaluation split (10% of data) |

## How Data Is Generated

1. **Seed examples** are written by hand, covering different Sugar activities (Write, TurtleArt, Pippy, Paint, etc.), age groups (5–16), and all three frameworks with their respective stages.

2. **Expansion** uses a teacher LLM (GPT-4o-mini via OpenAI API). The script `data/build_dataset.py` takes seed examples as few-shot demonstrations and asks the teacher to generate new examples for randomly sampled activity/age/framework combinations.

3. **Splitting** divides the data 90/10 into train and eval sets.

## Running the Dataset Builder

To expand the dataset:
```bash
python data/build_dataset.py --expand --api-key YOUR_OPENAI_KEY --target 500
```

To just split existing data:
```bash
python data/build_dataset.py --split-only
```

## Quality Considerations

- Generic titles like "Untitled" and "New Source File 1" are intentionally included because they appear frequently in real Sugar Journal data
- Framework selection in the training data follows age-based routing (under 10 → what_so_what, 10–12 → mixed, 12+ → gibbs/kolb) to match what the backend does in production
- Questions should be open-ended, warm, and never longer than 2 sentences
