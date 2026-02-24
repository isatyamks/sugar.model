# sugar.model

A fine-tuned open-source LLM that generates reflection questions for children using Sugar educational software.

Given activity context from the Sugar Journal (what the child did, how long, their age), the model produces a single age-appropriate reflection question to encourage thinking about their learning.

## Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

- [**Project Overview**](docs/overview.md) — What this project is, how it fits into the broader GSoC system, and the end-to-end flow from Sugar Shell to model response
- [**System Architecture**](docs/architecture.md) — The three system components (Sugar Shell, FastAPI backend, model), request flow, and data flow diagrams
- [**Reflection Frameworks**](docs/frameworks.md) — Detailed explanation of Gibbs, Kolb, and What-So-What frameworks, their stages, and age-based selection logic
- [**Dataset**](docs/dataset.md) — Training data format, input fields, generation process using a teacher LLM, and quality guidelines
- [**Training**](docs/training.md) — LoRA fine-tuning method, base model choice, configuration parameters, and how to run training
- [**Inference**](docs/inference.md) — `ReflectionModel` API, parameters, system prompts per framework, and how the backend consumes the model
- [**Evaluation**](docs/evaluation.md) — Scoring metrics (question detection, open-endedness, length, age-appropriateness), how to run evaluation
- [**Deployment**](docs/deployment.md) — Three deployment options (local model, merged export, Ollama), hardware requirements, and Sugar Shell configuration

## Mission

When a child finishes or pauses an activity in Sugar, the system should prompt them to reflect on what they did, why they did it, what they learned, and what they might try next. This model powers that experience — turning raw activity metadata into a meaningful, age-appropriate question that helps build reflective thinking habits.

The goal is not a general-purpose chatbot. It is a focused, lightweight model that does one thing well: generate a single reflection question from activity context.

## What Has Been Completed

- **Research** — Studied three reflective practice frameworks (What-So-What-Now-What, Gibbs Cycle, Kolb Cycle) and documented how each applies to different age groups and Sugar activities. See [`data/frameworks.md`](data/frameworks.md).
- **Training dataset** — Built a seed dataset of reflection prompt examples in JSONL format, covering various Sugar activities, age groups, and framework stages. Created an automated expansion pipeline using a teacher LLM. Split into train and eval sets.
- **Fine-tuning** — Trained a LoRA adapter on TinyLlama-1.1B-Chat using the dataset. Three epochs completed with checkpoints saved. Adapter weights are in `training/output/reflection-lora/`.
- **Inference module** — Built a `ReflectionModel` class that loads the adapter and generates reflection questions from activity context. Includes system prompt templates for each framework.
- **Evaluation module** — Implemented scoring metrics (question detection, open-endedness, length, age-appropriateness, activity reference) and an evaluation runner that reports aggregate scores.
- **Tooling** — Model export script for merging the adapter into a standalone model, and a smoke test script for quick validation.

## What Remains

- **Backend integration** — Connect the inference module to the FastAPI service in `architect/ai-reflection-service/`. The `LocalModelProvider` stub exists; it needs to be wired to `ReflectionModel`.
- **Sugar Journal integration** — Test the end-to-end flow: child closes or pauses an activity → Journal sends context to the API → model returns a question → the reflection dialog appears. The hooks exist in `sugar/src/jarabe/model/reflection.py`.
- **Model quality iteration** — Expand the dataset beyond current seed examples, retrain, and improve evaluation scores. Tune generation parameters (temperature, top-p) for more consistent output.
- **Multilingual support** — Extend the model and prompts to support languages beyond English, starting with Spanish and Portuguese (common in Sugar deployments).
- **Deployment packaging** — Create an Ollama Modelfile for easy self-hosted deployment. Write setup documentation for contributors.

## How We Will Cover It

| Phase | Work | Approach |
|-------|------|----------|
| Backend wiring | Point `LocalModelProvider` to the trained adapter, set `REFLECT_LLM_PROVIDER=local` | Straightforward integration — the API scaffolding already exists |
| End-to-end testing | Test in a running Sugar environment with real Journal entries | Use the existing `reflection.py` hooks and validate with multiple activity types |
| Dataset expansion | Generate 500+ additional examples using the teacher LLM pipeline in `build_dataset.py` | Run with `--expand`, review outputs, retrain |
| Model improvement | Experiment with Phi-2 or Llama-3 as base models if TinyLlama quality is insufficient | Swap `base_model` in config, retrain, compare eval scores |
| Multilingual | Add translated system prompts, generate multilingual training examples | Extend `prompts.py`, add language field to dataset format |
| Packaging | Create Ollama Modelfile, write contributor setup docs | Use `export_model.sh` to produce merged weights, build Modelfile on top |

## Structure

```
data/               Seed data, dataset builder, train/eval splits
training/           LoRA fine-tuning script and config
inference/          Model loading and prompt generation
evaluation/         Scoring metrics and evaluation runner
scripts/            Model export and smoke tests
docs/               Detailed project documentation
```

## Usage

```bash
pip install -r training/requirements.txt
python training/finetune.py
python scripts/test_inference.py --model-path training/output/reflection-lora
```
