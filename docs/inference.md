# Inference

## Interface

The inference module provides a single class, `ReflectionModel`, that loads the fine-tuned model and generates reflection questions.

```python
from inference.model import ReflectionModel

model = ReflectionModel("./training/output/reflection-lora")
model.load()

question = model.generate(
    title="Pippy Activity",
    mime_type="application/json",
    duration_min=25,
    age=14,
    framework="gibbs",
    stage="analysis",
    history_count=3,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| title | str | required | Activity title from the Journal |
| bundle_id | str | `""` | Sugar bundle ID |
| mime_type | str | `""` | Content MIME type |
| duration_min | int | None | Session duration in minutes |
| age | int | None | Learner's age |
| framework | str | `"what_so_what"` | Reflection framework |
| stage | str | `"what"` | Stage within the framework |
| history_count | int | `0` | Number of past sessions |
| temperature | float | `0.7` | Sampling temperature |
| max_new_tokens | int | `150` | Maximum output length |

## System Prompts

Each framework has a dedicated system prompt defined in `inference/prompts.py`. These prompts control the model's tone and approach:

- **what_so_what** — Friendly, warm, simple language for young children
- **gibbs** — Supportive but encouraging deeper structured analysis
- **kolb** — Encourages observation, pattern-finding, and experimentation

A default fallback prompt is used if the framework is unrecognized.

## Model Loading

On first call, `ReflectionModel` loads:
1. The tokenizer from the adapter directory
2. The base model (auto-detected from `adapter_config.json`, defaults to TinyLlama)
3. The LoRA adapter on top of the base model

Subsequent calls reuse the loaded model. The `health_check()` method verifies that loading and generation work correctly.

## How the Backend Uses This

The FastAPI backend in `architect/ai-reflection-service/` has a `LocalModelProvider` that wraps `ReflectionModel`. When `REFLECT_LLM_PROVIDER=local` is set, the backend instantiates the model at startup and calls `generate()` for each incoming reflection request.
