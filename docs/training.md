# Training

## Method

The model is fine-tuned using **LoRA (Low-Rank Adaptation)**. Instead of updating all model parameters, LoRA injects small trainable matrices into specific attention layers. This makes training fast and produces a compact adapter file (~9 MB) rather than a full model copy.

## Base Model

The current base model is **TinyLlama-1.1B-Chat-v1.0**. It was chosen because:
- It runs on consumer hardware without a dedicated GPU
- Its chat template supports system/user/assistant message formatting
- It is small enough for deployment in resource-constrained environments like school computers

Other base models can be used by changing `base_model` in `training/config.yaml`. Phi-2 (2.7B) and Llama-3-8B are viable alternatives if better quality is needed and hardware allows.

## Configuration

All training parameters are defined in `training/config.yaml`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| LoRA rank (r) | 16 | Controls adapter capacity |
| LoRA alpha | 32 | Scaling factor for LoRA updates |
| LoRA dropout | 0.05 | Regularization |
| Target modules | q_proj, v_proj | Which attention layers to adapt |
| Epochs | 3 | Number of full passes over training data |
| Batch size | 4 | Examples per gradient step |
| Gradient accumulation | 4 | Effective batch size = 16 |
| Learning rate | 2e-4 | AdamW optimizer learning rate |
| Scheduler | Cosine | Learning rate decay schedule |
| Max sequence length | 512 | Truncation length for tokenized examples |

## How Training Works

Each training example is formatted as a chat conversation:

```
System: You are a learning companion for children using Sugar educational software...
User: Activity: TurtleArt Activity | Duration: 15 min | Age: 9 | Framework: what_so_what | Stage: what | History: 0
Assistant: You just made something in TurtleArt! What was the most interesting part?
```

The model learns to generate the assistant response given the system prompt and user input.

## Running Training

```bash
cd sugar.model/training
pip install -r requirements.txt
python finetune.py
```

To resume from a checkpoint:
```bash
python finetune.py --resume
```

## Output

After training, the adapter weights are saved to `training/output/reflection-lora/`. This directory contains:
- `adapter_model.safetensors` — The trained LoRA weights
- `adapter_config.json` — LoRA configuration and base model reference
- `tokenizer.json` and `tokenizer_config.json` — Tokenizer files
- `checkpoint-*/` — Intermediate checkpoints from each epoch
