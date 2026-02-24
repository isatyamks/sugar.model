# -*- coding: utf-8 -*-
"""
Fine-tune a small LLM on reflection prompt generation using LoRA.

Usage:
    cd sugar.model/training
    pip install -r requirements.txt
    python finetune.py
"""

import json
import argparse
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


SCRIPT_DIR = Path(__file__).parent.resolve()

SYSTEM_PROMPT = (
    "You are a learning companion for children using Sugar educational "
    "software. Given information about an activity the child just completed, "
    "generate a single, age-appropriate reflection question. The question "
    "should be warm, specific to their work, and open-ended. Keep it to "
    "1-2 sentences."
)


def load_config(config_path=None):
    """Load training configuration from a YAML file."""
    if config_path is None:
        config_path = SCRIPT_DIR / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path):
    """Load a JSONL file into a HuggingFace Dataset."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_example(example, tokenizer):
    """Convert a training example to chat-formatted text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        text = (
            "System: " + SYSTEM_PROMPT + "\n"
            "User: " + example["input"] + "\n"
            "Assistant: " + example["output"]
        )

    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune reflection model")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Base model: {cfg['base_model']}")

    dataset_path = (SCRIPT_DIR / cfg["dataset_path"]).resolve()
    eval_dataset_path = (SCRIPT_DIR / cfg["eval_dataset_path"]).resolve() if cfg.get("eval_dataset_path") else None
    output_dir = (SCRIPT_DIR / cfg["output_dir"]).resolve()

    print(f"Dataset: {dataset_path}")
    print(f"Eval: {eval_dataset_path}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = load_dataset_from_jsonl(str(dataset_path))
    train_data = train_data.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=train_data.column_names,
    )
    print(f"Training examples: {len(train_data)}")

    eval_data = None
    if eval_dataset_path and eval_dataset_path.exists():
        eval_data = load_dataset_from_jsonl(str(eval_dataset_path))
        eval_data = eval_data.map(
            lambda ex: format_example(ex, tokenizer),
            remove_columns=eval_data.column_names,
        )
        print(f"Eval examples: {len(eval_data)}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float32,
        device_map="auto",
    )

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_cfg = cfg["training"]
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_length=train_cfg["max_seq_length"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        fp16=False,
        seed=train_cfg.get("seed", 42),
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    output_path = output_dir
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nModel saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
