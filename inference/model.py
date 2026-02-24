# -*- coding: utf-8 -*-
"""
Reflection model inference interface.

Usage:
    python -m inference.model --model-path ./training/output/reflection-lora
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from inference.prompts import get_system_prompt, build_user_prompt


class ReflectionModel:
    """Loads a fine-tuned LoRA model and generates reflection prompts."""

    def __init__(self, model_path, base_model=None, device="auto"):
        self._model_path = model_path
        self._device = device
        self._model = None
        self._tokenizer = None
        self._base_model_name = base_model
        self._loaded = False

    def load(self):
        """Load model and tokenizer into memory."""
        if self._loaded:
            return

        print(f"Loading tokenizer from {self._model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = self._base_model_name
        if base_model is None:
            import json
            from pathlib import Path
            config_path = Path(self._model_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    adapter_cfg = json.load(f)
                base_model = adapter_cfg.get("base_model_name_or_path")

        if base_model is None:
            base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        print(f"Loading base model: {base_model}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float32,
            device_map=self._device,
        )

        print(f"Loading LoRA adapter from {self._model_path}...")
        self._model = PeftModel.from_pretrained(base, self._model_path)
        self._model.eval()
        self._loaded = True
        print("Model ready.")

    def generate(self, title, bundle_id="", mime_type="",
                 duration_min=None, age=None,
                 framework="what_so_what", stage="what",
                 history_count=0, temperature=0.7, max_new_tokens=150):
        """Generate a reflection prompt for the given activity context."""
        if not self._loaded:
            self.load()

        system_prompt = get_system_prompt(framework)
        user_prompt = build_user_prompt(
            title, bundle_id, mime_type, duration_min,
            age, framework, stage, history_count
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = (
                "System: " + system_prompt + "\n"
                "User: " + user_prompt + "\n"
                "Assistant:"
            )

        inputs = self._tokenizer(
            input_text, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def health_check(self):
        """Return True if the model is loaded and can generate output."""
        if not self._loaded:
            return False
        try:
            self.generate(
                title="Paint Activity", mime_type="image/png",
                age=10, max_new_tokens=20
            )
            return True
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(description="Test reflection model")
    parser.add_argument(
        "--model-path", required=True,
        help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--base-model", default=None,
        help="Base model name (auto-detected if not provided)"
    )
    args = parser.parse_args()

    model = ReflectionModel(args.model_path, base_model=args.base_model)
    model.load()

    test_cases = [
        {
            "title": "TurtleArt Activity",
            "duration_min": 15,
            "age": 9,
            "framework": "what_so_what",
            "stage": "what",
        },
        {
            "title": "Pippy Activity",
            "mime_type": "application/json",
            "duration_min": 25,
            "age": 14,
            "framework": "gibbs",
            "stage": "analysis",
            "history_count": 3,
        },
        {
            "title": "Paint Activity",
            "mime_type": "image/png",
            "duration_min": 10,
            "age": 5,
            "framework": "what_so_what",
            "stage": "what",
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {tc}")
        result = model.generate(**tc)
        print(f"Output: {result}")


if __name__ == "__main__":
    main()
