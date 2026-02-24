#!/bin/bash
set -e

ADAPTER_PATH="${1:-../training/output/reflection-lora}"
OUTPUT_PATH="${2:-./exported-model}"

echo "=== Export Reflection Model ==="
echo "Adapter path: $ADAPTER_PATH"
echo "Output path:  $OUTPUT_PATH"
echo ""

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from pathlib import Path

adapter_path = '${ADAPTER_PATH}'
output_path = '${OUTPUT_PATH}'

config_path = Path(adapter_path) / 'adapter_config.json'
with open(config_path) as f:
    cfg = json.load(f)
base_model_name = cfg.get('base_model_name_or_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

print(f'Loading base model: {base_model_name}')
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=torch.float16
)

print(f'Loading adapter from: {adapter_path}')
model = PeftModel.from_pretrained(base_model, adapter_path)

print('Merging adapter into base model...')
merged = model.merge_and_unload()

print(f'Saving merged model to: {output_path}')
merged.save_pretrained(output_path)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.save_pretrained(output_path)

print('Done!')
print(f'Model size: {sum(p.numel() for p in merged.parameters()) / 1e6:.1f}M parameters')
"

echo ""
echo "Exported model saved to: $OUTPUT_PATH"
