# Deployment

## Option 1: Local Model via FastAPI

The simplest deployment path. The FastAPI backend loads the LoRA adapter directly using the `LocalModelProvider`.

**Steps:**

1. Train the model (or use the existing adapter in `training/output/reflection-lora/`)
2. Copy or symlink the adapter directory to a location accessible by the backend
3. In the backend's `.env` file, set:
   ```
   REFLECT_LLM_PROVIDER=local
   REFLECT_LLM_MODEL_NAME=path/to/reflection-lora
   ```
4. Start the backend:
   ```bash
   cd architect/ai-reflection-service
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Option 2: Merged Model Export

For environments where PEFT/LoRA libraries are not available, the adapter can be merged into the base model to produce a standalone model.

```bash
cd sugar.model/scripts
chmod +x export_model.sh
./export_model.sh ../training/output/reflection-lora ./exported-model
```

This produces a full model in `./exported-model` that can be loaded with standard `transformers` without the `peft` library.

## Option 3: Ollama

For self-hosted deployment using Ollama:

1. Export the merged model (Option 2)
2. Create an Ollama Modelfile:
   ```
   FROM ./exported-model
   SYSTEM "You are a learning companion for children using Sugar educational software..."
   PARAMETER temperature 0.7
   PARAMETER top_p 0.9
   ```
3. Build and run:
   ```bash
   ollama create reflection-model -f Modelfile
   ollama run reflection-model
   ```
4. Set the backend to use Ollama:
   ```
   REFLECT_LLM_PROVIDER=ollama
   REFLECT_LLM_MODEL_NAME=reflection-model
   ```

## Hardware Requirements

| Setup | RAM | GPU | Notes |
|-------|-----|-----|-------|
| TinyLlama (1.1B) | 4 GB | Not required | Runs on CPU, suitable for school computers |
| Phi-2 (2.7B) | 8 GB | Recommended | Noticeably better quality |
| Llama-3 (8B) | 16 GB | Required | Best quality, needs dedicated hardware |

## Sugar Shell Configuration

The Sugar Shell client sends requests to `http://localhost:8000/api/v1/reflect` by default. This URL is defined in `sugar/src/jarabe/model/reflection.py` as `AI_SERVICE_URL`. For production deployments, this should be configured via environment variable or Sugar settings.
