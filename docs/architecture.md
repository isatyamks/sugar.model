# System Architecture

## Components

The AI reflection system has three main components that communicate over HTTP:

### 1. Sugar Shell Client (`sugar/src/jarabe/model/reflection.py`)

The client-side service running inside the Sugar desktop environment. It:
- Detects when a child finishes or pauses an activity
- Reads activity metadata and past journal entries from the Sugar Datastore
- Sends an HTTP POST request to the backend with the activity context
- Displays the returned reflection question in a GTK dialog
- Saves the child's response back to the Journal

The client runs API calls in a background thread to avoid blocking the GTK main loop. If the backend is unreachable, it falls back to rule-based prompts.

### 2. FastAPI Backend (`architect/ai-reflection-service/`)

The backend receives reflection requests and orchestrates the pipeline:
1. **Routing** — Selects a reflection framework based on learner age and activity type
2. **Prompt construction** — Builds system and user prompts for the chosen framework and stage
3. **LLM inference** — Calls the configured LLM provider (local model, Ollama, OpenAI, or HuggingFace)
4. **Fallback** — If the primary provider fails, degrades to rule-based prompts
5. **Logging** — Records the interaction for analytics and model improvement

The backend supports multiple LLM providers through a pluggable interface. Setting `REFLECT_LLM_PROVIDER=local` uses the model trained in this repository.

### 3. Model (`sugar.model/` — this repository)

The fine-tuned language model that generates reflection questions. It provides:
- Training pipeline (dataset → LoRA fine-tuning → adapter weights)
- Inference module (`ReflectionModel` class consumed by the backend)
- Evaluation metrics for quality assurance

## Request Flow

```
1. Child closes "TurtleArt Activity" after 15 minutes

2. Sugar Shell reads Journal metadata:
   - title: "TurtleArt Activity"
   - bundle_id: "org.laptop.TurtleArtActivity"
   - mime_type: "application/x-turtle-art"
   - duration: 900 seconds
   - past entries: 3

3. Shell POSTs to http://localhost:8000/api/v1/reflect:
   {
     "context": {
       "activity_id": "act-123",
       "bundle_id": "org.laptop.TurtleArtActivity",
       "title": "TurtleArt Activity",
       "mime_type": "application/x-turtle-art",
       "duration_seconds": 900
     },
     "learner": {"age": 10, "language": "en"},
     "history": [...]
   }

4. Backend routes to Kolb framework (age 10, creative activity)
5. Backend selects "reflective_observation" stage
6. Backend calls ReflectionModel.generate(...)
7. Model returns: "When you look at your TurtleArt project, what part are you most proud of?"
8. Backend wraps response and returns it
9. Shell displays the question in a GTK dialog
10. Child types an answer, Shell saves it to Journal metadata
```

## Data Flow Diagram

```
Sugar Datastore ←──── save_reflection() ────── ReflectionService
       │                                              │
       │ get_activity_context()                       │ HTTP POST
       │ get_activity_history()                       │
       ▼                                              ▼
  Journal Metadata ──────────────────────→ FastAPI Backend
                                                │
                                   FrameworkRouter → PromptGenerator
                                                │
                                                ▼
                                          LLM Provider
                                          (sugar.model)
                                                │
                                                ▼
                                     Reflection Question
```
