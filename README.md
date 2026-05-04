# ResumReu

Local meeting summarization and Q&A platform.

## Stack

| Layer | Tooling |
|-------|---------|
| ASR | `faster-whisper` (HuggingFace) |
| Diarization | `pyannote.audio` (HuggingFace) |
| LLM | `google/gemma-4-26B-A4B-it` via `transformers` + `bitsandbytes` |
| Embeddings | `sentence-transformers` (multilingual) |
| Vector store | ChromaDB |
| API | FastAPI |
| Workers | Celery + Redis |
| UI | Streamlit |
| Docs | python-docx |

Hardware target: NVIDIA RTX 4070 Super (12 GB VRAM) + 32 GB RAM.

## Architecture

Clean architecture with strict layering:

```
domain          ← entities + interfaces (no external deps)
application     ← use cases + ports
infrastructure  ← adapters (whisper, pyannote, gemma, chroma, sqlite, docx)
api / workers   ← delivery layers (FastAPI, Celery)
```

Dependency rule: outer layers depend on inner layers, never the reverse.

## Setup

```bash
# 1. Create venv
python3 -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# Edit .env: set HF_TOKEN (https://huggingface.co/settings/tokens)
# Accept gating for: pyannote/speaker-diarization-3.1 and google/gemma-4-26B-A4B-it

# 4. Start Redis (Celery broker)
docker run -d --name resumreu-redis -p 6379:6379 redis:7-alpine

# 5. Run worker (one terminal)
celery -A src.workers.celery_app worker --loglevel=info

# 6. Run API (another terminal)
uvicorn src.api.main:app --reload

# 7. Run UI (another terminal)
streamlit run src/ui/app.py
```

## Tests

```bash
pytest
```
# ResumReu
