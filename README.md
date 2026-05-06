# ResumReu

Local meeting summarization and Q&A platform.

## Stack

| Layer | Tooling |
|-------|---------|
| ASR | `faster-whisper` (HuggingFace) |
| Diarization | `pyannote.audio` (HuggingFace) |
| LLM | `google/gemma-4-E4B-it` via `transformers` + `bitsandbytes` |
| Embeddings | HuggingFace `transformers` encoder (multilingual) |
| Vector store | ChromaDB |
| API | FastAPI |
| Workers | Celery + Redis |
| UI | Streamlit |
| Docs | python-docx |

Hardware target: NVIDIA RTX 5070 (8 GB VRAM) + 32 GB RAM.

## Architecture

Clean architecture with strict layering:

```
domain          ← entities + interfaces (no external deps)
application     ← use cases + ports
infrastructure  ← adapters (whisper, pyannote, gemma, chroma, sqlite, docx)
api / workers   ← delivery layers (FastAPI, Celery)
```

Dependency rule: outer layers depend on inner layers, never the reverse.

The pipeline (audio → transcript → summary → docx → index) runs sequentially in a single Celery task. Models are unloaded between steps to keep VRAM under the 8 GB budget; the LLM uses 4-bit quantization and is kept on GPU by default to avoid current split-device `bitsandbytes` issues — see `unload()` on each adapter and the wiring in [src/workers/tasks/audio_processing_task.py](src/workers/tasks/audio_processing_task.py).

---

## Prerequisites (one-time, host machine)

### 1. NVIDIA driver

RTX 5070 (Blackwell) requires **driver ≥ 570** (GeForce Game Ready or Studio).
Verify in PowerShell:

```powershell
nvidia-smi
```

### 2. Docker Desktop with GPU support

- Install **Docker Desktop ≥ 4.30** (Windows or Mac).
- Settings → General → enable *Use the WSL 2 based engine*.
- Settings → Resources → WSL Integration → enable for your distro.
- GPU passthrough is automatic in 4.30+ once the NVIDIA driver is up to date.

Verify GPU is visible from a container:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 3. HuggingFace account + token + licences

Generate a **classic Read token** at <https://huggingface.co/settings/tokens> (type *Read*).

Then accept the licence on each of these gated repos with the **same** account that owns the token:

- [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) — *Acknowledge license* (instant)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) — short form (instant)

> A fine-grained token works only if it explicitly grants *Read access to public gated repos* and lists the two repos above. The classic Read token avoids this trap.

### 4. Disk space

~35 GB free for a fresh E4B setup (image ~6–8 GB + HuggingFace model cache for Gemma/Whisper/pyannote). Allow more if you experiment with larger Gemma variants, because the persisted `hf_cache` volume keeps previous downloads.

---

## Run with Docker (recommended)

### 1. Configure `.env`

```powershell
Copy-Item .env.example .env
# Edit .env and set: HF_TOKEN=hf_xxxxxxxxxxxx
```

### 2. Build & start

```powershell
docker compose build
docker compose up -d
```

Three containers come up: `redis`, `api`, `worker`. The API listens on <http://localhost:8000>.

Verify everything is healthy:

```powershell
docker compose ps
docker compose exec api nvidia-smi   # RTX 5070 must show up
docker compose logs -f api worker    # no stack traces; "celery@... ready"
```

### 3. UI (run locally, not in Docker)

Streamlit is intentionally not containerised — run it from your venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

$env:API_URL = "http://localhost:8000"
streamlit run src/ui/app.py
```

### 4. First-call cold start

The very first `POST /meetings` triggers the model downloads into the persisted `hf_cache` volume:

- Whisper large-v3 (~3 GB)
- pyannote pipeline (~200 MB)
- Gemma 4 E4B (several GB; cached in `hf_cache`)
- mpnet embedder (~500 MB)

→ 5–30 minutes depending on bandwidth, **once**. Subsequent runs reuse the cache.

### 5. Smoke test

```powershell
curl http://localhost:8000/docs       # Swagger UI
curl http://localhost:8000/meetings   # []
```

Then via Swagger or `curl`: `POST /meetings` with a short French `.wav`, poll `GET /meetings/{id}` until status is `completed`, retrieve the document via `GET /meetings/{id}/document`.

### 6. Stop / restart

```powershell
docker compose down                   # keeps volumes (data/, hf_cache, redis_data)
docker compose down -v                # also wipes volumes (re-download required)
```

---

## Local development (without Docker)

For iterating on code without rebuilding the image:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
# Edit .env: HF_TOKEN, and keep CELERY_BROKER_URL pointing at redis://localhost:6379

# Start a Redis broker
docker run -d --name resumreu-redis -p 6379:6379 redis:7-alpine

# Three terminals:
celery -A src.workers.celery_app worker --loglevel=info --concurrency=1
uvicorn src.api.main:app --reload
streamlit run src/ui/app.py
```

---

## Tests

```powershell
pytest
```

---

## Troubleshooting

The detailed GPU/ML troubleshooting log is in [docs/ml-troubleshooting.md](docs/ml-troubleshooting.md).

| Symptom | Likely cause |
| --- | --- |
| `gated repo access denied` on first pipeline run | One of the HF licences above was not accepted with the account that owns `HF_TOKEN` |
| `CUBLAS_STATUS_NOT_SUPPORTED` from Whisper | Don't switch `WHISPER_COMPUTE_TYPE` to pure `float16` on Ada/Blackwell — keep `int8_float16` |
| API returns 200 but pipeline never reaches `completed` | Check `docker compose logs worker` — the worker likely OOM'd; reduce `LLM_MAX_NEW_TOKENS` or downgrade the LLM |
| `Some modules are dispatched on the CPU or the disk` while loading Gemma | Use `LLM_DEVICE_MAP=cuda` and `LLM_CPU_OFFLOAD=false` for E4B on 8 GB VRAM; if it OOMs, lower `LLM_MAX_NEW_TOKENS` or switch to E2B |
| `nvidia-smi` works on host but not in container | Update Docker Desktop to ≥ 4.30 and the NVIDIA driver to ≥ 570; restart Docker |
| Streamlit can't reach API | Confirm `API_URL=http://localhost:8000` in the env where Streamlit runs |


## Améliorations futures

- Amélioration de la pipeline de retranscription et de diarisation :
  - Tester WhisperX (Whisper + alignement de mots) pour une meilleure précision temporelle.
  - Expérimenter avec d’autres pipelines pyannote pour la diarisation, ou
  - Tester avec VibeVoice-ASR-HF qui combine diarisation + ASR, pour voir si la jointure améliore la précision globale.
- Amélioration de la synthèse de compte-rendu :
  - Tester d’autres LLMs plus petits et mieux adaptés à la synthèse.
  - Expérimenter avec des prompts plus sophistiqués, ou une approche en deux étapes (ex. extraire d’abord les points clés, puis synthétiser à partir de ces points).
- Amélioration du rag et de la recherche d’information :
  - Tester d’autres encodeurs d’embeddings, ou des modèles spécialisés dans les données conversationnelles.
  - Expérimenter avec des techniques de réduction de dimensionnalité pour optimiser les performances de ChromaDB.
- Amélioration de l’interface utilisateur :
  - Ajouter une visualisation de la transcription avec les segments temporels et les locuteurs identifiés.
  - Permettre de donner un nom a chaque speaker identifié
  - Permettre le téléchargement du compte-rendu au format Word ou PDF.
