# ML Stack Troubleshooting

Ce document regroupe les problèmes rencontrés pendant l’intégration Docker/GPU
et les corrections qui ont fonctionné sur la cible RTX 5070 8 Go, Python 3.11,
CUDA 13, Gemma 4, Whisper et pyannote.

## Configuration validée

Versions Python attendues par `pyproject.toml` :

- `torch>=2.11,<2.12`
- `torchaudio>=2.11,<2.12`
- `torchvision>=0.26,<0.27`
- `torchcodec==0.11.x` via `pyannote.audio`
- `pyannote.audio>=4.0,<5`
- `transformers>=5.5,<6`
- `accelerate>=1.1`
- `bitsandbytes>=0.49`

Modèles par défaut :

- ASR : `Systran/faster-whisper-large-v3`
- Diarization : `pyannote/speaker-diarization-community-1`
- LLM : `google/gemma-4-E4B-it`
- Embeddings : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Le modèle d’embedding garde un identifiant Hugging Face `sentence-transformers/...`,
mais le package Python `sentence-transformers` ne doit pas être installé par le
projet. L’embedder utilise directement `transformers`.

## Symptômes et corrections

### `Disabling PyTorch because PyTorch >= 2.1 is required but found 2.0.1`

Cause : les anciennes bornes `torch>=2.0,<2.1` forçaient une pile trop vieille
pour `transformers` moderne et Gemma.

Correction validée :

```toml
torch>=2.11,<2.12
torchaudio>=2.11,<2.12
torchvision>=0.26,<0.27
```

Puis reconstruire :

```powershell
docker compose build worker
docker compose up -d --force-recreate worker
```

### Erreur NumPy 1.x / NumPy 2.x ABI

Symptôme :

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Cause : mélange entre vieux modules compilés et stack moderne.

Correction validée : ne pas forcer `numpy<2` dans `pyproject.toml`. Mettre à
jour la stack Torch/pyannote/transformers ensemble plutôt que pinner NumPy vers
le bas.

### `NameError: name 'LRScheduler' is not defined`

Cause : `transformers` récent utilisé avec un PyTorch trop ancien.

Correction validée : remonter PyTorch et la famille Torch ensemble.

### `torchcodec` casse au démarrage du worker

Symptôme : import de `sentence_transformers` qui charge `torchcodec`, puis erreur
FFmpeg/libtorchcodec avant même de traiter une tâche.

Cause : le package Python `sentence-transformers` importe des modules audio/vidéo
inutiles pour nos embeddings texte.

Correction validée :

- retirer `sentence-transformers` des dépendances Python ;
- garder uniquement le modèle HF `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` ;
- implémenter l’embedder avec `AutoTokenizer`, `AutoModel` et mean pooling.

Vérification :

```powershell
docker compose run --rm --no-deps --entrypoint python worker -c "import importlib.util; print(importlib.util.find_spec('sentence_transformers') is not None)"
```

La sortie doit être `False`.

### `AttributeError: 'list' object has no attribute 'keys'` avec Gemma 4

Cause : Gemma 4 ne doit pas être chargé comme un simple tokenizer texte avec
`AutoTokenizer`. Le processor Gemma 4 expose le chat template attendu.

Correction validée : utiliser `AutoProcessor.from_pretrained(...)` puis
`processor.apply_chat_template(...)`.

### `Gemma4VideoProcessor requires the Torchvision library`

Cause : `AutoProcessor` Gemma 4 construit aussi des composants multimodaux,
même si l’usage courant est texte. `torchvision` doit être installé avec une
version compatible avec `torch`.

Correction validée :

```toml
torch>=2.11,<2.12
torchaudio>=2.11,<2.12
torchvision>=0.26,<0.27
```

Vérification :

```powershell
docker compose run --rm --no-deps --entrypoint python worker -c "import torch, torchaudio, torchvision; print(torch.__version__, torchaudio.__version__, torchvision.__version__)"
```

### Téléchargement Gemma apparemment bloqué à `Fetching 2 files: 0%`

Cause : la barre de progression Hugging Face/Xet ne se met pas correctement à
jour dans les logs Celery/Docker.

Vérifier la progression réelle dans le cache :

```powershell
docker compose exec -T worker python -c "from pathlib import Path; files=list(Path('/cache/huggingface/hub').rglob('*.incomplete')); print('\n'.join(f'{p.stat().st_size/(1024**3):.2f} GiB {p}' for p in files) or 'download complete')"
```

Si la taille augmente, le téléchargement progresse. Ne pas redémarrer le worker
tant que le fichier `.incomplete` grossit.

### `Some modules are dispatched on the CPU or the disk`

Cause : Gemma 4 E4B en 4-bit ne rentre pas entièrement dans 8 Go VRAM avec les
buffers et le contexte de génération. `device_map=auto` veut dispatcher des
modules ailleurs, mais bitsandbytes refuse si l’offload n’est pas activé
explicitement.

Première correction tentée :

```env
LLM_MODEL_ID=google/gemma-4-E4B-it
LLM_LOAD_IN_4BIT=true
LLM_DEVICE_MAP=auto
LLM_CPU_OFFLOAD=true
LLM_MAX_GPU_MEMORY=6GiB
LLM_MAX_CPU_MEMORY=24GiB
LLM_OFFLOAD_FOLDER=./data/models/llm-offload
LLM_MAX_NEW_TOKENS=1024
```

Dans Docker, `docker-compose.yml` peut forcer l’offload vers le volume persistant :

```yaml
LLM_OFFLOAD_FOLDER: /cache/torch/llm-offload
```

### `Params4bit.__new__() got an unexpected keyword argument '_is_hf_initialized'`

Cause : avec `accelerate==1.13.0`, le split CPU/GPU de paramètres
`bitsandbytes` 4-bit peut reconstruire des `Params4bit` en propageant un
attribut interne `_is_hf_initialized` que bitsandbytes ne sait pas accepter.
Le chargement arrive souvent à `Loading weights: 100%`, puis échoue pendant le
dispatch du modèle.

Correction validée pour Gemma 4 E4B sur RTX 5070 8 Go : éviter le split
CPU/GPU et forcer le modèle 4-bit sur le GPU.

```env
LLM_MODEL_ID=google/gemma-4-E4B-it
LLM_LOAD_IN_4BIT=true
LLM_DEVICE_MAP=cuda
LLM_CPU_OFFLOAD=false
LLM_MAX_NEW_TOKENS=512
```

Dans le code, `LLM_DEVICE_MAP=cuda` est traduit en `device_map={"": 0}` pour
`transformers`.

Si cela échoue par manque de VRAM :

```env
LLM_MAX_NEW_TOKENS=512
```

Puis, si nécessaire, passer à un modèle plus petit :

```env
LLM_MODEL_ID=google/gemma-4-E2B-it
```

### `JSONDecodeError: Expecting property name enclosed in double quotes`

Cause : le chargement Gemma fonctionne, mais la réponse de résumé n'est pas du
JSON strict. Gemma peut parfois produire un dictionnaire Python avec guillemets
simples (`{'summary': ...}`), `None`, ou du texte autour de l'objet.

Corrections validées :

- le prompt système interdit explicitement les blocs Markdown, les dictionnaires
  Python, les guillemets simples et `None/True/False` ;
- le résumé appelle le LLM avec `temperature=0.0` pour réduire les variations ;
- le parseur récupère l'objet structuré dans la réponse et accepte en secours les
  sorties de type dictionnaire Python courantes ;
- la réponse est validée par un `SummarySchema` Pydantic avant usage ;
- si le parsing ou la validation échoue, le use case redemande au LLM de corriger
  uniquement le format JSON. Budget actuel : 1 appel initial + 2 retries.

Si l'erreur revient, il faut enregistrer les 500 premiers caractères de la
réponse brute du LLM avant parsing pour voir la forme exacte produite.

### Nettoyer un téléchargement Gemma abandonné

Le cache Hugging Face est persistant dans le volume Docker `hf_cache`. Si un gros
modèle comme `gemma-4-26B-A4B-it` a été abandonné, il continue à consommer de
l’espace.

Lister les gros fichiers :

```powershell
docker compose exec -T worker python -c "from pathlib import Path; files=[]; root=Path('/cache/huggingface'); [files.append((p.stat().st_size,p)) for p in root.rglob('*') if p.is_file()]; print('\n'.join(f'{s/(1024**3):.2f} GiB {p}' for s,p in sorted(files, reverse=True)[:20]))"
```

Suppression prudente recommandée : arrêter les services puis supprimer seulement
le dossier du modèle abandonné dans le volume, pas tout `hf_cache`.

## Vérifications rapides

Dans le conteneur :

```powershell
docker compose run --rm --no-deps --entrypoint python worker -m pip check
docker compose run --rm --no-deps --entrypoint python worker -c "import torch, torchaudio, torchvision, transformers; print(torch.__version__, torchaudio.__version__, torchvision.__version__, transformers.__version__)"
docker compose run --rm --no-deps --entrypoint python worker -c "from transformers import AutoProcessor; from src.config.settings import get_settings; s=get_settings(); p=AutoProcessor.from_pretrained(s.llm.model_id, token=s.huggingface.token); print(type(p).__name__, hasattr(p, 'apply_chat_template'))"
```

Localement :

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```
