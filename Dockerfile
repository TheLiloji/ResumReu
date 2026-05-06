FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TORCH_COMPILE_DISABLE=1 \
    TORCHDYNAMO_DISABLE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip python3.11-dev \
        libpython3.11 ffmpeg libsndfile1 git curl \
        build-essential \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY prompts ./prompts

RUN pip install --upgrade pip setuptools wheel

# Install CUDA 12 libraries explicitly for CTranslate2 compatibility
# (faster-whisper backend is built against CUDA 12 even when PyTorch ships CUDA 13)
RUN pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

RUN pip install -e .

RUN mkdir -p /app/data/uploads \
             /app/data/outputs \
             /app/data/vector_store \
             /app/data/models \
             /app/data/glossary \
             /cache/huggingface \
             /cache/torch

COPY docker/entrypoint-api.sh /usr/local/bin/entrypoint-api.sh
COPY docker/entrypoint-worker.sh /usr/local/bin/entrypoint-worker.sh
RUN chmod +x /usr/local/bin/entrypoint-api.sh /usr/local/bin/entrypoint-worker.sh

EXPOSE 8000
