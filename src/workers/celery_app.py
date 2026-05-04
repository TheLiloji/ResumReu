"""Celery application — broker = Redis."""

from __future__ import annotations

from celery import Celery

from src.config.settings import get_settings

_settings = get_settings()

celery_app = Celery(
    "resumreu",
    broker=_settings.celery.broker_url,
    backend=_settings.celery.result_backend,
    include=["src.workers.tasks.audio_processing_task"],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # heavy tasks: avoid prefetch starvation
    worker_concurrency=1,          # single GPU — one worker process at a time,
                                   # otherwise each fork loads its own Whisper /
                                   # Gemma copy and VRAM explodes.
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60,
)
