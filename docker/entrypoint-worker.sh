#!/usr/bin/env bash
set -euo pipefail

exec celery -A src.workers.celery_app worker --loglevel=info --concurrency=1 --pool=solo
