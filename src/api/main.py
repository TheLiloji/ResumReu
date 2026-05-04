"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from src.api.container import get_container
from src.api.routers import documents, meetings, query

logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResumReu",
        version="0.1.0",
        description="Local meeting summarization & Q&A.",
    )
    # Touch the container once so SQL tables are created at startup.
    get_container()
    app.include_router(meetings.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    return app


app = create_app()
