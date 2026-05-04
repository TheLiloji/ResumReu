"""End-to-end pipeline task: audio → transcript → summary → docx → index."""

from __future__ import annotations

import logging
from uuid import UUID

from src.api.container import get_container
from src.application.use_cases.generate_document import GenerateDocumentInput
from src.application.use_cases.generate_summary import GenerateSummaryInput
from src.application.use_cases.index_meeting import IndexMeetingInput
from src.application.use_cases.process_audio import ProcessAudioInput
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="resumreu.run_pipeline", bind=True, max_retries=0)
def run_pipeline(self, meeting_id: str) -> dict[str, str]:
    """Run the full meeting-processing pipeline as a single Celery task."""
    container = get_container()
    mid = UUID(meeting_id)

    logger.info("[pipeline %s] step 1/4 — transcribe + diarize", meeting_id)
    container.process_audio_use_case().execute(ProcessAudioInput(meeting_id=mid))

    # Free Whisper + pyannote VRAM before loading the LLM (~13 GB in 4-bit).
    container.transcriber.unload()
    container.diarizer.unload()

    logger.info("[pipeline %s] step 2/4 — summarize", meeting_id)
    summary_out = container.generate_summary_use_case().execute(
        GenerateSummaryInput(meeting_id=mid)
    )

    logger.info("[pipeline %s] step 3/4 — generate document", meeting_id)
    doc_out = container.generate_document_use_case().execute(
        GenerateDocumentInput(
            meeting_id=mid,
            document=summary_out.document,
            output_dir=container.settings.app.outputs_dir,
        )
    )

    # Free LLM VRAM before the embedder loads for indexing.
    container.llm.unload()

    logger.info("[pipeline %s] step 4/4 — index", meeting_id)
    index_out = container.index_meeting_use_case().execute(
        IndexMeetingInput(meeting_id=mid, collection="meetings")
    )

    # Embedder can stay resident — it's small (~500 MB) and reused by Q&A.

    return {
        "meeting_id": meeting_id,
        "document_path": str(doc_out.document_path),
        "chunks_indexed": str(index_out.chunks_indexed),
    }
