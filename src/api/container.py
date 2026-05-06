"""Application container — wires concrete adapters to ports.

This is the *only* place where outer layers (api / workers) are allowed to
know about infrastructure modules. Everything else must depend on the ports.
"""

from __future__ import annotations

from functools import lru_cache

from src.application.use_cases.generate_document import GenerateDocumentUseCase
from src.application.use_cases.generate_summary import GenerateSummaryUseCase
from src.application.use_cases.index_meeting import IndexMeetingUseCase
from src.application.use_cases.process_audio import ProcessAudioUseCase
from src.application.use_cases.query_knowledge import QueryKnowledgeUseCase
from src.config.settings import Settings, get_settings
from src.domain.services.transcript_merger import TranscriptMerger
from src.infrastructure.document.docx_document_generator import DocxDocumentGenerator
from src.infrastructure.llm.gemma_llm_service import GemmaLlmService
from src.infrastructure.llm.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from src.infrastructure.persistence.database import make_engine, make_session_factory
from src.infrastructure.persistence.json_glossary_repository import (
    JsonGlossaryRepository,
)
from src.infrastructure.persistence.models import Base
from src.infrastructure.persistence.sqlite_meeting_repository import (
    SqliteMeetingRepository,
)
from src.infrastructure.persistence.sqlite_transcript_repository import (
    SqliteTranscriptRepository,
)
from src.infrastructure.speech.faster_whisper_transcriber import FasterWhisperTranscriber
from src.infrastructure.speech.pyannote_diarizer import PyannoteDiarizer
from src.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore


class Container:
    """Composition root: builds and caches all singletons."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Database
        self._engine = make_engine(settings.database.url)
        Base.metadata.create_all(self._engine)
        self._session_factory = make_session_factory(self._engine)

        # Prompts (read eagerly so a missing file fails at boot, not on first call)
        self.summary_system_prompt = settings.llm.summary_prompt_file.read_text(
            encoding="utf-8"
        )

        # Repositories
        self.meeting_repo = SqliteMeetingRepository(self._session_factory)
        self.transcript_repo = SqliteTranscriptRepository(self._session_factory)
        self.glossary_repo = JsonGlossaryRepository(
            settings.app.data_dir / "glossary" / "cegid_terms.json"
        )

        # Domain services
        self.transcript_merger = TranscriptMerger()

        # Adapters (lazy-loaded inside the services on first use)
        self.transcriber = FasterWhisperTranscriber(settings.whisper)
        self.diarizer = PyannoteDiarizer(settings.pyannote, settings.huggingface)
        self.llm = GemmaLlmService(settings.llm, settings.huggingface)
        self.embedder = SentenceTransformerEmbedder(settings.embedding)
        self.vector_store = ChromaVectorStore(
            persist_dir=settings.vector_store.persist_dir,
            embedder=self.embedder,
        )
        self.document_generator = DocxDocumentGenerator()

    # --- use cases ---
    def process_audio_use_case(self) -> ProcessAudioUseCase:
        return ProcessAudioUseCase(
            transcriber=self.transcriber,
            diarizer=self.diarizer,
            merger=self.transcript_merger,
            meetings=self.meeting_repo,
            transcripts=self.transcript_repo,
        )

    def generate_summary_use_case(self) -> GenerateSummaryUseCase:
        return GenerateSummaryUseCase(
            llm=self.llm,
            meetings=self.meeting_repo,
            transcripts=self.transcript_repo,
            glossary_repo=self.glossary_repo,
            summary_system_prompt=self.summary_system_prompt,
        )

    def generate_document_use_case(self) -> GenerateDocumentUseCase:
        return GenerateDocumentUseCase(
            generator=self.document_generator,
            meetings=self.meeting_repo,
        )

    def index_meeting_use_case(self) -> IndexMeetingUseCase:
        return IndexMeetingUseCase(
            vector_store=self.vector_store,
            meetings=self.meeting_repo,
            transcripts=self.transcript_repo,
        )

    def query_knowledge_use_case(self) -> QueryKnowledgeUseCase:
        return QueryKnowledgeUseCase(
            llm=self.llm,
            vector_store=self.vector_store,
        )


@lru_cache(maxsize=1)
def get_container() -> Container:
    return Container(get_settings())
