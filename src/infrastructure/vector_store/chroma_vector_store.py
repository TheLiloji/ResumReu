"""VectorStorePort implementation backed by ChromaDB (PersistentClient)."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.application.ports.embedder_port import EmbedderPort
from src.application.ports.vector_store_port import (
    RetrievalHit,
    VectorRecord,
    VectorStorePort,
)

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStorePort):
    """ChromaDB persistent backend.

    Embeddings are computed externally via the injected EmbedderPort, so the
    store and the embedding model stay decoupled.
    """

    def __init__(self, persist_dir: Path, embedder: EmbedderPort) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client: ClientAPI = chromadb.PersistentClient(path=str(persist_dir))
        self._embedder = embedder
        self._collections: dict[str, Collection] = {}

    def _collection(self, name: str) -> Collection:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        coll = self._collection(collection)
        embeddings = self._embedder.embed([r.text for r in records])
        coll.upsert(
            ids=[r.id for r in records],
            documents=[r.text for r in records],
            metadatas=[dict(r.metadata) for r in records],
            embeddings=embeddings,
        )
        logger.info("Upserted %d records into '%s'", len(records), collection)

    def query(
        self,
        collection: str,
        query_text: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[RetrievalHit]:
        coll = self._collection(collection)
        embedding = self._embedder.embed([query_text])[0]
        result = coll.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=metadata_filter,
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[RetrievalHit] = []
        for rid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            hits.append(
                RetrievalHit(
                    record=VectorRecord(id=rid, text=doc, metadata=dict(meta or {})),
                    score=1.0 - float(dist),  # cosine distance → similarity
                )
            )
        return hits

    def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        self._collection(collection).delete(ids=ids)
