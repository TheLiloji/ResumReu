"""GlossaryRepository implementation that loads terms from a JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.domain.repositories.glossary_repository import GlossaryRepository
from src.domain.value_objects.technical_term import Glossary, TechnicalTerm

logger = logging.getLogger(__name__)


class JsonGlossaryRepository(GlossaryRepository):
    """Loads the technical glossary from a JSON file with this shape:

        [
          {"term": "Y2", "definition": "...", "aliases": ["Cegid Y2"], "category": "ERP"},
          ...
        ]

    Returns an empty Glossary if the file does not exist.
    """

    def __init__(self, glossary_path: Path) -> None:
        self._path = glossary_path

    def load(self) -> Glossary:
        if not self._path.exists():
            logger.warning("Glossary file not found at %s — using empty glossary", self._path)
            return Glossary(terms=())
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        terms = tuple(
            TechnicalTerm(
                term=item["term"],
                definition=item["definition"],
                aliases=tuple(item.get("aliases", [])),
                category=item.get("category"),
            )
            for item in raw
        )
        logger.info("Loaded %d glossary terms from %s", len(terms), self._path)
        return Glossary(terms=terms)
