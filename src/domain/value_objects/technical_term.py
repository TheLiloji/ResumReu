"""Domain value objects for the technical glossary (e.g. Cegid-specific terms)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TechnicalTerm:
    """A domain-specific term with its definition and optional aliases."""

    term: str
    definition: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    category: str | None = None

    def __post_init__(self) -> None:
        if not self.term.strip():
            raise ValueError("TechnicalTerm.term must not be empty")
        if not self.definition.strip():
            raise ValueError("TechnicalTerm.definition must not be empty")

    def matches(self, candidate: str) -> bool:
        """True if `candidate` matches the term or one of its aliases (case-insensitive)."""
        candidate_norm = candidate.strip().lower()
        if candidate_norm == self.term.lower():
            return True
        return any(candidate_norm == alias.lower() for alias in self.aliases)


@dataclass(frozen=True, slots=True)
class Glossary:
    """A read-only collection of technical terms, indexable by name."""

    terms: tuple[TechnicalTerm, ...]

    def find(self, candidate: str) -> TechnicalTerm | None:
        for term in self.terms:
            if term.matches(candidate):
                return term
        return None

    def detect_in_text(self, text: str) -> tuple[TechnicalTerm, ...]:
        """Return every glossary term whose name or alias appears in `text`."""
        text_lower = text.lower()
        hits: list[TechnicalTerm] = []
        for term in self.terms:
            needles = (term.term, *term.aliases)
            if any(needle.lower() in text_lower for needle in needles):
                hits.append(term)
        return tuple(hits)
