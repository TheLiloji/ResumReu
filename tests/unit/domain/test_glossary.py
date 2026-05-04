import pytest

from src.domain.value_objects.technical_term import Glossary, TechnicalTerm


def test_term_rejects_empty() -> None:
    with pytest.raises(ValueError):
        TechnicalTerm(term="", definition="x")
    with pytest.raises(ValueError):
        TechnicalTerm(term="x", definition="  ")


def test_matches_term_or_alias_case_insensitive() -> None:
    term = TechnicalTerm(term="Y2", definition="ERP", aliases=("Cegid Y2",))
    assert term.matches("y2")
    assert term.matches("CEGID Y2")
    assert not term.matches("Loop")


def test_glossary_detects_terms_in_text() -> None:
    glossary = Glossary(
        terms=(
            TechnicalTerm("Y2", "ERP"),
            TechnicalTerm("Loop", "Expertise"),
            TechnicalTerm("Retail", "Commerce", aliases=("Cegid Retail",)),
        )
    )
    text = "On évoque Y2 et Cegid Retail dans cette réunion."
    hits = glossary.detect_in_text(text)
    found = {t.term for t in hits}
    assert found == {"Y2", "Retail"}
