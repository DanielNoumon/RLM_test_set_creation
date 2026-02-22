"""
Entity extraction from document sections.

Extracts facts, dates, numbers, terms, contact info, etc.
using regex patterns. These entities are used by the selection
module to find suitable passages for each question type.
"""
import re
from typing import List, Set
from dataclasses import dataclass, field

from ..parsing.document import Section


@dataclass
class SectionEntities:
    """Entities extracted from a single section."""
    section: Section
    source_filename: str
    dates: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    money_amounts: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    defined_terms: List[str] = field(default_factory=list)
    key_nouns: Set[str] = field(default_factory=set)

    @property
    def has_factual_content(self) -> bool:
        """Does this section contain extractable facts?"""
        return bool(
            self.dates or self.numbers or self.money_amounts
            or self.emails or self.phones
        )

    @property
    def entity_count(self) -> int:
        return (
            len(self.dates) + len(self.numbers)
            + len(self.money_amounts) + len(self.emails)
            + len(self.phones) + len(self.urls)
            + len(self.defined_terms)
        )


# ── Patterns ────────────────────────────────────────────

_DATE_PATTERNS = re.compile(
    r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b"
    r"|\b\d{1,2}\s+(?:januari|februari|maart|april|mei"
    r"|juni|juli|augustus|september|oktober|november"
    r"|december)\s+\d{4}\b"
    r"|\b(?:january|february|march|april|may|june|july"
    r"|august|september|october|november|december)"
    r"\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

_MONEY_RE = re.compile(
    r"€\s*[\d.,]+"
    r"|\$\s*[\d.,]+"
    r"|\b\d+(?:[.,]\d+)?\s*(?:euro|EUR)\b",
    re.IGNORECASE,
)

_PERCENTAGE_RE = re.compile(r"\b\d+(?:[.,]\d+)?%")

_NUMBER_FACT_RE = re.compile(
    r"\b\d{2,}(?:[.,]\d+)?\b"
)

_EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.\w{2,}")
_PHONE_RE = re.compile(
    r"\b0\d{1,2}[-\s]?\d{7,8}\b"
    r"|\b\+\d{1,3}[-\s]?\d{1,4}[-\s]?\d{4,8}\b"
)
_URL_RE = re.compile(r"https?://\S+")

# Definition-like patterns (Dutch + English)
_DEFINITION_RE = re.compile(
    r"(?P<term>[A-Z\u00C0-\u024F][\w\s&\-]{2,30}?)"
    r"\s+(?:houdt\s+in|betekent|staat\s+voor|is\s+een"
    r"|refers?\s+to|means?|is\s+defined\s+as)",
    re.IGNORECASE,
)

# Capitalized noun phrases (potential key concepts)
_KEY_NOUN_RE = re.compile(
    r"\b([A-Z\u00C0-\u024F][a-z\u00C0-\u024F]{2,}(?:\s+[A-Z\u00C0-\u024F][a-z\u00C0-\u024F]{2,})*)\b"
)

# Abbreviations in parentheses: "Risico-Inventarisatie & Evaluatie (Ri&E)"
_ABBREVIATION_RE = re.compile(
    r"\(([A-Z][A-Za-z&]{1,10})\)"
)


def extract_entities(
    section: Section, source_filename: str
) -> SectionEntities:
    """Extract all entities from a section."""
    text = section.full_text
    entities = SectionEntities(
        section=section,
        source_filename=source_filename,
    )

    entities.dates = _DATE_PATTERNS.findall(text)
    entities.money_amounts = _MONEY_RE.findall(text)
    entities.numbers = (
        _PERCENTAGE_RE.findall(text)
        + [n for n in _NUMBER_FACT_RE.findall(text)
           if n not in [d.replace("-", "").replace("/", "")
                        for d in entities.dates]]
    )
    entities.emails = _EMAIL_RE.findall(text)
    entities.phones = _PHONE_RE.findall(text)
    entities.urls = _URL_RE.findall(text)

    # Defined terms
    for m in _DEFINITION_RE.finditer(text):
        term = m.group("term").strip()
        if len(term) > 2:
            entities.defined_terms.append(term)

    # Abbreviations
    for m in _ABBREVIATION_RE.finditer(text):
        entities.defined_terms.append(m.group(1))

    # Key noun phrases (capitalized multi-word)
    for m in _KEY_NOUN_RE.finditer(text):
        noun = m.group(1)
        if len(noun) > 3 and not noun.isdigit():
            entities.key_nouns.add(noun)

    return entities


def find_shared_entities(
    entities_a: SectionEntities,
    entities_b: SectionEntities,
) -> Set[str]:
    """Find entities shared between two sections.

    Useful for identifying multi-hop candidates: sections that
    share entities likely discuss related topics.
    """
    shared = set()

    # Shared key nouns
    shared.update(
        entities_a.key_nouns & entities_b.key_nouns
    )

    # Shared defined terms
    terms_a = set(t.lower() for t in entities_a.defined_terms)
    terms_b = set(t.lower() for t in entities_b.defined_terms)
    shared.update(terms_a & terms_b)

    return shared
