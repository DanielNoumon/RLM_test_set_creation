"""
Document model: structured representation of parsed documents.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Section:
    """A section (chapter/subsection/paragraph) of a document."""
    heading: str
    content: str
    level: int = 0  # 0 = top-level chapter, 1 = subsection, etc.
    page_start: int = 0
    page_end: int = 0
    char_start: int = 0  # offset in the full document text
    char_end: int = 0
    children: List["Section"] = field(default_factory=list)
    # Structural tags detected during parsing
    has_table: bool = False
    has_list: bool = False
    has_definition: bool = False
    has_numbers: bool = False  # contains numeric facts
    has_dates: bool = False
    has_contact_info: bool = False  # emails, phone numbers, URLs

    @property
    def full_text(self) -> str:
        """Heading + content combined."""
        if self.heading:
            return f"{self.heading}\n{self.content}"
        return self.content

    @property
    def all_text(self) -> str:
        """Full text including all children recursively."""
        parts = [self.full_text]
        for child in self.children:
            parts.append(child.all_text)
        return "\n\n".join(parts)

    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class Document:
    """A parsed document with hierarchical structure."""
    filename: str
    raw_text: str
    sections: List[Section] = field(default_factory=list)
    page_count: int = 0
    doc_type: str = "unknown"  # pdf, txt, md

    def all_sections_flat(self) -> List[Section]:
        """Return all sections (including nested) as a flat list."""
        result = []
        for sec in self.sections:
            result.append(sec)
            result.extend(self._flatten(sec.children))
        return result

    def _flatten(self, sections: List[Section]) -> List[Section]:
        result = []
        for sec in sections:
            result.append(sec)
            result.extend(self._flatten(sec.children))
        return result

    def get_text_at(self, start: int, end: int) -> str:
        """Extract verbatim text from the raw document."""
        return self.raw_text[start:end]
