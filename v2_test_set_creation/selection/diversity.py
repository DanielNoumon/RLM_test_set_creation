"""
Diversity tracker: ensures generated questions cover different
topics, sections, and documents without repetition.
"""
from typing import List, Set

from ..parsing.document import Section


class DiversityTracker:
    """Track which sections, topics, and documents have been used."""

    def __init__(self):
        self._used_section_ids: Set[int] = set()  # id(section)
        self._used_headings: Set[str] = set()
        self._used_documents: List[str] = []
        self._used_questions: List[str] = []

    def mark_used(
        self,
        section: Section,
        source_filename: str,
        question_text: str = "",
    ) -> None:
        """Mark a section as used."""
        self._used_section_ids.add(id(section))
        self._used_headings.add(section.heading.lower().strip())
        if source_filename not in self._used_documents:
            self._used_documents.append(source_filename)
        if question_text:
            self._used_questions.append(question_text)

    def is_used(self, section: Section) -> bool:
        """Check if a section has already been used."""
        return id(section) in self._used_section_ids

    def heading_is_used(self, heading: str) -> bool:
        """Check if a heading (topic) has been used."""
        return heading.lower().strip() in self._used_headings

    @property
    def used_documents(self) -> List[str]:
        return list(self._used_documents)

    @property
    def used_count(self) -> int:
        return len(self._used_section_ids)

    def least_used_document(
        self, all_filenames: List[str]
    ) -> str:
        """Return the filename that has been used the least."""
        counts = {f: 0 for f in all_filenames}
        for f in self._used_documents:
            if f in counts:
                counts[f] += 1
        return min(counts, key=counts.get)
