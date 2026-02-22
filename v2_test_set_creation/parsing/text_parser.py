"""
Text/Markdown parser: extracts structured sections from plain text files.
"""
import re
from typing import List
from pathlib import Path

from .document import Document, Section
from .pdf_parser import _tag_section


# Markdown heading pattern: # Heading, ## Subheading, etc.
_MD_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Blank-line separated paragraphs
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def parse_text_file(filepath: str) -> Document:
    """Parse a .txt or .md file into a structured Document."""
    path = Path(filepath)
    filename = path.name
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    doc_type = "md" if path.suffix == ".md" else "txt"

    if doc_type == "md":
        sections = _parse_markdown(raw_text)
    else:
        sections = _parse_plain_text(raw_text)

    for sec in _flatten(sections):
        _tag_section(sec)

    return Document(
        filename=filename,
        raw_text=raw_text,
        sections=sections,
        page_count=1,
        doc_type=doc_type,
    )


def _parse_markdown(raw_text: str) -> List[Section]:
    """Split markdown by heading levels."""
    headings = []
    for m in _MD_HEADING_RE.finditer(raw_text):
        level = len(m.group(1)) - 1  # # = 0, ## = 1, etc.
        headings.append((
            m.start(), m.end(), m.group(2).strip(), level
        ))

    if not headings:
        return _parse_plain_text(raw_text)

    sections = []
    for i, (h_start, h_end, h_text, level) in enumerate(headings):
        content_end = (
            headings[i + 1][0] if i + 1 < len(headings)
            else len(raw_text)
        )
        content = raw_text[h_end:content_end].strip()
        sections.append(Section(
            heading=h_text,
            content=content,
            level=level,
            page_start=1,
            page_end=1,
            char_start=h_start,
            char_end=content_end,
        ))
    return sections


def _parse_plain_text(raw_text: str) -> List[Section]:
    """Split plain text into paragraph-based sections."""
    paragraphs = _PARAGRAPH_SPLIT_RE.split(raw_text)
    sections = []
    pos = 0
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            pos += len(para) + 2
            continue
        lines = para.split("\n")
        heading = lines[0].strip()[:80]
        content = "\n".join(lines[1:]).strip() if len(lines) > 1 else para
        sections.append(Section(
            heading=heading,
            content=content,
            level=0,
            page_start=1,
            page_end=1,
            char_start=pos,
            char_end=pos + len(para),
        ))
        pos += len(para) + 2
    return sections


def _flatten(sections: List[Section]) -> List[Section]:
    result = []
    for sec in sections:
        result.append(sec)
        result.extend(_flatten(sec.children))
    return result
