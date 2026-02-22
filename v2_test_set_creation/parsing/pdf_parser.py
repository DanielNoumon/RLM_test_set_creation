"""
PDF parser: extracts structured sections from PDF files.

Tier 1: Heuristic heading detection (numbered chapters, role titles, etc.)
Tier 2: LLM-assisted fallback when heuristics fail.
"""
import re
from typing import List, Tuple, Optional

import fitz  # PyMuPDF

from .document import Document, Section


# ── Heading detection patterns ──────────────────────────

# Numbered chapters: "01. Title", "02. Title", "1. Title"
_NUMBERED_CHAPTER_RE = re.compile(
    r"^\s*(\d{1,2})\.?\s+([A-Z\u00C0-\u024F].{2,80})$", re.MULTILINE
)

# Numbered subsections: "2.4 Title", "5.1 Title", "10.2 Title"
_NUMBERED_SUBSECTION_RE = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2})\s+([A-Z\u00C0-\u024F].{2,80})$",
    re.MULTILINE,
)

# Standalone title-like lines (short, no trailing punctuation)
# e.g. "Strategy Consultant", "Interne vertrouwenspersoon"
_TITLE_LINE_RE = re.compile(
    r"^([A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F &\-/()]{4,60})$",
    re.MULTILINE,
)

# ── Structural tag detection ────────────────────────────

_BULLET_RE = re.compile(
    r"^\s*[•\-\*]\s", re.MULTILINE
)
_NUMBERED_LIST_RE = re.compile(
    r"^\s*[a-z]\)\s|^\s*\d+[\.\)]\s", re.MULTILINE
)
_DATE_RE = re.compile(
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
    r"|\b\d{4}\b"
    r"|\b(?:januari|februari|maart|april|mei|juni|juli"
    r"|augustus|september|oktober|november|december"
    r"|january|february|march|april|may|june|july"
    r"|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_NUMBER_FACT_RE = re.compile(
    r"€\s*[\d.,]+"
    r"|\d+(?:[.,]\d+)?%"
    r"|\b\d{2,}\b",
)
_EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.\w+")
_PHONE_RE = re.compile(r"\b\d{2,3}[-\s]?\d{7,8}\b")
_URL_RE = re.compile(r"https?://\S+")
_DEFINITION_RE = re.compile(
    r"\bhoudt\s+in\b|\bbetekent\b|\bstaat\s+voor\b"
    r"|\bwordt\s+bedoeld\b|\bis\s+gedefinieerd\b"
    r"|\brefers?\s+to\b|\bmeans?\b|\bdefined\s+as\b",
    re.IGNORECASE,
)


def parse_pdf(filepath: str) -> Document:
    """Parse a PDF file into a structured Document."""
    doc = fitz.open(filepath)
    filename = filepath.replace("\\", "/").split("/")[-1]

    # Extract full text with page boundaries
    pages_text: List[str] = []
    for page in doc:
        text = page.get_text()
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        pages_text.append(text)

    page_count = doc.page_count
    doc.close()

    raw_text = "\n".join(pages_text)

    # Build page offset map: char offset -> page number
    page_offsets = _build_page_offsets(pages_text)

    # Try heuristic section detection
    sections = _detect_sections_heuristic(
        raw_text, pages_text, page_offsets
    )

    # If heuristics found very few sections, fall back to
    # page-based splitting (the LLM fallback can be added later)
    if len(sections) < 2:
        sections = _fallback_page_sections(
            pages_text, page_offsets
        )

    # Tag each section with structural metadata
    for sec in _flatten_all(sections):
        _tag_section(sec)

    return Document(
        filename=filename,
        raw_text=raw_text,
        sections=sections,
        page_count=page_count,
        doc_type="pdf",
    )


def _build_page_offsets(
    pages_text: List[str],
) -> List[Tuple[int, int]]:
    """Build list of (start_char, end_char) per page."""
    offsets = []
    pos = 0
    for page_text in pages_text:
        start = pos
        end = pos + len(page_text)
        offsets.append((start, end))
        pos = end + 1  # +1 for the \n join separator
    return offsets


def _char_to_page(
    char_pos: int, page_offsets: List[Tuple[int, int]]
) -> int:
    """Map a character position to a page number (1-indexed)."""
    for i, (start, end) in enumerate(page_offsets):
        if start <= char_pos < end:
            return i + 1
    return len(page_offsets)


def _detect_sections_heuristic(
    raw_text: str,
    pages_text: List[str],
    page_offsets: List[Tuple[int, int]],
) -> List[Section]:
    """Detect sections using regex heading patterns.

    Strategy:
    1. Find all numbered chapter headings (level 0).
    2. Find all numbered subsection headings (level 1).
    3. If no numbered headings, try title-line detection.
    4. Split text at heading positions and build hierarchy.
    """
    headings: List[Tuple[int, int, str, int]] = []
    # (char_start, char_end_of_heading, heading_text, level)

    # Numbered chapters
    for m in _NUMBERED_CHAPTER_RE.finditer(raw_text):
        headings.append((
            m.start(), m.end(), m.group(0).strip(), 0
        ))

    # Numbered subsections
    for m in _NUMBERED_SUBSECTION_RE.finditer(raw_text):
        headings.append((
            m.start(), m.end(), m.group(0).strip(), 1
        ))

    # If no numbered headings found, try title lines
    if not headings:
        for m in _TITLE_LINE_RE.finditer(raw_text):
            heading_text = m.group(1).strip()
            # Skip page numbers (standalone digits)
            if heading_text.isdigit():
                continue
            # Skip very common non-heading lines
            if len(heading_text.split()) < 2:
                continue
            headings.append((
                m.start(), m.end(), heading_text, 0
            ))

    if not headings:
        return []

    # Sort by position
    headings.sort(key=lambda h: h[0])

    # Deduplicate: if a subsection heading overlaps with a
    # chapter heading at the same position, keep the subsection
    deduped = []
    seen_positions = set()
    for h in headings:
        # Check if this position is too close to an existing one
        too_close = False
        for sp in seen_positions:
            if abs(h[0] - sp) < 5:
                too_close = True
                break
        if not too_close:
            deduped.append(h)
            seen_positions.add(h[0])
    headings = deduped

    # Build sections from headings
    sections: List[Section] = []
    for i, (h_start, h_end, h_text, level) in enumerate(headings):
        # Content runs from end of heading to start of next heading
        if i + 1 < len(headings):
            content_end = headings[i + 1][0]
        else:
            content_end = len(raw_text)

        content = raw_text[h_end:content_end].strip()

        # Skip ToC entries (very short content between headings)
        if len(content) < 20 and i < len(headings) - 1:
            continue

        page_start = _char_to_page(h_start, page_offsets)
        page_end = _char_to_page(
            content_end - 1, page_offsets
        )

        sec = Section(
            heading=h_text,
            content=content,
            level=level,
            page_start=page_start,
            page_end=page_end,
            char_start=h_start,
            char_end=content_end,
        )
        sections.append(sec)

    # Build hierarchy: nest level-1 sections under level-0
    return _build_hierarchy(sections)


def _build_hierarchy(
    flat_sections: List[Section],
) -> List[Section]:
    """Nest subsections under their parent chapters."""
    if not flat_sections:
        return []

    top_level: List[Section] = []
    current_parent: Optional[Section] = None

    for sec in flat_sections:
        if sec.level == 0:
            top_level.append(sec)
            current_parent = sec
        elif current_parent is not None:
            current_parent.children.append(sec)
        else:
            # Orphan subsection — promote to top level
            top_level.append(sec)

    return top_level


def _fallback_page_sections(
    pages_text: List[str],
    page_offsets: List[Tuple[int, int]],
) -> List[Section]:
    """Fallback: treat each page as a section."""
    sections = []
    pos = 0
    for i, page_text in enumerate(pages_text):
        text = page_text.strip()
        if not text or len(text) < 30:
            pos += len(page_text) + 1
            continue
        # Use first non-empty line as heading
        lines = [ln for ln in text.split("\n") if ln.strip()]
        heading = lines[0].strip() if lines else f"Page {i + 1}"
        content = "\n".join(lines[1:]).strip() if lines else ""

        sections.append(Section(
            heading=heading,
            content=content,
            level=0,
            page_start=i + 1,
            page_end=i + 1,
            char_start=pos,
            char_end=pos + len(page_text),
        ))
        pos += len(page_text) + 1
    return sections


def _flatten_all(sections: List[Section]) -> List[Section]:
    """Flatten a nested section list."""
    result = []
    for sec in sections:
        result.append(sec)
        result.extend(_flatten_all(sec.children))
    return result


def _tag_section(sec: Section) -> None:
    """Detect and tag structural features of a section."""
    text = sec.full_text
    sec.has_list = bool(
        _BULLET_RE.search(text) or _NUMBERED_LIST_RE.search(text)
    )
    sec.has_dates = bool(_DATE_RE.search(text))
    sec.has_numbers = bool(_NUMBER_FACT_RE.search(text))
    sec.has_contact_info = bool(
        _EMAIL_RE.search(text)
        or _PHONE_RE.search(text)
        or _URL_RE.search(text)
    )
    sec.has_definition = bool(_DEFINITION_RE.search(text))
    # Table detection: lines with multiple tab/pipe separators
    lines = text.split("\n")
    table_lines = sum(
        1 for ln in lines
        if ln.count("\t") >= 2 or ln.count("|") >= 2
    )
    sec.has_table = table_lines >= 2
