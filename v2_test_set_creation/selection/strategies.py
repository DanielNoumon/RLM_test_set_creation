"""
Per-question-type passage selection strategies.

Each strategy uses the search index + entity data to find
candidate passages, then returns them ranked by suitability.
No LLM calls — pure Python logic.
"""
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..config import QuestionType
from ..parsing.document import Section, Document
from ..indexing.search_index import SearchIndex
from ..indexing.entity_extractor import (
    SectionEntities, find_shared_entities,
)
from .diversity import DiversityTracker


@dataclass
class PassageCandidate:
    """A candidate passage selected for question generation."""
    passage: str  # verbatim text from the document
    source_documents: List[str]
    chapter: str
    subchapters: List[str]
    page_start: int
    page_end: int
    section: Section
    # For multi-hop: second section info
    section_b: Optional[Section] = None
    source_b: Optional[str] = None


def select_passages(
    question_type: QuestionType,
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Select passages suitable for a question type.

    Dispatches to type-specific strategy functions.
    """
    strategy = _STRATEGIES.get(question_type, _generic_strategy)
    return strategy(
        count, index, entities_map, documents, tracker, rng
    )


# ── Strategy implementations ────────────────────────────


def _direct_lookup_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find sections with specific, extractable facts.

    Prioritizes sections containing numbers, dates, money,
    contact info — the kind of content that makes good
    "what is the exact X?" questions.
    """
    # Score sections by factual density
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 15:
            continue
        score = (
            len(ent.numbers) * 2
            + len(ent.money_amounts) * 3
            + len(ent.dates) * 2
            + len(ent.emails) * 2
            + len(ent.phones) * 3
            + (1 if sec.has_list else 0)
        )
        if score > 0:
            scored.append((ent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    # Add some randomness to avoid always picking the same top
    rng.shuffle(scored[:min(len(scored), count * 3)])

    return _build_candidates(scored[:count * 2], count, tracker)


def _paraphrase_lookup_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find sections with rich descriptive content.

    Good paraphrase questions come from sections with
    substantive explanations (not just lists of facts).
    """
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 25:
            continue
        # Prefer longer, descriptive sections with some entities
        score = (
            min(sec.word_count(), 200) / 50
            + ent.entity_count
            + (2 if sec.has_definition else 0)
        )
        scored.append((ent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    rng.shuffle(scored[:min(len(scored), count * 3)])
    return _build_candidates(scored[:count * 2], count, tracker)


def _specific_jargon_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find sections with defined terms and abbreviations."""
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 15:
            continue
        score = (
            len(ent.defined_terms) * 3
            + (2 if sec.has_definition else 0)
            + len(ent.key_nouns) * 0.5
        )
        if score > 0:
            scored.append((ent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    rng.shuffle(scored[:min(len(scored), count * 3)])
    return _build_candidates(scored[:count * 2], count, tracker)


def _multi_hop_within_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find pairs of sections within the SAME document
    that share entities (related topics).
    """
    candidates = []
    filenames = index.get_source_filenames()

    for fname in filenames:
        # Get entities for sections in this document
        doc_entities = [
            e for e in entities_map
            if e.source_filename == fname
        ]
        # Find pairs with shared entities
        for i in range(len(doc_entities)):
            for j in range(i + 1, len(doc_entities)):
                ea, eb = doc_entities[i], doc_entities[j]
                if tracker.is_used(ea.section) and tracker.is_used(eb.section):
                    continue
                if ea.section.word_count() < 15 or eb.section.word_count() < 15:
                    continue
                # Must be from different sections/pages
                if ea.section.heading == eb.section.heading:
                    continue
                shared = find_shared_entities(ea, eb)
                if shared:
                    candidates.append((ea, eb, fname, len(shared)))

    candidates.sort(key=lambda x: x[3], reverse=True)
    rng.shuffle(candidates[:min(len(candidates), count * 3)])

    results = []
    used_docs = {}  # doc -> count, to spread across docs
    for ea, eb, fname, _ in candidates:
        if len(results) >= count:
            break
        if tracker.is_used(ea.section) and tracker.is_used(eb.section):
            continue
        # Limit to 1 pair per document to force diversity
        if used_docs.get(fname, 0) >= max(1, count // len(
            {c[2] for c in candidates}
        )):
            continue
        passage_a = ea.section.full_text
        passage_b = eb.section.full_text
        # Truncate each to ~800 chars
        passage_a = _truncate_passage(passage_a, 800)
        passage_b = _truncate_passage(passage_b, 800)
        combined = f"{passage_a}\n\n---\n\n{passage_b}"

        results.append(PassageCandidate(
            passage=combined,
            source_documents=[fname],
            chapter=ea.section.heading,
            subchapters=[eb.section.heading],
            page_start=ea.section.page_start,
            page_end=eb.section.page_end,
            section=ea.section,
            section_b=eb.section,
        ))
        tracker.mark_used(ea.section, fname)
        tracker.mark_used(eb.section, fname)
        used_docs[fname] = used_docs.get(fname, 0) + 1

    # If diversity limit blocked us, fill remaining
    if len(results) < count:
        for ea, eb, fname, _ in candidates:
            if len(results) >= count:
                break
            if tracker.is_used(ea.section) and tracker.is_used(eb.section):
                continue
            passage_a = _truncate_passage(ea.section.full_text, 800)
            passage_b = _truncate_passage(eb.section.full_text, 800)
            combined = f"{passage_a}\n\n---\n\n{passage_b}"
            results.append(PassageCandidate(
                passage=combined,
                source_documents=[fname],
                chapter=ea.section.heading,
                subchapters=[eb.section.heading],
                page_start=ea.section.page_start,
                page_end=eb.section.page_end,
                section=ea.section,
                section_b=eb.section,
            ))
            tracker.mark_used(ea.section, fname)
            tracker.mark_used(eb.section, fname)

    return results


def _multi_hop_between_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find pairs of sections from DIFFERENT documents
    that share entities.
    """
    filenames = index.get_source_filenames()
    if len(filenames) < 2:
        return []

    candidates = []
    # Group entities by source document
    by_doc = {}
    for e in entities_map:
        by_doc.setdefault(e.source_filename, []).append(e)

    fnames = list(by_doc.keys())
    for i in range(len(fnames)):
        for j in range(i + 1, len(fnames)):
            for ea in by_doc[fnames[i]]:
                if ea.section.word_count() < 15:
                    continue
                for eb in by_doc[fnames[j]]:
                    if eb.section.word_count() < 15:
                        continue
                    shared = find_shared_entities(ea, eb)
                    if shared:
                        candidates.append((
                            ea, eb,
                            fnames[i], fnames[j],
                            len(shared),
                        ))

    candidates.sort(key=lambda x: x[4], reverse=True)
    rng.shuffle(candidates[:min(len(candidates), count * 3)])

    results = []
    for ea, eb, fname_a, fname_b, _ in candidates:
        if len(results) >= count:
            break
        if tracker.is_used(ea.section) and tracker.is_used(eb.section):
            continue
        passage_a = _truncate_passage(ea.section.full_text, 800)
        passage_b = _truncate_passage(eb.section.full_text, 800)
        combined = f"{passage_a}\n\n---\n\n{passage_b}"

        results.append(PassageCandidate(
            passage=combined,
            source_documents=[fname_a, fname_b],
            chapter=ea.section.heading,
            subchapters=[eb.section.heading],
            page_start=ea.section.page_start,
            page_end=eb.section.page_end,
            section=ea.section,
            section_b=eb.section,
            source_b=fname_b,
        ))
        tracker.mark_used(ea.section, fname_a)
        tracker.mark_used(eb.section, fname_b)

    return results


def _needle_in_haystack_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find sections with small, specific, easy-to-miss details.

    Long sections with only 1-2 specific facts are ideal:
    the fact is a "needle" buried in the "haystack" of text.
    """
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 40:
            continue
        # Ideal: long section, few but specific entities
        if 1 <= ent.entity_count <= 4 and sec.word_count() > 50:
            density = ent.entity_count / sec.word_count()
            # Lower density = more hidden = better needle
            score = (1 - density) * sec.word_count()
            scored.append((ent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    rng.shuffle(scored[:min(len(scored), count * 3)])
    return _build_candidates(scored[:count * 2], count, tracker)


def _temporal_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find sections with dates, deadlines, or time-based info."""
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 15:
            continue
        score = len(ent.dates) * 3 + (1 if sec.has_dates else 0)
        if score > 0:
            scored.append((ent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    rng.shuffle(scored[:min(len(scored), count * 3)])
    return _build_candidates(scored[:count * 2], count, tracker)


def _lists_extraction_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find sections containing bullet or numbered lists."""
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 15:
            continue
        if sec.has_list:
            score = sec.word_count()
            scored.append((ent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    rng.shuffle(scored[:min(len(scored), count * 3)])
    return _build_candidates(scored[:count * 2], count, tracker)


def _hallucination_test_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """For hallucination tests, we still need a passage as context
    for the LLM to know what the documents ARE about, so it can
    generate a question about something NOT in them.

    Pick diverse sections from different documents.
    """
    all_sections = index.get_all_sections()
    if not all_sections:
        return []

    rng.shuffle(all_sections)
    results = []
    for sec, fname in all_sections:
        if len(results) >= count:
            break
        if sec.word_count() < 20:
            continue
        passage = _truncate_passage(sec.full_text, 600)
        results.append(PassageCandidate(
            passage=passage,
            source_documents=[fname],
            chapter=sec.heading,
            subchapters=[],
            page_start=sec.page_start,
            page_end=sec.page_end,
            section=sec,
        ))

    return results


def _tables_extraction_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Find sections with tabular or structured numeric data.

    Tries has_table first; falls back to sections dense in
    numbers, money amounts, or percentages (quasi-tabular).
    """
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 15:
            continue
        if sec.has_table:
            score = sec.word_count() + 100
            scored.append((ent, float(score)))
            continue
        num_density = (
            len(ent.numbers) + len(ent.money_amounts)
        )
        if num_density >= 2:
            scored.append((ent, float(num_density * 10)))

    scored.sort(key=lambda x: x[1], reverse=True)
    rng.shuffle(scored[:min(len(scored), count * 3)])
    return _build_candidates(scored[:count * 2], count, tracker)


def _generic_strategy(
    count: int,
    index: SearchIndex,
    entities_map: List[SectionEntities],
    documents: List[Document],
    tracker: DiversityTracker,
    rng: random.Random,
) -> List[PassageCandidate]:
    """Fallback: pick diverse, content-rich sections."""
    scored = []
    for ent in entities_map:
        sec = ent.section
        if tracker.is_used(sec) or sec.word_count() < 15:
            continue
        score = (
            min(sec.word_count(), 200) / 50
            + ent.entity_count * 0.5
        )
        scored.append((ent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    rng.shuffle(scored[:min(len(scored), count * 3)])
    return _build_candidates(scored[:count * 2], count, tracker)


# ── Helpers ─────────────────────────────────────────────


def _build_candidates(
    scored: List[Tuple[SectionEntities, float]],
    count: int,
    tracker: DiversityTracker,
) -> List[PassageCandidate]:
    """Convert scored entity entries into PassageCandidates."""
    results = []
    for ent, _ in scored:
        if len(results) >= count:
            break
        sec = ent.section
        if tracker.is_used(sec):
            continue

        passage = _truncate_passage(sec.full_text, 1200)
        results.append(PassageCandidate(
            passage=passage,
            source_documents=[ent.source_filename],
            chapter=sec.heading,
            subchapters=[
                c.heading for c in sec.children
            ],
            page_start=sec.page_start,
            page_end=sec.page_end,
            section=sec,
        ))
        tracker.mark_used(sec, ent.source_filename)

    return results


def _truncate_passage(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last sentence-ending punctuation
    last_period = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_period > max_chars // 2:
        return truncated[:last_period + 1]
    return truncated


# ── Strategy dispatch table ─────────────────────────────

_STRATEGIES = {
    QuestionType.DIRECT_LOOKUP: _direct_lookup_strategy,
    QuestionType.PARAPHRASE_LOOKUP: _paraphrase_lookup_strategy,
    QuestionType.SPECIFIC_JARGON: _specific_jargon_strategy,
    QuestionType.MULTI_HOP_WITHIN_CORPUS: _multi_hop_within_strategy,
    QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS: _multi_hop_between_strategy,
    QuestionType.NEEDLE_IN_HAYSTACK: _needle_in_haystack_strategy,
    QuestionType.TEMPORAL_QUESTIONS: _temporal_strategy,
    QuestionType.LISTS_EXTRACTION: _lists_extraction_strategy,
    QuestionType.HALLUCINATION_TEST: _hallucination_test_strategy,
    QuestionType.PINPOINTING_QUOTING: _direct_lookup_strategy,
    QuestionType.LONG_CONTEXT_SYNTHESIS: _paraphrase_lookup_strategy,
    QuestionType.TABLES_EXTRACTION: _tables_extraction_strategy,
    QuestionType.CROSS_DOCUMENT_CONFLICT: _multi_hop_between_strategy,
    QuestionType.AMBIGUOUS_QUESTIONS: _paraphrase_lookup_strategy,
    QuestionType.ADVERSARIAL_AGGRO: _generic_strategy,
    QuestionType.PROMPT_INJECTION: _generic_strategy,
    QuestionType.TOOL_CALL_CHECK: _direct_lookup_strategy,
    QuestionType.MULTI_TURN_FOLLOWUP: _paraphrase_lookup_strategy,
    QuestionType.ACCESS_CONTROL: _generic_strategy,
    QuestionType.INFOGRAPHIC_EXTRACTION: _generic_strategy,
}
