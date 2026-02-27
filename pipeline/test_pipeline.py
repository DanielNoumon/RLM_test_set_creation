"""Quick smoke test for the pipeline phases (no LLM calls)."""
import random
from pipeline.parsing.pdf_parser import parse_pdf
from pipeline.indexing.search_index import SearchIndex
from pipeline.indexing.entity_extractor import extract_entities
from pipeline.selection.strategies import select_passages
from pipeline.selection.diversity import DiversityTracker
from pipeline.config import QuestionType

DATA = "data/files_for_test_set"

# Parse
docs = [
    parse_pdf(f"{DATA}/Contactgegevens vertrouwenspersoon.pdf"),
    parse_pdf(f"{DATA}/DEF Rolomschrijvingen DSL 2025.pdf"),
    parse_pdf(f"{DATA}/DSL_Handboek_mei_2024.pdf"),
]
for d in docs:
    secs = d.all_sections_flat()
    print(f"{d.filename}: {d.page_count}p, {len(secs)} sections")
    for s in secs[:3]:
        tags = []
        if s.has_list: tags.append("list")
        if s.has_numbers: tags.append("nums")
        if s.has_dates: tags.append("dates")
        if s.has_contact_info: tags.append("contact")
        if s.has_definition: tags.append("def")
        print(f"  [{s.level}] {s.heading[:50]} ({s.word_count()}w, p{s.page_start}) [{', '.join(tags)}]")

# Index
index = SearchIndex()
index.build(docs)
all_secs = index.get_all_sections()
print(f"\nIndexed {len(all_secs)} sections")

# Entity extraction
entities_map = []
for sec, fname in all_secs:
    entities_map.append(extract_entities(sec, fname))

rich = [(e, e.entity_count) for e in entities_map if e.entity_count > 0]
rich.sort(key=lambda x: x[1], reverse=True)
print(f"Sections with entities: {len(rich)}/{len(entities_map)}")
for e, cnt in rich[:5]:
    print(f"  {e.source_filename}: {e.section.heading[:40]} ({cnt} entities)")
    if e.defined_terms:
        print(f"    terms: {e.defined_terms[:3]}")
    if e.money_amounts:
        print(f"    money: {e.money_amounts[:3]}")
    if e.phones:
        print(f"    phones: {e.phones[:3]}")

# Selection test per question type
rng = random.Random(42)
tracker = DiversityTracker()
test_types = [
    QuestionType.DIRECT_LOOKUP,
    QuestionType.PARAPHRASE_LOOKUP,
    QuestionType.SPECIFIC_JARGON,
    QuestionType.MULTI_HOP_WITHIN_CORPUS,
    QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS,
]

print("\n--- Selection Test ---")
for qtype in test_types:
    candidates = select_passages(
        question_type=qtype,
        count=2,
        index=index,
        entities_map=entities_map,
        documents=docs,
        tracker=tracker,
        rng=rng,
    )
    print(f"\n{qtype.value}: {len(candidates)} candidates")
    for c in candidates:
        print(f"  src={c.source_documents}, ch={c.chapter[:40]}")
        print(f"  passage[:100]: {c.passage[:100].replace(chr(10), ' ')}...")

print("\nAll phases OK (no LLM calls needed)")
