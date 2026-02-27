"""
Pipeline orchestrator: Parse → Index → Select → Generate → Validate.

Ties all modules together into a single run() call.
"""
import os
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .config import TestSetConfig, QuestionType
from .parsing.document import Document
from .parsing.pdf_parser import parse_pdf
from .parsing.text_parser import parse_text_file
from .indexing.search_index import SearchIndex
from .indexing.entity_extractor import (
    SectionEntities, extract_entities,
)
from .selection.strategies import select_passages
from .selection.diversity import DiversityTracker
from .generation.llm_client import LLMClient
from .generation.qa_generator import QAGenerator
from .generation.prompts import (
    EXPECTED_BEHAVIORS, TYPE_INSTRUCTIONS,
)
from .validation.validator import Validator


class Pipeline:
    """v2 Test Set Creation Pipeline."""

    def __init__(self, config: TestSetConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.validator = Validator(config.validation)

        # LLM client — only used for Q+A generation
        self.llm = LLMClient(
            api_key=config.llm.api_key,
            model=config.llm.model,
            azure_endpoint=config.llm.azure_endpoint,
            azure_api_version=config.llm.azure_api_version,
        )
        self.qa_gen = QAGenerator(self.llm)

    def run(self) -> Tuple[str, Dict[str, Any]]:
        """Run the full pipeline. Returns (output_path, test_set)."""
        total_start = time.time()
        enabled = self.config.get_enabled_question_types()
        total_q = self.config.get_total_enabled_questions()

        print("v2 Test Set Creator")
        print("=" * 50)
        print(f"Enabled question types: {len(enabled)}")
        print(f"Total questions to generate: {total_q}")
        print(f"Input path: {self.config.input_documents_path}")
        print(f"LLM model: {self.config.llm.model}")
        print("=" * 50)

        # ── Phase 0: Parse documents ────────────────────
        print("\n[Phase 0] Parsing documents...")
        phase_start = time.time()
        documents = self._load_documents()
        parse_time = time.time() - phase_start
        print(
            f"  Parsed {len(documents)} documents "
            f"({sum(d.page_count for d in documents)} pages) "
            f"in {parse_time:.1f}s"
        )
        for doc in documents:
            secs = doc.all_sections_flat()
            print(
                f"  - {doc.filename}: {doc.page_count} pages, "
                f"{len(secs)} sections"
            )

        # ── Phase 1: Index ──────────────────────────────
        print("\n[Phase 1] Building search index...")
        phase_start = time.time()
        index = SearchIndex()
        index.build(documents)
        entities_map = self._extract_all_entities(
            index, documents
        )
        index_time = time.time() - phase_start
        print(
            f"  Indexed {len(index.get_all_sections())} sections, "
            f"extracted entities from {len(entities_map)} sections "
            f"in {index_time:.1f}s"
        )

        # ── Build document metadata (dates) ─────────────
        doc_metadata = self._extract_doc_metadata(documents)

        # ── Phase 2-4: Select → Generate → Validate ────
        print("\n[Phase 2-4] Generating questions...")
        tracker = DiversityTracker()
        all_questions = []
        metrics = {
            "generation_time": 0,
            "questions_by_type": {},
            "total_questions": 0,
            "parse_time": parse_time,
            "index_time": index_time,
            "llm_calls": 0,
        }

        for qtype in enabled:
            qcfg = self.config.question_types[qtype]
            print(
                f"\n  [{qtype.value}] Selecting passages "
                f"for {qcfg.count} questions..."
            )

            # Phase 2: Select passages
            # Types with high rejection rates need more candidates
            _HIGH_REJECT = {
                QuestionType.HALLUCINATION_TEST,
                QuestionType.LONG_CONTEXT_SYNTHESIS,
            }
            extra = 6 if qtype in _HIGH_REJECT else 2
            candidates = select_passages(
                question_type=qtype,
                count=qcfg.count + extra,  # extras for failures
                index=index,
                entities_map=entities_map,
                documents=documents,
                tracker=tracker,
                rng=self.rng,
                cfg=self.config.selection,
            )
            print(f"    Selected {len(candidates)} candidates")

            # Phase 3+4: Generate & Validate
            type_questions = []
            for p_idx, candidate in enumerate(candidates):
                if len(type_questions) >= qcfg.count:
                    break

                print(
                    f"    Generating Q+A from passage "
                    f"{p_idx + 1}/{len(candidates)}..."
                )

                # Phase 3: LLM generates Q+A
                # Pass doc_metadata for temporal/versioning
                candidate_meta = None
                if qtype == QuestionType.TEMPORAL_QUESTIONS:
                    candidate_meta = {
                        s: doc_metadata.get(s, {})
                        for s in candidate.source_documents
                    }
                q_data = self.qa_gen.generate(
                    passage=candidate.passage,
                    source_documents=candidate.source_documents,
                    chapter=candidate.chapter,
                    question_type=qtype,
                    difficulty=qcfg.difficulty,
                    doc_metadata=candidate_meta,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                )
                metrics["llm_calls"] += 1

                if not q_data:
                    print("      No valid Q+A generated, skip")
                    continue

                # Phase 4: Validate
                validation = self.validator.validate(
                    q_data=q_data,
                    golden_ctx=candidate.passage,
                    sources=candidate.source_documents,
                    documents=documents,
                    question_type=qtype,
                )

                if not validation["valid"]:
                    print(
                        f"      REJECTED: "
                        f"{validation['reason']}"
                    )
                    continue

                # Hallucination: verify topic isn't elsewhere
                if qtype == QuestionType.HALLUCINATION_TEST:
                    if self._hallucination_topic_exists(
                        q_data["question"], index,
                        candidate.passage,
                    ):
                        print(
                            "      REJECTED: hallucination "
                            "topic found elsewhere in corpus"
                        )
                        continue

                # Build question record
                idx = len(all_questions) + len(type_questions)
                instruction = TYPE_INSTRUCTIONS.get(
                    qtype,
                    f"Generate {qtype.value} questions."
                )
                question_record = {
                    "id": f"v2_{qtype.value}_{idx}",
                    "type": qtype.value,
                    "question": q_data["question"],
                    "expected_behavior": EXPECTED_BEHAVIORS.get(
                        qtype,
                        f"Answer the {qtype.value} question."
                    ),
                    "golden_answer": q_data["answer"],
                    "golden_context": candidate.passage,
                    "source_documents": candidate.source_documents,
                    "difficulty": qcfg.difficulty,
                    "generation_prompt": instruction,
                    "hallucination_detected": False,
                    "context_repaired": False,
                    "metadata": {
                        "generated_by": "v2_pipeline",
                        "question_type": qtype.value,
                        "llm_model": self.config.llm.model,
                        "chapter": candidate.chapter,
                        "subchapters": candidate.subchapters,
                        "start_pages": {
                            s: candidate.page_start
                            for s in candidate.source_documents
                        },
                        "end_pages": {
                            s: candidate.page_end
                            for s in candidate.source_documents
                        },
                        "context_match_ratio": validation.get(
                            "match_ratio", 0
                        ),
                        "answer_grounded": validation.get(
                            "answer_grounded", False
                        ),
                    },
                }

                # Multi-turn followup: add turn_2 fields
                if qtype == QuestionType.MULTI_TURN_FOLLOWUP:
                    question_record["question_turn_2"] = (
                        q_data.get("question_turn_2", "")
                    )
                    question_record["answer_turn_2"] = (
                        q_data.get("answer_turn_2", "")
                    )

                type_questions.append(question_record)
                tracker.mark_used(
                    candidate.section,
                    candidate.source_documents[0],
                    q_data["question"],
                )
                print(
                    f"      PASSED: "
                    f"'{q_data['question'][:60]}' "
                    f"(match="
                    f"{validation.get('match_ratio', 0):.0%})"
                )

            all_questions.extend(type_questions)
            metrics["questions_by_type"][qtype.value] = len(
                type_questions
            )
            print(
                f"    Generated {len(type_questions)}/{qcfg.count} "
                f"{qtype.value} questions"
            )

        total_time = time.time() - total_start
        metrics["generation_time"] = total_time
        metrics["total_questions"] = len(all_questions)

        # ── Build test set ──────────────────────────────
        test_set = {
            "metadata": {
                "created_at": time.time(),
                "pipeline_version": "v2",
                "config": {
                    "llm_model": self.config.llm.model,
                    "rlm_model": self.config.llm.model,
                    "enabled_types": [
                        t.value for t in enabled
                    ],
                    "total_questions": len(all_questions),
                },
                "metrics": metrics,
            },
            "questions": all_questions,
        }

        # ── Save ────────────────────────────────────────
        output_file = self._save(test_set)

        print(f"\n{'=' * 50}")
        print(f"Pipeline completed in {total_time:.1f}s")
        print(f"LLM calls: {metrics['llm_calls']}")
        print(f"Questions generated: {len(all_questions)}")
        print(f"Output: {output_file}")

        return output_file, test_set

    # ── Internal methods ────────────────────────────────

    def _load_documents(self) -> List[Document]:
        """Load and parse all documents from input path."""
        path = Path(self.config.input_documents_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Input path not found: {path}"
            )

        documents = []
        for fpath in sorted(path.rglob("*")):
            if not fpath.is_file():
                continue
            try:
                if fpath.suffix.lower() == ".pdf":
                    documents.append(parse_pdf(str(fpath)))
                elif fpath.suffix.lower() in (".txt", ".md"):
                    documents.append(
                        parse_text_file(str(fpath))
                    )
                elif fpath.suffix.lower() == ".json":
                    # JSON documents: load as text
                    text = fpath.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    documents.append(Document(
                        filename=fpath.name,
                        raw_text=text,
                        page_count=1,
                        doc_type="json",
                    ))
            except Exception as e:
                print(f"  Error loading {fpath}: {e}")

        return documents

    def _extract_all_entities(
        self,
        index: SearchIndex,
        documents: List[Document],
    ) -> List[SectionEntities]:
        """Extract entities from all indexed sections."""
        entities_map = []
        for sec, fname in index.get_all_sections():
            entities = extract_entities(sec, fname)
            entities_map.append(entities)
        return entities_map

    def _extract_doc_metadata(
        self, documents: List[Document]
    ) -> Dict[str, Dict[str, str]]:
        """Extract file-level metadata (dates) for each document.

        Tries PDF metadata first, then falls back to filesystem
        modification time.  Used by temporal/versioning questions.
        """
        import re as _re
        path = Path(self.config.input_documents_path)
        meta = {}
        for doc in documents:
            info: Dict[str, str] = {}
            # Try to find the file on disk
            for fpath in path.rglob("*"):
                if fpath.name == doc.filename:
                    # Filesystem modified time
                    mtime = datetime.fromtimestamp(
                        fpath.stat().st_mtime
                    )
                    info["modified"] = mtime.strftime(
                        "%Y-%m-%d"
                    )
                    # Try extracting date from filename
                    m = _re.search(
                        r"(20\d{2})", doc.filename
                    )
                    if m:
                        info["year_in_name"] = m.group(1)
                    break
            meta[doc.filename] = info
        return meta

    def _hallucination_topic_exists(
        self,
        question: str,
        index: SearchIndex,
        source_passage: str,
    ) -> bool:
        """Check if the hallucination question's topic appears
        elsewhere in the corpus (outside the source passage).

        Uses BM25 search to find potential matches.  Only rejects
        when a *clearly different* section scores high AND contains
        multiple topic-specific keywords from the question.
        """
        import re as _re
        # Extract topic-specific nouns (>= 6 chars, skip stopwords)
        q_norm = _re.sub(r"[^\w\s]", " ", question.lower())
        stopwords = {
            "welke", "welk", "wanneer", "waarom", "hoeveel",
            "welken", "wordt", "worden", "heeft", "hebben",
            "kunnen", "volgens", "hierbij", "hiervan", "daarbij",
            "daarvan", "medewerker", "medewerkers", "binnen",
            "beleid", "regeling", "informatie", "document",
            "richtlijn", "organisatie", "datasciencelab",
        }
        keywords = [
            w for w in q_norm.split()
            if len(w) >= 6 and w not in stopwords
        ]
        if len(keywords) < 2:
            return False

        query = " ".join(keywords[:5])
        results = index.search(query, top_k=3)

        # Check if any top result is clearly from a DIFFERENT section
        src_norm = _re.sub(
            r"\s+", " ", source_passage.lower()[:300]
        )
        src_words = set(src_norm.split())
        for sec, fname, score in results:
            bm25_thresh = self.config.pipeline.hallucination_bm25_threshold
            if score < bm25_thresh:
                continue
            sec_text = sec.full_text.lower()
            sec_norm = _re.sub(r"\s+", " ", sec_text[:300])
            sec_words = set(sec_norm.split())
            # Must be a genuinely different section
            overlap = len(src_words & sec_words)
            total = max(len(sec_words), 1)
            overlap_thresh = self.config.pipeline.hallucination_overlap_threshold
            if overlap / total > overlap_thresh:
                continue  # same/very similar section, skip
            # Count how many topic keywords appear in this section
            kw_hits = sum(
                1 for kw in keywords if kw in sec_text
            )
            if kw_hits >= 2:
                return True

        return False

    def _save(self, test_set: Dict[str, Any]) -> str:
        """Save test set to the configured output path."""
        name = self.config.corpus_name.replace(" ", "_")
        date_str = datetime.now().strftime(
            "%d_%m_%y_T%H_%M"
        )
        filename = f"v2_{name}_{date_str}.json"

        corpus_dir = os.path.join(
            self.config.output_path, name
        )
        os.makedirs(corpus_dir, exist_ok=True)
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                test_set, f, indent=2, ensure_ascii=False
            )

        return filepath
