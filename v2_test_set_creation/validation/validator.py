"""
Deterministic validation for generated Q+A pairs.

Checks that golden_context is real document text and that
the answer is grounded in the passage. No LLM calls needed
because passages come directly from the parsed document index.
"""
import re
from typing import Dict, Any, List, Optional

from ..parsing.document import Document
from ..config import QuestionType


class Validator:
    """Validate generated questions against source documents."""

    # Minimum fraction of answer keywords found in passage
    MIN_KEYWORD_RATIO = 0.25

    def validate(
        self,
        q_data: Dict[str, str],
        golden_ctx: str,
        sources: List[str],
        documents: List[Document],
        question_type: Optional[QuestionType] = None,
    ) -> Dict[str, Any]:
        """Validate a generated question.

        Checks:
        1. golden_context exists in source documents
        2. golden_context is not a ToC region
        3. golden_answer is grounded in golden_context
        4. golden_context length is reasonable
        5. multi_hop_between excerpts come from different docs

        Returns dict with 'valid', 'reason', and metrics.
        """
        # Check 1: context exists in source documents
        ctx_ratio = self._compute_context_match(
            golden_ctx, sources, documents
        )
        if ctx_ratio < 0.3:
            return {
                "valid": False,
                "reason": (
                    f"context not found in docs "
                    f"(match={ctx_ratio:.0%})"
                ),
                "match_ratio": ctx_ratio,
                "answer_grounded": False,
            }

        # Check 2: not a ToC region
        if self._is_toc_region(golden_ctx):
            return {
                "valid": False,
                "reason": "context is a table-of-contents",
                "match_ratio": ctx_ratio,
                "answer_grounded": False,
            }

        # Check 3: answer grounded in passage
        # Skip grounding for types where answer intentionally
        # diverges from passage content.
        _SKIP_GROUNDING = {
            QuestionType.HALLUCINATION_TEST,
            QuestionType.ADVERSARIAL_AGGRO,
            QuestionType.PROMPT_INJECTION,
            QuestionType.TOOL_CALL_CHECK,
            QuestionType.TEMPORAL_QUESTIONS,
            QuestionType.LONG_CONTEXT_SYNTHESIS,
        }
        answer = q_data.get("answer", "")
        if question_type in _SKIP_GROUNDING:
            answer_grounded = True
        else:
            answer_grounded = self._is_answer_grounded(
                answer, golden_ctx
            )
            if not answer_grounded:
                return {
                    "valid": False,
                    "reason": "answer not grounded in passage",
                    "match_ratio": ctx_ratio,
                    "answer_grounded": False,
                }

        # Check 4: context length
        if len(golden_ctx.strip()) < 30:
            return {
                "valid": False,
                "reason": "context too short (<30 chars)",
                "match_ratio": ctx_ratio,
                "answer_grounded": answer_grounded,
            }

        # Check 5: multi_hop_between excerpts from diff docs
        if (
            question_type
            == QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS
            and "\n\n---\n\n" in golden_ctx
        ):
            parts = golden_ctx.split("\n\n---\n\n", 1)
            if len(parts) == 2 and len(sources) >= 2:
                doc0 = self._best_matching_doc(
                    parts[0], sources, documents
                )
                doc1 = self._best_matching_doc(
                    parts[1], sources, documents
                )
                if doc0 and doc1 and doc0 == doc1:
                    return {
                        "valid": False,
                        "reason": (
                            f"both excerpts from same doc "
                            f"({doc0})"
                        ),
                        "match_ratio": ctx_ratio,
                        "answer_grounded": answer_grounded,
                    }

        return {
            "valid": True,
            "reason": "passed",
            "match_ratio": ctx_ratio,
            "answer_grounded": answer_grounded,
        }

    # ── Helper methods ──────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _compute_context_match(
        self,
        golden_ctx: str,
        sources: List[str],
        documents: List[Document],
    ) -> float:
        """Fraction of context chunks found in source docs."""
        if not golden_ctx or not sources or not documents:
            return 0.0

        doc_texts = self._get_doc_texts(sources, documents)
        if not doc_texts:
            return 0.0

        combined = self._normalize(" ".join(doc_texts))
        ctx_norm = self._normalize(golden_ctx)
        words = ctx_norm.split()

        chunk_size = 4
        if len(words) < chunk_size:
            return 1.0 if ctx_norm in combined else 0.0

        chunks_total = 0
        chunks_found = 0
        step = max(1, chunk_size // 2)
        for i in range(0, len(words) - chunk_size + 1, step):
            chunk = " ".join(words[i:i + chunk_size])
            chunks_total += 1
            if chunk in combined:
                chunks_found += 1

        return chunks_found / chunks_total if chunks_total else 0.0

    def _is_answer_grounded(
        self, answer: str, passage: str
    ) -> bool:
        """Check if answer keywords appear in the passage."""
        if not answer or not passage:
            return False

        a_norm = self._normalize(answer)
        p_norm = self._normalize(passage)

        words = [w for w in a_norm.split() if len(w) >= 4]
        if not words:
            return a_norm in p_norm

        found = sum(1 for w in words if w in p_norm)
        return (found / len(words)) >= self.MIN_KEYWORD_RATIO

    @staticmethod
    def _is_toc_region(text: str) -> bool:
        lines = [
            ln.strip() for ln in text.split("\n") if ln.strip()
        ]
        if len(lines) < 2:
            return False
        num_ending = sum(
            1 for ln in lines if re.search(r"\d+\s*$", ln)
        )
        short_lines = sum(1 for ln in lines if len(ln) < 80)
        toc_score = num_ending / len(lines)
        short_score = short_lines / len(lines)
        return toc_score > 0.3 and short_score > 0.6

    def _best_matching_doc(
        self,
        excerpt: str,
        sources: List[str],
        documents: List[Document],
    ) -> Optional[str]:
        keywords = set(
            w for w in self._normalize(excerpt).split()
            if len(w) >= 4
        )
        if not keywords:
            return None

        best_src = None
        best_score = 0.0
        for src in sources:
            texts = self._get_doc_texts([src], documents)
            if not texts:
                continue
            combined = self._normalize(" ".join(texts))
            found = sum(1 for kw in keywords if kw in combined)
            score = found / len(keywords)
            if score > best_score:
                best_score = score
                best_src = src
        return best_src

    @staticmethod
    def _get_doc_texts(
        sources: List[str], documents: List[Document]
    ) -> List[str]:
        """Get raw text for matching source filenames."""
        import os
        texts = []
        for doc in documents:
            doc_base = os.path.basename(doc.filename).lower()
            for src in sources:
                src_base = os.path.basename(src).lower()
                if doc_base == src_base:
                    texts.append(doc.raw_text)
                    break
        return texts
