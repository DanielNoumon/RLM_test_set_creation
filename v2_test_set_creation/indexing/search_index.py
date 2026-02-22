"""
Lightweight BM25 search index over document sections.

No external dependencies — implements BM25 from scratch using
only Python builtins and math. This replaces the RLM REPL's
search_documents() with a proper ranked retrieval system.
"""
import math
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter

from ..parsing.document import Document, Section


class SearchIndex:
    """BM25 search index over document sections."""

    # BM25 parameters
    K1 = 1.5
    B = 0.75

    def __init__(self):
        self._sections: List[Section] = []
        self._doc_sources: List[str] = []  # filename per section
        self._term_freqs: List[Counter] = []
        self._doc_freqs: Counter = Counter()
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    def build(self, documents: List[Document]) -> None:
        """Build the index from parsed documents."""
        self._sections = []
        self._doc_sources = []
        self._term_freqs = []

        for doc in documents:
            for sec in doc.all_sections_flat():
                # Skip very short sections (headers, page nums)
                if sec.word_count() < 10:
                    continue
                self._sections.append(sec)
                self._doc_sources.append(doc.filename)
                tokens = _tokenize(sec.full_text)
                self._term_freqs.append(Counter(tokens))

        self._n_docs = len(self._sections)
        if self._n_docs == 0:
            return

        # Document frequencies
        self._doc_freqs = Counter()
        for tf in self._term_freqs:
            for term in tf:
                self._doc_freqs[term] += 1

        # Average document length
        total_tokens = sum(
            sum(tf.values()) for tf in self._term_freqs
        )
        self._avg_dl = total_tokens / self._n_docs

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Section, str, float]]:
        """Search for sections matching the query.

        Returns list of (Section, filename, score) sorted by score.
        """
        if self._n_docs == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for idx in range(self._n_docs):
            if source_filter and self._doc_sources[idx] != source_filter:
                continue
            score = self._bm25_score(idx, query_tokens)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append((
                self._sections[idx],
                self._doc_sources[idx],
                score,
            ))
        return results

    def get_all_sections(
        self,
    ) -> List[Tuple[Section, str]]:
        """Return all indexed sections with their source filename."""
        return list(zip(self._sections, self._doc_sources))

    def get_sections_by_source(
        self, filename: str
    ) -> List[Section]:
        """Get all sections from a specific document."""
        return [
            sec for sec, src in zip(
                self._sections, self._doc_sources
            )
            if src == filename
        ]

    def get_source_filenames(self) -> List[str]:
        """Get unique source filenames in the index."""
        seen = []
        for src in self._doc_sources:
            if src not in seen:
                seen.append(src)
        return seen

    def _bm25_score(
        self, doc_idx: int, query_tokens: List[str]
    ) -> float:
        """Compute BM25 score for a document given query tokens."""
        tf = self._term_freqs[doc_idx]
        dl = sum(tf.values())
        score = 0.0

        for term in query_tokens:
            if term not in self._doc_freqs:
                continue
            df = self._doc_freqs[term]
            idf = math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1
            )
            term_freq = tf.get(term, 0)
            numerator = term_freq * (self.K1 + 1)
            denominator = (
                term_freq
                + self.K1
                * (1 - self.B + self.B * dl / self._avg_dl)
            )
            score += idf * numerator / denominator

        return score


# ── Tokenization ────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z\u00C0-\u024F0-9]+", re.IGNORECASE)

# Dutch + English stopwords (minimal set)
_STOPWORDS = frozenset({
    "de", "het", "een", "en", "van", "in", "is", "dat",
    "op", "te", "aan", "met", "voor", "er", "zijn", "die",
    "wordt", "door", "om", "als", "bij", "ook", "niet",
    "maar", "kan", "naar", "dan", "nog", "wel", "ze",
    "je", "we", "uit", "al", "was", "dit", "heeft", "of",
    "over", "tot", "worden", "meer", "hun", "been",
    "the", "a", "an", "and", "or", "is", "in", "to",
    "of", "for", "on", "with", "at", "by", "from",
    "this", "that", "it", "be", "are", "was", "were",
    "has", "have", "had", "not", "but", "can", "will",
})


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase terms, removing stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
