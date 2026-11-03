"""Deterministic retrieval over processed Lantus chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import math
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

logger = logging.getLogger(__name__)


# Light token normalization: strips common English suffixes so "stored" vs
# "storage", "effects" vs "effect" etc. match. Applied identically to both
# corpus and query tokens, so over-stripping is harmless.
_SUFFIXES = (
    "ations",
    "ation",
    "itions",
    "ition",
    "ings",
    "ing",
    "ies",
    "ied",
    "ed",
    "es",
    "s",
    "e",
)

_NOISE_WORDS = frozenset(
    {
        "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
        "does", "did", "do", "is", "are", "was", "were", "be", "been", "being",
        "can", "could", "should", "would", "will", "may", "might", "shall",
        "the", "a", "an", "and", "or", "but", "for", "with", "to", "from",
        "in", "on", "at", "by", "of", "not", "this", "that", "these", "those",
        "i", "we", "you", "he", "she", "it", "they", "them", "your", "my",
        "me", "him", "her", "us", "our", "its", "their", "than", "then",
        "as", "if", "into", "about", "after", "before", "out", "up", "down",
        "no", "yes", "oh", "so", "too", "very", "more", "most", "have", "has",
        "had", "haven't", "hasn't", "can't", "dont", "don't", "shouldn't",
        "n't", "nt",
    }
)


def normalize_token(token: str) -> str:
    """Normalize a single lowercase token (suffix stripping) for matching."""
    t = token
    if len(t) <= 3:
        return t
    for suffix in _SUFFIXES:
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            return t[: -len(suffix)]
    return t


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into normalized, non-noise tokens."""
    result = []
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        token = normalize_token(raw)
        if token and token not in _NOISE_WORDS:
            result.append(token)
    return result


# Deterministic query expansion: maps user phrasing to canonical corpus terms
# (e.g. the package insert says "adverse reaction", never "side effect").
_QUERY_EXPANSIONS = (
    (re.compile(r"(?i)used\s+for\b"), "indication indicated therapeutic"),
    (re.compile(r"(?i)side\s+effects?\b"), "adverse reaction undesirable"),
    (re.compile(r"(?i)how\s+does\b.*\bwork\b|mechanism\s+of\s+action"),
     "mode action pharmacodynamic insulin glargine"),
    (re.compile(r"(?i)stor(?:e|ed|age|ing)\b|refrigerat\w*|fridge"), "storage refrigerat freezer"),
    (re.compile(r"(?i)miss(?:ed)?\s+a\s+dose|forget"), "missed dose skip injection"),
)


def expand_query(query: str) -> str:
    """Append canonical corpus terms when the query matches a known phrasing."""
    expanded = query
    for pattern, terms in _QUERY_EXPANSIONS:
        if pattern.search(query):
            expanded = f"{expanded} {terms}"
    return expanded


@dataclass
class RetrievedChunk:
    chunk_id: str
    page_number: int
    section_heading: Optional[str]
    relevance_score: float
    text: str
    metadata: Optional[Dict[str, Any]] = None


# Relevance blending for candidate reranking: cosine over a short chunk can be
# dominated by repetitions of the product name, letting packaging/labeling
# chunks outrank the section that actually answers the question. Candidates
# from the cosine top-pool are therefore re-scored as
# cosine + _RERANK_ALPHA * (idf-weighted coverage of the expanded query
# terms), where coverage counts DISTINCT matched terms (text + section
# heading) so term frequency and chunk length cannot swamp subject relevance.
_RERANK_POOL = 10
_RERANK_ALPHA = 0.07


class LantusRetriever:
    """Lightweight TF-IDF based retriever."""

    def __init__(self, chunks_path: Path | str | None = None) -> None:
        self.chunks_path = Path(chunks_path or "data/processed/lantus_chunks.json")
        self.chunks: List[Dict[str, Any]] = self._load_chunks()
        self.vectorizer = None
        self.tfidf_matrix = None
        self._idf: Dict[str, float] = {}
        self._chunk_terms: List[set] = []
        self._build_index()

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            logger.warning(
                "mi_retriever chunk file not found at %s; retrieval has no "
                "evidence to answer from (operators: verify the chunks file "
                "is deployed)",
                self.chunks_path,
            )
            return []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self) -> None:
        if TfidfVectorizer is None or cosine_similarity is None:
            logger.warning(
                "mi_retriever degraded: scikit-learn unavailable, falling back "
                "to keyword-overlap retrieval (operators: verify scikit-learn "
                "is installed in the deployment environment)"
            )
            self._build_chunk_terms()
            return
        texts = [c["text"] for c in self.chunks]
        if not texts:
            logger.warning(
                "mi_retriever degraded: no reference chunks loaded at %s; "
                "retrieval cannot answer product questions",
                self.chunks_path,
            )
            return
        self.vectorizer = TfidfVectorizer(analyzer=tokenize_text, max_features=2000)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._build_chunk_terms()
        self._idf = dict(
            zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_)
        )

    def _build_chunk_terms(self) -> None:
        # Per-chunk term sets (body text + section heading): the rerank
        # coverage in TF-IDF mode and the keyword-overlap scoring in
        # degraded mode share this same normalized token representation.
        self._chunk_terms = [
            set(tokenize_text(c["text"]))
            | set(tokenize_text(c.get("section_heading") or ""))
            for c in self.chunks
        ]

    def _coverage(self, query_terms: set, index: int) -> float:
        """idf-weighted count of distinct query terms covered by one chunk."""
        if not query_terms or index >= len(self._chunk_terms):
            return 0.0
        matched = query_terms & self._chunk_terms[index]
        return sum(self._idf.get(term, 0.0) for term in matched)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []
        query = query.strip()
        k = max(1, k) if k else 3

        if self.vectorizer is None or self.tfidf_matrix is None:
            # Degraded keyword-overlap retrieval (scikit-learn unavailable).
            # Deterministic scoring over the SAME normalized tokens the TF-IDF
            # analyzer uses, with IDF weights computed straight from the
            # reference chunks themselves (pure Python; no sklearn needed):
            # a term that appears in every chunk (e.g. the product name) gets
            # no weight, mirroring how the TF-IDF subject-coverage rerank
            # discounts boilerplate. Rank by distinct matched-term weight,
            # then by matched-token density (occurrences per chunk character)
            # so long chunks cannot swamp subject relevance. Never invents
            # relevance an overlap cannot support.
            query_terms = set(tokenize_text(expand_query(query)))
            if not query_terms:
                return []
            n_docs = len(self._chunk_terms)
            doc_freq = {
                term: sum(1 for terms in self._chunk_terms if term in terms)
                for term in query_terms
            }
            term_weight = {
                term: math.log((n_docs + 1.0) / (doc_freq[term] + 1.0))
                for term in query_terms
            }
            scored = []
            for index in range(len(self.chunks)):
                matched = query_terms & self._chunk_terms[index]
                if not matched:
                    continue
                c = self.chunks[index]
                text_lower = (
                    f"{c['text']}\n{c.get('section_heading') or ''}".lower()
                )
                occurrences = sum(text_lower.count(term) for term in matched)
                weight = sum(term_weight[term] for term in matched)
                density = occurrences / max(1, len(c["text"]))
                scored.append((weight, density, index))
            scored.sort(key=lambda entry: (-entry[0], -entry[1]))
            results = []
            for weight, _density, index in scored[:k]:
                c = self.chunks[index]
                results.append(
                    RetrievedChunk(
                        chunk_id=c["chunk_id"],
                        page_number=c["page_number"],
                        section_heading=c.get("section_heading"),
                        relevance_score=round(weight, 6),
                        text=c["text"],
                    )
                )
            if results:
                logger.warning(
                    "mi_retriever degraded retrieval in use: matched %d "
                    "chunk(s) by idf-weighted keyword overlap",
                    len(results),
                )
            return results

        # TF-IDF similarity, then a subject-coverage rerank of the top pool
        # (see _RERANK_* above). `relevance_score` stays the cosine value.
        expanded = expand_query(query)
        query_vec = self.vectorizer.transform([expanded])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        pool = [
            idx for idx in sims.argsort()[::-1][:_RERANK_POOL] if sims[idx] > 0
        ]
        query_terms = set(tokenize_text(expanded))
        pool.sort(
            key=lambda idx: (
                -(sims[idx] + _RERANK_ALPHA * self._coverage(query_terms, idx)),
                -sims[idx],
            )
        )
        results = []
        for idx in pool[:k]:
            c = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    page_number=c["page_number"],
                    section_heading=c.get("section_heading"),
                    relevance_score=float(sims[idx]),
                    text=c["text"],
                )
            )
        return results

    def get_chunk_count(self) -> int:
        return len(self.chunks)
