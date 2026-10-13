"""Deterministic retrieval over processed Lantus chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


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
    (re.compile(r"(?i)how\s+does\b.*\bwork\b|mechanism\s+of\s+action"), "action long acting analogue"),
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


class LantusRetriever:
    """Lightweight TF-IDF based retriever."""

    def __init__(self, chunks_path: Path | str | None = None) -> None:
        self.chunks_path = Path(chunks_path or "data/processed/lantus_chunks.json")
        self.chunks: List[Dict[str, Any]] = self._load_chunks()
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self) -> None:
        if TfidfVectorizer is None or cosine_similarity is None:
            return  # fallback mode
        texts = [c["text"] for c in self.chunks]
        if not texts:
            return
        self.vectorizer = TfidfVectorizer(analyzer=tokenize_text, max_features=2000)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []
        query = query.strip()
        k = max(1, k) if k else 3

        if self.vectorizer is None or self.tfidf_matrix is None:
            # Simple keyword fallback
            results = []
            query_lower = query.lower()
            for c in self.chunks:
                text_lower = c["text"].lower()
                score = text_lower.count(query_lower) / max(1, len(text_lower))
                if query_lower in text_lower:
                    results.append((score, c))
            results.sort(key=lambda x: x[0], reverse=True)
            top = results[:k]
            return [
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    page_number=c["page_number"],
                    section_heading=c.get("section_heading"),
                    relevance_score=float(s),
                    text=c["text"],
                )
                for s, c in top
            ]

        # TF-IDF similarity
        query_vec = self.vectorizer.transform([expand_query(query)])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            if sims[idx] <= 0:
                continue
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
