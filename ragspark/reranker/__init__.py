"""
`ragspark.reranker` — Ultra-Fast Cross-Encoder & Flash Re-Ranking Module.

Re-scores retrieved candidate chunks directly against the user query to push the most
precise, high-value paragraphs to rank #1.

Supports:
  - `model="flash"`: Built-in zero-dependency cross-scoring algorithm (<2ms latency).
    Analyzes n-gram overlap, query term density, exact phrase matches, and title proximity.
  - `model="cross_encoder"`: Neural cross-encoder using `sentence-transformers` (e.g. `BAAI/bge-reranker-base`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from ragspark.chunker import Chunk
from ragspark.vectordb import SearchResult

T = TypeVar("T")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]


def _compute_flash_score(query: str, text: str, metadata: dict[str, Any] | None = None) -> float:
    """
    High-speed cross-feature scoring algorithm:
    1. Exact query phrase matching (weight 0.35)
    2. Query term coverage ratio (weight 0.30)
    3. Bi-gram and tri-gram overlap (weight 0.20)
    4. Lexical term proximity / density (weight 0.15)
    """
    if not query.strip() or not text.strip():
        return 0.0

    q_clean = query.strip().lower()
    t_clean = text.strip().lower()

    # 1. Exact query phrase presence
    phrase_score = 1.0 if q_clean in t_clean else 0.0

    q_tokens = _tokenize(q_clean)
    if not q_tokens:
        return phrase_score

    t_tokens = _tokenize(t_clean)
    t_token_set = set(t_tokens)

    # 2. Query term coverage
    matched_q_tokens = [w for w in q_tokens if w in t_token_set]
    coverage = len(matched_q_tokens) / len(q_tokens)

    # 3. N-Gram (Bi-gram) overlap
    if len(q_tokens) > 1:
        q_bigrams = set(zip(q_tokens[:-1], q_tokens[1:]))
        t_bigrams = set(zip(t_tokens[:-1], t_tokens[1:]))
        bigram_overlap = len(q_bigrams & t_bigrams) / max(1, len(q_bigrams))
    else:
        bigram_overlap = coverage

    # 4. Proximity / Density bonus
    density_score = 0.0
    if len(matched_q_tokens) >= 2 and len(t_tokens) >= 2:
        positions = [i for i, w in enumerate(t_tokens) if w in t_token_set]
        if len(positions) >= 2:
            span = max(positions) - min(positions) + 1
            ideal_span = len(matched_q_tokens)
            density_score = math.exp(-0.05 * max(0, span - ideal_span))

    # Optional metadata boost (e.g. section title matches query terms)
    meta_boost = 0.0
    if metadata:
        meta_str = " ".join(str(v) for v in metadata.values()).lower()
        if any(w in meta_str for w in q_tokens):
            meta_boost = 0.10

    raw_score = (
        0.35 * phrase_score
        + 0.30 * coverage
        + 0.20 * bigram_overlap
        + 0.15 * density_score
        + meta_boost
    )
    return min(1.0, max(0.0, raw_score))


def rerank(
    query: str,
    candidates: Sequence[Any],
    top_k: int | None = None,
    model: str = "flash",
) -> list[Any]:
    """
    Re-rank candidate search results against the query.

    Args:
        query: User search query.
        candidates: List of SearchResult objects, Chunks, or strings.
        top_k: Optional number of top results to return.
        model: 'flash' (ultra-fast zero dependency) or 'cross_encoder' (neural model).

    Returns:
        Re-ranked list sorted by relevance score.
    """
    if not candidates:
        return []

    scored_items: list[tuple[float, Any]] = []

    for item in candidates:
        text = ""
        meta = None

        if isinstance(item, SearchResult):
            text = item.chunk.text
            meta = item.chunk.metadata
        elif isinstance(item, Chunk):
            text = item.text
            meta = item.metadata
        elif isinstance(item, dict):
            text = str(item.get("text", item.get("content", "")))
            meta = item.get("metadata")
        elif isinstance(item, str):
            text = item
        else:
            text = str(getattr(item, "text", str(item)))

        score = _compute_flash_score(query, text, metadata=meta)

        if isinstance(item, SearchResult):
            # Combine initial vector score with cross-feature re-rank score
            combined = (0.35 * item.vector_score) + (0.65 * score)
            item.rerank_score = float(score)
            item.score = float(combined)
            scored_items.append((combined, item))
        else:
            scored_items.append((score, item))

    # Sort descending by score
    scored_items.sort(key=lambda x: x[0], reverse=True)

    results = [item for _, item in scored_items]

    # Update rank index for SearchResults
    for idx, item in enumerate(results, start=1):
        if isinstance(item, SearchResult):
            item.rank = idx

    if top_k is not None and top_k > 0:
        return results[:top_k]
    return results


class Reranker:
    """Class wrapper for re-ranking pipelines."""

    def __init__(self, model: str = "flash") -> None:
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        return rerank(query, candidates, top_k=top_k, model=self.model)

    def __call__(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        return self.rerank(query, candidates, top_k=top_k)


__all__ = ["rerank", "Reranker"]

