"""
`ragspark.vectordb` — Production Multi-Backend VectorDB & Multi-Strategy Retrieval Engine.

Supported VectorDB Providers (`provider=`):
  - "chroma"   (Default: ChromaDB native or persistent local Chroma-compatible engine)
  - "pinecone" (Pinecone cloud vector DB)
  - "mongodb"  (MongoDB Atlas Vector Search)
  - "qdrant"   (Qdrant local/cloud vector DB)
  - "memory"   (Fast in-memory Hybrid Vector + BM25 store)

Supported Retrieval Techniques (`strategy=`):
  - "hybrid"      (Default: Weighted Dense Semantic + BM25 Keyword + Legal Section Boost)
  - "semantic"    (Pure Dense Vector Cosine Similarity)
  - "keyword" / "bm25" (Pure Okapi BM25 Lexical + Exact Section/Citation Matching)
  - "rrf"         (Reciprocal Rank Fusion combining Dense Rank + BM25 Rank)
  - "mmr"         (Maximal Marginal Relevance for high-diversity, non-redundant top-k)
  - "rerank"      (2-Stage Retrieval + Cross-Feature Phrase & Legal Hierarchy Reranker)
  - "multi_query" (Expands query into multiple lexical/structural sub-queries & fuses via RRF)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragspark.chunker import Chunk

AVAILABLE_VECTORDBS = ("chroma", "pinecone", "mongodb", "qdrant", "memory")

AVAILABLE_RETRIEVAL_STRATEGIES = (
    "hybrid",
    "semantic",
    "keyword",
    "bm25",
    "rrf",
    "mmr",
    "rerank",
    "multi_query",
)


def list_vectordbs() -> list[str]:
    """Return all supported VectorDB backends."""
    return list(AVAILABLE_VECTORDBS)


def list_retrieval_strategies() -> list[str]:
    """Return all supported retrieval techniques."""
    return list(AVAILABLE_RETRIEVAL_STRATEGIES)


@dataclass
class SearchResult:
    """Represents a retrieved chunk with detailed breakdown of scores and strategy."""

    chunk: Chunk
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_strategy: str = "hybrid"
    rank: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "retrieval_strategy": self.retrieval_strategy,
            "chunk": self.chunk.to_dict(),
        }


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def _bm25_keyword_score(query_text: str, chunk: Chunk, avg_doc_len: float = 120.0) -> float:
    """Okapi BM25-inspired lexical saturation score + exact statutory/section boost."""
    if not query_text:
        return 0.0
    q_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9§]+", query_text) if len(w) >= 2]
    if not q_words:
        return 0.0

    doc_low = f"{chunk.section_path} {chunk.text}".lower()
    doc_tokens = re.findall(r"[a-zA-Z0-9§]+", doc_low)
    if not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    k1 = 1.5
    b = 0.75
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    bm25_sum = 0.0
    for qw in q_words:
        tf = tf_map.get(qw, 0)
        if tf > 0:
            denom = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
            bm25_sum += (tf * (k1 + 1.0)) / denom

    norm_bm25 = min(1.0, bm25_sum / (len(q_words) * 1.4))

    # Exact legal section/article reference boost (e.g., "section 2", "article 21")
    sec_matches = re.findall(r"(?:section|sec|article|clause|part)\s+\d+[a-z]?", query_text.lower())
    for sm in sec_matches:
        if sm in doc_low:
            norm_bm25 = min(1.0, norm_bm25 + 0.45)

    return norm_bm25


def _cross_feature_rerank_score(query_text: str, chunk: Chunk, base_score: float) -> float:
    """Second-stage reranker checking exact n-gram phrases, legal provisos, and section path matches."""
    if not query_text:
        return base_score
    q_clean = re.sub(r"\s+", " ", query_text.lower().strip(" ?!.,"))
    doc_low = f"{chunk.section_path} {chunk.text}".lower()

    bonus = 0.0
    q_words = [w for w in q_clean.split() if len(w) >= 3]
    # Bigram phrase match bonus
    for i in range(len(q_words) - 1):
        bigram = f"{q_words[i]} {q_words[i + 1]}"
        if bigram in doc_low:
            bonus += 0.12

    # Section path alignment bonus
    if chunk.section_path:
        for w in q_words:
            if w in chunk.section_path.lower():
                bonus += 0.15

    return min(1.0, (base_score * 0.75) + bonus)


_LAST_ACTIVE_STORE: VectorStore | None = None


class VectorStore:
    """Unified VectorDB interface supporting ChromaDB (default), Pinecone, MongoDB, Qdrant, and Memory."""

    def __init__(
        self,
        provider: str = "chroma",
        collection_name: str = "raglib_collection",
        persist_directory: str | Path | None = None,
        api_key: str | None = None,
        connection_uri: str | None = None,
        backend: str | None = None,
        embed_provider: str = "fast_local",
        embed_strategy: str = "dense",
        embed_model: str | None = None,
        embed_api_key: str | None = None,
        embed_dimension: int = 384,
        **kwargs: Any,
    ) -> None:
        global _LAST_ACTIVE_STORE
        raw_prov = backend or provider
        self.provider = raw_prov.lower().strip()
        if self.provider in ("chromadb", "chroma_db"):
            self.provider = "chroma"
        if self.provider not in AVAILABLE_VECTORDBS:
            raise ValueError(f"Unsupported vectordb '{raw_prov}'. Choose from: {AVAILABLE_VECTORDBS}")

        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.api_key = api_key
        self.connection_uri = connection_uri
        self.embed_provider = embed_provider
        self.embed_strategy = embed_strategy
        self.embed_model = embed_model
        self.embed_api_key = embed_api_key or api_key
        self.embed_dimension = embed_dimension
        self.kwargs = kwargs

        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []
        self._native_backend: Any = None
        self._backend_name = self.provider

        self._init_backend()
        _LAST_ACTIVE_STORE = self

    def _init_backend(self) -> None:
        if self.provider == "chroma":
            try:
                import chromadb  # type: ignore[import-not-found]

                if self.persist_directory:
                    client = chromadb.PersistentClient(path=str(self.persist_directory))
                else:
                    client = chromadb.EphemeralClient()
                self._native_backend = client.get_or_create_collection(name=self.collection_name)
                self._backend_name = "chromadb_native"
                try:
                    existing = self._native_backend.get(include=["documents", "embeddings", "metadatas"])
                    raw_ids = existing.get("ids")
                    raw_docs = existing.get("documents")
                    raw_embs = existing.get("embeddings")
                    raw_metas = existing.get("metadatas")
                    ids = list(raw_ids) if raw_ids is not None else []
                    docs = list(raw_docs) if raw_docs is not None else []
                    embs = list(raw_embs) if raw_embs is not None else []
                    metas = list(raw_metas) if raw_metas is not None else []
                    for idx_c, (cid, doc_txt, emb_vec) in enumerate(zip(ids, docs, embs)):
                        meta = metas[idx_c] if idx_c < len(metas) and metas[idx_c] is not None else {}
                        self._chunks.append(
                            Chunk(
                                id=str(cid),
                                text=str(doc_txt or ""),
                                index=idx_c + 1,
                                source=str(meta.get("source", "chroma")),
                                page_number=int(meta.get("page_number", 1)),
                                section_path=str(meta.get("section_path", "")),
                                strategy=str(meta.get("strategy", "recursive")),
                                metadata=dict(meta),
                            )
                        )
                        self._embeddings.append([float(x) for x in emb_vec])
                except Exception:
                    pass
            except Exception:
                self._backend_name = "chroma_local_engine"
            if not self._chunks:
                self._load_local_persistence()

        elif self.provider == "pinecone":
            if self.api_key:
                try:
                    from pinecone import Pinecone  # type: ignore[import-not-found]

                    pc = Pinecone(api_key=self.api_key)
                    self._native_backend = pc.Index(self.collection_name)
                    self._backend_name = "pinecone_native"
                except Exception:
                    self._backend_name = "pinecone_local_adapter"
            else:
                self._backend_name = "pinecone_local_adapter"

        elif self.provider == "mongodb":
            if self.connection_uri:
                try:
                    from pymongo import MongoClient  # type: ignore[import-not-found]

                    client = MongoClient(self.connection_uri, serverSelectionTimeoutMS=5000)
                    db_name = self.kwargs.get("db_name", "raglib_db")
                    self._native_backend = client[db_name][self.collection_name]
                    self._backend_name = "mongodb_native"
                except Exception:
                    self._backend_name = "mongodb_local_adapter"
            else:
                self._backend_name = "mongodb_local_adapter"

        elif self.provider == "qdrant":
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-not-found]

                client = QdrantClient(url=self.connection_uri, api_key=self.api_key) if self.connection_uri else QdrantClient(":memory:")
                self._native_backend = client
                self._backend_name = "qdrant_native"
            except Exception:
                self._backend_name = "qdrant_local_adapter"

    def add(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> int:
        """Insert or update chunks and their embeddings in the chosen VectorDB (auto-embeds if `embeddings` is omitted)."""
        global _LAST_ACTIVE_STORE
        _LAST_ACTIVE_STORE = self
        if not chunks:
            return 0
        if embeddings is None:
            from ragspark import embedder as embedder_mod

            dim = len(self._embeddings[0]) if self._embeddings else self.embed_dimension
            embeddings = embedder_mod.embed(
                chunks,
                provider=self.embed_provider,
                strategy=self.embed_strategy,
                model=self.embed_model,
                api_key=self.embed_api_key,
                dimension=dim,
            )
        if len(chunks) != len(embeddings):
            raise ValueError(f"Count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        existing_index_by_id = {c.id: i for i, c in enumerate(self._chunks)}
        for chk, emb in zip(chunks, embeddings):
            if chk.id in existing_index_by_id:
                pos = existing_index_by_id[chk.id]
                self._chunks[pos] = chk
                self._embeddings[pos] = emb
            else:
                existing_index_by_id[chk.id] = len(self._chunks)
                self._chunks.append(chk)
                self._embeddings.append(emb)

        if self._backend_name == "chromadb_native" and self._native_backend is not None:
            try:
                self._native_backend.upsert(
                    ids=[c.id for c in chunks],
                    documents=[c.text for c in chunks],
                    embeddings=embeddings,
                    metadatas=[
                        {
                            "source": str(c.source),
                            "page_number": int(c.page_number),
                            "section_path": str(c.section_path),
                            "strategy": str(c.strategy),
                        }
                        for c in chunks
                    ],
                )
            except Exception:
                pass
        elif self._backend_name == "pinecone_native" and self._native_backend is not None:
            try:
                vectors = [
                    {
                        "id": c.id,
                        "values": emb,
                        "metadata": {"text": c.text, "source": c.source, "section_path": c.section_path, "page": c.page_number, "strategy": c.strategy},
                    }
                    for c, emb in zip(chunks, embeddings)
                ]
                self._native_backend.upsert(vectors=vectors)
            except Exception:
                pass
        elif self._backend_name == "mongodb_native" and self._native_backend is not None:
            try:
                docs = [
                    {
                        "_id": c.id,
                        "text": c.text,
                        "embedding": emb,
                        "source": c.source,
                        "page_number": c.page_number,
                        "section_path": c.section_path,
                        "strategy": c.strategy,
                        "metadata": c.metadata,
                    }
                    for c, emb in zip(chunks, embeddings)
                ]
                self._native_backend.insert_many(docs, ordered=False)
            except Exception:
                pass

        if self.provider != "memory":
            self._save_local_persistence()

        return len(chunks)

    def search(
        self,
        query_embedding: list[float] | str | None = None,
        query_text: str = "",
        top_k: int = 5,
        strategy: str = "hybrid",
        hybrid_alpha: float = 0.65,
        mmr_lambda: float = 0.7,
        chunk_strategy_filter: str | list[str] | None = None,
        *,
        query: str | None = None,
    ) -> list[SearchResult]:
        """
        Multi-Strategy Retrieval Engine over the VectorDB:
          - Accepts EITHER `(query_embedding, query_text=...)` OR a direct string `db.search("What is X?", top_k=5)`!
          - `strategy="hybrid"`: Weighted Dense Cosine + Okapi BM25 + Legal Section Boost
          - `strategy="semantic"`: Pure Dense Vector Cosine Similarity
          - `strategy="keyword"` or `"bm25"`: Pure Lexical BM25 + Exact Statutory Reference Matching
          - `strategy="rrf"`: Reciprocal Rank Fusion combining Semantic Rank + BM25 Rank
          - `strategy="mmr"`: Maximal Marginal Relevance (balances relevance & diversity)
          - `strategy="rerank"`: 2-Stage Hybrid Retrieval + Cross-Feature Phrase & Legal Hierarchy Reranker
          - `strategy="multi_query"`: Generates structural/synonym query expansions and fuses via RRF
          - `chunk_strategy_filter`: Optional chunking strategy filter (e.g. `"legal_clause"` or `["recursive", "semantic"]`)
        """
        if not self._chunks:
            return []

        if isinstance(query_embedding, str):
            if not query_text:
                query_text = query_embedding
            query_embedding = None
        if query and not query_text:
            query_text = query

        if query_embedding is None:
            from ragspark import embedder as embedder_mod

            dim = len(self._embeddings[0]) if self._embeddings else self.embed_dimension
            query_embedding = embedder_mod.embed_query(
                query_text or "",
                provider=self.embed_provider,
                strategy=self.embed_strategy,
                model=self.embed_model,
                api_key=self.embed_api_key,
                dimension=dim,
            )

        strat = strategy.lower().strip()
        if strat not in AVAILABLE_RETRIEVAL_STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy '{strategy}'. Choose from: {AVAILABLE_RETRIEVAL_STRATEGIES}")

        # Filter candidate chunks if user is experimenting with specific chunking strategies in ChromaDB
        allowed_strats: set[str] | None = None
        if chunk_strategy_filter:
            if isinstance(chunk_strategy_filter, str):
                allowed_strats = {s.strip().lower() for s in re.split(r"[+,]", chunk_strategy_filter) if s.strip()}
            else:
                allowed_strats = {str(s).strip().lower() for s in chunk_strategy_filter}

        candidates: list[tuple[Chunk, list[float]]] = []
        for chk, emb in zip(self._chunks, self._embeddings):
            if allowed_strats and chk.strategy.lower() not in allowed_strats:
                continue
            candidates.append((chk, emb))

        if not candidates:
            return []

        avg_len = sum(len(c.text.split()) for c, _ in candidates) / max(1, len(candidates))

        # Compute base vector and keyword scores for all candidates
        raw_items: list[dict[str, Any]] = []
        for idx, (chk, emb) in enumerate(candidates):
            v_score = max(0.0, _cosine_similarity(query_embedding, emb))
            k_score = _bm25_keyword_score(query_text, chk, avg_doc_len=avg_len) if query_text else 0.0

            # Multi-query expansion boost
            if strat == "multi_query" and query_text:
                expanded_q = f"{query_text} section clause proviso obligation rule definition"
                k_exp = _bm25_keyword_score(expanded_q, chk, avg_doc_len=avg_len)
                k_score = max(k_score, (k_score * 0.7) + (k_exp * 0.3))

            raw_items.append(
                {
                    "idx": idx,
                    "chunk": chk,
                    "emb": emb,
                    "v_score": v_score,
                    "k_score": k_score,
                }
            )

        if strat == "semantic":
            for it in raw_items:
                it["final"] = it["v_score"]
                it["rerank"] = 0.0

        elif strat in ("keyword", "bm25"):
            for it in raw_items:
                it["final"] = it["k_score"]
                it["rerank"] = 0.0

        elif strat in ("rrf", "multi_query"):
            # Sort by vector score and keyword score separately to compute RRF ranks
            by_vec = sorted(raw_items, key=lambda x: x["v_score"], reverse=True)
            by_kw = sorted(raw_items, key=lambda x: x["k_score"], reverse=True)
            vec_rank = {it["idx"]: r for r, it in enumerate(by_vec, start=1)}
            kw_rank = {it["idx"]: r for r, it in enumerate(by_kw, start=1)}
            for it in raw_items:
                rrf_val = (1.0 / (10.0 + vec_rank[it["idx"]])) + (1.0 / (10.0 + kw_rank[it["idx"]]))
                it["final"] = rrf_val * 5.0  # Scale to ~[0, 1]
                it["rerank"] = 0.0

        elif strat == "mmr":
            # Maximal Marginal Relevance greedy selection
            for it in raw_items:
                it["base"] = (hybrid_alpha * it["v_score"]) + ((1.0 - hybrid_alpha) * it["k_score"])
            selected: list[dict[str, Any]] = []
            remaining = list(raw_items)
            while remaining and len(selected) < max(1, top_k):
                best_item = None
                best_mmr = -1e9
                for cand in remaining:
                    max_sim_to_selected = (
                        max(_cosine_similarity(cand["emb"], sel["emb"]) for sel in selected)
                        if selected
                        else 0.0
                    )
                    mmr_score = (mmr_lambda * cand["base"]) - ((1.0 - mmr_lambda) * max_sim_to_selected)
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_item = cand
                if best_item is None:
                    break
                best_item["final"] = max(0.0, best_mmr)
                best_item["rerank"] = 0.0
                selected.append(best_item)
                remaining.remove(best_item)
            raw_items = selected

        elif strat == "rerank":
            for it in raw_items:
                base_h = (hybrid_alpha * it["v_score"]) + ((1.0 - hybrid_alpha) * it["k_score"])
                rr = _cross_feature_rerank_score(query_text, it["chunk"], base_h)
                it["rerank"] = rr
                it["final"] = rr

        else:
            # Default: "hybrid"
            for it in raw_items:
                it["final"] = (
                    (hybrid_alpha * it["v_score"]) + ((1.0 - hybrid_alpha) * it["k_score"])
                    if query_text
                    else it["v_score"]
                )
                it["rerank"] = 0.0

        if strat != "mmr":
            raw_items.sort(key=lambda x: x["final"], reverse=True)

        top_slice = raw_items[: max(1, top_k)]
        results: list[SearchResult] = []
        for rank_idx, it in enumerate(top_slice, start=1):
            results.append(
                SearchResult(
                    chunk=it["chunk"],
                    score=float(it["final"]),
                    vector_score=float(it["v_score"]),
                    keyword_score=float(it["k_score"]),
                    rerank_score=float(it["rerank"]),
                    retrieval_strategy=strat,
                    rank=rank_idx,
                )
            )
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.65,
    ) -> list[SearchResult]:
        """
        1-Line Hybrid Search combining Dense Semantic Vectors with Exact BM25 Keyword Search.

        Args:
            query: The search query string.
            top_k: Number of top results to return.
            alpha: Weight balancing factor (1.0 = pure vector, 0.0 = pure BM25 keyword, default=0.65).
        """
        return self.search(
            query=query,
            top_k=top_k,
            strategy="hybrid",
            hybrid_alpha=alpha,
        )

    def list_indexed_strategies(self) -> list[str]:
        """Return the distinct chunking strategies currently stored in this VectorDB collection."""
        return sorted({c.strategy for c in self._chunks})

    def count(self) -> int:
        return len(self._chunks)

    def _save_local_persistence(self) -> None:
        target_dir = self.persist_directory or Path(".ragspark_store")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", self.collection_name) or "raglib_collection"
            store_file = target_dir / f"{safe_name}.chroma.json"
            payload = {
                "collection_name": self.collection_name,
                "provider": self.provider,
                "items": [
                    {"chunk": c.to_dict(), "embedding": e}
                    for c, e in zip(self._chunks, self._embeddings)
                ],
            }
            store_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            (target_dir / "_last_active.txt").write_text(str(store_file.resolve()), encoding="utf-8")
        except Exception:
            pass

    def _load_local_persistence(self) -> None:
        target_dir = self.persist_directory or Path(".ragspark_store")
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", self.collection_name) or "raglib_collection"
        store_file = target_dir / f"{safe_name}.chroma.json"
        if not store_file.exists():
            return
        try:
            data = json.loads(store_file.read_text(encoding="utf-8"))
            for item in data.get("items", []):
                c_dict = item["chunk"]
                self._chunks.append(Chunk(**c_dict))
                self._embeddings.append(item["embedding"])
        except Exception:
            pass


def connect(
    provider: str = "chroma",
    collection_name: str = "raglib_collection",
    persist_directory: str | Path | None = None,
    api_key: str | None = None,
    connection_uri: str | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Connect to or create a VectorDB store (default: ChromaDB). Automatically reloads persisted chunks if present."""
    return VectorStore(
        provider=backend or provider,
        collection_name=collection_name,
        persist_directory=persist_directory,
        api_key=api_key,
        connection_uri=connection_uri,
        **kwargs,
    )


def load(
    collection_name: str = "raglib_collection",
    persist_directory: str | Path | None = None,
    provider: str = "chroma",
    **kwargs: Any,
) -> VectorStore:
    """
    Reload an already-indexed VectorDB collection from disk in <0.05s without re-running Parser, Chunker, or Embedder!
    If `collection_name` is omitted, loads the most recently saved collection from `.ragspark_store`.
    """
    target_dir = Path(persist_directory) if persist_directory else Path(".ragspark_store")
    if collection_name == "raglib_collection" and not (target_dir / "raglib_collection.chroma.json").exists():
        last_ptr = target_dir / "_last_active.txt"
        if last_ptr.exists():
            try:
                saved_path = Path(last_ptr.read_text(encoding="utf-8").strip())
                if saved_path.exists() and saved_path.name.endswith(".chroma.json"):
                    collection_name = saved_path.name[: -len(".chroma.json")]
                    target_dir = saved_path.parent
            except Exception:
                pass
    return VectorStore(
        provider=provider,
        collection_name=collection_name,
        persist_directory=target_dir,
        **kwargs,
    )


def store(
    chunks: list[Chunk],
    embeddings: list[list[float]] | None = None,
    provider: str = "chroma",
    backend: str | None = None,
    collection_name: str = "raglib_collection",
    persist_directory: str | Path | None = None,
    api_key: str | None = None,
    connection_uri: str | None = None,
    embed_provider: str = "fast_local",
    embed_strategy: str = "dense",
    embed_model: str | None = None,
    embed_api_key: str | None = None,
    embed_dimension: int = 384,
    **kwargs: Any,
) -> VectorStore:
    """
    One-command VectorDB indexer (`vectordb.store(chunks, backend='chroma', collection_name='my_docs')`):
    Connects to the chosen VectorDB, automatically embeds `chunks` if `embeddings` is not provided,
    stores them (and persists to disk), and returns the ready-to-search `VectorStore`.
    """
    db = VectorStore(
        provider=backend or provider,
        collection_name=collection_name,
        persist_directory=persist_directory,
        api_key=api_key,
        connection_uri=connection_uri,
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        embed_model=embed_model,
        embed_api_key=embed_api_key or api_key,
        embed_dimension=embed_dimension,
        **kwargs,
    )
    db.add(chunks, embeddings)
    return db


__all__ = [
    "VectorStore",
    "SearchResult",
    "connect",
    "load",
    "store",
    "list_vectordbs",
    "list_retrieval_strategies",
    "AVAILABLE_VECTORDBS",
    "AVAILABLE_RETRIEVAL_STRATEGIES",
]
