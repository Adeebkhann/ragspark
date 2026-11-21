"""
`ragspark.vectordb` — Production Multi-Backend VectorDB & Multi-Strategy Retrieval Engine.

Supported VectorDB Providers (`provider=`):
  - "memory"   (Default: in-memory Hybrid Vector + BM25 store; zero files written)
  - "sqlite"   (Same engine, intended for use with `persist_dir=`)
  - "chroma"   (ChromaDB native when installed, else the local engine)
  - "pinecone" (Pinecone cloud vector DB)
  - "mongodb"  (MongoDB Atlas Vector Search)
  - "qdrant"   (Qdrant local/cloud vector DB)

Storage rules:
  - Nothing touches the disk unless `persist_dir=` is passed. Then rows go to
    `<persist_dir>/ragspark.db` (SQLite, WAL mode), one transaction per `add()`/`save()`.
  - In-memory stores past 10,000 chunks spill embeddings to a SQLite file under
    `tempfile.gettempdir()`, deleted when the store is closed or garbage-collected.

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
import os
import re
import sqlite3
import tempfile
import uuid
import weakref
from array import array
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ragspark.chunker import Chunk

AVAILABLE_VECTORDBS = ("memory", "sqlite", "chroma", "pinecone", "mongodb", "qdrant")

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

SPILL_THRESHOLD = 10_000
"""In-memory stores without `persist_dir` move embeddings to a temp-dir SQLite file past this many chunks."""

_DB_FILENAME = "ragspark.db"
_LEGACY_STORE_DIR = Path(".ragspark_store")


def _pack(vec: list[float]) -> bytes:
    return array("f", vec).tobytes() if vec else b""


def _unpack(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    arr = array("f")
    arr.frombytes(blob)
    return arr.tolist()


class _SQLiteStore:
    """
    Minimal SQLite (WAL) persistence for one collection.

    A connection is opened per operation and closed right after, so no file handle outlives
    a call (Windows can then delete the directory) and every write is one atomic transaction.
    """

    def __init__(self, path: Path, collection: str) -> None:
        self.path = path
        self.collection = collection
        self._schema_ready = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if not self._schema_ready:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS chunks ("
                    " collection TEXT NOT NULL, id TEXT NOT NULL, seq INTEGER NOT NULL,"
                    " data TEXT NOT NULL, embedding BLOB, PRIMARY KEY (collection, id))"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS chunks_seq ON chunks (collection, seq)")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta (collection TEXT NOT NULL, key TEXT NOT NULL,"
                    " value TEXT, PRIMARY KEY (collection, key))"
                )
            self._schema_ready = True
        return conn

    def upsert(self, rows: list[tuple[int, Chunk, list[float]]], meta: dict[str, str] | None = None) -> None:
        """Write all rows (and metadata) in ONE transaction: a single disk commit per call."""
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO chunks (collection, id, seq, data, embedding) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(collection, id) DO UPDATE SET data=excluded.data, embedding=excluded.embedding",
                [
                    (self.collection, c.id, seq, json.dumps(c.to_dict(), ensure_ascii=False), _pack(e))
                    for seq, c, e in rows
                ],
            )
            if meta:
                conn.executemany(
                    "INSERT OR REPLACE INTO meta (collection, key, value) VALUES (?, ?, ?)",
                    [(self.collection, k, v) for k, v in meta.items()],
                )

    def load(self) -> tuple[list[Chunk], list[list[float]], dict[str, str]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT data, embedding FROM chunks WHERE collection=? ORDER BY seq", (self.collection,)
            ).fetchall()
            meta = dict(conn.execute("SELECT key, value FROM meta WHERE collection=?", (self.collection,)).fetchall())
        chunks = [Chunk(**json.loads(d)) for d, _ in rows]
        embs = [_unpack(e) for _, e in rows]
        return chunks, embs, meta

    def iter_embeddings(self, batch: int = 2048) -> Iterator[list[float]]:
        """Stream embeddings in insertion order without loading them all into RAM."""
        with closing(self._connect()) as conn:
            cur = conn.execute("SELECT embedding FROM chunks WHERE collection=? ORDER BY seq", (self.collection,))
            while True:
                rows = cur.fetchmany(batch)
                if not rows:
                    break
                for (blob,) in rows:
                    yield _unpack(blob)

    def destroy(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(f"{self.path}{suffix}").unlink()
            except OSError:
                pass


class VectorStore:
    """
    Unified VectorDB interface: in-memory by default, SQLite WAL when `persist_dir` is given,
    plus native ChromaDB / Pinecone / MongoDB / Qdrant backends when their clients are installed.

    Nothing is written to disk unless `persist_dir` (alias `persist_directory`) is passed. A
    memory store that grows past 10,000 chunks moves its embeddings to a SQLite file under
    `tempfile.gettempdir()`, which is deleted when the store is closed or garbage-collected.
    """

    def __init__(
        self,
        provider: str = "memory",
        collection_name: str = "ragspark_collection",
        persist_directory: str | Path | None = None,
        api_key: str | None = None,
        connection_uri: str | None = None,
        backend: str | None = None,
        embed_provider: str = "auto",
        embed_strategy: str = "dense",
        embed_model: str | None = None,
        embed_api_key: str | None = None,
        embed_dimension: int = 384,
        persist_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        global _LAST_ACTIVE_STORE
        raw_prov = backend or provider or "memory"
        self.provider = raw_prov.lower().strip()
        if self.provider in ("chromadb", "chroma_db"):
            self.provider = "chroma"
        if self.provider not in AVAILABLE_VECTORDBS:
            raise ValueError(f"Unsupported vectordb '{raw_prov}'. Choose from: {AVAILABLE_VECTORDBS}")

        target = persist_dir if persist_dir is not None else persist_directory
        self.collection_name = collection_name
        self.persist_directory = Path(target) if target else None
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
        self._sqlite: _SQLiteStore | None = None
        self._spill: _SQLiteStore | None = None
        self._pending: list[tuple[int, Chunk, list[float]]] = []
        self._seq = 0

        self._init_backend()
        _LAST_ACTIVE_STORE = self

    @property
    def persist_dir(self) -> Path | None:
        return self.persist_directory

    @property
    def is_spilled(self) -> bool:
        """True once embeddings have moved to the temp-dir SQLite spill file."""
        return self._spill is not None

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
                self._load_from_chroma()
            except Exception:
                self._backend_name = "chroma_local_engine"

        elif self.provider == "pinecone":
            self._backend_name = "pinecone_local_adapter"
            if self.api_key:
                try:
                    from pinecone import Pinecone  # type: ignore[import-not-found]

                    self._native_backend = Pinecone(api_key=self.api_key).Index(self.collection_name)
                    self._backend_name = "pinecone_native"
                except Exception:
                    pass

        elif self.provider == "mongodb":
            self._backend_name = "mongodb_local_adapter"
            if self.connection_uri:
                try:
                    from pymongo import MongoClient  # type: ignore[import-not-found]

                    client = MongoClient(self.connection_uri, serverSelectionTimeoutMS=5000)
                    db_name = self.kwargs.get("db_name", "raglib_db")
                    self._native_backend = client[db_name][self.collection_name]
                    self._backend_name = "mongodb_native"
                except Exception:
                    pass

        elif self.provider == "qdrant":
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-not-found]

                self._native_backend = (
                    QdrantClient(url=self.connection_uri, api_key=self.api_key)
                    if self.connection_uri
                    else QdrantClient(":memory:")
                )
                self._backend_name = "qdrant_native"
            except Exception:
                self._backend_name = "qdrant_local_adapter"

        # Local persistence: only when the caller asked for it, and only for backends
        # that do not persist on their own.
        if self.persist_directory and self._backend_name != "chromadb_native":
            self._sqlite = _SQLiteStore(self.persist_directory / _DB_FILENAME, self.collection_name)
            chunks, embs, meta = self._sqlite.load()
            if chunks:
                self._chunks, self._embeddings = chunks, embs
                self._seq = len(chunks)
                self._adopt_meta(meta)
            else:
                self._import_legacy_json(self.persist_directory)

    def _adopt_meta(self, meta: dict[str, str]) -> None:
        """Reuse the embedder that built a persisted collection so query vectors match."""
        if self.embed_provider == "auto" and meta.get("embed_provider"):
            self.embed_provider = meta["embed_provider"]
        if not self.embed_model and meta.get("embed_model"):
            self.embed_model = meta["embed_model"]
        if meta.get("embed_strategy"):
            self.embed_strategy = meta["embed_strategy"]

    def _load_from_chroma(self) -> None:
        try:
            existing = self._native_backend.get(include=["documents", "embeddings", "metadatas"])
        except Exception:
            return
        ids = list(existing.get("ids") or [])
        docs = list(existing.get("documents") or [])
        embs = existing.get("embeddings")
        embs = list(embs) if embs is not None else []
        metas = list(existing.get("metadatas") or [])
        for i, (cid, doc_txt, emb_vec) in enumerate(zip(ids, docs, embs)):
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            self._chunks.append(
                Chunk(
                    id=str(cid),
                    text=str(doc_txt or ""),
                    index=i + 1,
                    source=str(meta.get("source", "chroma")),
                    page_number=int(meta.get("page_number", 1)),
                    section_path=str(meta.get("section_path", "")),
                    strategy=str(meta.get("strategy", "recursive")),
                    metadata=dict(meta),
                )
            )
            self._embeddings.append([float(x) for x in emb_vec])
        self._seq = len(self._chunks)

    def _import_legacy_json(self, directory: Path) -> None:
        """Read (never write) a pre-0.4 `<collection>.chroma.json` file, if one exists."""
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", self.collection_name) or "raglib_collection"
        legacy = directory / f"{safe_name}.chroma.json"
        if not legacy.exists():
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            for item in data.get("items", []):
                self._chunks.append(Chunk(**item["chunk"]))
                self._embeddings.append(item["embedding"])
            self._seq = len(self._chunks)
        except Exception:
            pass

    # ------------------------------------------------------------------ writes

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
        auto_save: bool = True,
    ) -> int:
        """
        Insert or update chunks (auto-embeds them when `embeddings` is omitted).

        With `persist_dir`, all rows are written in a single SQLite transaction. Pass
        `auto_save=False` inside bulk loops and call `save()` once at the end.
        """
        global _LAST_ACTIVE_STORE
        _LAST_ACTIVE_STORE = self
        if not chunks:
            return 0
        if embeddings is None:
            from ragspark import embedder as embedder_mod

            embeddings, used = embedder_mod.embed_with_provider(
                chunks,
                provider=self.embed_provider,
                strategy=self.embed_strategy,
                model=self.embed_model,
                api_key=self.embed_api_key,
                dimension=self._dimension(),
            )
            self.embed_provider = used  # pin it so query vectors come from the same model
        if len(chunks) != len(embeddings):
            raise ValueError(f"Count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        existing_index_by_id = {c.id: i for i, c in enumerate(self._chunks)}
        new_rows: list[tuple[int, Chunk, list[float]]] = []
        for chk, emb in zip(chunks, embeddings):
            emb = list(emb) if emb is not None else []
            if chk.id in existing_index_by_id:
                pos = existing_index_by_id[chk.id]
                self._chunks[pos] = chk
                if self._spill is None:
                    self._embeddings[pos] = emb
                new_rows.append((pos, chk, emb))
            else:
                pos = self._seq
                self._seq += 1
                existing_index_by_id[chk.id] = len(self._chunks)
                self._chunks.append(chk)
                if self._spill is None:
                    self._embeddings.append(emb)
                new_rows.append((pos, chk, emb))

        if self._spill is not None:
            self._spill.upsert(new_rows)

        self._write_native(chunks, embeddings)

        if self._sqlite is not None:
            self._pending.extend(new_rows)
            if auto_save:
                self.save()
        elif self._spill is None and len(self._chunks) > SPILL_THRESHOLD:
            self._spill_to_tempdir()

        return len(chunks)

    def _write_native(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if self._native_backend is None:
            return
        try:
            if self._backend_name == "chromadb_native" and all(embeddings):
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
            elif self._backend_name == "pinecone_native" and all(embeddings):
                self._native_backend.upsert(
                    vectors=[
                        {
                            "id": c.id,
                            "values": emb,
                            "metadata": {"text": c.text, "source": c.source, "section_path": c.section_path, "page": c.page_number, "strategy": c.strategy},
                        }
                        for c, emb in zip(chunks, embeddings)
                    ]
                )
            elif self._backend_name == "mongodb_native":
                self._native_backend.insert_many(
                    [
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
                    ],
                    ordered=False,
                )
        except Exception:
            pass

    def _spill_to_tempdir(self) -> None:
        """Move embeddings out of RAM into a SQLite file in the OS temp dir (never the CWD)."""
        spill_dir = Path(tempfile.gettempdir()) / "ragspark"
        path = spill_dir / f"spill-{os.getpid()}-{uuid.uuid4().hex[:12]}.db"
        store = _SQLiteStore(path, self.collection_name)
        store.upsert([(i, c, e) for i, (c, e) in enumerate(zip(self._chunks, self._embeddings))])
        self._spill = store
        self._embeddings = []
        self._spill_finalizer = weakref.finalize(self, store.destroy)

    def save(self) -> None:
        """Flush pending writes to `persist_dir` in one transaction (no-op for pure in-memory stores)."""
        if self._sqlite is None or not self._pending:
            return
        self._sqlite.upsert(
            self._pending,
            meta={
                "embed_provider": str(self.embed_provider),
                "embed_model": str(self.embed_model or ""),
                "embed_strategy": str(self.embed_strategy),
            },
        )
        self._pending = []

    def close(self) -> None:
        """Flush pending writes and delete any temp-dir spill file."""
        self.save()
        if self._spill is not None:
            self._spill_finalizer()
            self._spill = None

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    # ------------------------------------------------------------------ reads

    def _dimension(self) -> int:
        if self._embeddings and self._embeddings[0]:
            return len(self._embeddings[0])
        return self.embed_dimension

    def _iter_embeddings(self) -> Iterator[list[float]]:
        if self._spill is not None:
            return self._spill.iter_embeddings()
        return iter(self._embeddings)

    def _has_vectors(self) -> bool:
        if self._spill is not None:
            return next(self._spill.iter_embeddings(batch=1), []) != []
        return bool(self._embeddings) and bool(self._embeddings[0])

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

        When the collection has no vectors (lexical embedder) or the query cannot be embedded,
        every strategy degrades to BM25 keyword scoring instead of failing.
        """
        if not self._chunks:
            return []

        if isinstance(query_embedding, str):
            if not query_text:
                query_text = query_embedding
            query_embedding = None
        if query and not query_text:
            query_text = query

        strat = strategy.lower().strip()
        if strat not in AVAILABLE_RETRIEVAL_STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy '{strategy}'. Choose from: {AVAILABLE_RETRIEVAL_STRATEGIES}")

        has_vectors = self._has_vectors()
        if query_embedding is None and has_vectors:
            try:
                from ragspark import embedder as embedder_mod

                query_embedding = embedder_mod.embed_query(
                    query_text or "",
                    provider=self.embed_provider,
                    strategy=self.embed_strategy,
                    model=self.embed_model,
                    api_key=self.embed_api_key,
                    dimension=self._dimension(),
                )
            except Exception:
                query_embedding = []
        lexical = not has_vectors or not query_embedding or len(query_embedding) != self._dimension()
        if lexical:
            query_embedding = []
            if strat == "semantic":
                strat = "bm25"

        allowed_strats: set[str] | None = None
        if chunk_strategy_filter:
            if isinstance(chunk_strategy_filter, str):
                allowed_strats = {s.strip().lower() for s in re.split(r"[+,]", chunk_strategy_filter) if s.strip()}
            else:
                allowed_strats = {str(s).strip().lower() for s in chunk_strategy_filter}

        candidates = [
            c for c in self._chunks if not allowed_strats or c.strategy.lower() in allowed_strats
        ]
        if not candidates:
            return []
        avg_len = sum(len(c.text.split()) for c in candidates) / max(1, len(candidates))
        keep_emb = strat == "mmr"

        raw_items: list[dict[str, Any]] = []
        for idx, (chk, emb) in enumerate(zip(self._chunks, self._iter_embeddings())):
            if allowed_strats and chk.strategy.lower() not in allowed_strats:
                continue
            v_score = 0.0 if lexical else max(0.0, _cosine_similarity(query_embedding, emb))  # type: ignore[arg-type]
            k_score = _bm25_keyword_score(query_text, chk, avg_doc_len=avg_len) if query_text else 0.0
            if strat == "multi_query" and query_text:
                expanded_q = f"{query_text} section clause proviso obligation rule definition"
                k_exp = _bm25_keyword_score(expanded_q, chk, avg_doc_len=avg_len)
                k_score = max(k_score, (k_score * 0.7) + (k_exp * 0.3))
            raw_items.append(
                {"idx": idx, "chunk": chk, "emb": emb if keep_emb else None, "v_score": v_score, "k_score": k_score}
            )

        alpha = 0.0 if lexical else hybrid_alpha

        if strat == "semantic":
            for it in raw_items:
                it["final"] = it["v_score"]
                it["rerank"] = 0.0

        elif strat in ("keyword", "bm25"):
            for it in raw_items:
                it["final"] = it["k_score"]
                it["rerank"] = 0.0

        elif strat in ("rrf", "multi_query"):
            by_vec = sorted(raw_items, key=lambda x: x["v_score"], reverse=True)
            by_kw = sorted(raw_items, key=lambda x: x["k_score"], reverse=True)
            vec_rank = {it["idx"]: r for r, it in enumerate(by_vec, start=1)}
            kw_rank = {it["idx"]: r for r, it in enumerate(by_kw, start=1)}
            for it in raw_items:
                vec_part = 0.0 if lexical else 1.0 / (10.0 + vec_rank[it["idx"]])
                rrf_val = vec_part + (1.0 / (10.0 + kw_rank[it["idx"]]))
                it["final"] = rrf_val * (10.0 if lexical else 5.0)  # Scale to ~[0, 1]
                it["rerank"] = 0.0

        elif strat == "mmr":
            for it in raw_items:
                it["base"] = (alpha * it["v_score"]) + ((1.0 - alpha) * it["k_score"])
            # MMR is quadratic: run it over the strongest candidates only.
            pool = sorted(raw_items, key=lambda x: x["base"], reverse=True)[: max(50, top_k * 5)]
            selected: list[dict[str, Any]] = []
            while pool and len(selected) < max(1, top_k):
                best_item = None
                best_mmr = -1e9
                for cand in pool:
                    max_sim = (
                        max(_cosine_similarity(cand["emb"], sel["emb"]) for sel in selected) if selected else 0.0
                    )
                    mmr_score = (mmr_lambda * cand["base"]) - ((1.0 - mmr_lambda) * max_sim)
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_item = cand
                if best_item is None:
                    break
                best_item["final"] = max(0.0, best_mmr)
                best_item["rerank"] = 0.0
                selected.append(best_item)
                pool.remove(best_item)
            raw_items = selected

        elif strat == "rerank":
            for it in raw_items:
                base_h = (alpha * it["v_score"]) + ((1.0 - alpha) * it["k_score"])
                rr = _cross_feature_rerank_score(query_text, it["chunk"], base_h)
                it["rerank"] = rr
                it["final"] = rr

        else:  # "hybrid"
            for it in raw_items:
                it["final"] = (
                    (alpha * it["v_score"]) + ((1.0 - alpha) * it["k_score"]) if query_text else it["v_score"]
                )
                it["rerank"] = 0.0

        if strat != "mmr":
            raw_items.sort(key=lambda x: x["final"], reverse=True)

        results: list[SearchResult] = []
        for rank_idx, it in enumerate(raw_items[: max(1, top_k)], start=1):
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
        return self.search(query=query, top_k=top_k, strategy="hybrid", hybrid_alpha=alpha)

    def list_indexed_strategies(self) -> list[str]:
        """Return the distinct chunking strategies currently stored in this VectorDB collection."""
        return sorted({c.strategy for c in self._chunks})

    def count(self) -> int:
        return len(self._chunks)


def connect(
    provider: str = "memory",
    collection_name: str = "ragspark_collection",
    persist_directory: str | Path | None = None,
    api_key: str | None = None,
    connection_uri: str | None = None,
    backend: str | None = None,
    persist_dir: str | Path | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Connect to or create a VectorDB store (default: in-memory). Reloads persisted chunks when `persist_dir` is given."""
    return VectorStore(
        provider=backend or provider,
        collection_name=collection_name,
        persist_directory=persist_dir if persist_dir is not None else persist_directory,
        api_key=api_key,
        connection_uri=connection_uri,
        **kwargs,
    )


def load(
    collection_name: str = "ragspark_collection",
    persist_directory: str | Path | None = None,
    provider: str = "memory",
    persist_dir: str | Path | None = None,
    **kwargs: Any,
) -> VectorStore:
    """
    Reload an already-indexed collection from `persist_dir` without re-running parsing, chunking or embedding.

    Without `persist_dir`: a pre-0.4 `.ragspark_store` folder in the working directory is read into
    memory (nothing is written there); failing that, the store already active in this session is
    returned (handy after re-running a notebook cell).
    """
    target = persist_dir if persist_dir is not None else persist_directory
    if target is not None:
        return VectorStore(provider=provider, collection_name=collection_name, persist_directory=target, **kwargs)

    if _LEGACY_STORE_DIR.is_dir():
        last_ptr = _LEGACY_STORE_DIR / "_last_active.txt"
        if collection_name == "ragspark_collection" and last_ptr.exists():
            try:
                saved = Path(last_ptr.read_text(encoding="utf-8").strip())
                if saved.name.endswith(".chroma.json"):
                    collection_name = saved.name[: -len(".chroma.json")]
            except Exception:
                pass
        legacy = VectorStore(provider=provider, collection_name=collection_name, **kwargs)
        legacy._import_legacy_json(_LEGACY_STORE_DIR)  # read-only
        if len(legacy):
            return legacy
    if _LAST_ACTIVE_STORE is not None:
        return _LAST_ACTIVE_STORE
    raise ValueError("vectordb.load() needs `persist_dir=` (the folder passed when the collection was built).")


def store(
    chunks: list[Chunk],
    embeddings: list[list[float]] | None = None,
    provider: str = "memory",
    backend: str | None = None,
    collection_name: str = "ragspark_collection",
    persist_directory: str | Path | None = None,
    api_key: str | None = None,
    connection_uri: str | None = None,
    embed_provider: str = "auto",
    embed_strategy: str = "dense",
    embed_model: str | None = None,
    embed_api_key: str | None = None,
    embed_dimension: int = 384,
    persist_dir: str | Path | None = None,
    **kwargs: Any,
) -> VectorStore:
    """
    One-command indexer: `vectordb.store(chunks)` embeds `chunks` (unless `embeddings` is given),
    stores them in memory (or in `persist_dir`), and returns the ready-to-search `VectorStore`.
    """
    db = VectorStore(
        provider=backend or provider,
        collection_name=collection_name,
        persist_directory=persist_dir if persist_dir is not None else persist_directory,
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
    "SPILL_THRESHOLD",
]
