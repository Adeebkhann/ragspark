"""
`ragspark.embedder` — Multi-Provider Embedding Engine with a never-crash fallback ladder.

Default provider `"auto"` walks this ladder and uses the first rung that works:
  1. `"onnx"`   — local quantized ONNX model via `fastembed` (BAAI/bge-small-en-v1.5), if installed
                  (`pip install ragspark[onnx]`). Runs on CPU, no API key, no data leaves the machine.
  2. Cloud API  — `"gemini"` (GEMINI_API_KEY / GOOGLE_API_KEY) then `"openai"` (OPENAI_API_KEY).
                  Groq does not offer an embeddings endpoint, so it is not a rung.
  3. `"lexical"` — no vectors at all; the vector store switches to pure BM25 keyword retrieval.
Set `RAGSPARK_EMBED_PROVIDER` to pin a rung (e.g. `lexical` for offline tests).

Other providers: `"ollama"`, `"sentence_transformers"` / `"huggingface"`, custom functions via
`embedder.register_custom("name", fn)`.

`"fast_local"` is feature hashing, kept only for backward compatibility. It is NOT a semantic
embedding model and is never chosen automatically.

Embedding strategies (`strategy=`): "dense", "hybrid_sparse_dense", "legal_domain_boost", "context_enriched".
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import urllib.request
from typing import Any, Callable

from ragspark.chunker import Chunk

_log = logging.getLogger("ragspark.embedder")

AVAILABLE_PROVIDERS = (
    "auto",
    "onnx",
    "lexical",
    "fast_local",
    "ollama",
    "gemini",
    "openai",
    "sentence_transformers",
    "huggingface",
)

AVAILABLE_EMBED_STRATEGIES = (
    "dense",
    "hybrid_sparse_dense",
    "legal_domain_boost",
    "context_enriched",
)

CUSTOM_EMBEDDERS: dict[str, Callable[[list[str], int], list[list[float]]]] = {}

DEFAULT_ONNX_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def auto_chain(api_key: str | None = None) -> list[str]:
    """Return the providers the `"auto"` ladder will try, in order (always ends with `"lexical"`)."""
    pinned = os.environ.get("RAGSPARK_EMBED_PROVIDER", "").strip().lower()
    if pinned and pinned != "auto":
        return [pinned] if pinned == "lexical" else [pinned, "lexical"]
    chain: list[str] = []
    if _has_module("fastembed"):
        chain.append("onnx")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or (api_key or "").startswith("AIza"):
        chain.append("gemini")
    if os.environ.get("OPENAI_API_KEY") or (api_key or "").startswith("sk-"):
        chain.append("openai")
    chain.append("lexical")
    return chain


def list_providers() -> list[str]:
    """Return all supported embedding providers."""
    return list(AVAILABLE_PROVIDERS) + list(CUSTOM_EMBEDDERS.keys())


def list_strategies() -> list[str]:
    """Return all supported embedding strategies."""
    return list(AVAILABLE_EMBED_STRATEGIES)


def register_custom(name: str, fn: Callable[[list[str], int], list[list[float]]]) -> None:
    """Register a custom embedding function `fn(texts: list[str], dimension: int) -> list[list[float]]`."""
    CUSTOM_EMBEDDERS[name.lower().strip()] = fn


def _fast_local_embed_one(text: str, dim: int = 384, strategy: str = "dense") -> list[float]:
    """
    Laptop-safe deterministic feature-hashing embedder supporting multiple strategies:
      - "dense": balanced unigram + bigram + 4-gram semantic projection
      - "hybrid_sparse_dense": 70% dense semantic + 30% exact keyword/number hash slots
      - "legal_domain_boost": 2.5x boost on statutory sections, articles, clauses, and obligations
      - "context_enriched": emphasizes structural prefixes
    """
    vec = [0.0] * dim
    words = re.findall(r"[a-zA-Z0-9§]+", text.lower())
    if not words:
        return vec

    legal_terms = {
        "section", "sec", "article", "clause", "part", "chapter",
        "shall", "must", "provided", "proviso", "notwithstanding",
        "explanation", "act", "court", "appeal", "petition", "tribunal",
    }

    features: list[tuple[str, float]] = []
    for i, w in enumerate(words):
        if strategy == "legal_domain_boost":
            weight = 2.6 if (w.isdigit() or w in legal_terms) else 1.0
        elif strategy == "hybrid_sparse_dense":
            weight = 2.2 if (w.isdigit() or len(w) >= 7) else 1.0
        else:
            weight = 1.6 if (w.isdigit() or w in legal_terms) else 1.0

        features.append((f"u:{w}", weight))
        if i + 1 < len(words):
            features.append((f"b:{w}_{words[i + 1]}", 1.25))
        if len(w) >= 5 and strategy != "hybrid_sparse_dense":
            features.append((f"c4:{w[:4]}", 0.5))

    for feat, weight in features:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [round(v / norm, 6) for v in vec]
    return vec


class SmartEmbedder:
    """Multi-backend, multi-strategy embedding engine."""

    def __init__(
        self,
        provider: str = "fast_local",
        strategy: str = "dense",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int = 384,
    ) -> None:
        self.provider = provider.lower().strip()
        self.strategy = strategy.lower().strip()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dimension = dimension
        self._st_model: Any = None
        self._st_model_name: str | None = None
        self._onnx_model: Any = None
        self._onnx_model_name: str | None = None
        self.last_provider: str | None = None

    def embed(
        self,
        items: list[Chunk] | list[str] | str,
        provider: str | None = None,
        strategy: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
        is_query: bool = False,
    ) -> list[list[float]]:
        """Embed a list of `Chunk` objects, a list of strings, or a single string using the chosen provider & strategy."""
        prov = (provider or self.provider).lower().strip()
        strat = (strategy or self.strategy).lower().strip()
        mdl = model or self.model
        key = api_key or self.api_key
        url = base_url or self.base_url
        dim = dimension or self.dimension

        # If provider matches one of the strategy names (e.g. user passed provider="legal_domain_boost"), handle gracefully
        if prov in AVAILABLE_EMBED_STRATEGIES and prov not in AVAILABLE_PROVIDERS:
            strat = prov
            prov = "fast_local"

        if isinstance(items, str):
            texts = [items]
        elif items and isinstance(items[0], Chunk):
            if strat == "context_enriched":
                texts = [
                    f"[Source: {c.source} | Page: {c.page_number} | Section: {c.section_path or 'General'} | Strategy: {c.strategy}]\n{c.text}"  # type: ignore[union-attr]
                    for c in items
                ]
            else:
                texts = [c.text for c in items]  # type: ignore[union-attr]
        else:
            texts = [str(x) for x in items]

        if not texts:
            return []

        if prov == "auto":
            last_exc: Exception | None = None
            for rung in auto_chain(key):
                try:
                    vectors = self._embed_texts(texts, rung, strat, mdl, key, url, dim, is_query)
                    self.last_provider = rung
                    return vectors
                except Exception as exc:  # never crash: fall to the next rung
                    last_exc = exc
                    _log.warning("ragspark embedder: provider %r failed (%s); trying next fallback", rung, type(exc).__name__)
            self.last_provider = "lexical"
            _log.warning("ragspark embedder: all providers failed (%s); using lexical BM25", type(last_exc).__name__)
            return [[] for _ in texts]

        self.last_provider = prov
        return self._embed_texts(texts, prov, strat, mdl, key, url, dim, is_query)

    def _embed_texts(
        self,
        texts: list[str],
        prov: str,
        strat: str,
        mdl: str | None,
        key: str | None,
        url: str | None,
        dim: int,
        is_query: bool = False,
    ) -> list[list[float]]:
        if prov == "lexical":
            return [[] for _ in texts]
        if prov in CUSTOM_EMBEDDERS:
            return CUSTOM_EMBEDDERS[prov](texts, dim)
        if prov == "onnx":
            return self._embed_onnx(texts, model=mdl or DEFAULT_ONNX_MODEL, is_query=is_query)
        if prov == "fast_local":
            return [_fast_local_embed_one(t, dim=dim, strategy=strat) for t in texts]
        if prov == "ollama":
            return self._embed_ollama(texts, model=mdl or "nomic-embed-text", base_url=url or "http://localhost:11434")
        if prov == "gemini":
            return self._embed_gemini(texts, model=mdl or DEFAULT_GEMINI_EMBED_MODEL, api_key=key, is_query=is_query)
        if prov == "openai":
            return self._embed_openai(texts, model=mdl or DEFAULT_OPENAI_EMBED_MODEL, api_key=key, base_url=url)
        if prov in ("sentence_transformers", "huggingface"):
            return self._embed_sentence_transformers(texts, model=mdl or "all-MiniLM-L6-v2")

        raise ValueError(f"Unsupported embedding provider '{prov}'. Choose from: {list_providers()}")

    def embed_query(
        self,
        query: str,
        provider: str | None = None,
        strategy: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
    ) -> list[float]:
        """Embed a single search query string."""
        res = self.embed(
            [query],
            provider=provider,
            strategy=strategy,
            model=model,
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
            is_query=True,
        )
        return res[0] if res else []

    def _embed_onnx(self, texts: list[str], model: str, is_query: bool = False) -> list[list[float]]:
        """Quantized ONNX embeddings on CPU via `fastembed` (`pip install ragspark[onnx]`)."""
        from fastembed import TextEmbedding  # type: ignore[import-not-found]

        if self._onnx_model is None or self._onnx_model_name != model:
            self._onnx_model = TextEmbedding(model_name=model)
            self._onnx_model_name = model
        fn = getattr(self._onnx_model, "query_embed", None) if is_query else None
        vectors = (fn or self._onnx_model.embed)(texts)
        return [[float(x) for x in v] for v in vectors]

    def _embed_ollama(self, texts: list[str], model: str, base_url: str) -> list[list[float]]:
        endpoint = f"{base_url.rstrip('/')}/api/embed"
        payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "embeddings" in data:
                    return data["embeddings"]
        except Exception:
            vectors: list[list[float]] = []
            for t in texts:
                ep_legacy = f"{base_url.rstrip('/')}/api/embeddings"
                p_leg = json.dumps({"model": model, "prompt": t}).encode("utf-8")
                req_leg = urllib.request.Request(ep_legacy, data=p_leg, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req_leg, timeout=30) as resp:
                    d_leg = json.loads(resp.read().decode("utf-8"))
                    vectors.append(d_leg["embedding"])
            return vectors
        raise RuntimeError(f"Failed to retrieve embeddings from Ollama at {base_url}")

    def _embed_gemini(self, texts: list[str], model: str, api_key: str | None, is_query: bool = False) -> list[list[float]]:
        key = (api_key if (api_key or "").startswith("AIza") else None) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Gemini API key required. Pass `api_key='...'` or set GEMINI_API_KEY environment variable.")
        model_path = model if model.startswith("models/") else f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:batchEmbedContents"
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        all_vectors: list[list[float]] = []
        # Google Gemini batchEmbedContents allows at most 100 texts per batch request
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            requests_list = []
            for t in batch_texts:
                item: dict[str, Any] = {"model": model_path, "content": {"parts": [{"text": t}]}, "taskType": task}
                if "gemini-embedding" in model_path:
                    item["outputDimensionality"] = 768
                requests_list.append(item)
            payload = json.dumps({"requests": requests_list}).encode("utf-8")
            # Key goes in a header, never in the URL, so it cannot leak into logs or tracebacks.
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json", "x-goog-api-key": key}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                all_vectors.extend(item["values"] for item in data.get("embeddings", []))
        return all_vectors

    def _embed_openai(self, texts: list[str], model: str, api_key: str | None, base_url: str | None) -> list[list[float]]:
        key = (api_key if (api_key or "").startswith("sk-") else None) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required. Pass `api_key='...'` or set OPENAI_API_KEY environment variable.")
        url = f"{(base_url or 'https://api.openai.com/v1').rstrip('/')}/embeddings"
        payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [item["embedding"] for item in data.get("data", [])]

    def _embed_sentence_transformers(self, texts: list[str], model: str) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers` or use `provider='fast_local'`."
            ) from exc
        if self._st_model is None or self._st_model_name != model:
            self._st_model = SentenceTransformer(model)
            self._st_model_name = model
        encoded = self._st_model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in encoded]


default_embedder = SmartEmbedder()


def embed(
    items: list[Chunk] | list[str] | str,
    provider: str = "auto",
    strategy: str = "dense",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    dimension: int = 384,
) -> list[list[float]]:
    """One-command embedding. `provider="auto"` uses the ONNX -> cloud -> lexical ladder (see module docs)."""
    return default_embedder.embed(
        items=items,
        provider=provider,
        strategy=strategy,
        model=model,
        api_key=api_key,
        base_url=base_url,
        dimension=dimension,
    )


def embed_query(
    query: str,
    provider: str = "auto",
    strategy: str = "dense",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    dimension: int = 384,
) -> list[float]:
    """Embed a single query string."""
    return default_embedder.embed_query(
        query=query,
        provider=provider,
        strategy=strategy,
        model=model,
        api_key=api_key,
        base_url=base_url,
        dimension=dimension,
    )


def embed_with_provider(
    items: list[Chunk] | list[str] | str,
    provider: str = "auto",
    strategy: str = "dense",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    dimension: int = 384,
) -> tuple[list[list[float]], str]:
    """Like `embed`, but also returns the provider that actually produced the vectors."""
    vectors = default_embedder.embed(
        items, provider=provider, strategy=strategy, model=model, api_key=api_key, base_url=base_url, dimension=dimension
    )
    return vectors, default_embedder.last_provider or provider


__all__ = [
    "SmartEmbedder",
    "auto_chain",
    "embed_with_provider",
    "embed",
    "embed_query",
    "register_custom",
    "list_providers",
    "list_strategies",
    "AVAILABLE_PROVIDERS",
    "AVAILABLE_EMBED_STRATEGIES",
]
