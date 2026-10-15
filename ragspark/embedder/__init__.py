"""
`ragspark.embedder` — Production Multi-Strategy & Multi-Provider Embedding Engine.

Supports:
  1. Embedding Providers (`provider=`):
     - "fast_local"            (Default: zero-GPU, instant CPU feature-hashing embedder)
     - "ollama"                (Local Ollama embedding runtime: nomic-embed-text, mxbai-embed-large, bge-m3)
     - "gemini"                (Google Gemini Embedding API: text-embedding-004, gemini-embedding-001)
     - "openai"                (OpenAI Embedding API: text-embedding-3-small, text-embedding-3-large)
     - "sentence_transformers" (HuggingFace local models: all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, legal-bert)
  2. Embedding Strategies (`strategy=`):
     - "dense"                 (Standard semantic representation)
     - "hybrid_sparse_dense"   (Fuses dense semantic dims + exact lexical/identifier dims)
     - "legal_domain_boost"    (Amplifies statutory sections, Articles, Clauses, Case Citations, and defined terms)
     - "context_enriched"      (Prepends `[Source | Page | Section Path]` metadata into the text before embedding!)
  3. Custom User Embedding Functions:
     - `embedder.register_custom("my_embedder", fn)`
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from typing import Any, Callable

from ragspark.chunker import Chunk

AVAILABLE_PROVIDERS = (
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

    def embed(
        self,
        items: list[Chunk] | list[str] | str,
        provider: str | None = None,
        strategy: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
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

        if prov in CUSTOM_EMBEDDERS:
            return CUSTOM_EMBEDDERS[prov](texts, dim)
        if prov == "fast_local":
            return [_fast_local_embed_one(t, dim=dim, strategy=strat) for t in texts]
        if prov == "ollama":
            return self._embed_ollama(texts, model=mdl or "nomic-embed-text", base_url=url or "http://localhost:11434")
        if prov == "gemini":
            return self._embed_gemini(texts, model=mdl or "text-embedding-004", api_key=key)
        if prov == "openai":
            return self._embed_openai(texts, model=mdl or "text-embedding-3-small", api_key=key, base_url=url)
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
        )
        return res[0] if res else []

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

    def _embed_gemini(self, texts: list[str], model: str, api_key: str | None) -> list[list[float]]:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Gemini API key required. Pass `api_key='...'` or set GEMINI_API_KEY environment variable.")
        model_path = model if model.startswith("models/") else f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:batchEmbedContents?key={key}"
        all_vectors: list[list[float]] = []
        # Google Gemini batchEmbedContents allows at most 100 texts per batch request
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            requests_list = [
                {"model": model_path, "content": {"parts": [{"text": t}]}}
                for t in batch_texts
            ]
            payload = json.dumps({"requests": requests_list}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                all_vectors.extend(item["values"] for item in data.get("embeddings", []))
        return all_vectors

    def _embed_openai(self, texts: list[str], model: str, api_key: str | None, base_url: str | None) -> list[list[float]]:
        key = api_key or os.environ.get("OPENAI_API_KEY")
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
    provider: str = "fast_local",
    strategy: str = "dense",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    dimension: int = 384,
) -> list[list[float]]:
    """One-command embedding function supporting user-selectable `provider` and `strategy`."""
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
    provider: str = "fast_local",
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


__all__ = [
    "SmartEmbedder",
    "embed",
    "embed_query",
    "register_custom",
    "list_providers",
    "list_strategies",
    "AVAILABLE_PROVIDERS",
    "AVAILABLE_EMBED_STRATEGIES",
]
