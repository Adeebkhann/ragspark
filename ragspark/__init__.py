"""
`ragspark` — Unified One-Command RAG Pipeline Library (General + Specialized Law RAG + Chatbot Studio UI).

Supports both `ragspark` and `raglib` imports seamlessly:
    from ragspark import parser, chunker, embedder, vectordb, prompt_eng, llm, rag, ui
"""

from __future__ import annotations

import sys

import raglib
from raglib import (
    BuiltPrompt,
    Chunk,
    ParsedDocument,
    RAGPipeline,
    RAGResponse,
    SearchResult,
    VectorStore,
    __all__,
    __version__,
    ask,
    ask_law,
    build_law_rag,
    build_rag,
    chunk,
    chunk_combination,
    chunk_law,
    chunker,
    embed,
    embed_query,
    embedder,
    launch_ui,
    llm,
    notebook_chat,
    parse,
    parse_law,
    parse_text,
    parse_unstructured,
    parser,
    prompt_eng,
    rag,
    store,
    ui,
    vectordb,
)

# Register submodule aliases so `import ragspark.parser`, `from ragspark.rag import ...` work natively
for _sub in ("parser", "chunker", "embedder", "vectordb", "prompt_eng", "llm", "rag", "ui", "cli"):
    _mod = getattr(raglib, _sub, None)
    if _mod is not None:
        sys.modules[f"ragspark.{_sub}"] = _mod

