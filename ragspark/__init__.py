"""
`ragspark` — Production Unified RAG Pipeline Library (General + Specialized Law RAG + Chatbot Studio UI).

Import any pipeline step directly:
    from ragspark import parser
    from ragspark import chunker
    from ragspark import embedder
    from ragspark import vectordb
    from ragspark import prompt_eng
    from ragspark import llm
    from ragspark import rag
    from ragspark import ui
"""

from __future__ import annotations

from ragspark import chunker, embedder, llm, parser, prompt_eng, rag, reranker, ui, vectordb
from ragspark.chunker import (
    Chunk,
    chunk,
    chunk_combination,
    chunk_law,
)
from ragspark.embedder import embed, embed_query
from ragspark.parser import ParsedDocument, parse, parse_law, parse_text, parse_unstructured
from ragspark.prompt_eng import BuiltPrompt, detect_language
from ragspark.rag import RAGPipeline, RAGResponse, ask, ask_law, chat
from ragspark.rag import create as build_rag
from ragspark.rag import create_law as build_law_rag
from ragspark.reranker import Reranker, rerank
from ragspark.ui import launch as launch_ui
from ragspark.ui import notebook_chat
from ragspark.vectordb import SearchResult, VectorStore, store

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "parser",
    "chunker",
    "embedder",
    "vectordb",
    "prompt_eng",
    "llm",
    "rag",
    "ui",
    "reranker",
    "rerank",
    "Reranker",
    "launch_ui",
    "notebook_chat",
    "store",
    "parse",
    "parse_law",
    "parse_unstructured",
    "parse_text",
    "chunk",
    "chunk_combination",
    "chunk_law",
    "embed",
    "embed_query",
    "build_rag",
    "build_law_rag",
    "ask",
    "chat",
    "ask_law",
    "detect_language",
    "ParsedDocument",
    "Chunk",
    "VectorStore",
    "SearchResult",
    "BuiltPrompt",
    "RAGPipeline",
    "RAGResponse",
]
