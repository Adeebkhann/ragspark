"""
`raglib` — Production Unified RAG Pipeline Library (General + Specialized Law RAG + Chatbot Studio UI).

Import any pipeline step directly:
    from raglib import parser
    from raglib import chunker
    from raglib import embedder
    from raglib import vectordb
    from raglib import prompt_eng
    from raglib import llm
    from raglib import rag
    from raglib import ui
"""

from __future__ import annotations

from raglib import chunker, embedder, llm, parser, prompt_eng, rag, ui, vectordb
from raglib.chunker import Chunk, chunk, chunk_combination, chunk_law
from raglib.embedder import embed, embed_query
from raglib.parser import ParsedDocument, parse, parse_law, parse_text, parse_unstructured
from raglib.prompt_eng import BuiltPrompt
from raglib.rag import RAGPipeline, RAGResponse, ask, ask_law
from raglib.rag import create as build_rag
from raglib.rag import create_law as build_law_rag
from raglib.ui import launch as launch_ui
from raglib.ui import notebook_chat
from raglib.vectordb import SearchResult, VectorStore, store

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
    "ask_law",
    "ParsedDocument",
    "Chunk",
    "VectorStore",
    "SearchResult",
    "BuiltPrompt",
    "RAGPipeline",
    "RAGResponse",
]
