import pytest
import ragspark
from ragspark import chunker, parser, vectordb, rag, reranker

SAMPLE_DOC = """RAGSPARK ENTERPRISE PRODUCT MANUAL
Chapter 1: Platform Overview
Section 1.1: Architecture & Capabilities
RagSpark is an enterprise Retrieval-Augmented Generation library built for multi-format document ingestion, hybrid search, and deterministic grounded reasoning.
Section 1.2: Security & Privacy Guarantees
All embeddings and chunks are stored locally in the persistent vector store. Zero proprietary data leaves your environment.
Section 1.3: Performance & Scalability
RagSpark supports multi-threading, dynamic batching, and sub-50ms hybrid retrieval across enterprise-scale repositories.
Section 2: Support and SLA
Enterprise customers receive 24/7 dedicated support with response times under one hour."""


def test_hybrid_search():
    doc = parser.parse_text(SAMPLE_DOC)
    chunks = chunker.chunk(doc, strategy="recursive", chunk_size=200, overlap=30)
    db = vectordb.store(chunks, backend="memory")
    
    # Test hybrid_search method
    results = db.hybrid_search("What are the security and privacy guarantees?", top_k=2, alpha=0.5)
    assert len(results) > 0
    assert any("privacy" in r.chunk.text.lower() for r in results)


def test_flash_reranker():
    doc = parser.parse_text(SAMPLE_DOC)
    chunks = chunker.chunk(doc, strategy="recursive", chunk_size=150, overlap=20)
    db = vectordb.store(chunks, backend="memory")
    
    hits = db.search("dedicated support with response times under one hour", top_k=4)
    assert len(hits) > 0
    
    # Rerank candidates
    reranked = reranker.rerank("dedicated support with response times under one hour", hits, top_k=2)
    assert len(reranked) == 2
    assert "support" in reranked[0].chunk.text.lower()
    assert reranked[0].rerank_score > 0.0


def test_multi_turn_chat_memory():
    bot = rag.create(SAMPLE_DOC, chunk_strategy="recursive", vectordb="memory")
    
    # Turn 1
    res1 = bot.chat("What are the security guarantees?")
    assert len(res1.sources) > 0
    assert len(bot.history) == 2
    assert bot.history[0]["role"] == "user"
    assert bot.history[1]["role"] == "assistant"
    
    # Turn 2: Follow-up question with pronoun
    res2 = bot.chat("And what about its performance?")
    assert len(bot.history) == 4
    
    # Test clear_history
    bot.clear_history()
    assert len(bot.history) == 0


def test_sentence_citations_and_evidence():
    bot = rag.create(SAMPLE_DOC, chunk_strategy="recursive", vectordb="memory")
    res = bot.ask("What is the response time for dedicated support?", top_k=2, rerank=True)
    
    # Check citations
    cites = res.citations
    assert len(cites) > 0
    assert "evidence" in cites[0]
    assert "source" in cites[0]
    assert "page" in cites[0]
    
    # Check evidence list
    assert len(res.evidence) > 0
    
    # Check formatted markdown string
    formatted = res.format_citations()
    assert "Verified Sources & Evidence" in formatted
    assert ">" in formatted


def test_ragspark_imports():
    from ragspark import reranker, rerank, Reranker, chat
    assert reranker is not None
    assert callable(rerank)
    assert callable(chat)

