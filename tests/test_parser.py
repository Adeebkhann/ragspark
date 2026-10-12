"""Comprehensive test suite for `raglib` v0.3.0 (parser, 10 chunkers + combinations, embed strategies, 7 retrievers, UI)."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pymupdf as fitz
import pytest

import raglib
from raglib import chunker, embedder, llm, parser, prompt_eng, rag, ui, vectordb
from raglib.cli import main as cli_main


SAMPLE_LAW_TEXT = """
SUPREME COURT OF INDIA - REPORTABLE
Page 1 of 2

IN THE SUPREME COURT OF INDIA
CIVIL APPEAL NO. 4821 OF 2024

Alpha Technologies Pvt. Ltd. v. Union of India

Dated: 14th October, 2024

PART I - PRELIMINARY AND DEFINITIONS

CHAPTER I - GENERAL PROVISIONS

Section 1. Short title and definitions.—
(1) This Act may be called THE DIGITAL DATA PROTECTION AND ARBITRATION ACT, 2024.
(2) In this Act, unless the context otherwise requires:
(a) "Confidential Information" means any proprietary data, trade secret, or algorithm disclosed by a Data Fiduciary;
(b) "Tribunal" shall mean the Appellate Tribunal established under Section 18 of the Telecom Regulatory Authority of India Act, 1997.

Section 2. Obligations of Data Fiduciary.—
(1) Every Data Fiduciary shall implement reasonable security safeguards to prevent personal data breach.
(a) The Data Fiduciary shall not transfer sensitive personal data outside the territory without prior approval.
Provided that nothing in this sub-section shall restrict cross-border transfers notified under Article 21 of the Constitution of India.
Explanation 1.— For the purposes of this section, as held in Justice K.S. Puttaswamy v. Union of India, (2017) 10 SCC 1 and AIR 1973 SC 1461, privacy is an intrinsic right.
Notwithstanding anything contained in Section 14 of the Arbitration and Conciliation Act, 1996, any dispute shall be resolved within ninety days.
"""

SAMPLE_GENERAL_TEXT = """
# Cloud Architecture Guide

## Microservices Overview
Microservices split backend applications into independent services that communicate over HTTP or gRPC.
They improve deployment velocity and team autonomy across distributed engineering organizations.

## Database Scaling
Horizontal sharding partitions rows across multiple database nodes based on a shard key.
Read replicas offload analytical queries from the primary write node.
"""


def _create_multi_page_law_pdf(pdf_path: Path, num_pages: int = 30) -> Path:
    """Create a 30-page Law PDF to verify sub-second C-level parsing speed."""
    doc = fitz.open()
    for p in range(1, num_pages + 1):
        page = doc.new_page()
        content = (
            f"CONFIDENTIAL COURT RECORD - HEADER\n"
            f"IN THE SUPREME COURT OF INDIA\n"
            f"Section {p}. Statutory Provision {p}.-\n"
            f"(1) Every entity under clause {p} shall maintain compliance records.\n"
            f'(a) "Protected Asset {p}" means any cryptographic key or juris-\n'
            f"dictional ledger maintained under Section 14 of the Arbitration and Conciliation Act, 1996.\n"
            f"Provided that nothing herein shall apply where exempted under Article 21 of the Constitution of India.\n"
            f"See Kesavananda Bharati v. State of Kerala, AIR 1973 SC 1461.\n"
            f"Page {p} of {num_pages}\n"
        )
        page.insert_textbox(fitz.Rect(40, 40, 555, 800), content, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_import_style_and_all_modules() -> None:
    """Verify `from raglib import parser, chunker, embedder, vectordb, prompt_eng, llm, rag, ui` works."""
    assert hasattr(parser, "parse")
    assert hasattr(chunker, "chunk")
    assert hasattr(chunker, "chunk_combination")
    assert hasattr(embedder, "embed")
    assert hasattr(vectordb, "connect")
    assert hasattr(prompt_eng, "build")
    assert hasattr(llm, "generate")
    assert hasattr(rag, "create")
    assert hasattr(rag, "create_law")
    assert hasattr(ui, "launch")
    assert raglib.__version__ == "0.3.0"


def test_fast_30_page_pdf_speed_and_cleaning(tmp_path: Path) -> None:
    """Verify a 30-page Law PDF parses in under 1.5 seconds without pdfplumber bottleneck."""
    pdf_file = _create_multi_page_law_pdf(tmp_path / "fast_30_pages.pdf", num_pages=30)

    t0 = time.perf_counter()
    doc = parser.parse(pdf_file)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.5, f"Expected <1.5s for 30 pages, took {elapsed:.2f}s"
    assert doc.num_pages == 30
    assert doc.mode_used == "law"
    assert "jurisdictional" in doc.text
    assert "Page 1 of 30" not in doc.text
    assert len(doc.flat_sections) >= 30


def test_all_10_chunker_strategies_and_combinations() -> None:
    """Verify all 10 chunking strategies + Ensemble & Chain combinations for ChromaDB experimentation."""
    for strat in (
        "recursive",
        "fixed",
        "sentence",
        "token",
        "paragraph",
        "markdown",
        "semantic",
        "sliding_window",
        "parent_child",
        "legal_clause",
    ):
        chunks = chunker.chunk(SAMPLE_LAW_TEXT, strategy=strat, chunk_size=220, chunk_overlap=30)
        assert len(chunks) >= 1
        assert chunks[0].strategy == strat

    # Test Ensemble Combination (list of strategies or '+' joined)
    combo_chunks = chunker.chunk(
        SAMPLE_LAW_TEXT,
        strategy=["legal_clause", "semantic", "sliding_window"],
        chunk_size=300,
    )
    distinct_strats = {c.strategy for c in combo_chunks}
    assert "legal_clause" in distinct_strats
    assert ("semantic" in distinct_strats) or ("sliding_window" in distinct_strats)

    # Test Chain Combination ('>' coarse-to-fine pipeline)
    chained = chunker.chunk(SAMPLE_LAW_TEXT, strategy="legal_clause>sentence", chunk_size=200)
    assert len(chained) >= 1
    assert chained[0].strategy == "legal_clause>sentence"


def test_embedding_strategies_and_all_7_retrieval_techniques(tmp_path: Path) -> None:
    """Verify all 4 embedding strategies and all 7 retrieval techniques (semantic, bm25, hybrid, rrf, mmr, rerank, multi_query)."""
    combo_chunks = chunker.chunk(SAMPLE_LAW_TEXT, strategy="legal_clause+semantic", chunk_size=400)

    for emb_strat in ("dense", "hybrid_sparse_dense", "legal_domain_boost", "context_enriched"):
        vecs = embedder.embed(combo_chunks, provider="fast_local", strategy=emb_strat, dimension=256)
        assert len(vecs) == len(combo_chunks)
        assert len(vecs[0]) == 256

    store = vectordb.connect(
        provider="chroma",
        collection_name="exp_col",
        persist_directory=tmp_path / "chroma_exp",
    )
    store.add(combo_chunks, vecs)
    assert "legal_clause" in store.list_indexed_strategies()

    q_text = "What are the obligations under Section 2 regarding cross-border data transfers?"
    q_vec = embedder.embed_query(q_text, strategy="legal_domain_boost", dimension=256)

    for ret_strat in ("semantic", "bm25", "hybrid", "rrf", "mmr", "rerank", "multi_query"):
        hits = store.search(q_vec, query_text=q_text, top_k=3, strategy=ret_strat)
        assert len(hits) >= 1
        assert hits[0].retrieval_strategy == ret_strat

    # Filter ChromaDB retrieval by a specific chunking strategy
    filtered_hits = store.search(
        q_vec,
        query_text=q_text,
        top_k=3,
        strategy="rerank",
        chunk_strategy_filter="legal_clause",
    )
    assert all(h.chunk.strategy == "legal_clause" for h in filtered_hits)


def test_end_to_end_experimentation_and_ui_server(tmp_path: Path) -> None:
    """Verify RAGPipeline.compare_retrieval, student cell attributes, and background UI server."""
    doc = parser.parse_law(SAMPLE_LAW_TEXT)
    assert doc.total_pages == doc.num_pages
    assert len(doc.clauses) == len(doc.flat_sections)
    assert len(doc.articles) == len(doc.flat_sections)

    law_chks = chunker.chunk_law(doc)
    built_from_chks = prompt_eng.build(
        law_chks,
        technique="legal_irac",
        system_prompt="""You are a Constitutional Law Bench Senior Counsel of the Supreme Court of India.""",
    )
    assert built_from_chks.technique == "legal_irac"

    ans = llm.generate(built_from_chks, model="extractive")
    assert isinstance(ans, str)
    assert ans.text == str(ans)
    assert ans.answer == str(ans)


    law_pipe = rag.create_law(
        SAMPLE_LAW_TEXT,
        chunk_strategy=["legal_clause", "sliding_window"],
        embed_strategy="legal_domain_boost",
        retrieval_strategy="rerank",
        vectordb="chroma",
        prompt_technique="legal_irac",
        system_prompt="Explain clearly for law students.",
        llm_provider="extractive",
        llm_model="extractive",
        llm_api_key="test-key-optional",
    )
    resp = law_pipe.ask("What is Confidential Information?")
    assert resp.text == resp.answer
    assert "### 1. Issue" not in resp.answer  # Natural sentences by default!

    # When user explicitly asks in IRAC format in their query, generate IRAC:
    irac_resp = law_pipe.ask("Explain Confidential Information in IRAC format")
    assert irac_resp.wants_irac is True
    assert "### 1. Issue" in irac_resp.answer

    # Test General & Specialized Law Query Rewriting + Story Question Solver
    gen_rw = prompt_eng.rewrite_query("Can you please tell me how data breach safeguards work?")
    assert "data" in str(gen_rw).lower()

    story_q = (
        "Ramesh runs a tech company in Delhi. His company suffered a personal data breach and transferred "
        "sensitive personal data abroad without approval. What are Ramesh's obligations?"
    )
    law_rw = prompt_eng.rewrite_law_query(story_q)
    assert law_rw.is_story is True
    assert "Ramesh" in law_rw.actors

    story_resp = law_pipe.ask(story_q)
    assert story_resp.is_story is True
    assert "Ramesh" in story_resp.answer

    comparison = law_pipe.compare_retrieval("What is Confidential Information?", top_k=2)
    assert "semantic" in comparison
    assert "bm25" in comparison
    assert "rerank" in comparison


    # Test launching the clean customizable Chatbot UI in background mode and hitting GET /
    server = ui.launch(
        law_pipe,
        host="127.0.0.1",
        port=18599,
        open_browser=False,
        background=True,
        title="Constitution Chatbot",
        theme="emerald",
        accent_color="#f59e0b",
    )
    try:
        with urllib.request.urlopen("http://127.0.0.1:18599/", timeout=5) as resp_http:
            html = resp_http.read().decode("utf-8")
            assert "Constitution Chatbot" in html
            assert "#f59e0b" in html
    finally:
        server.shutdown()
        server.server_close()


def test_readme_quickstart_and_ragspark_aliases(tmp_path: Path) -> None:
    """Verify the exact 1-line Quickstart in README.md and ragspark compatibility aliases."""
    from ragspark import chunker as rs_chunker
    from ragspark import llm as rs_llm
    from ragspark import parser as rs_parser
    from ragspark import prompt_eng as rs_prompt_eng
    from ragspark import rag as rs_rag
    from ragspark import ui as rs_ui
    from ragspark import vectordb as rs_vectordb

    doc = rs_parser.parse(SAMPLE_GENERAL_TEXT)
    chunks = rs_chunker.chunk(doc, strategy="recursive", chunk_size=500, overlap=80)
    assert len(chunks) >= 1

    db = rs_vectordb.store(
        chunks,
        backend="chroma",
        collection_name="my_docs",
        persist_directory=tmp_path / "qs_chroma",
    )
    hits = db.search("What are the key takeaways?", top_k=5)
    assert len(hits) >= 1

    rewritten = rs_prompt_eng.rewrite_query("tell me about microservices")
    prompt = rs_prompt_eng.build(hits, query=rewritten.rewritten_query, technique="standard")
    answer = rs_llm.generate(prompt, provider="extractive")
    assert len(answer.text) > 10

    bot = rs_rag.create(SAMPLE_GENERAL_TEXT)
    nb_srv = rs_ui.notebook_chat(bot, port=18610)
    try:
        assert nb_srv.server_address[1] >= 18610
    finally:
        nb_srv.shutdown()
        nb_srv.server_close()


def test_step7_notebook_screenshot_regression() -> None:
    """
    Reproduce the exact Jupyter Notebook Step-7 call from the user's screenshot:
      answer = llm.generate(
          prompt,
          query="Explain freedom of speech and expression under Article 19(1)(a).",
          provider="gemini",
          model="gemini 3.8 flash",
          api_key="test-placeholder-key"
      )
      print(answer.text)
    Verify it NEVER outputs 'I could not find relevant passages in the document to answer your question.'
    even when `prompt` was built without hits or without a query in Step 6!
    """
    from raglib import llm, prompt_eng

    # Case 1: User built `prompt` in Step 6 with empty/missing hits or without passing `query`
    empty_prompt = prompt_eng.build(technique="standard")
    answer = llm.generate(
        empty_prompt,
        query="Explain freedom of speech and expression under Article 19(1)(a).",
        provider="gemini",
        model="gemini 3.8 flash",
        api_key="test-placeholder-key",
    )
    assert "I could not find relevant passages" not in answer.text
    assert "Article 19(1)(a)" in answer.text
    assert "freedom of speech and expression" in answer.text.lower()



