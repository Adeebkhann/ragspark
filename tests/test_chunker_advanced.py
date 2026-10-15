"""
Comprehensive tests for advanced chunking strategies, combinations, and graph visualizations in ragspark.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from ragspark import chunker


def test_table_aware_chunking():
    doc_text = """
    # Enterprise Service Level Agreement

    Here are the core uptime guarantees:

    | Tier | Availability | Response Time | Penalty Credit |
    | :--- | :--- | :--- | :--- |
    | Basic | 99.0% | 24 Hours | 5% |
    | Pro | 99.9% | 4 Hours | 15% |
    | Enterprise | 99.99% | 15 Minutes | 30% |

    Termination conditions apply to repeated breach of SLA.
    """
    chunks = chunker.chunk(doc_text, strategy="table_aware", chunk_size=400)
    assert len(chunks) >= 1
    # Check that the table block exists with its headers intact
    table_chunk = [c for c in chunks if "| Basic |" in c.text]
    assert len(table_chunk) == 1
    assert "| Tier | Availability |" in table_chunk[0].text
    assert "| Enterprise |" in table_chunk[0].text


def test_code_aware_chunking():
    doc_text = """
    # Authentication Module Documentation

    Here is the implementation:

    ```python
    def authenticate(username, token):
        if not token:
            raise ValueError("Token required")
        return verify_jwt(token)

    class SessionManager:
        def __init__(self, ttl=3600):
            self.ttl = ttl
            self.sessions = {}

        def get_session(self, sid):
            return self.sessions.get(sid)
    ```

    Ensure tokens are signed with RS256.
    """
    chunks = chunker.chunk(doc_text, strategy="code_aware", chunk_size=300)
    assert len(chunks) >= 1
    # Verify that code fences are intact
    code_chunks = [c for c in chunks if "def authenticate" in c.text]
    assert len(code_chunks) >= 1
    assert "```python" in code_chunks[0].text


def test_contextual_chunking():
    doc_text = """
    # Product Roadmap 2026

    ## Feature Stream A
    We will launch real-time agentic collaboration across all mobile platforms.

    ## Feature Stream B
    We will add distributed vector indexing with sub-millisecond retrieval.
    """
    chunks = chunker.chunk(doc_text, strategy="contextual", chunk_size=200)
    assert len(chunks) >= 1
    # Verify [Context: ...] breadcrumb is injected
    assert any("[Context:" in c.text for c in chunks)
    assert any("Feature Stream A" in c.text for c in chunks)


def test_multi_resolution_combination():
    doc_text = """
    # Section 1. Governance Overview
    The board of directors shall meet quarterly to review company performance and financial audits.
    Every committee is appointed with explicit charter responsibilities.
    Risk committee oversees operational and cyber threats.

    # Section 2. Audit Requirements
    External audits must be conducted annually by an independent certified public accountant firm.
    Any material weaknesses must be remediated within ninety days of disclosure.
    """
    chunks = chunker.chunk_combination(
        doc_text,
        strategies=["markdown", "sentence"],
        mode="multi_resolution",
        chunk_size=300,
    )
    assert len(chunks) >= 2
    # Verify micro chunks link to parent macro chunk
    micro_chunks = [c for c in chunks if c.metadata.get("parent_macro_id")]
    assert len(micro_chunks) >= 1


def test_interleave_combination():
    doc_text = "Sentence one is here. Sentence two is here. Sentence three is here. Sentence four is here. Sentence five is here."
    chunks = chunker.chunk_combination(
        doc_text,
        strategies=["sentence", "recursive"],
        mode="interleave",
        chunk_size=50,
    )
    assert len(chunks) >= 2

