"""
`ragspark.chunker` — Universal Domain-Aware Chunker (v0.4.0).

Supports 24 strategies across ALL domains — auto-detected, no config needed.

Invariants (every strategy):
  - No word is ever cut in half: splits happen only at paragraph/sentence/line breaks or whitespace.
  - When `chunk_size` is omitted (None) with `strategy="auto"`, chunks follow natural document
    boundaries (Article/Section headings, Markdown headings, functions/classes, paragraphs).
  - Natural blocks over 2,500 chars are subdivided; no strategy returns a runaway chunk.

NATURAL STRATEGIES (v0.4.0):
  - "natural"        (Natural boundaries: headings / code defs / tables / paragraphs, 2,500-char ceiling)
  - "micro_sentence" (Small whole-sentence windows inside each natural block, for high-precision recall)

ORIGINAL 15 STRATEGIES:
  - "auto"           (Smart auto: detects domain, picks best strategy automatically)
  - "recursive"      (Paragraph -> Line -> Sentence -> Word recursive splitter)
  - "fixed"          (Deterministic fixed character window with overlap)
  - "sentence"       (Sentence-boundary window splitter)
  - "token"          (Word/token window splitter)
  - "paragraph"      (Paragraph-block splitter)
  - "markdown"       (Heading-hierarchy # / ## / ### splitter)
  - "semantic"       (CPU lexical-cohesion topic-shift splitter)
  - "sliding_window" (Dense overlapping sliding window for high-recall retrieval)
  - "parent_child"   (Hierarchical parent context stored in metadata + granular child text)
  - "legal_clause"   (Law: Part > Chapter > Section > Clause > Proviso + definitions)
  - "table_aware"    (Preserves Markdown/HTML table integrity)
  - "code_aware"     (AST/syntax-conscious code block splitter)
  - "contextual"     (Prepends document-level contextual headers)
  - "adaptive"       (Dynamically routes by content: code/table/markdown/prose)

NEW DOMAIN-SPECIFIC STRATEGIES (v0.4.0):
  - "medical_chunk"    (Clinical sections: History/Diagnosis/Treatment/References)
  - "finance_chunk"    (Financial tables atomic, fiscal period context attached)
  - "research_chunk"   (Abstract > Methods > Results > Discussion, citations preserved)
  - "news_chunk"       (Inverted pyramid: headline+lead atomic, paragraphs as chunks)
  - "qa_chunk"         (Q&A pairs always kept together, never split)
  - "ecommerce_chunk"  (Product title+attributes atomic, reviews separate)
  - "policy_chunk"     (Policy ID+Scope+Statement atomic, sections separate)

SMART AUTO (uses detect_domain() — zero config, works on ANY domain):
  chunker.chunk(doc, strategy="auto")  → detects & routes automatically

Combination of Strategies:
  - Pass a list: chunker.chunk(doc, strategy=["recursive", "semantic", "legal_clause"])
  - Or + string: chunker.chunk(doc, strategy="legal_clause+sliding_window")
  - mode="ensemble": runs each strategy, tags chunks by strategy for A/B testing
  - mode="chain": coarse → fine pipeline
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragspark.parser import ParsedDocument, parse, parse_law, parse_text

# v0.4.0: Universal domain-aware chunker — import all domain strategies + detector
from ragspark.chunker.smart_universal import (  # noqa: E402
    detect_domain,
    DOMAIN_STRATEGY_MAP,
    split_sentences_fast,
    chunk_medical,
    chunk_finance,
    chunk_research,
    chunk_news,
    chunk_qa,
    chunk_ecommerce,
    chunk_policy,
    smart_chunk,
    ALL_STRATEGIES as _ALL_DOMAIN_STRATEGIES,
)
from ragspark.chunker.natural import (  # noqa: E402
    MICRO_SENTENCE_SIZE,
    NATURAL_CEILING,
    _looks_like_code,
    filter_noise,
    is_noise,
    micro_sentence_spans,
    natural_spans,
    safe_split,
    safe_split_spans,
)

DEFAULT_CHUNK_SIZE = 800
"""Size used by size-based strategies when `chunk_size` is omitted."""


@dataclass
class Chunk:
    """Represents a single chunk ready for embedding and ChromaDB/VectorDB indexing."""

    id: str
    text: str
    index: int
    source: str = "inline"
    page_number: int = 1
    section_path: str = ""
    strategy: str = "recursive"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AVAILABLE_STRATEGIES = (
    "auto",
    "natural",
    "micro_sentence",
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
    "table_aware",
    "code_aware",
    "contextual",
    "adaptive",
    # v0.4.0 — Domain-specific strategies
    "medical_chunk",
    "finance_chunk",
    "research_chunk",
    "news_chunk",
    "qa_chunk",
    "ecommerce_chunk",
    "policy_chunk",
)


def list_strategies() -> list[str]:
    """Return all supported chunking strategies."""
    return list(AVAILABLE_STRATEGIES)


def _make_chunk_id(source: str, strategy: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{strategy}:{index}:{text[:160]}".encode("utf-8")).hexdigest()[:12]
    return f"chk_{strategy}_{index}_{digest}"


def _split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fixed-size character windows, snapped to whitespace so no word is cut."""
    return safe_split(text, chunk_size, chunk_overlap)


def _split_paragraph(text: str, chunk_size: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paras:
        return []
    out: list[str] = []
    for p in paras:
        if len(p) <= chunk_size * 1.5:
            out.append(p)
        else:
            out.extend(_split_recursive(p, chunk_size, chunk_size // 8))
    return out


def _split_sliding_window(text: str, chunk_size: int, stride: int | None = None) -> list[str]:
    """Dense overlapping sliding window over sentences (50% stride by default)."""
    overlap = stride if stride is not None else max(40, chunk_size // 2)
    return _split_sentences(text, chunk_size=chunk_size, chunk_overlap=overlap)


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursive paragraph -> sentence -> line -> word splitter (never cuts a word)."""
    return safe_split(text, chunk_size, chunk_overlap)


def _fit(pieces: list[str], chunk_size: int) -> list[str]:
    """Subdivide any piece longer than `chunk_size` at word-safe boundaries."""
    out: list[str] = []
    for p in pieces:
        out.extend(safe_split(p, chunk_size) if len(p) > chunk_size else [p])
    return out


def _split_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = _fit([s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()], chunk_size)
    if not sentences:
        return []
    chunks: list[str] = []
    current_sents: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > chunk_size and current_sents:
            chunks.append(" ".join(current_sents))
            overlap_sents: list[str] = []
            ov_len = 0
            for prev_s in reversed(current_sents):
                if ov_len + len(prev_s) <= chunk_overlap:
                    overlap_sents.insert(0, prev_s)
                    ov_len += len(prev_s) + 1
                else:
                    break
            current_sents = overlap_sents
            current_len = ov_len
        current_sents.append(sent)
        current_len += len(sent) + 1

    if current_sents:
        chunks.append(" ".join(current_sents))
    return chunks


def _split_tokens(text: str, token_size: int = 200, token_overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, token_size - max(0, token_overlap))
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + token_size]
        if window:
            chunks.append(" ".join(window))
        if start + token_size >= len(words):
            break
    return chunks


def _split_markdown(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, str]]:
    """Split by Markdown headers (`#`, `##`, `###`) and return (header_path, body_chunk)."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_header = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, current_lines))

    results: list[tuple[str, str]] = []
    for header, sec_lines in sections:
        body = "\n".join(sec_lines).strip()
        if not body:
            continue
        for sub in _split_recursive(body, chunk_size, chunk_overlap):
            results.append((header, sub))
    return results


def _split_semantic_cpu(text: str, chunk_size: int) -> list[str]:
    """Laptop-safe CPU Semantic Chunker splitting at lexical cohesion drops."""
    sentences = _fit([s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()], chunk_size)
    if not sentences:
        return []

    def _tokens(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", s)}

    chunks: list[str] = []
    curr: list[str] = [sentences[0]]
    curr_len = len(sentences[0])
    prev_toks = _tokens(sentences[0])

    for sent in sentences[1:]:
        toks = _tokens(sent)
        union = len(prev_toks | toks)
        jaccard = (len(prev_toks & toks) / union) if union > 0 else 0.0

        if curr_len + len(sent) > chunk_size or (curr_len >= chunk_size * 0.4 and jaccard < 0.05):
            chunks.append(" ".join(curr))
            curr = [sent]
            curr_len = len(sent)
        else:
            curr.append(sent)
            curr_len += len(sent) + 1
        prev_toks = toks

    if curr:
        chunks.append(" ".join(curr))
    return chunks


def _split_table_aware(text: str, chunk_size: int) -> list[str]:
    """Preserves Markdown and delimited tables intact, splitting rows with repeated headers if oversized."""
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current_type = "text"
    current_lines: list[str] = []

    def is_table_line(line: str) -> bool:
        s = line.strip()
        return (s.startswith("|") and s.endswith("|")) or bool(re.match(r"^\|?\s*[-:]+[-| :]*\|?$", s))

    for line in lines:
        if is_table_line(line):
            if current_type != "table":
                if current_lines:
                    blocks.append(("text", current_lines))
                current_lines = [line]
                current_type = "table"
            else:
                current_lines.append(line)
        else:
            if current_type == "table":
                if current_lines:
                    blocks.append(("table", current_lines))
                current_lines = [line]
                current_type = "text"
            else:
                current_lines.append(line)

    if current_lines:
        blocks.append((current_type, current_lines))

    chunks: list[str] = []
    for b_type, b_lines in blocks:
        block_txt = "\n".join(b_lines).strip()
        if not block_txt:
            continue
        if b_type == "text":
            chunks.extend(_split_recursive(block_txt, chunk_size, chunk_size // 8))
        else:
            if len(block_txt) <= int(chunk_size * 1.5) or len(b_lines) <= 3:
                chunks.append(block_txt)
            else:
                header_lines = b_lines[:2]
                header_prefix = "\n".join(header_lines)
                curr_rows: list[str] = []
                curr_len = len(header_prefix)
                for row in b_lines[2:]:
                    if curr_len + len(row) + 1 > chunk_size and curr_rows:
                        chunks.append(f"{header_prefix}\n" + "\n".join(curr_rows))
                        curr_rows = [row]
                        curr_len = len(header_prefix) + len(row) + 1
                    else:
                        curr_rows.append(row)
                        curr_len += len(row) + 1
                if curr_rows:
                    chunks.append(f"{header_prefix}\n" + "\n".join(curr_rows))
    return chunks


def _split_code_aware(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """AST/syntax-conscious code splitter preserving function/class definitions and code blocks intact."""
    pattern = re.compile(r"(```[\w]*\n[\s\S]*?```)")
    tokens = pattern.split(text)
    chunks: list[str] = []

    for piece in tokens:
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("```") and piece.endswith("```"):
            if len(piece) <= int(chunk_size * 1.5):
                chunks.append(piece)
            else:
                lines = piece.splitlines()
                lang_line = lines[0] if lines else "```"
                code_lines = lines[1:-1] if len(lines) > 2 else lines
                sub_blocks: list[list[str]] = []
                cur_block: list[str] = []
                for line in code_lines:
                    if re.match(r"^(def\s+|class\s+|async\s+def\s+|function\s+|export\s+|pub\s+fn\s+)", line):
                        if cur_block:
                            sub_blocks.append(cur_block)
                            cur_block = [line]
                        else:
                            cur_block.append(line)
                    else:
                        cur_block.append(line)
                if cur_block:
                    sub_blocks.append(cur_block)

                for sb in sub_blocks:
                    sb_text = f"{lang_line}\n" + "\n".join(sb) + "\n```"
                    if len(sb_text) <= chunk_size:
                        chunks.append(sb_text)
                    else:
                        chunks.extend(_split_recursive(sb_text, chunk_size, chunk_overlap))
        else:
            chunks.extend(_split_recursive(piece, chunk_size, chunk_overlap))
    return chunks


def _split_contextual(text: str, doc_name: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, str]]:
    """Context-Enriched Splitter (Anthropic Contextual Retrieval pattern).
    Prepends document title and section breadcrumbs to ensure zero ambiguity during retrieval."""
    pairs = _split_markdown(text, chunk_size=max(200, chunk_size - 120), chunk_overlap=chunk_overlap)
    if not pairs:
        base_chunks = _split_recursive(text, chunk_size=max(200, chunk_size - 120), chunk_overlap=chunk_overlap)
        pairs = [("General", c) for c in base_chunks]

    results: list[tuple[str, str]] = []
    clean_doc = Path(doc_name).stem.replace("_", " ").title() if doc_name else "Document"
    for heading, chunk_txt in pairs:
        context_header = f"[Context: {clean_doc} > {heading}]"
        enriched_text = f"{context_header}\n{chunk_txt}"
        results.append((heading, enriched_text))
    return results


def _split_adaptive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Smart content router: routes code to code_aware, tables to table_aware, markdown to markdown, prose to recursive."""
    if "```" in text:
        return _split_code_aware(text, chunk_size, chunk_overlap)
    if re.search(r"\|[-:]+\|", text):
        return _split_table_aware(text, chunk_size)
    if re.search(r"^#{1,6}\s+", text, flags=re.MULTILINE):
        return [c for _, c in _split_markdown(text, chunk_size, chunk_overlap)]
    return _split_recursive(text, chunk_size, chunk_overlap)


def _page_locator(doc: ParsedDocument):
    """Return a function mapping a character offset in `doc.text` to a page number."""
    pages = [p for p in (doc.pages or []) if (p.clean_text or "").strip()]
    if len(pages) <= 1:
        only = pages[0].page_number if pages else 1
        return lambda _offset: only
    text = doc.text or ""
    starts: list[int] = []
    numbers: list[int] = []
    pos = 0
    for page in pages:
        probe = page.clean_text.strip()[:60]
        found = text.find(probe, pos) if probe else -1
        if found >= 0:
            starts.append(found)
            numbers.append(page.page_number)
            pos = found + 1
    if not starts:
        first = pages[0].page_number
        return lambda _offset: first

    def locate(offset: int) -> int:
        i = bisect.bisect_right(starts, offset) - 1
        return numbers[max(0, i)]

    return locate


def _enforce_ceiling(chunks: list[Chunk], cap: int) -> list[Chunk]:
    """Subdivide (word-safely) any chunk longer than `cap` characters."""
    if all(len(c.text) <= cap for c in chunks):
        return chunks
    out: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= cap:
            out.append(c)
            continue
        parts = safe_split(c.text, cap)
        for k, piece in enumerate(parts, start=1):
            meta = dict(c.metadata)
            meta.update({"part": k, "parts": len(parts), "parent_chunk_id": c.id})
            out.append(
                Chunk(
                    id=f"{c.id}_p{k}",
                    text=piece,
                    index=c.index,
                    source=c.source,
                    page_number=c.page_number,
                    section_path=c.section_path,
                    strategy=c.strategy,
                    metadata=meta,
                )
            )
    for i, c in enumerate(out, start=1):
        c.index = i
    return out


class SmartChunker:
    """Multi-Strategy & Combination Chunker for General & Legal RAG."""

    def chunk(
        self,
        doc_or_source: ParsedDocument | str | Path,
        strategy: str | list[str] = "auto",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        combination_mode: str = "ensemble",
        overlap: int | None = None,
    ) -> list[Chunk]:
        """
        Chunk any ParsedDocument, file path, or raw string using a single strategy OR a combination of strategies!

        Args:
            doc_or_source: A `ParsedDocument`, a file path (`.pdf`, `.txt`, `.md`), or a raw text string.
            strategy:
                - `"auto"` (default): with `chunk_size=None` chunks by natural document boundaries;
                  with a `chunk_size`, detects the domain and routes to the best strategy.
                - `"natural"` / `"micro_sentence"`: natural blocks, or small sentence windows inside them.
                - Single strategy string: `"recursive"`, `"fixed"`, `"sentence"`, `"token"`,
                  `"paragraph"`, `"markdown"`, `"semantic"`, `"sliding_window"`, `"parent_child"`, `"legal_clause"`
                - Or combination list: `["legal_clause", "semantic"]`
                - Or `+`-separated combination string: `"legal_clause+semantic"` or `"recursive+sliding_window"`
            chunk_size: Target chunk size in characters (or words when `strategy="token"`).
                `None` (default) means natural boundaries for `"auto"`/`"natural"` (2,500-char ceiling)
                and 800 for size-based strategies.
            chunk_overlap: Overlap between consecutive chunks (also accepts `overlap=`). Defaults to
                `min(120, chunk_size // 5)`; natural chunks carry no overlap.
            combination_mode:
                - `"ensemble"` (default for list): runs each strategy and merges all resulting chunks into one list
                  (with each chunk tagged by its `strategy` for ChromaDB experimentation).
                - `"chain"` (default for `"a>b"`): splits first with strategy 1 (coarse) and then refines oversized
                  chunks with strategy 2 (fine).
        """
        if overlap is not None:
            chunk_overlap = overlap
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        natural_mode = chunk_size is None
        size = DEFAULT_CHUNK_SIZE if chunk_size is None else int(chunk_size)
        if chunk_overlap is None:
            chunk_overlap = min(120, size // 5)
        if chunk_overlap < 0 or chunk_overlap >= size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be non-negative and strictly less than chunk_size ({size}).")
        result = self._chunk_impl(doc_or_source, strategy, chunk_size, size, chunk_overlap, combination_mode, natural_mode)
        return _enforce_ceiling(result, max(NATURAL_CEILING, 2 * size))

    def _chunk_impl(
        self,
        doc_or_source: ParsedDocument | str | Path,
        strategy: str | list[str],
        raw_chunk_size: int | None,
        chunk_size: int,
        chunk_overlap: int,
        combination_mode: str,
        natural_mode: bool,
    ) -> list[Chunk]:
        # Handle list of strategies or "+" / ">" combined strategies
        if isinstance(strategy, (list, tuple)):
            strat_list = [str(s).strip().lower() for s in strategy if str(s).strip()]
            return self.chunk_combination(
                doc_or_source,
                strategies=strat_list,
                mode=combination_mode,
                chunk_size=raw_chunk_size,
                chunk_overlap=chunk_overlap,
            )

        chosen_raw = str(strategy).lower().strip()
        if "+" in chosen_raw or ">" in chosen_raw or "," in chosen_raw:
            mode_comb = "chain" if ">" in chosen_raw else combination_mode
            parts = [p.strip() for p in re.split(r"[+>,]", chosen_raw) if p.strip()]
            return self.chunk_combination(
                doc_or_source,
                strategies=parts,
                mode=mode_comb,
                chunk_size=raw_chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if chosen_raw not in AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {AVAILABLE_STRATEGIES}")

        doc = self._coerce_document(doc_or_source, force_law=(chosen_raw == "legal_clause"))
        chosen = chosen_raw

        if chosen == "auto" and natural_mode:
            # No size given: follow the document's own structure.
            if doc.mode_used == "law" and doc.flat_sections:
                return self.chunk_law(doc, chunk_size=NATURAL_CEILING, chunk_overlap=0)
            chosen = "natural"
        elif chosen == "auto":
            # v0.4.0: Universal domain detection — works for ANY domain, not just law
            domain = detect_domain(doc.text, doc.source)
            if domain == "law" and doc.mode_used == "law" and doc.flat_sections:
                chosen = "legal_clause"
            else:
                chosen = DOMAIN_STRATEGY_MAP.get(domain, "adaptive")

        if chosen in ("natural", "micro_sentence"):
            return self._chunk_natural(
                doc,
                micro=(chosen == "micro_sentence"),
                chunk_size=None if natural_mode else chunk_size,
                chunk_overlap=0 if natural_mode else chunk_overlap,
            )

        # v0.4.0: Route to new domain-specific chunkers
        _DOMAIN_CHUNKERS = {
            "medical_chunk":   lambda text, src: chunk_medical(text, src, chunk_size),
            "finance_chunk":   lambda text, src: chunk_finance(text, src, chunk_size),
            "research_chunk":  lambda text, src: chunk_research(text, src, chunk_size),
            "news_chunk":      lambda text, src: chunk_news(text, src, chunk_size),
            "qa_chunk":        lambda text, src: chunk_qa(text, src, chunk_size),
            "ecommerce_chunk": lambda text, src: chunk_ecommerce(text, src, chunk_size),
            "policy_chunk":    lambda text, src: chunk_policy(text, src, chunk_size),
        }
        if chosen in _DOMAIN_CHUNKERS:
            raw_dicts = _DOMAIN_CHUNKERS[chosen](doc.text, doc.source)
            return [
                Chunk(
                    id=d["id"],
                    text=d["text"],
                    index=d["index"],
                    source=d["source"],
                    page_number=d.get("page_number", 1),
                    section_path=d.get("section_path", ""),
                    strategy=d.get("strategy", chosen),
                    metadata=d.get("metadata", {}),
                )
                for d in raw_dicts
            ]

        if chosen == "legal_clause":
            return self.chunk_law(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks: list[Chunk] = []
        idx = 0

        if chosen == "markdown":
            md_pairs = _split_markdown(doc.text, chunk_size, chunk_overlap)
            for header, piece in md_pairs:
                idx += 1
                chunks.append(
                    Chunk(
                        id=_make_chunk_id(doc.source, "markdown", idx, piece),
                        text=piece,
                        index=idx,
                        source=doc.source,
                        page_number=1,
                        section_path=header,
                        strategy="markdown",
                        metadata={"doc_type": doc.doc_type, "heading": header, "strategy": "markdown"},
                    )
                )
            return chunks

        if chosen == "contextual":
            ctx_pairs = _split_contextual(doc.text, doc.source, chunk_size, chunk_overlap)
            for header, piece in ctx_pairs:
                idx += 1
                chunks.append(
                    Chunk(
                        id=_make_chunk_id(doc.source, "contextual", idx, piece),
                        text=piece,
                        index=idx,
                        source=doc.source,
                        page_number=1,
                        section_path=header,
                        strategy="contextual",
                        metadata={"doc_type": doc.doc_type, "heading": header, "strategy": "contextual"},
                    )
                )
            return chunks

        if chosen == "parent_child":
            return self._chunk_parent_child(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        page_blocks = doc.pages if doc.pages else []
        if not page_blocks:
            raw_pieces = self._run_text_strategy(doc.text, chosen, chunk_size, chunk_overlap)
            for piece in raw_pieces:
                idx += 1
                chunks.append(
                    Chunk(
                        id=_make_chunk_id(doc.source, chosen, idx, piece),
                        text=piece,
                        index=idx,
                        source=doc.source,
                        page_number=1,
                        strategy=chosen,
                        metadata={"doc_type": doc.doc_type, "strategy": chosen},
                    )
                )
            return chunks

        for page in page_blocks:
            if not page.clean_text.strip():
                continue
            pieces = self._run_text_strategy(page.clean_text, chosen, chunk_size, chunk_overlap)
            for piece in pieces:
                idx += 1
                chunks.append(
                    Chunk(
                        id=_make_chunk_id(doc.source, chosen, idx, piece),
                        text=piece,
                        index=idx,
                        source=doc.source,
                        page_number=page.page_number,
                        strategy=chosen,
                        metadata={"doc_type": doc.doc_type, "page": page.page_number, "strategy": chosen},
                    )
                )

        return chunks

    def chunk_combination(
        self,
        doc_or_source: ParsedDocument | str | Path,
        strategies: list[str],
        mode: str = "ensemble",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Chunk]:
        """
        Combine multiple chunking strategies for ChromaDB experimentation:
          - `mode="ensemble"`: Runs each strategy in `strategies` and returns the combined set of chunks
            (each tagged with `chunk.strategy` so ChromaDB stores multi-resolution/multi-strategy representations).
          - `mode="chain"`: Coarse-to-fine pipeline — runs `strategies[0]` at `chunk_size * 2`, then splits each
            resulting chunk with `strategies[1]` at `chunk_size`, preserving parent `section_path`.
        """
        clean_strats = [s.strip().lower() for s in strategies if s.strip()]
        if not clean_strats:
            return self.chunk(doc_or_source, strategy="auto", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if len(clean_strats) == 1:
            return self.chunk(doc_or_source, strategy=clean_strats[0], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        needs_law = "legal_clause" in clean_strats
        doc = self._coerce_document(doc_or_source, force_law=needs_law)
        raw_size = chunk_size
        chunk_size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
        if chunk_overlap is None:
            chunk_overlap = min(120, chunk_size // 5)

        if mode.lower() == "chain":
            coarse_chunks = self.chunk(
                doc,
                strategy=clean_strats[0],
                chunk_size=int(chunk_size * 1.8),
                chunk_overlap=chunk_overlap,
            )
            chained_label = ">".join(clean_strats)
            refined: list[Chunk] = []
            idx = 0
            for c_chk in coarse_chunks:
                fine_pieces = self._run_text_strategy(
                    c_chk.text,
                    clean_strats[1],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for fp in fine_pieces:
                    idx += 1
                    meta = dict(c_chk.metadata)
                    meta["strategy"] = chained_label
                    meta["parent_chunk_id"] = c_chk.id
                    refined.append(
                        Chunk(
                            id=_make_chunk_id(doc.source, chained_label, idx, fp),
                            text=fp,
                            index=idx,
                            source=doc.source,
                            page_number=c_chk.page_number,
                            section_path=c_chk.section_path,
                            strategy=chained_label,
                            metadata=meta,
                        )
                    )
            return refined

        if mode.lower() in ("multi_resolution", "multiresolution", "hierarchical"):
            macro_strat = clean_strats[0]
            micro_strat = clean_strats[1] if len(clean_strats) > 1 else "sentence"
            macro_chunks = self.chunk(doc, strategy=macro_strat, chunk_size=int(chunk_size * 2), chunk_overlap=chunk_overlap)
            micro_chunks = self.chunk(doc, strategy=micro_strat, chunk_size=max(160, chunk_size // 2), chunk_overlap=max(20, chunk_overlap // 2))
            for mic in micro_chunks:
                for mac in macro_chunks:
                    if mic.text[:60] in mac.text:
                        mic.metadata["parent_macro_id"] = mac.id
                        mic.metadata["parent_macro_preview"] = mac.text[:120]
                        break
            return macro_chunks + micro_chunks

        if mode.lower() == "interleave":
            groups = [self.chunk(doc, strategy=s, chunk_size=raw_size, chunk_overlap=chunk_overlap) for s in clean_strats]
            interleaved: list[Chunk] = []
            max_len = max(len(g) for g in groups) if groups else 0
            seen_texts_i: set[tuple[int, str]] = set()
            for i in range(max_len):
                for g in groups:
                    if i < len(g):
                        chk = g[i]
                        key = (chk.page_number, chk.text.strip())
                        if key not in seen_texts_i:
                            seen_texts_i.add(key)
                            interleaved.append(chk)
            return interleaved

        # Default: Ensemble mode — combine chunks from all selected strategies (deduplicating exact identical text per page)
        combined: list[Chunk] = []
        seen_texts: set[tuple[int, str]] = set()
        for strat in clean_strats:
            sub_chunks = self.chunk(doc, strategy=strat, chunk_size=raw_size, chunk_overlap=chunk_overlap)
            for sc in sub_chunks:
                dedup_key = (sc.page_number, sc.text.strip())
                if dedup_key in seen_texts:
                    continue
                seen_texts.add(dedup_key)
                sc.metadata["ensemble_strategies"] = clean_strats
                combined.append(sc)
        return combined

    def _chunk_natural(
        self,
        doc: ParsedDocument,
        micro: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
    ) -> list[Chunk]:
        """Natural-boundary chunks (or `micro_sentence` windows inside them), with TOC/gazette noise removed."""
        text = doc.text or ""
        if doc.raw_text and _looks_like_code(doc.raw_text):
            text = doc.raw_text  # the cleaner flattens indentation; code needs it intact
        is_law = doc.mode_used == "law" or str(doc.doc_type).startswith("law")
        blocks = natural_spans(text, chunk_size=chunk_size, overlap=chunk_overlap, is_law=is_law)
        page_of = _page_locator(doc)
        chunks: list[Chunk] = []
        nat_idx = 0
        for blk in blocks:
            block_text = text[blk.start:blk.end].strip()
            if not block_text or is_noise(block_text):
                continue
            nat_idx += 1
            parent_id = _make_chunk_id(doc.source, "natural", nat_idx, block_text)
            base_meta = {"doc_type": doc.doc_type, "block_type": blk.kind, **blk.metadata}
            if blk.heading:
                base_meta["heading"] = blk.heading
            if not micro:
                chunks.append(
                    Chunk(
                        id=parent_id,
                        text=block_text,
                        index=nat_idx,
                        source=doc.source,
                        page_number=page_of(blk.start),
                        section_path=blk.section_path,
                        strategy="natural",
                        metadata={**base_meta, "page": page_of(blk.start), "strategy": "natural"},
                    )
                )
                continue
            micro_size = chunk_size or MICRO_SENTENCE_SIZE
            for a, b in micro_sentence_spans(text, blk.start, blk.end, size=micro_size):
                piece = text[a:b].strip()
                if not piece:
                    continue
                idx = len(chunks) + 1
                chunks.append(
                    Chunk(
                        id=_make_chunk_id(doc.source, "micro_sentence", idx, piece),
                        text=piece,
                        index=idx,
                        source=doc.source,
                        page_number=page_of(a),
                        section_path=blk.section_path,
                        strategy="micro_sentence",
                        metadata={
                            **base_meta,
                            "page": page_of(a),
                            "strategy": "micro_sentence",
                            "parent_id": parent_id,
                        },
                    )
                )
        return chunks

    def chunk_law(
        self,
        doc_or_source: ParsedDocument | str | Path,
        chunk_size: int = 900,
        chunk_overlap: int = 100,
        include_hierarchy: bool = True,
        attach_definitions: bool = True,
    ) -> list[Chunk]:
        """
        Specialized Law Command: Chunks legal documents along statutory/contractual hierarchy boundaries
        (`Part > Chapter > Section > Subsection > Clause > Proviso`) and enriches each chunk with
        its exact legal lineage and matching defined terms.
        """
        doc = self._coerce_document(doc_or_source, force_law=True)
        if not doc.flat_sections:
            return self.chunk(doc, strategy="recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        defs_map = {d.term.lower(): f"{d.term}: {d.definition}" for d in doc.definitions}
        chunks: list[Chunk] = []
        idx = 0

        for node in doc.flat_sections:
            header_line = f"[{node.full_path}]"
            if node.title:
                header_line += f" {node.title}"
            body_text = (node.text or "").strip()

            if not body_text and node.children:
                continue

            sub_pieces = (
                _split_recursive(body_text, max(250, chunk_size - len(header_line) - 20), chunk_overlap)
                if body_text
                else [header_line]
            )

            for piece in sub_pieces:
                idx += 1
                formatted = (
                    f"{header_line}\n{piece}".strip()
                    if include_hierarchy and piece != header_line
                    else piece
                )

                matched_defs: list[str] = []
                if attach_definitions and defs_map:
                    piece_low = formatted.lower()
                    for term_low, def_str in defs_map.items():
                        if re.search(rf"\b{re.escape(term_low)}\b", piece_low):
                            matched_defs.append(def_str)

                chunks.append(
                    Chunk(
                        id=_make_chunk_id(doc.source, "legal_clause", idx, formatted),
                        text=formatted,
                        index=idx,
                        source=doc.source,
                        page_number=node.page_number,
                        section_path=node.full_path,
                        strategy="legal_clause",
                        metadata={
                            "doc_type": doc.doc_type,
                            "node_type": node.node_type,
                            "clause_number": node.number,
                            "section_path": node.full_path,
                            "page": node.page_number,
                            "definitions_in_scope": matched_defs,
                            "court": doc.legal_metadata.get("court"),
                            "strategy": "legal_clause",
                        },
                    )
                )

        return chunks

    def _chunk_parent_child(self, doc: ParsedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        """Create granular child chunks (size = chunk_size // 2) that carry their parent chunk context in metadata."""
        parent_size = max(400, chunk_size)
        child_size = max(180, chunk_size // 2)
        chunks: list[Chunk] = []
        idx = 0

        pages = doc.pages if doc.pages else []
        source_blocks = [(p.page_number, p.clean_text) for p in pages] if pages else [(1, doc.text)]

        for pno, ptext in source_blocks:
            parents = _split_recursive(ptext, parent_size, chunk_overlap)
            for p_idx, parent_txt in enumerate(parents, start=1):
                children = _split_sentences(parent_txt, child_size, max(20, chunk_overlap // 2))
                for child_txt in children:
                    idx += 1
                    chunks.append(
                        Chunk(
                            id=_make_chunk_id(doc.source, "parent_child", idx, child_txt),
                            text=child_txt,
                            index=idx,
                            source=doc.source,
                            page_number=pno,
                            section_path=f"Parent-{p_idx}",
                            strategy="parent_child",
                            metadata={
                                "doc_type": doc.doc_type,
                                "page": pno,
                                "parent_text": parent_txt,
                                "strategy": "parent_child",
                            },
                        )
                    )
        return chunks

    def _run_text_strategy(self, text: str, chosen: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if chosen == "fixed":
            return _split_fixed(text, chunk_size, chunk_overlap)
        if chosen == "paragraph":
            return _split_paragraph(text, chunk_size)
        if chosen == "sliding_window":
            return _split_sliding_window(text, chunk_size, chunk_overlap or (chunk_size // 2))
        if chosen == "sentence":
            return _split_sentences(text, chunk_size, chunk_overlap)
        if chosen == "token":
            tok_size = max(20, chunk_size // 4) if chunk_size > 250 else chunk_size
            tok_ov = max(5, chunk_overlap // 4) if chunk_overlap > 50 else chunk_overlap
            return _split_tokens(text, tok_size, tok_ov)
        if chosen == "semantic":
            return _split_semantic_cpu(text, chunk_size)
        if chosen == "table_aware":
            return _split_table_aware(text, chunk_size)
        if chosen == "code_aware":
            return _split_code_aware(text, chunk_size, chunk_overlap)
        if chosen == "adaptive":
            return _split_adaptive(text, chunk_size, chunk_overlap)
        return _split_recursive(text, chunk_size, chunk_overlap)

    def _coerce_document(self, doc_or_source: ParsedDocument | str | Path, force_law: bool = False) -> ParsedDocument:
        if isinstance(doc_or_source, ParsedDocument):
            return doc_or_source
        if isinstance(doc_or_source, Path):
            return parse_law(doc_or_source) if force_law else parse(doc_or_source)
        if isinstance(doc_or_source, str) and "\n" not in doc_or_source and len(doc_or_source) < 260:
            cand = Path(doc_or_source)
            if cand.exists():
                return parse_law(cand) if force_law else parse(cand)
            if cand.suffix.lower() in (".pdf", ".txt", ".md", ".markdown"):
                raise FileNotFoundError(f"File not found: {cand}")
        return parse_text(str(doc_or_source), mode="law" if force_law else "auto")


default_chunker = SmartChunker()
_LAST_ACTIVE_CHUNKS: list[Chunk] = []


def chunk(
    doc_or_source: ParsedDocument | str | Path,
    strategy: str | list[str] = "auto",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    combination_mode: str = "ensemble",
    overlap: int | None = None,
) -> list[Chunk]:
    """
    One-command chunker supporting single strategies OR combinations (`['natural', 'micro_sentence']`).

    With `chunk_size` omitted, `strategy="auto"` chunks by natural document boundaries.
    """
    global _LAST_ACTIVE_CHUNKS
    res = default_chunker.chunk(
        doc_or_source=doc_or_source,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap if overlap is not None else chunk_overlap,
        combination_mode=combination_mode,
    )
    if res:
        _LAST_ACTIVE_CHUNKS = res
    return res


def chunk_combination(
    doc_or_source: ParsedDocument | str | Path,
    strategies: list[str],
    mode: str = "ensemble",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Explicitly combine multiple chunking strategies ('ensemble' or 'chain') for ChromaDB experimentation."""
    global _LAST_ACTIVE_CHUNKS
    res = default_chunker.chunk_combination(
        doc_or_source=doc_or_source,
        strategies=strategies,
        mode=mode,
        chunk_size=chunk_size,
        chunk_overlap=overlap if overlap is not None else chunk_overlap,
    )
    if res:
        _LAST_ACTIVE_CHUNKS = res
    return res


def chunk_law(
    doc_or_source: ParsedDocument | str | Path,
    chunk_size: int = 900,
    chunk_overlap: int = 100,
    include_hierarchy: bool = True,
    attach_definitions: bool = True,
    overlap: int | None = None,
) -> list[Chunk]:
    """Specialized one-command Law Chunker preserving statutory/contract hierarchy and definitions."""
    global _LAST_ACTIVE_CHUNKS
    res = default_chunker.chunk_law(
        doc_or_source=doc_or_source,
        chunk_size=chunk_size,
        chunk_overlap=overlap if overlap is not None else chunk_overlap,
        include_hierarchy=include_hierarchy,
        attach_definitions=attach_definitions,
    )
    if res:
        _LAST_ACTIVE_CHUNKS = res
    return res


__all__ = [
    "Chunk",
    "SmartChunker",
    "chunk",
    "chunk_combination",
    "chunk_law",
    "list_strategies",
    "AVAILABLE_STRATEGIES",
    "DEFAULT_CHUNK_SIZE",
    "NATURAL_CEILING",
    "safe_split",
    "filter_noise",
    "is_noise",
    # v0.4.0 — Universal domain-aware chunker
    "detect_domain",
    "DOMAIN_STRATEGY_MAP",
    "split_sentences_fast",
    "smart_chunk",
    "chunk_medical",
    "chunk_finance",
    "chunk_research",
    "chunk_news",
    "chunk_qa",
    "chunk_ecommerce",
    "chunk_policy",
]

