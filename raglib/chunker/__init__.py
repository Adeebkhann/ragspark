"""
`raglib.chunker` — Production Multi-Strategy & Combination Chunker (General + Law).

Supports 10 Individual Strategies + Multi-Strategy Combinations:
  - "auto"           (Picks "legal_clause" for law docs, "recursive" for general)
  - "recursive"      (Paragraph -> Line -> Sentence -> Word recursive splitter)
  - "fixed"          (Deterministic fixed character window with overlap)
  - "sentence"       (Sentence-boundary window splitter)
  - "token"          (Word/token window splitter)
  - "paragraph"      (Paragraph-block splitter)
  - "markdown"       (Heading-hierarchy `# / ## / ###` splitter)
  - "semantic"       (CPU lexical-cohesion topic-shift splitter)
  - "sliding_window" (Dense overlapping sliding window for high-recall retrieval)
  - "parent_child"   (Hierarchical parent context stored in metadata + granular child text)
  - "legal_clause"   (Specialized Law hierarchy: Part > Chapter > Section > Clause > Proviso + definitions)

Combination of Strategies (for ChromaDB Experimentation):
  - Pass a list: `chunker.chunk(doc, strategy=["recursive", "semantic", "legal_clause"])`
  - Or `+` string: `chunker.chunk(doc, strategy="legal_clause+sliding_window")`
  - Mode `"ensemble"` (default for lists): runs each strategy and tags each chunk's `strategy` field in ChromaDB
    so you can compare or retrieve across combined strategies!
  - Mode `"chain"` (or `"+"` syntax): pipelines Coarse Strategy -> Fine Strategy sequentially.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from raglib.parser import ParsedDocument, parse, parse_law, parse_text


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
)


def list_strategies() -> list[str]:
    """Return all supported chunking strategies."""
    return list(AVAILABLE_STRATEGIES)


def _make_chunk_id(source: str, strategy: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{strategy}:{index}:{text[:160]}".encode("utf-8")).hexdigest()[:12]
    return f"chk_{strategy}_{index}_{digest}"


def _split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - max(0, chunk_overlap))
    out: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i : i + chunk_size].strip()
        if piece:
            out.append(piece)
        if i + chunk_size >= len(text):
            break
    return out


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
    """Industry-standard recursive paragraph/line/sentence/word splitter."""
    if not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
                overlap_tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = f"{overlap_tail}\n{para}".strip() if overlap_tail else para
            else:
                current = para

            while len(current) > chunk_size:
                cut = current.rfind(". ", 0, chunk_size)
                if cut == -1 or cut < chunk_size // 2:
                    cut = current.rfind(" ", 0, chunk_size)
                if cut == -1 or cut < chunk_size // 3:
                    cut = chunk_size
                else:
                    cut += 1
                piece = current[:cut].strip()
                if piece:
                    chunks.append(piece)
                step = max(1, cut - max(0, chunk_overlap))
                current = current[step:].strip()

    if current:
        chunks.append(current)
    return chunks


def _split_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
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
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
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


class SmartChunker:
    """Multi-Strategy & Combination Chunker for General & Legal RAG."""

    def chunk(
        self,
        doc_or_source: ParsedDocument | str | Path,
        strategy: str | list[str] = "auto",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        combination_mode: str = "ensemble",
        overlap: int | None = None,
    ) -> list[Chunk]:
        """
        Chunk any ParsedDocument, file path, or raw string using a single strategy OR a combination of strategies!

        Args:
            doc_or_source: A `ParsedDocument`, a file path (`.pdf`, `.txt`, `.md`), or a raw text string.
            strategy:
                - Single strategy string: `"auto"`, `"recursive"`, `"fixed"`, `"sentence"`, `"token"`,
                  `"paragraph"`, `"markdown"`, `"semantic"`, `"sliding_window"`, `"parent_child"`, `"legal_clause"`
                - Or combination list: `["legal_clause", "semantic"]`
                - Or `+`-separated combination string: `"legal_clause+semantic"` or `"recursive+sliding_window"`
            chunk_size: Target chunk size in characters (or words when `strategy="token"`).
            chunk_overlap: Overlap between consecutive chunks (also accepts `overlap=`).
            combination_mode:
                - `"ensemble"` (default for list): runs each strategy and merges all resulting chunks into one list
                  (with each chunk tagged by its `strategy` for ChromaDB experimentation).
                - `"chain"` (default for `"a>b"`): splits first with strategy 1 (coarse) and then refines oversized
                  chunks with strategy 2 (fine).
        """
        if overlap is not None:
            chunk_overlap = overlap
        # Handle list of strategies or "+" / ">" combined strategies
        if isinstance(strategy, (list, tuple)):
            strat_list = [str(s).strip().lower() for s in strategy if str(s).strip()]
            return self.chunk_combination(
                doc_or_source,
                strategies=strat_list,
                mode=combination_mode,
                chunk_size=chunk_size,
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
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if chosen_raw not in AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {AVAILABLE_STRATEGIES}")

        doc = self._coerce_document(doc_or_source, force_law=(chosen_raw == "legal_clause"))
        chosen = chosen_raw

        if chosen == "auto":
            if doc.mode_used == "law" and doc.flat_sections:
                chosen = "legal_clause"
            else:
                chosen = "recursive"

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
        chunk_size: int = 800,
        chunk_overlap: int = 120,
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

        # Default: Ensemble mode — combine chunks from all selected strategies (deduplicating exact identical text per page)
        combined: list[Chunk] = []
        seen_texts: set[tuple[int, str]] = set()
        for strat in clean_strats:
            sub_chunks = self.chunk(doc, strategy=strat, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for sc in sub_chunks:
                dedup_key = (sc.page_number, sc.text.strip())
                if dedup_key in seen_texts:
                    continue
                seen_texts.add(dedup_key)
                sc.metadata["ensemble_strategies"] = clean_strats
                combined.append(sc)
        return combined

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
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    combination_mode: str = "ensemble",
    overlap: int | None = None,
) -> list[Chunk]:
    """One-command chunker supporting single strategies OR combinations (`['legal_clause', 'semantic']` or `'recursive+sentence'`)."""
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
    chunk_size: int = 800,
    chunk_overlap: int = 120,
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
]
