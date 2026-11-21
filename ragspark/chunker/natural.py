"""
ragspark.chunker.natural
========================
Natural-boundary chunking with a word-safe boundary walker (v0.4.0).

Three guarantees every function in this module upholds:

1. **No word-slicing.** Text is only ever cut inside a whitespace run. The preferred
   cut is a paragraph break, then a sentence end (`. `, `? `, `! `), then a line break,
   then any whitespace. A single token longer than the limit is kept whole rather
   than cut.
2. **Natural boundaries first.** `NaturalChunker` splits on Article/Section/Part
   headings, Markdown headings, top-level functions/classes, fenced code blocks,
   tables, and finally paragraphs.
3. **Natural ceiling.** Any natural block longer than `NATURAL_CEILING` (2,500 chars)
   is subdivided with the boundary walker, so a 50,000-character unformatted
   schedule never becomes a single chunk.

`filter_noise()` drops table-of-contents and gazette boilerplate blocks so they
never reach the vector store.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

NATURAL_CEILING = 2500
"""Maximum characters for a natural block before it is subdivided."""

MIN_NATURAL_BLOCK = 160
"""Paragraph blocks shorter than this are merged with the following block in the same section."""

MICRO_SENTENCE_SIZE = 320
"""Default size for `micro_sentence` chunks."""

# ---------------------------------------------------------------------------
# Word-safe boundary walker
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PARA_RE = re.compile(r"\n[ \t\r\f\v]*\n\s*")
_SENT_RE = re.compile(r"(?<=[.!?。！？।])[\"'”’)\]]*(\s+)")
_LINE_RE = re.compile(r"\n\s*")


class _Breaks:
    """Sorted whitespace-run spans for one break level, searchable with bisect."""

    __slots__ = ("starts", "ends")

    def __init__(self, spans: Iterable[tuple[int, int]]) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []
        for s, e in spans:
            self.starts.append(s)
            self.ends.append(e)

    def last_in(self, lo: int, hi: int) -> tuple[int, int] | None:
        """Return the last break whose start lies in (lo, hi]."""
        i = bisect.bisect_right(self.starts, hi) - 1
        if i >= 0 and self.starts[i] > lo:
            return self.starts[i], self.ends[i]
        return None

    def first_after(self, pos: int) -> tuple[int, int] | None:
        """Return the first break whose start is >= pos."""
        i = bisect.bisect_left(self.starts, pos)
        if i < len(self.starts):
            return self.starts[i], self.ends[i]
        return None


def _build_breaks(text: str, start: int, end: int) -> tuple[_Breaks, _Breaks, _Breaks, _Breaks]:
    para = _Breaks((m.start(), m.end()) for m in _PARA_RE.finditer(text, start, end))
    sent = _Breaks((m.start(1), m.end(1)) for m in _SENT_RE.finditer(text, start, end))
    line = _Breaks((m.start(), m.end()) for m in _LINE_RE.finditer(text, start, end))
    space = _Breaks((m.start(), m.end()) for m in _WS_RE.finditer(text, start, end))
    return para, sent, line, space


def safe_split_spans(
    text: str,
    max_chars: int,
    overlap: int = 0,
    start: int = 0,
    end: int | None = None,
) -> list[tuple[int, int]]:
    """
    Split ``text[start:end]`` into ``(a, b)`` spans of at most ``max_chars`` characters.

    Cuts happen only inside whitespace runs, preferring paragraph > sentence > line > space.
    Each returned span is trimmed of surrounding whitespace. A single token longer than
    ``max_chars`` is returned as its own (oversized) span instead of being cut.
    ``overlap`` repeats up to that many trailing characters of the previous span at the
    start of the next one, snapped forward to a word boundary.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    end = len(text) if end is None else end
    overlap = max(0, min(overlap, max_chars - 1))

    # Trim leading/trailing whitespace of the region.
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return []
    if end - start <= max_chars:
        return [(start, end)]

    para, sent, line, space = _build_breaks(text, start, end)
    spans: list[tuple[int, int]] = []
    s = start
    while s < end:
        if end - s <= max_chars:
            spans.append((s, end))
            break
        hi = s + max_chars
        min_fill = s + max_chars // 2
        cut: tuple[int, int] | None = None
        for level in (para, sent, line):
            cut = level.last_in(min_fill, hi)
            if cut:
                break
        if cut is None:
            cut = space.last_in(s, hi)
        if cut is None:
            # One token longer than max_chars: keep it whole, cut at the next whitespace.
            cut = space.first_after(hi)
            if cut is None or cut[0] >= end:
                spans.append((s, end))
                break
        cut_start, cut_end = cut
        spans.append((s, cut_start))
        nxt = cut_end
        if overlap:
            m = _WS_RE.search(text, max(s, cut_start - overlap), cut_start)
            if m and s < m.end() < cut_start:
                nxt = m.end()
        if nxt <= s:
            nxt = cut_end
        s = nxt
        while s < end and text[s].isspace():
            s += 1
    return [(a, b) for a, b in spans if b > a]


def safe_split(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """Word-safe splitter: returns pieces of ``text`` no longer than ``max_chars`` (see `safe_split_spans`)."""
    return [text[a:b] for a, b in safe_split_spans(text, max_chars, overlap)]


# ---------------------------------------------------------------------------
# Natural block detection
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LEGAL_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:PART|Part|CHAPTER|Chapter|SCHEDULE|Schedule|ARTICLE|Article|SECTION|Section|ANNEXURE|Annexure)"
    r"\s+[0-9IVXLCDM]+[A-Z]?\b"
    r"|§+\s*\d+[A-Za-z]?\b"
    r"|(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE\b"
    r")"
)
_NUMBERED_SECTION_RE = re.compile(r"^\s*\d{1,3}[A-Z]?\.\s+[A-Z][^\n]{2,110}$")
_CODE_DEF_RE = re.compile(
    r"^(?:@\w|(?:async\s+)?def\s+\w|class\s+\w|function\s+\w|export\s+(?:default\s+)?(?:function|class|const)\b"
    r"|(?:pub\s+)?fn\s+\w|func\s+\w|(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\()"
)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class NaturalBlock:
    """A natural unit of a document, addressed by character offsets into the source text."""

    start: int
    end: int
    kind: str  # "section" | "markdown" | "code" | "table" | "paragraph"
    heading: str = ""
    section_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _looks_like_code(text: str) -> bool:
    sample = text[:6000]
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    defs = sum(1 for ln in lines if _CODE_DEF_RE.match(ln))
    syntax = sum(1 for ln in lines if re.search(r"[{};]\s*$|^\s*(?:import|from|return|if|for|while)\b", ln))
    return defs >= 1 and (defs + syntax) / len(lines) >= 0.25


def _label(line: str, limit: int = 90) -> str:
    """Short heading label: the line up to its first sentence end, trimmed at a word boundary."""
    title = re.sub(r"\s+", " ", line.strip())
    # An unwrapped heading runs straight into its body ("...the Union India, that is Bharat"):
    # cut where a lowercase word is followed by a Capitalised word that starts a lowercase clause.
    glued = re.search(r"^(.{12,}?\b[A-Za-z][a-z]+)\s+(?=[A-Z][a-z]*,?\s+[a-z])", title)
    if glued and len(glued.group(1)) <= limit:
        return glued.group(1)
    m = re.match(r"^(.{8,}?[a-z\)])\.\s", title)
    if m and len(m.group(1)) <= limit:
        return m.group(1)
    if len(title) <= limit:
        return title
    cut = title.rfind(" ", 0, limit)
    return title[: cut if cut > 0 else limit].rstrip(" ,;:") + "..."


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        spans.append((pos, pos + len(line.rstrip("\r\n"))))
        pos += len(line)
    return spans


def detect_blocks(text: str, is_law: bool = False) -> list[NaturalBlock]:
    """
    Detect natural blocks in ``text`` and return them in document order.

    Heading lines (Markdown `#`, Part/Chapter/Article/Section/Schedule, numbered law sections,
    top-level code definitions) start a new section that runs until the next heading.
    Fenced code blocks and pipe tables are kept atomic. Text with no headings falls back to
    paragraph blocks.
    """
    if not text.strip():
        return []

    is_code = _looks_like_code(text)
    lines = _line_spans(text)
    numbered_ok = is_law or sum(1 for a, b in lines if _NUMBERED_SECTION_RE.match(text[a:b])) >= 3

    def heading_of(line: str) -> tuple[str, int] | None:
        m = _MD_HEADING_RE.match(line)
        if m:
            return m.group(2).strip(), len(m.group(1))
        if _TOC_LINE_RE.match(line.strip()):
            return None  # "Article 1. Name ........ 1" is a TOC entry, not a heading
        if _LEGAL_HEADING_RE.match(line):
            # PDF cleaners often unwrap the heading into its body, so any length is accepted.
            title = _label(line)
            word = title.split()[0].upper()
            if word in ("PART", "SCHEDULE") or "SCHEDULE" in title.split(".")[0].upper():
                return title, 1
            return title, 2 if word == "CHAPTER" else 3
        if numbered_ok and not is_code and _NUMBERED_SECTION_RE.match(line):
            return line.strip(), 3
        if is_code and _CODE_DEF_RE.match(line):
            return line.strip().rstrip(":{ ").strip(), 1
        return None

    blocks: list[NaturalBlock] = []
    md_stack: list[tuple[int, str]] = []
    legal_stack: list[tuple[int, str]] = []
    cur_start: int | None = None
    cur_heading = ""
    cur_path = ""
    cur_kind = "paragraph"

    def flush(upto: int) -> None:
        nonlocal cur_start
        if cur_start is not None and text[cur_start:upto].strip():
            blocks.append(NaturalBlock(cur_start, upto, cur_kind, cur_heading, cur_path))
        cur_start = None

    i = 0
    n = len(lines)
    while i < n:
        a, b = lines[i]
        line = text[a:b]

        # Atomic fenced code block.
        if _FENCE_RE.match(line):
            fence = _FENCE_RE.match(line).group(1)  # type: ignore[union-attr]
            j = i + 1
            while j < n and not text[lines[j][0]:lines[j][1]].strip().startswith(fence):
                j += 1
            j = min(j, n - 1)
            flush(a)
            blocks.append(NaturalBlock(a, lines[j][1], "code", cur_heading, cur_path))
            i = j + 1
            continue

        # Atomic pipe table.
        if _TABLE_LINE_RE.match(line):
            j = i
            while j + 1 < n and _TABLE_LINE_RE.match(text[lines[j + 1][0]:lines[j + 1][1]]):
                j += 1
            if j > i:
                flush(a)
                blocks.append(NaturalBlock(a, lines[j][1], "table", cur_heading, cur_path))
                i = j + 1
                continue

        head = heading_of(line)
        if head:
            title, level = head
            flush(a)
            if _MD_HEADING_RE.match(line):
                while md_stack and md_stack[-1][0] >= level:
                    md_stack.pop()
                md_stack.append((level, title))
                cur_path = " > ".join(t for _, t in md_stack)
                cur_kind = "markdown"
            elif is_code:
                cur_path = title
                cur_kind = "code"
            else:
                while legal_stack and legal_stack[-1][0] >= level:
                    legal_stack.pop()
                legal_stack.append((level, title))
                cur_path = " > ".join(t for _, t in legal_stack)
                cur_kind = "section"
            cur_heading = title
            cur_start = a
            i += 1
            continue

        if cur_start is None and line.strip():
            cur_start = a
        i += 1

    flush(len(text))

    has_structure = any(blk.kind in ("section", "markdown") or (blk.kind == "code" and is_code) for blk in blocks)
    if has_structure:
        return blocks

    # No headings: split prose blocks into paragraphs (tables/fences stay atomic).
    out: list[NaturalBlock] = []
    for blk in blocks:
        if blk.kind != "paragraph":
            out.append(blk)
            continue
        pos = blk.start
        for m in _PARA_RE.finditer(text, blk.start, blk.end):
            if text[pos:m.start()].strip():
                out.append(NaturalBlock(pos, m.start(), "paragraph"))
            pos = m.end()
        if text[pos:blk.end].strip():
            out.append(NaturalBlock(pos, blk.end, "paragraph"))
    return out


def _merge_small(blocks: list[NaturalBlock], text: str, ceiling: int) -> list[NaturalBlock]:
    """Merge tiny blocks (or heading-only sections) with the next block of the same section."""
    merged: list[NaturalBlock] = []
    for blk in blocks:
        if merged:
            prev = merged[-1]
            prev_len = len(text[prev.start:prev.end].strip())
            heading_only = prev.kind in ("section", "markdown", "code") and prev_len <= len(prev.heading) + 2
            preamble = prev.kind == "paragraph" and not prev.section_path
            child = bool(prev.section_path) and blk.section_path.startswith(prev.section_path)
            same_section = prev.section_path == blk.section_path or heading_only or preamble or child
            if is_noise(text[prev.start:prev.end]) or is_noise(text[blk.start:blk.end]):
                same_section = False
            mergeable = ("paragraph", "section", "markdown", "code")
            if (
                prev_len < MIN_NATURAL_BLOCK
                and prev.kind in mergeable
                and blk.kind in mergeable
                and same_section
                and blk.end - prev.start <= ceiling
            ):
                merged[-1] = NaturalBlock(
                    prev.start,
                    blk.end,
                    blk.kind if prev.kind == "paragraph" else prev.kind,
                    blk.heading or prev.heading,
                    blk.section_path or prev.section_path,
                )
                continue
        merged.append(blk)
    return merged


def natural_spans(
    text: str,
    chunk_size: int | None = None,
    overlap: int = 0,
    is_law: bool = False,
) -> list[NaturalBlock]:
    """
    Return the final natural chunk spans for ``text``.

    With ``chunk_size=None`` every natural block becomes one chunk, and blocks over
    `NATURAL_CEILING` are subdivided. With a ``chunk_size``, adjacent blocks of the same
    section are packed together up to that size and larger blocks are subdivided at it.
    """
    ceiling = NATURAL_CEILING if chunk_size is None else max(1, int(chunk_size))
    blocks = _merge_small(detect_blocks(text, is_law=is_law), text, ceiling)

    if chunk_size is not None:
        packed: list[NaturalBlock] = []
        for blk in blocks:
            if packed:
                prev = packed[-1]
                if (
                    prev.section_path == blk.section_path
                    and prev.kind == blk.kind == "paragraph"
                    and blk.end - prev.start <= ceiling
                ):
                    packed[-1] = NaturalBlock(prev.start, blk.end, prev.kind, prev.heading, prev.section_path)
                    continue
            packed.append(blk)
        blocks = packed

    out: list[NaturalBlock] = []
    for blk in blocks:
        if blk.end - blk.start <= ceiling:
            out.append(blk)
            continue
        parts = safe_split_spans(text, ceiling, overlap=overlap, start=blk.start, end=blk.end)
        for p_idx, (a, b) in enumerate(parts, start=1):
            out.append(
                NaturalBlock(a, b, blk.kind, blk.heading, blk.section_path, {"part": p_idx, "parts": len(parts)})
            )
    return out


def micro_sentence_spans(text: str, start: int, end: int, size: int = MICRO_SENTENCE_SIZE) -> list[tuple[int, int]]:
    """Group whole sentences inside ``text[start:end]`` into windows of up to ``size`` chars (no overlap)."""
    spans: list[tuple[int, int]] = []
    sent_starts = [start] + [m.end(1) for m in _SENT_RE.finditer(text, start, end)]
    sent_bounds: list[tuple[int, int]] = []
    for i, s in enumerate(sent_starts):
        e = sent_starts[i + 1] if i + 1 < len(sent_starts) else end
        sent_bounds.append((s, e))

    cur_a: int | None = None
    cur_b = start
    for s, e in sent_bounds:
        if cur_a is not None and e - cur_a > size:
            spans.extend(safe_split_spans(text, size, start=cur_a, end=cur_b))
            cur_a = None
        if cur_a is None:
            cur_a = s
        cur_b = e
    if cur_a is not None:
        spans.extend(safe_split_spans(text, size, start=cur_a, end=cur_b))
    return spans


# ---------------------------------------------------------------------------
# TOC / gazette noise filter
# ---------------------------------------------------------------------------

_TOC_LINE_RE = re.compile(r"^.{2,160}?(?:\.{3,}|…{2,}|\s{3,}|\t+)\s*\d{1,4}\s*$")
_TOC_HEADING_RE = re.compile(r"^\s*(?:table\s+of\s+contents|contents|arrangement\s+of\s+sections|index)\s*$", re.IGNORECASE)
_GAZETTE_RE = re.compile(
    r"(?:registered\s+no\.?\s*[A-Z]{1,3}[\s—-]*\(?\w*|"
    r"\bEXTRAORDINARY\b|"
    r"published\s+by\s+authority|"
    r"\bgazette\s+of\s+india\b|"
    r"भारत\s+का\s+राजपत्र|"
    r"असाधारण|"
    r"controller\s+of\s+publications|"
    r"printed\s+by\s+the\s+(?:manager|director)|"
    r"government\s+of\s+india\s+press|"
    r"\bCG-[A-Z]{2}-[A-Z]-\d{3,}|"
    r"\bsl\.\s*no\.\s*\d+\b)",
    re.IGNORECASE,
)


def is_noise(text: str) -> bool:
    """True when ``text`` is a table of contents or gazette publication boilerplate."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    toc_lines = sum(1 for ln in lines if _TOC_LINE_RE.match(ln))
    if len(lines) >= 3 and toc_lines / len(lines) >= 0.6:
        return True
    if _TOC_HEADING_RE.match(lines[0]) and (len(lines) == 1 or toc_lines / max(1, len(lines) - 1) >= 0.4):
        return True
    gazette_lines = sum(1 for ln in lines if _GAZETTE_RE.search(ln))
    if gazette_lines and gazette_lines / len(lines) >= 0.5:
        return True
    if gazette_lines >= 2 and len(text) < 600:
        return True
    return False


def filter_noise(chunks: list[Any]) -> list[Any]:
    """Drop TOC and gazette boilerplate chunks (objects with ``.text`` or plain strings)."""
    kept = [c for c in chunks if not is_noise(getattr(c, "text", c) if not isinstance(c, str) else c)]
    return kept if kept else list(chunks)


__all__ = [
    "NATURAL_CEILING",
    "MIN_NATURAL_BLOCK",
    "MICRO_SENTENCE_SIZE",
    "NaturalBlock",
    "safe_split",
    "safe_split_spans",
    "detect_blocks",
    "natural_spans",
    "micro_sentence_spans",
    "is_noise",
    "filter_noise",
]
