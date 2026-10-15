"""Unstructured PDF text cleaning pipeline (ligature normalization, header/footer removal, de-hyphenation, line unwrap)."""

from __future__ import annotations

import re
from collections import Counter

# Common PDF ligatures & symbol normalizations
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
    "\u200b": "",
    "\r\n": "\n",
    "\r": "\n",
}

# Page number patterns that appear alone on a line
PAGE_NUMBER_LINE_RE = re.compile(
    r"^\s*(?:"
    r"page\s+\d+(?:\s+of\s+\d+)?"
    r"|-\s*\d+\s*-"
    r"|\[\s*\d+\s*\]"
    r"|\d+\s*/\s*\d+"
    r"|\d{1,4}"
    r")\s*$",
    re.IGNORECASE,
)

# Hyphenated word split across newline: e.g. "juris-\ndiction"
HYPHEN_BREAK_RE = re.compile(r"([a-zA-Z]{2,})-\n\s*([a-z]{2,})")

# Structural line starts that should NEVER be merged onto the previous line
STRUCTURAL_START_RE = re.compile(
    r"^\s*(?:"
    r"(?:PART|CHAPTER|ARTICLE|SECTION|SEC\.|SCHEDULE|ANNEXURE|CLAUSE)\s+[A-Z0-9IVXLCDM]+"
    r"|§+\s*\d+"
    r"|\d+\.\s+[A-Z]"
    r"|\(\d+\)\s+"
    r"|\([a-z]{1,3}\)\s+"
    r"|\([ivxlcdm]+\)\s+"
    r"|(?:Provided\s+(?:further\s+)?that|Explanation\b|Notwithstanding\b|WHEREAS\b|NOW\s+THEREFORE\b)"
    r"|[-*•]\s+"
    r"|#{1,6}\s+"
    r")",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """Replace PDF ligatures, non-breaking spaces, and smart quotes."""
    for src, dst in LIGATURE_MAP.items():
        text = text.replace(src, dst)
    return text


def detect_repeating_headers_footers(pages_raw: list[str], min_pages: int = 3) -> set[str]:
    """
    Detect identical header/footer lines that repeat across >= 50% of pages.
    Returns a set of normalized line strings to strip.
    """
    if len(pages_raw) < min_pages:
        return set()

    boundary_counter: Counter[str] = Counter()
    for page_text in pages_raw:
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        if not lines:
            continue
        candidates = set(lines[:2] + lines[-2:])
        for cand in candidates:
            # Normalize digits so "Case No. 123 - Page 1" and "Case No. 123 - Page 2" match
            norm = re.sub(r"\d+", "#", cand)
            if len(norm) >= 4:
                boundary_counter[norm] += 1

    threshold = max(2, int(len(pages_raw) * 0.5))
    return {norm for norm, count in boundary_counter.items() if count >= threshold}


def clean_page_text(
    raw_text: str,
    repeating_norms: set[str] | None = None,
    unwrap_lines: bool = True,
) -> str:
    """Clean a single page of unstructured/legal PDF text."""
    text = normalize_unicode(raw_text)
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)

    raw_lines = text.splitlines()
    filtered_lines: list[str] = []

    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue

        # Check top/bottom lines for page numbers or repeating headers/footers
        is_boundary = idx < 2 or idx >= len(raw_lines) - 2
        if is_boundary:
            if PAGE_NUMBER_LINE_RE.match(stripped):
                continue
            if repeating_norms:
                norm = re.sub(r"\d+", "#", stripped)
                if norm in repeating_norms:
                    continue

        # Collapse excessive horizontal whitespace inside a line
        cleaned_line = re.sub(r"[ \t]{2,}", " ", stripped)
        filtered_lines.append(cleaned_line)

    if not unwrap_lines:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(filtered_lines)).strip()

    # Smart line unwrapping: join soft wraps inside paragraphs while keeping structural boundaries intact
    merged_lines: list[str] = []
    for line in filtered_lines:
        if not line:
            if merged_lines and merged_lines[-1] != "":
                merged_lines.append("")
            continue

        if (
            not merged_lines
            or merged_lines[-1] == ""
            or STRUCTURAL_START_RE.match(line)
            or merged_lines[-1].endswith((":", ";", "—", ".-"))
            or merged_lines[-1].isupper()
        ):
            merged_lines.append(line)
        else:
            prev = merged_lines[-1]
            # If previous line does not end with terminal punctuation and current line looks like a continuation
            if not prev.endswith((".", "?", "!", '"', "'")) or (line[0].islower()):
                merged_lines[-1] = f"{prev} {line}"
            else:
                merged_lines.append(line)

    result = "\n".join(merged_lines)
    return re.sub(r"\n{3,}", "\n\n", result).strip()

