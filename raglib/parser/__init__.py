"""
`raglib.parser` — Phase 1 Smart Unstructured & Legal Regex PDF Parser.

Usage:
    from raglib import parser

    doc = parser.parse("contract_or_statute.pdf")       # Auto-detects law vs unstructured
    law_doc = parser.parse_law("judgment.pdf")          # Forces Legal Regex Intelligence
    raw_doc = parser.parse_unstructured("report.pdf")   # Unstructured layout + table extraction
"""

from __future__ import annotations

import re
from pathlib import Path

from raglib.parser.core import SmartParser, default_parser
from raglib.parser.models import (
    LegalCitation,
    LegalDefinition,
    LegalNode,
    ObligationItem,
    PageBlock,
    ParsedDocument,
    TableBlock,
)


def parse(
    source: str | Path,
    mode: str = "auto",
    extract_tables: bool | str = False,
    unwrap_lines: bool = True,
    custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
    ocr: bool | str = "auto",
    api_key: str | None = None,
) -> ParsedDocument:
    """One-command smart parser for unstructured & legal PDFs (blazing fast C-level extraction + Per-Page Hybrid OCR)."""
    return default_parser.parse(
        source=source,
        mode=mode,
        extract_tables=extract_tables,
        unwrap_lines=unwrap_lines,
        custom_regexes=custom_regexes,
        ocr=ocr,
        api_key=api_key,
    )


def parse_law(
    source: str | Path,
    extract_tables: bool | str = False,
    unwrap_lines: bool = True,
    custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
    ocr: bool | str = "auto",
    api_key: str | None = None,
) -> ParsedDocument:
    """One-command legal PDF parser using Hierarchical & Citation Regex Intelligence + Per-Page Hybrid OCR."""
    return default_parser.parse_law(
        source=source,
        extract_tables=extract_tables,
        unwrap_lines=unwrap_lines,
        custom_regexes=custom_regexes,
        ocr=ocr,
        api_key=api_key,
    )


def parse_unstructured(
    source: str | Path,
    extract_tables: bool | str = False,
    unwrap_lines: bool = True,
    custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
    ocr: bool | str = "auto",
    api_key: str | None = None,
) -> ParsedDocument:
    """One-command unstructured PDF parser (layout + headings + text cleaner + Per-Page Hybrid OCR)."""
    return default_parser.parse_unstructured(
        source=source,
        extract_tables=extract_tables,
        unwrap_lines=unwrap_lines,
        custom_regexes=custom_regexes,
        ocr=ocr,
        api_key=api_key,
    )


def parse_text(
    text: str,
    mode: str = "auto",
    source_name: str = "inline_text",
    unwrap_lines: bool = True,
    custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
) -> ParsedDocument:
    """One-command raw text parser using Unstructured Cleaner + Legal Regex Intelligence."""
    return default_parser.parse_text(
        text=text,
        mode=mode,
        source_name=source_name,
        unwrap_lines=unwrap_lines,
        custom_regexes=custom_regexes,
    )


def parse_dir(
    directory: str | Path,
    mode: str = "auto",
    pattern: str = "*.pdf",
    extract_tables: bool = True,
) -> list[ParsedDocument]:
    """Batch-parse an entire folder of PDFs in one command."""
    return default_parser.parse_dir(
        directory=directory,
        mode=mode,
        pattern=pattern,
        extract_tables=extract_tables,
    )


def register_regex(name: str, pattern: str | re.Pattern[str], flags: int = 0) -> None:
    """Register a custom named regex pattern to extract from documents."""
    default_parser.register_regex(name=name, pattern=pattern, flags=flags)


def clear_regexes() -> None:
    """Clear all custom registered regex patterns."""
    default_parser.clear_regexes()


__all__ = [
    "SmartParser",
    "parse",
    "parse_law",
    "parse_unstructured",
    "parse_text",
    "parse_dir",
    "register_regex",
    "clear_regexes",
    "ParsedDocument",
    "PageBlock",
    "TableBlock",
    "LegalNode",
    "LegalCitation",
    "LegalDefinition",
    "ObligationItem",
]

