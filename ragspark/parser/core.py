"""Core orchestration for `ragspark.parser` — uniting Unstructured PDF extraction and Legal Regex Intelligence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ragspark.parser.cleaner import clean_page_text, detect_repeating_headers_footers
from ragspark.parser.engines import extract_pdf_hybrid
from ragspark.parser.legal_regex import (
    clear_custom_regexes,
    detect_legal_document_type,
    extract_legal_citations,
    extract_legal_definitions,
    extract_legal_hierarchy,
    extract_legal_metadata,
    extract_legal_obligations,
    register_custom_regex,
    run_custom_regexes,
)
from ragspark.parser.models import PageBlock, ParsedDocument, TableBlock


class SmartParser:
    """
    Unified Smart Parser for Unstructured PDFs & Legal Documents.
    Accessible directly via `from ragspark import parser`.
    """

    def parse(
        self,
        source: str | Path,
        mode: str = "auto",
        extract_tables: bool | str = False,
        unwrap_lines: bool = True,
        custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
        ocr: bool | str = "auto",
        api_key: str | None = None,
    ) -> ParsedDocument:
        """
        One-command smart parser for PDFs (and .txt/.md files) with Smart Per-Page Hybrid OCR fallback.

        Args:
            source: Path to the PDF (or text/markdown) file.
            mode: "auto" (auto-detects law vs unstructured), "law" (forces Legal Regex engine),
                  or "unstructured" (general unstructured extraction only).
            extract_tables: False (default, blazing fast <0.2s), True/"fast" (PyMuPDF C-level tables),
                  or "deep" (pdfplumber full page inspection).
            unwrap_lines: Whether to unwrap broken mid-sentence lines while preserving structure.
            custom_regexes: Optional dictionary of `{name: regex_pattern}` to extract custom matches.
            ocr: "auto" (default — runs OCR only on scanned/image-only pages), "force" (runs OCR on all pages),
                 "tesseract", "rapidocr", "gemini_vision", or False.
            api_key: Optional Gemini API key (`AIza...` or `AQ....`) for multimodal Gemini Vision OCR fallback.

        Returns:
            ParsedDocument with clean text, Markdown, tables, and (if legal) hierarchical sections,
            citations, definitions, obligations, and legal metadata.
        """
        if isinstance(source, str) and ("\n" in source or len(source) > 260):
            return self.parse_text(
                text=source,
                mode=mode,
                unwrap_lines=unwrap_lines,
                custom_regexes=custom_regexes,
            )

        path = Path(source)
        if path.suffix.lower() != ".pdf" and path.exists():
            text_content = path.read_text(encoding="utf-8", errors="ignore")
            parsed = self.parse_text(
                text=text_content,
                mode=mode,
                unwrap_lines=unwrap_lines,
                custom_regexes=custom_regexes,
            )
            parsed.source = str(path)
            return parsed

        pages_raw_data, tables_by_page, doc_meta, engine_used = extract_pdf_hybrid(
            path, extract_tables=extract_tables, ocr=ocr, api_key=api_key
        )


        raw_page_strings = [p["raw_text"] for p in pages_raw_data]
        repeating_norms = detect_repeating_headers_footers(raw_page_strings)

        pages: list[PageBlock] = []
        all_headings: list[str] = []
        all_tables: list[TableBlock] = []

        for p_dict in pages_raw_data:
            pno = p_dict["page_number"]
            raw_t = p_dict["raw_text"]
            clean_t = clean_page_text(
                raw_t,
                repeating_norms=repeating_norms,
                unwrap_lines=unwrap_lines,
            )
            p_tables = tables_by_page.get(pno, [])
            p_headings = p_dict.get("headings", [])

            for h in p_headings:
                if h not in all_headings:
                    all_headings.append(h)
            all_tables.extend(p_tables)

            pages.append(
                PageBlock(
                    page_number=pno,
                    raw_text=raw_t,
                    clean_text=clean_t,
                    headings=p_headings,
                    tables=p_tables,
                )
            )

        full_raw_text = "\n\n".join(raw_page_strings)
        full_clean_text = "\n\n".join(p.clean_text for p in pages if p.clean_text)

        return self._build_parsed_document(
            source=str(path),
            pages=pages,
            full_raw_text=full_raw_text,
            full_clean_text=full_clean_text,
            all_headings=all_headings,
            all_tables=all_tables,
            doc_meta=doc_meta,
            engine_used=engine_used,
            mode=mode,
            custom_regexes=custom_regexes,
        )

    def parse_law(
        self,
        source: str | Path,
        extract_tables: bool | str = False,
        unwrap_lines: bool = True,
        custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
        ocr: bool | str = "auto",
        api_key: str | None = None,
    ) -> ParsedDocument:
        """Explicitly parse a PDF or text file using the full Legal Regex Intelligence pipeline + Hybrid OCR."""
        return self.parse(
            source=source,
            mode="law",
            extract_tables=extract_tables,
            unwrap_lines=unwrap_lines,
            custom_regexes=custom_regexes,
            ocr=ocr,
            api_key=api_key,
        )

    def parse_unstructured(
        self,
        source: str | Path,
        extract_tables: bool | str = False,
        unwrap_lines: bool = True,
        custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
        ocr: bool | str = "auto",
        api_key: str | None = None,
    ) -> ParsedDocument:
        """Explicitly parse a PDF or text file using only the Unstructured Layout & Table pipeline + Hybrid OCR."""
        return self.parse(
            source=source,
            mode="unstructured",
            extract_tables=extract_tables,
            unwrap_lines=unwrap_lines,
            custom_regexes=custom_regexes,
            ocr=ocr,
            api_key=api_key,
        )

    def parse_text(
        self,
        text: str,
        mode: str = "auto",
        source_name: str = "inline_text",
        unwrap_lines: bool = True,
        custom_regexes: dict[str, str | re.Pattern[str]] | None = None,
    ) -> ParsedDocument:
        """Parse raw text directly through the Unstructured Cleaner + Legal Regex pipeline."""
        clean_t = clean_page_text(text, repeating_norms=set(), unwrap_lines=unwrap_lines)
        page = PageBlock(
            page_number=1,
            raw_text=text,
            clean_text=clean_t,
            headings=[],
            tables=[],
        )
        return self._build_parsed_document(
            source=source_name,
            pages=[page],
            full_raw_text=text,
            full_clean_text=clean_t,
            all_headings=[],
            all_tables=[],
            doc_meta={"title": source_name, "page_count": 1},
            engine_used="inline_text",
            mode=mode,
            custom_regexes=custom_regexes,
        )

    def parse_dir(
        self,
        directory: str | Path,
        mode: str = "auto",
        pattern: str = "*.pdf",
        extract_tables: bool = True,
    ) -> list[ParsedDocument]:
        """Batch-parse all matching files in a directory in one command."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        results: list[ParsedDocument] = []
        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file():
                results.append(
                    self.parse(file_path, mode=mode, extract_tables=extract_tables)
                )
        return results

    def register_regex(self, name: str, pattern: str | re.Pattern[str], flags: int = 0) -> None:
        """Register a custom named regex pattern to extract from all parsed documents."""
        register_custom_regex(name, pattern, flags)

    def clear_regexes(self) -> None:
        """Clear all user-registered custom regexes."""
        clear_custom_regexes()

    def _build_parsed_document(
        self,
        source: str,
        pages: list[PageBlock],
        full_raw_text: str,
        full_clean_text: str,
        all_headings: list[str],
        all_tables: list[TableBlock],
        doc_meta: dict[str, Any],
        engine_used: str,
        mode: str,
        custom_regexes: dict[str, str | re.Pattern[str]] | None,
    ) -> ParsedDocument:
        detected_type = detect_legal_document_type(full_clean_text)

        if mode.lower() == "law":
            effective_mode = "law"
            doc_type = detected_type if detected_type.startswith("law_") else "law_general"
        elif mode.lower() == "unstructured":
            effective_mode = "unstructured"
            doc_type = "unstructured"
        else:
            # Auto mode
            effective_mode = "law" if detected_type.startswith("law_") else "unstructured"
            doc_type = detected_type

        sections = []
        flat_sections = []
        citations = []
        definitions = []
        obligations = []
        legal_meta: dict[str, Any] = {}

        if effective_mode == "law":
            sections, flat_sections = extract_legal_hierarchy(pages)
            citations = extract_legal_citations(pages)
            definitions = extract_legal_definitions(pages)
            obligations = extract_legal_obligations(pages, flat_sections)
            legal_meta = extract_legal_metadata(full_clean_text)
            engine_used = f"{engine_used}+legal_regex"

        custom_matches = run_custom_regexes(full_clean_text, extra_patterns=custom_regexes)

        global _LAST_PARSED_DOC
        doc_obj = ParsedDocument(
            source=source,
            doc_type=doc_type,
            mode_used=effective_mode,
            engine_used=engine_used,
            raw_text=full_raw_text,
            text=full_clean_text,
            pages=pages,
            headings=all_headings,
            tables=all_tables,
            sections=sections,
            flat_sections=flat_sections,
            citations=citations,
            definitions=definitions,
            obligations=obligations,
            legal_metadata=legal_meta,
            custom_matches=custom_matches,
            metadata=doc_meta,
        )
        _LAST_PARSED_DOC = doc_obj
        return doc_obj


_LAST_PARSED_DOC: ParsedDocument | None = None

# Singleton instance backing `from ragspark import parser`
default_parser = SmartParser()


