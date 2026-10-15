"""Data models for ragspark.parser — representing parsed unstructured & legal documents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TableBlock:
    """Represents a table extracted from a PDF page."""

    page_number: int
    headers: list[str]
    rows: list[list[str]]
    markdown: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageBlock:
    """Represents a single parsed page from a document."""

    page_number: int
    raw_text: str
    clean_text: str
    headings: list[str] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LegalNode:
    """Represents a hierarchical legal structural unit (Part, Chapter, Section, Clause, Proviso, etc.)."""

    id: str
    node_type: str  # PART, CHAPTER, ARTICLE, SECTION, SUBSECTION, CLAUSE, SUBCLAUSE, PROVISO, EXPLANATION, SCHEDULE
    number: str
    title: str
    text: str
    full_path: str
    page_number: int = 1
    parent_id: str | None = None
    children: list[LegalNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "number": self.number,
            "title": self.title,
            "text": self.text,
            "full_path": self.full_path,
            "page_number": self.page_number,
            "parent_id": self.parent_id,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class LegalCitation:
    """Represents a legal citation (case law, reporter, or statutory reference) extracted via Regex."""

    citation_type: str  # CASE_VERSUS, REPORTER_CITATION, STATUTE_REFERENCE, CONSTITUTIONAL_REF
    raw_match: str
    normalized: str
    context: str
    page_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LegalDefinition:
    """Represents a defined legal term extracted via Regex (e.g., '"Act" means...')."""

    term: str
    definition: str
    raw_match: str
    page_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObligationItem:
    """Represents a binding obligation, prohibition, or proviso extracted from legal text."""

    category: str  # MANDATORY_SHALL, PROHIBITION, EXCEPTION_NOTWITHSTANDING, PROVISO, CONDITION_SUBJECT_TO
    trigger_phrase: str
    statement: str
    section_path: str = ""
    page_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    """
    Unified output of `parser.parse(...)`.
    Works seamlessly for both unstructured PDFs and structured Legal PDFs,
    and feeds directly into future `ragspark.chunker`, `ragspark.embedder`, and `ragspark.rag` modules.
    """

    source: str
    doc_type: str  # "law_statute", "law_judgment", "law_contract", "unstructured"
    mode_used: str  # "law" or "unstructured"
    engine_used: str
    raw_text: str
    text: str
    pages: list[PageBlock] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    # Legal Regex Enrichment fields
    sections: list[LegalNode] = field(default_factory=list)
    flat_sections: list[LegalNode] = field(default_factory=list)
    citations: list[LegalCitation] = field(default_factory=list)
    definitions: list[LegalDefinition] = field(default_factory=list)
    obligations: list[ObligationItem] = field(default_factory=list)
    legal_metadata: dict[str, Any] = field(default_factory=dict)
    custom_matches: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def total_pages(self) -> int:
        """Alias for `num_pages` so `doc.total_pages` and `doc.num_pages` both work."""
        return len(self.pages)

    @property
    def clauses(self) -> list[LegalNode]:
        """Alias for `flat_sections` so `doc.clauses` and `doc.flat_sections` both work."""
        return self.flat_sections

    @property
    def articles(self) -> list[LegalNode]:
        """Alias for `flat_sections` for constitutional/statutory documents."""
        return self.flat_sections


    def to_markdown(self) -> str:
        """Render the parsed document into clean, RAG-ready Markdown with hierarchy & tables."""
        lines: list[str] = []
        title = self.metadata.get("title") or Path(self.source).stem
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"- **Document Type:** `{self.doc_type}`")
        lines.append(f"- **Pages:** {self.num_pages}")
        lines.append(f"- **Engine:** `{self.engine_used}`")

        if self.legal_metadata:
            if self.legal_metadata.get("court"):
                lines.append(f"- **Court:** {self.legal_metadata['court']}")
            if self.legal_metadata.get("case_numbers"):
                lines.append(f"- **Case No:** {', '.join(self.legal_metadata['case_numbers'])}")
            if self.legal_metadata.get("parties"):
                lines.append(f"- **Parties:** {', '.join(self.legal_metadata['parties'])}")
            if self.legal_metadata.get("dates"):
                lines.append(f"- **Dates:** {', '.join(self.legal_metadata['dates'][:5])}")

        lines.append("\n---\n")

        if self.flat_sections:
            lines.append("## Legal Structure & Clauses\n")
            for node in self.flat_sections:
                header_prefix = "###" if node.node_type in ("PART", "CHAPTER", "ARTICLE", "SECTION") else "####"
                heading_label = f"{node.node_type} {node.number}".strip()
                if node.title:
                    heading_label += f": {node.title}"
                lines.append(f"{header_prefix} {heading_label} `[{node.full_path}]`")
                if node.text:
                    lines.append(node.text)
                lines.append("")
        else:
            lines.append("## Document Content\n")
            lines.append(self.text)
            lines.append("")

        if self.tables:
            lines.append("## Extracted Tables\n")
            for idx, tbl in enumerate(self.tables, 1):
                lines.append(f"### Table {idx} (Page {tbl.page_number})\n")
                lines.append(tbl.markdown)
                lines.append("")

        if self.definitions:
            lines.append("## Extracted Legal Definitions\n")
            for d in self.definitions:
                lines.append(f"- **{d.term}**: {d.definition}")
            lines.append("")

        if self.citations:
            lines.append("## Extracted Legal Citations\n")
            for c in self.citations:
                lines.append(f"- `[{c.citation_type}]` **{c.normalized}** (Page {c.page_number})")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire ParsedDocument into a Python dictionary."""
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "mode_used": self.mode_used,
            "engine_used": self.engine_used,
            "num_pages": self.num_pages,
            "text": self.text,
            "headings": self.headings,
            "pages": [p.to_dict() for p in self.pages],
            "tables": [t.to_dict() for t in self.tables],
            "sections": [s.to_dict() for s in self.sections],
            "flat_sections": [s.to_dict() for s in self.flat_sections],
            "citations": [c.to_dict() for c in self.citations],
            "definitions": [d.to_dict() for d in self.definitions],
            "obligations": [o.to_dict() for o in self.obligations],
            "legal_metadata": self.legal_metadata,
            "custom_matches": self.custom_matches,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the entire ParsedDocument into a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, output_path: str | Path) -> Path:
        """Save the parsed document as `.json` or `.md` based on file extension."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in (".md", ".markdown"):
            out.write_text(self.to_markdown(), encoding="utf-8")
        else:
            out.write_text(self.to_json(indent=2), encoding="utf-8")
        return out

    def summary(self) -> dict[str, Any]:
        """Return a quick summary of what was extracted."""
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "mode_used": self.mode_used,
            "engine_used": self.engine_used,
            "pages": self.num_pages,
            "characters": len(self.text),
            "headings_found": len(self.headings),
            "tables_found": len(self.tables),
            "legal_sections_found": len(self.flat_sections),
            "citations_found": len(self.citations),
            "definitions_found": len(self.definitions),
            "obligations_found": len(self.obligations),
        }

