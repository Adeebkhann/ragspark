"""
Legal Regex Intelligence Engine for raglib.parser.
Extracts legal hierarchy (Parts, Chapters, Articles, Sections, Subsections, Clauses, Provisos),
Case & Statute Citations, Legal Definitions, Obligations/Prohibitions, and Court/Contract Metadata.
"""

from __future__ import annotations

import re
from typing import Any

from raglib.parser.models import (
    LegalCitation,
    LegalDefinition,
    LegalNode,
    ObligationItem,
    PageBlock,
)

# ---------------------------------------------------------------------------
# 1. HIERARCHICAL LEGAL STRUCTURE REGEX PATTERNS
# ---------------------------------------------------------------------------

# Hierarchy rank (lower number = higher in the tree)
LEVEL_RANK = {
    "PART": 1,
    "CHAPTER": 2,
    "SCHEDULE": 2,
    "ARTICLE": 3,
    "SECTION": 3,
    "CLAUSE": 3,
    "SUBSECTION": 4,
    "SUBCLAUSE": 5,
    "ROMAN_CLAUSE": 6,
    "PROVISO": 7,
    "EXPLANATION": 7,
}

# Ordered pattern matchers for line-start structural boundaries
HIERARCHY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "PART",
        re.compile(
            r"^\s*PART\s+([IVXLCDM0-9]+[A-Z]?)(?:\s*[-:—.]*\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    (
        "CHAPTER",
        re.compile(
            r"^\s*CHAPTER\s+([IVXLCDM0-9A-Z]+)(?:\s*[-:—.]*\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    (
        "SCHEDULE",
        re.compile(
            r"^\s*(?:THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|[IVXLCDM0-9]+)?\s*SCHEDULE\b(?:\s*[-:—.]*\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    (
        "ARTICLE",
        re.compile(
            r"^\s*(?:ARTICLE|ART\.)\s*([0-9]+[A-Z]?(?:\([0-9a-zA-Z]+\))*)(?:\s*[-:—.]\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    (
        "SECTION",
        re.compile(
            r"^\s*(?:SECTION|SEC\.|§+)\s*([0-9]+[A-Z]?(?:\([0-9a-zA-Z]+\))*)(?:\s*[-:—.]\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    (
        "CLAUSE",
        re.compile(
            r"^\s*CLAUSE\s+([0-9]+(?:\.[0-9]+)*[A-Z]?)(?:\s*[-:—.]\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    # Numbered Article / Statutory Section style used in the Constitution of India PDF:
    # e.g. "14. Equality before law.—The State shall not..." or "21A. Right to education.—The State..."
    (
        "ARTICLE",
        re.compile(
            r"^\s*(\d{1,3}[A-Z]?(?:\.\d{1,3})?)\.\s+([A-Z][^\n]{2,110}?)(?:\.—|\.-|—|–|:-|\.\s+|\s+-\s+)(.*)$"
        ),
    ),
    (
        "PROVISO",
        re.compile(
            r"^\s*(Provided\s+(?:further\s+|also\s+)?that)\b\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "EXPLANATION",
        re.compile(
            r"^\s*(Explanation(?:\s*[0-9IVX]+)?)\s*(?:\.—|—|:|-|\.)\s*(.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "SUBSECTION",
        re.compile(r"^\s*\((\d{1,3})\)\s+(.*)$"),
    ),
    (
        "ROMAN_CLAUSE",
        re.compile(r"^\s*\(([ivxlcdm]{1,6})\)\s+(.*)$", re.IGNORECASE),
    ),
    (
        "SUBCLAUSE",
        re.compile(r"^\s*\(([a-z]{1,2})\)\s+(.*)$"),
    ),
]

# Roman numerals set to disambiguate `(i)`, `(v)`, `(x)` between SUBCLAUSE and ROMAN_CLAUSE
ROMAN_SET = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}


# ---------------------------------------------------------------------------
# 2. LEGAL CITATION REGEX PATTERNS
# ---------------------------------------------------------------------------

# Case Versus: e.g., "Kesavananda Bharati v. State of Kerala", "Roe v. Wade", "Alpha Corp. vs. Beta LLC"
CASE_VERSUS_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.,'\- ]{2,45}?)\s+(?:v\.|vs\.|versus)\s+([A-Z][A-Za-z0-9&.,'\- ]{2,45}?)(?=[,;.\)\n]|\s+\(|\s+AIR\b|\s+\d{4}\b|\s+on\b|$)"
)

# Reporter Citations: US Reports, SCC, AIR, INSC, UKSC, F.2d/F.3d, All ER
REPORTER_CITATION_RE = re.compile(
    r"(?:"
    r"\bAIR\s+\d{4}\s+[A-Z]{2,6}\s+\d+\b"
    r"|\(\d{4}\)\s+\d+\s+SCC\s+(?:\([A-Za-z]+\)\s+)?\d+\b"
    r"|\b\d{4}\s+SCC\s+OnLine\s+[A-Z]+\s+\d+\b"
    r"|\b\d{4}\s+INSC\s+\d+\b"
    r"|\b\d{1,3}\s+U\.?\s*S\.?\s+\d{1,4}(?:\s*\(\d{4}\))?"
    r"|\b\d{1,3}\s+F\.\s*(?:2d|3d|4th|Supp\.)\s+\d{1,4}\b"
    r"|\[\d{4}\]\s+(?:UKSC|EWCA|EWHC|All\s+ER)\s+\d+\b"
    r")"
)

# Statutory Reference: e.g., "Section 12(1)(a) of the Companies Act, 2013", "Sections 302 and 34 of the Indian Penal Code"
STATUTE_REF_RE = re.compile(
    r"\b((?:Sections?|Secs?\.|Articles?|Rules?|Orders?|Clauses?)\s+\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*"
    r"(?:\s*(?:,|and|read\s+with|r/w|to)\s*(?:Sections?|Secs?\.|Articles?|Rules?)?\s*\d+[A-Za-z]?(?:\([0-9a-zA-Z]+\))*)*"
    r"\s+of\s+the\s+[A-Z][A-Za-z0-9 ,\-()]{3,70}?(?:Act|Code|Constitution|Rules|Regulations|Ordinance)(?:,\s*\d{4})?)"
)

# Constitutional Reference: e.g., "Article 21 of the Constitution of India", "Constitution (Forty-second Amendment) Act, 1976"
CONSTITUTIONAL_REF_RE = re.compile(
    r"\b(?:"
    r"Articles?\s+\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?(?:\s+of\s+the\s+Constitution(?:\s+of\s+India)?)?"
    r"|Constitution\s*\([A-Za-z0-9\-\s]+Amendment\)\s*Act,\s*\d{4}"
    r"|(?:First|Second|Fourth|Fifth|Sixth|Eighth|Fourteenth|Forty-second|Forty-fourth|Seventy-third|Eighty-sixth|One\s+Hundred\s+and\s+[A-Za-z]+)\s+Amendment"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 3. DEFINED TERMS & OBLIGATIONS REGEX PATTERNS
# ---------------------------------------------------------------------------

# Legal Definitions: e.g., '"Confidential Information" means ...', '(a) "Act" shall mean ...'
LEGAL_DEFINITION_RE = re.compile(
    r'["“]([A-Z][A-Za-z0-9 /\-]{1,50})["”]\s+(?:shall\s+)?(?:mean|means|includes|has\s+the\s+meaning)\s+([^.;\n]{10,350}[.;])'
)

# Hereinafter referred to as: e.g., 'ABC Private Limited (hereinafter referred to as the "Company")'
HEREINAFTER_DEF_RE = re.compile(
    r'([A-Z][A-Za-z0-9 &.,\-]{3,80}?)\s*\(\s*(?:hereinafter\s+)?(?:referred\s+to\s+as|called)\s+(?:the\s+)?["“]([A-Za-z0-9 /\-]{2,40})["”]\s*\)'
)

# Obligation / Prohibition / Proviso / Non-obstante sentence triggers
OBLIGATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EXCEPTION_NOTWITHSTANDING", re.compile(r"\b(Notwithstanding\s+anything\s+contained\s+in\b[^.]{10,300}\.)", re.IGNORECASE)),
    ("PROVISO", re.compile(r"\b(Provided\s+(?:further\s+)?that\b[^.]{10,300}\.)", re.IGNORECASE)),
    ("CONDITION_SUBJECT_TO", re.compile(r"\b(Subject\s+to\s+the\s+provisions\s+of\b[^.]{10,300}\.)", re.IGNORECASE)),
    ("PROHIBITION", re.compile(r"([^.\n]{3,120}\b(?:shall\s+not|must\s+not|is\s+prohibited\s+from|no\s+person\s+shall)\b[^.\n]{5,220}\.)", re.IGNORECASE)),
    ("MANDATORY_SHALL", re.compile(r"([^.\n]{3,120}\b(?:shall|must)\b(?!\s+not\b)[^.\n]{5,220}\.)")),
]


# ---------------------------------------------------------------------------
# 4. METADATA REGEX PATTERNS (COURTS, CASE NO, DATES, PARTIES)
# ---------------------------------------------------------------------------

COURT_HEADER_RE = re.compile(
    r"\b(IN\s+THE\s+(?:SUPREME\s+COURT|HIGH\s+COURT|DISTRICT\s+COURT|UNITED\s+STATES\s+(?:SUPREME|DISTRICT)\s+COURT|COURT\s+OF\s+APPEALS)[^\n]{0,80})",
    re.IGNORECASE,
)

CASE_NUMBER_RE = re.compile(
    r"\b((?:CIVIL|CRIMINAL|WRIT|SPECIAL\s+LEAVE|ARBITRATION|COMPANY)\s+(?:APPEAL|PETITION|APPLICATION|SUIT)\s+(?:(?:\([A-Z]+\)\s+)?NO\.?\s*\d+(?:\s*(?:OF|/|-)\s*\d{2,4})?))",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)

ACT_TITLE_RE = re.compile(
    r"\b(THE\s+[A-Z][A-Z\s,&\-]{3,65}\s+ACT,\s*\d{4})\b"
)


# ---------------------------------------------------------------------------
# 5. CUSTOM USER REGEX REGISTRY
# ---------------------------------------------------------------------------

CUSTOM_REGEX_REGISTRY: dict[str, re.Pattern[str]] = {}


def register_custom_regex(name: str, pattern: str | re.Pattern[str], flags: int = 0) -> None:
    """Register a custom named regex pattern to run automatically during legal/unstructured parsing."""
    if isinstance(pattern, re.Pattern):
        CUSTOM_REGEX_REGISTRY[name] = pattern
    else:
        CUSTOM_REGEX_REGISTRY[name] = re.compile(pattern, flags)


def clear_custom_regexes() -> None:
    """Clear all user-registered custom regexes."""
    CUSTOM_REGEX_REGISTRY.clear()


# ---------------------------------------------------------------------------
# 6. EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------

def detect_legal_document_type(text: str) -> str:
    """
    Auto-detect if a document is a Legal Statute/Act, Court Judgment, Legal Contract, or general Unstructured document.
    """
    sample = text[:15000]
    judgment_Hits = len(COURT_HEADER_RE.findall(sample)) + len(CASE_NUMBER_RE.findall(sample)) + len(CASE_VERSUS_RE.findall(sample))
    statute_hits = len(re.findall(r"\b(?:BE\s+it\s+enacted|Short\s+title,?\s+extent\s+and\s+commencement|Provided\s+that|Explanation\.—|Section\s+\d+)\b", sample, re.IGNORECASE))
    contract_hits = len(re.findall(r"\b(?:WHEREAS|NOW,?\s+THEREFORE|IN\s+WITNESS\s+WHEREOF| Indemnif|Confidential\s+Information|Governing\s+Law|Force\s+Majeure)\b", sample, re.IGNORECASE))

    if judgment_Hits >= 2:
        return "law_judgment"
    if statute_hits >= 2:
        return "law_statute"
    if contract_hits >= 2:
        return "law_contract"
    if (judgment_Hits + statute_hits + contract_hits) >= 2:
        return "law_general"
    return "unstructured"


def extract_legal_hierarchy(pages: list[PageBlock]) -> tuple[list[LegalNode], list[LegalNode]]:
    """
    Parse pages line-by-line using hierarchical legal regexes.
    Returns (root_tree_nodes, flat_nodes_list).
    """
    root_nodes: list[LegalNode] = []
    flat_nodes: list[LegalNode] = []
    # Active ancestor stack: list of (rank, LegalNode)
    stack: list[tuple[int, LegalNode]] = []
    node_counter = 0

    for page in pages:
        lines = page.clean_text.splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            matched_type: str | None = None
            matched_num: str = ""
            matched_title: str = ""
            matched_body: str = ""

            for node_type, pattern in HIERARCHY_PATTERNS:
                m = pattern.match(line)
                if not m:
                    continue

                groups = m.groups()
                if node_type in ("PART", "CHAPTER", "CLAUSE"):
                    matched_type = node_type
                    matched_num = (groups[0] or "").strip()
                    rest = (groups[1] or "").strip()
                    matched_title = rest
                elif node_type == "SCHEDULE":
                    matched_type = "SCHEDULE"
                    matched_num = (groups[0] or "I").strip()
                    matched_title = (groups[1] or "").strip()
                elif node_type in ("SECTION", "ARTICLE"):
                    matched_type = node_type
                    if len(groups) == 3:
                        # Numbered Article/Section with title and inline body: "14. Equality before law.—The State..."
                        matched_num = groups[0].strip()
                        matched_title = groups[1].strip()
                        matched_body = groups[2].strip()
                    else:
                        matched_num = (groups[0] or "").strip()
                        rest = (groups[1] or "").strip()
                        if "—" in rest:
                            t_part, b_part = rest.split("—", 1)
                            matched_title = t_part.strip(" .-:")
                            matched_body = b_part.strip()
                        else:
                            matched_title = rest
                elif node_type in ("PROVISO", "EXPLANATION"):
                    matched_type = node_type
                    matched_num = groups[0].strip()
                    matched_body = (groups[1] or "").strip()
                elif node_type == "SUBSECTION":
                    matched_type = "SUBSECTION"
                    matched_num = f"({groups[0].strip()})"
                    matched_body = (groups[1] or "").strip()
                elif node_type == "ROMAN_CLAUSE":
                    roman_val = groups[0].strip().lower()
                    if roman_val not in ROMAN_SET:
                        continue
                    matched_type = "ROMAN_CLAUSE"
                    matched_num = f"({groups[0].strip()})"
                    matched_body = (groups[1] or "").strip()
                elif node_type == "SUBCLAUSE":
                    matched_type = "SUBCLAUSE"
                    matched_num = f"({groups[0].strip()})"
                    matched_body = (groups[1] or "").strip()

                if matched_type:
                    break

            if matched_type:
                node_counter += 1
                rank = LEVEL_RANK.get(matched_type, 5)

                while stack and stack[-1][0] >= rank:
                    stack.pop()

                parent_node = stack[-1][1] if stack else None
                label_piece = (
                    f"{matched_type.title()} {matched_num}".strip()
                    if matched_type in ("PART", "CHAPTER", "ARTICLE", "SECTION", "CLAUSE", "SCHEDULE")
                    else matched_num
                )
                full_path = f"{parent_node.full_path} > {label_piece}" if parent_node else label_piece

                new_node = LegalNode(
                    id=f"node_{node_counter}",
                    node_type=matched_type,
                    number=matched_num,
                    title=matched_title,
                    text=matched_body,
                    full_path=full_path,
                    page_number=page.page_number,
                    parent_id=parent_node.id if parent_node else None,
                )

                if parent_node:
                    parent_node.children.append(new_node)
                else:
                    root_nodes.append(new_node)

                stack.append((rank, new_node))
                flat_nodes.append(new_node)

                # Check if an inline clause/subsection e.g. "(1) All citizens..." starts inside matched_body
                if matched_type in ("SECTION", "ARTICLE") and matched_body:
                    sub_m = re.match(r"^\((\d{1,3})\)\s+(.*)$", matched_body)
                    if sub_m:
                        new_node.text = ""
                        node_counter += 1
                        sub_num = f"({sub_m.group(1)})"
                        sub_node = LegalNode(
                            id=f"node_{node_counter}",
                            node_type="SUBSECTION",
                            number=sub_num,
                            title="",
                            text=sub_m.group(2).strip(),
                            full_path=f"{new_node.full_path} > {sub_num}",
                            page_number=page.page_number,
                            parent_id=new_node.id,
                        )
                        new_node.children.append(sub_node)
                        stack.append((LEVEL_RANK["SUBSECTION"], sub_node))
                        flat_nodes.append(sub_node)
            else:
                # Continuation line: append to the currently active node on the stack
                if stack:
                    active_node = stack[-1][1]
                    if active_node.text:
                        active_node.text = f"{active_node.text}\n{line}"
                    else:
                        active_node.text = line

    return root_nodes, flat_nodes


def extract_legal_citations(pages: list[PageBlock]) -> list[LegalCitation]:
    """Extract Case Versus citations, Law Reporter citations, Statutory references, and Constitutional references."""
    citations: list[LegalCitation] = []
    seen: set[tuple[str, str]] = set()

    for page in pages:
        text = page.clean_text

        # 1. Case Versus (e.g., X v. Y)
        for m in CASE_VERSUS_RE.finditer(text):
            raw = m.group(0).strip(" ,;.")
            norm = f"{m.group(1).strip()} v. {m.group(2).strip()}"
            key = ("CASE_VERSUS", norm.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 60)
                citations.append(
                    LegalCitation(
                        citation_type="CASE_VERSUS",
                        raw_match=raw,
                        normalized=norm,
                        context=text[start:end].replace("\n", " ").strip(),
                        page_number=page.page_number,
                    )
                )

        # 2. Reporter Citations (SCC, AIR, US, INSC, UKSC, etc.)
        for m in REPORTER_CITATION_RE.finditer(text):
            raw = m.group(0).strip()
            norm = re.sub(r"\s+", " ", raw)
            key = ("REPORTER_CITATION", norm.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 60)
                citations.append(
                    LegalCitation(
                        citation_type="REPORTER_CITATION",
                        raw_match=raw,
                        normalized=norm,
                        context=text[start:end].replace("\n", " ").strip(),
                        page_number=page.page_number,
                    )
                )

        # 3. Statutory References ("Section 14 of the ... Act, 2013")
        for m in STATUTE_REF_RE.finditer(text):
            raw = m.group(1).strip()
            norm = re.sub(r"\s+", " ", raw)
            key = ("STATUTE_REFERENCE", norm.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 50)
                citations.append(
                    LegalCitation(
                        citation_type="STATUTE_REFERENCE",
                        raw_match=raw,
                        normalized=norm,
                        context=text[start:end].replace("\n", " ").strip(),
                        page_number=page.page_number,
                    )
                )

        # 4. Constitutional References ("Article 21 of the Constitution")
        for m in CONSTITUTIONAL_REF_RE.finditer(text):
            raw = m.group(0).strip()
            norm = re.sub(r"\s+", " ", raw)
            key = ("CONSTITUTIONAL_REF", norm.lower())
            if key not in seen:
                seen.add(key)
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 50)
                citations.append(
                    LegalCitation(
                        citation_type="CONSTITUTIONAL_REF",
                        raw_match=raw,
                        normalized=norm,
                        context=text[start:end].replace("\n", " ").strip(),
                        page_number=page.page_number,
                    )
                )

    return citations


def extract_legal_definitions(pages: list[PageBlock]) -> list[LegalDefinition]:
    """Extract defined terms ('"Term" means...', 'hereinafter referred to as "Term"')."""
    definitions: list[LegalDefinition] = []
    seen_terms: set[str] = set()

    for page in pages:
        text = page.clean_text
        for m in LEGAL_DEFINITION_RE.finditer(text):
            term = m.group(1).strip()
            defn = re.sub(r"\s+", " ", m.group(2).strip())
            if term.lower() not in seen_terms:
                seen_terms.add(term.lower())
                definitions.append(
                    LegalDefinition(
                        term=term,
                        definition=defn,
                        raw_match=m.group(0).strip(),
                        page_number=page.page_number,
                    )
                )

        for m in HEREINAFTER_DEF_RE.finditer(text):
            entity = re.sub(r"\s+", " ", m.group(1).strip(" ,("))
            alias = m.group(2).strip()
            if alias.lower() not in seen_terms:
                seen_terms.add(alias.lower())
                definitions.append(
                    LegalDefinition(
                        term=alias,
                        definition=f"Refers to {entity}",
                        raw_match=m.group(0).strip(),
                        page_number=page.page_number,
                    )
                )

    return definitions


def extract_legal_obligations(pages: list[PageBlock], flat_sections: list[LegalNode]) -> list[ObligationItem]:
    """
    Fast O(N) obligation, prohibition, proviso, and non-obstante extractor.
    Splits pages into sentences once and uses C-level keyword guards before regex evaluation.
    """
    obligations: list[ObligationItem] = []
    seen_statements: set[str] = set()

    # Pre-index sections by page_number for O(1) page lookup
    sections_by_page: dict[int, list[LegalNode]] = {}
    for node in flat_sections:
        sections_by_page.setdefault(node.page_number, []).append(node)

    sentence_split_re = re.compile(r"(?<=[.;])\s+|\n+")

    for page in pages:
        text = page.clean_text
        text_low = text.lower()
        # Fast C-string guard: skip page entirely if no obligation triggers exist
        if not any(
            kw in text_low
            for kw in ("shall", "must", "provided that", "provided further that", "notwithstanding", "subject to", "prohibited")
        ):
            continue

        page_nodes = sections_by_page.get(page.page_number, [])
        for raw_sent in sentence_split_re.split(text):
            stmt = re.sub(r"\s+", " ", raw_sent.strip())
            if len(stmt) < 15 or len(stmt) > 600:
                continue
            low = stmt.lower()

            category: str | None = None
            if "notwithstanding anything contained" in low:
                category = "EXCEPTION_NOTWITHSTANDING"
            elif "provided that" in low or "provided further that" in low:
                category = "PROVISO"
            elif "subject to the provisions of" in low:
                category = "CONDITION_SUBJECT_TO"
            elif "shall not" in low or "must not" in low or "is prohibited from" in low or "no person shall" in low:
                category = "PROHIBITION"
            elif " shall " in f" {low} " or " must " in f" {low} ":
                category = "MANDATORY_SHALL"

            if not category:
                continue

            if not stmt.endswith("."):
                stmt += "."
            if stmt.lower() in seen_statements:
                continue
            seen_statements.add(stmt.lower())

            sec_path = ""
            probe = stmt[:24]
            for node in reversed(page_nodes):
                if (probe and probe in node.text) or (node.text and node.text[:24] in stmt):
                    sec_path = node.full_path
                    break

            obligations.append(
                ObligationItem(
                    category=category,
                    trigger_phrase=category.replace("_", " ").title(),
                    statement=stmt,
                    section_path=sec_path,
                    page_number=page.page_number,
                )
            )

    return obligations


def extract_legal_metadata(text: str) -> dict[str, Any]:
    """Extract court name, case numbers, acts, parties, and dates from legal text."""
    courts = list(dict.fromkeys(re.sub(r"\s+", " ", m.strip()) for m in COURT_HEADER_RE.findall(text)))
    case_numbers = list(dict.fromkeys(re.sub(r"\s+", " ", m.strip()) for m in CASE_NUMBER_RE.findall(text)))
    dates = list(dict.fromkeys(re.sub(r"\s+", " ", m.strip()) for m in DATE_RE.findall(text)))
    acts = list(dict.fromkeys(re.sub(r"\s+", " ", m.strip()) for m in ACT_TITLE_RE.findall(text)))

    parties: list[str] = []
    for m in CASE_VERSUS_RE.finditer(text[:5000]):
        pair = f"{m.group(1).strip()} v. {m.group(2).strip()}"
        if pair not in parties:
            parties.append(pair)

    return {
        "court": courts[0] if courts else None,
        "all_courts": courts,
        "case_numbers": case_numbers,
        "acts_mentioned": acts,
        "parties": parties,
        "dates": dates,
    }


def run_custom_regexes(text: str, extra_patterns: dict[str, str | re.Pattern[str]] | None = None) -> dict[str, list[str]]:
    """Run all registered custom regexes + any call-time custom regex patterns."""
    combined: dict[str, re.Pattern[str]] = dict(CUSTOM_REGEX_REGISTRY)
    if extra_patterns:
        for k, pat in extra_patterns.items():
            combined[k] = pat if isinstance(pat, re.Pattern) else re.compile(pat)

    results: dict[str, list[str]] = {}
    for name, compiled in combined.items():
        matches = [m.group(0).strip() for m in compiled.finditer(text)]
        results[name] = list(dict.fromkeys(matches))
    return results

