"""
`raglib.prompt_eng` — Phase 5: Customizable Prompt Engineering, Query Rewriting & Legal Story Solver.

Supports:
  - General Query Rewriting (`prompt_eng.rewrite_query(query)`)
  - Specialized Legal & Constitutional Query Rewriting (`prompt_eng.rewrite_law_query(query)`)
  - Story / Fact-Pattern Scenario Detection (e.g., real-life hypotheticals & case scenarios)
  - Adaptive Answer Formatting:
      * Generates natural, fluent ChatGPT-style sentences & paragraphs by default
      * Automatically switches to strict **IRAC** (`Issue, Rule, Application, Conclusion`) when the user asks for `"IRAC"` in their query!
  - User-defined `system_prompt` and selectable `technique` (`"standard"`, `"natural_law"`, `"legal_irac"`, `"cot"`, `"citation_strict"`, `"few_shot"`, `"concise"`)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from raglib.chunker import Chunk
from raglib.vectordb import SearchResult

DEFAULT_GENERAL_SYSTEM_PROMPT = (
    "You are an accurate, helpful AI assistant. Answer the user's question in clear, natural, well-structured "
    "sentences strictly using the provided context. Cite the source page or section naturally inline. "
    "Only use IRAC (Issue, Rule, Application, Conclusion) headings if the user explicitly asks for IRAC in their query."
)

DEFAULT_LAW_SYSTEM_PROMPT = (
    "You are a Senior Constitutional & Legal Counsel. Answer legal and hypothetical story questions in clear, "
    "fluent, natural sentences and paragraphs grounded strictly in the provided statutory/constitutional provisions. "
    "Cite exact Part, Article, Section, Clause, Proviso, and Schedule numbers naturally within your sentences "
    "(e.g., under [Part III > Article 21]). If the user presents a story or hypothetical scenario, directly "
    "address the characters and facts of the story and explain how the law applies to them. "
    "IMPORTANT: Do NOT use rigid IRAC (Issue, Rule, Application, Conclusion) headings unless the user explicitly "
    "writes 'IRAC' in their question."
)

TECHNIQUE_INSTRUCTIONS: dict[str, str] = {
    "standard": (
        "Write a clear, natural, flowing explanation in well-crafted sentences grounded in the context, "
        "citing `[Page Y]` or `[Section]` naturally inline."
    ),
    "natural_law": (
        "Write a fluent, authoritative legal explanation in natural sentences and paragraphs (like a Senior Counsel "
        "explaining to a bench or student). If the question is a story/scenario, apply the relevant Articles/Sections "
        "directly to the characters and events in the story. Only use IRAC headings if the user's query explicitly requests IRAC."
    ),
    "cot": (
        "Reason step-by-step through the context in clear, natural paragraphs, checking any exceptions or provisos, "
        "and present a cohesive final answer with inline citations."
    ),
    "few_shot": (
        "Follow this natural, grounded answer style:\n"
        "Example Question: What is the notice period?\n"
        "Example Answer: Under `[Clause 4.2]`, a party must provide at least **30 days' prior written notice** "
        "before terminating the agreement, unless immediate termination is triggered under the breach proviso in `[Clause 4.3]`.\n\n"
        "Now answer the user's question in the same fluent, sentence-based style."
    ),
    "citation_strict": (
        "STRICT CITATION MODE:\n"
        "- Write in clear sentences where every claim ends with an explicit citation `[Chunk N, Page P]`.\n"
        "- Include brief verbatim quotes in quotation marks for key rules."
    ),
    "legal_irac": (
        "ADAPTIVE LEGAL MODE:\n"
        "- If the user's question explicitly asks for **IRAC** (e.g., mentions 'IRAC' or 'Issue Rule Application Conclusion'), "
        "structure your response with `### 1. Issue`, `### 2. Rule`, `### 3. Application`, and `### 4. Conclusion`.\n"
        "- Otherwise, do NOT use IRAC headings—write a natural, fluent, ChatGPT-style explanation in cohesive sentences "
        "and paragraphs, citing exact Articles, Sections, Clauses, and Provisos naturally inline."
    ),
    "strict_irac": (
        "Structure your legal response strictly using the **IRAC** framework:\n"
        "1. **ISSUE:** State the exact legal question presented.\n"
        "2. **RULE:** Cite the governing Articles, Sections, Sub-clauses, and Provisos from the context.\n"
        "3. **APPLICATION:** Apply the statutory rules and exceptions to the facts or question.\n"
        "4. **CONCLUSION:** State the definitive legal outcome with exact citations."
    ),
    "concise": (
        "Respond in 2–4 crisp, natural sentences with inline citations. No filler text."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Specialized Legal & Constitutional Concept Mapping for Story & Query Rewriting
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_SCENARIO_MAPPINGS: list[tuple[tuple[str, ...], str]] = [
    (
        ("arrest", "arrested", "custody", "detained", "detention", "police", "magistrate", "24 hours", "twenty-four hours", "warrant", "grounds of arrest", "jail", "lockup", "lawyer"),
        "Part III Article 22 protection against arrest and detention grounds of arrest produced before nearest magistrate within twenty-four hours consult legal practitioner Article 21 personal liberty",
    ),
    (
        ("speech", "expression", "journalist", "newspaper", "media", "press", "publish", "article criticizing", "protest", "rally", "assembly", "association", "union", "movement", "reside", "profession", "trade", "business", "censorship", "ban"),
        "Part III Article 19(1)(a) freedom of speech and expression peaceably assemble without arms associations union move freely reside practice profession reasonable restrictions Article 19(2)",
    ),
    (
        ("discriminate", "discrimination", "caste", "religion", "sex", "gender", "place of birth", "race", "equal", "equality", "arbitrary", "shop", "hotel", "well", "public restaurant", "admission", "reservation", "public employment", "job"),
        "Part III Article 14 equality before law equal protection of laws Article 15 prohibition of discrimination on grounds of religion race caste sex place of birth Article 16 equality of opportunity in public employment",
    ),
    (
        ("child", "children", "minor", "12-year-old", "10-year-old", "13-year-old", "boy", "girl", "factory", "mine", "hazardous", "fireworks", "chemical factory", "forced labour", "begar", "trafficking", "bonded labour"),
        "Part III Article 24 prohibition of employment of children in factories mines hazardous employment below age of fourteen years Article 23 prohibition of traffic in human beings and forced labour",
    ),
    (
        ("school", "education", "free education", "primary education", "6 to 14", "six to fourteen"),
        "Part III Article 21A right to education free and compulsory education to all children of age six to fourteen years Article 45 Article 51A(k)",
    ),
    (
        ("life", "liberty", "privacy", "dignity", "livelihood", "shelter", "health", "clean environment", "pollution", "passport", "travel abroad", "telephone tapping", "surveillance", "encounter", "torture"),
        "Part III Article 21 protection of life and personal liberty procedure established by law right to privacy human dignity",
    ),
    (
        ("untouchability", "outcaste", "dalit", "title", "maharaja", "knighthood"),
        "Part III Article 17 abolition of untouchability forbidden offense Article 18 abolition of titles",
    ),
    (
        ("double jeopardy", "punished twice", "same offence", "witness against himself", "self-incrimination", "confession", "forced to testify", "retrospective", "ex post facto", "new criminal law"),
        "Part III Article 20 protection in respect of conviction for offences ex-post facto law double jeopardy prosecuted and punished for same offence witness against himself",
    ),
    (
        ("temple", "mosque", "church", "worship", "pray", "religion", "religious", "conscience", "propagate", "forced conversion", "minority", "linguistic", "cultural", "educational institution"),
        "Part III Article 25 freedom of conscience and free profession practice and propagation of religion Article 26 religious affairs Article 29 Article 30 right of minorities to establish and administer educational institutions",
    ),
    (
        ("writ", "supreme court", "high court", "habeas corpus", "mandamus", "certiorari", "prohibition", "quo warranto", "constitutional remedy", "petition", "pil", "violation of fundamental right"),
        "Part III Article 32 right to constitutional remedies move the Supreme Court by appropriate proceedings writs habeas corpus mandamus prohibition quo warranto certiorari Article 226 High Court",
    ),
    (
        ("duty", "duties", "national flag", "national anthem", "scientific temper", "environment", "forest", "wildlife", "public property", "sovereignty"),
        "Part IVA Article 51A fundamental duties of every citizen of India abide by Constitution respect ideal institutions National Flag National Anthem",
    ),
    (
        ("directive principles", "dpsp", "uniform civil code", "ucc", "panchayat", "equal pay for equal work", "maternity", "living wage", "free legal aid", "cattle", "alcohol", "international peace"),
        "Part IV Directive Principles of State Policy Article 36 Article 37 Article 38 Article 39A free legal aid Article 40 village panchayats Article 44 Uniform Civil Code Article 48",
    ),
    (
        ("president", "governor", "impeachment", "pardon", "mercy petition", "death sentence", "ordinance", "council of ministers", "prime minister", "attorney general"),
        "Part V Article 52 President of India Article 61 impeachment Article 72 power of President to grant pardons reprieves respites remissions Article 74 Council of Ministers Article 123 promulgate Ordinances",
    ),
    (
        ("parliament", "lok sabha", "rajya sabha", "money bill", "speaker", "joint sitting", "budget", "amendment", "amend constitution", "basic structure"),
        "Part V Article 79 Constitution of Parliament Rajya Sabha Lok Sabha Article 108 joint sitting Article 110 definition of Money Bills Article 368 power of Parliament to amend the Constitution",
    ),
    (
        ("emergency", "national emergency", "president rule", "state emergency", "financial emergency", "armed rebellion", "external aggression", "failure of constitutional machinery"),
        "Part XVIII Emergency Provisions Article 352 Proclamation of Emergency war external aggression armed rebellion Article 356 failure of constitutional machinery in States Article 359 Article 360 Financial Emergency",
    ),
    (
        ("confidential", "trade secret", "data fiduciary", "personal data", "data breach", "cross-border", "transfer", "arbitration", "tribunal", "dispute", "ninety days"),
        "Confidential Information Data Fiduciary obligations security safeguards personal data breach cross-border transfer Article 21 Section 14 Arbitration and Conciliation Act ninety days Tribunal",
    ),
]


class QueryRewriteResult(str):
    """
    String-compatible rewritten query returned by `prompt_eng.rewrite_query(...)` and `prompt_eng.rewrite_law_query(...)`.
    Works directly as a `str` (`print(rq)`) AND exposes `.original`, `.rewritten`, `.is_story`, `.wants_irac`,
    `.detected_concepts`, `.actors`, and `.sub_queries`.
    """

    original: str
    rewritten: str
    is_story: bool
    wants_irac: bool
    detected_concepts: list[str]
    actors: list[str]
    sub_queries: list[str]

    def __new__(
        cls,
        rewritten: str,
        original: str = "",
        is_story: bool = False,
        wants_irac: bool = False,
        detected_concepts: list[str] | None = None,
        actors: list[str] | None = None,
        sub_queries: list[str] | None = None,
    ) -> QueryRewriteResult:
        obj = super().__new__(cls, rewritten)
        obj.original = original or rewritten
        obj.rewritten = rewritten
        obj.is_story = is_story
        obj.wants_irac = wants_irac
        obj.detected_concepts = detected_concepts or []
        obj.actors = actors or []
        obj.sub_queries = sub_queries or [rewritten]
        return obj

    @property
    def text(self) -> str:
        return str(self)

    @property
    def rewritten_query(self) -> str:
        return self.rewritten

    @property
    def original_query(self) -> str:
        return self.original

    @property
    def detected_articles(self) -> list[str]:
        return self.detected_concepts

    @property
    def legal_concepts(self) -> list[str]:
        return self.detected_concepts

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "is_story": self.is_story,
            "wants_irac": self.wants_irac,
            "detected_concepts": self.detected_concepts,
            "actors": self.actors,
            "sub_queries": self.sub_queries,
        }


def detect_query_intent(query: str) -> dict[str, Any]:
    """
    Detect whether a user query:
      1. Explicitly asks for IRAC format (`wants_irac=True` only when 'irac' or 'issue rule application conclusion' is in the query)
      2. Is a Story / Scenario / Hypothetical Fact-Pattern question (`is_story=True`)
    """
    q_clean = (query or "").strip()
    q_low = q_clean.lower()

    wants_irac = bool(
        re.search(r"\b(?:irac|i\.r\.a\.c\.?)\b", q_low)
        or ("issue" in q_low and "rule" in q_low and "application" in q_low and "conclusion" in q_low)
    )

    # Detect story/scenario questions: multi-sentence narrative, named person/entity, or scenario markers
    words = q_clean.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", q_clean) if s.strip()]
    story_markers = (
        "suppose ",
        "imagine ",
        "scenario",
        "story",
        "hypothetical",
        "was arrested",
        "is arrested",
        "police arrested",
        "hired a",
        "working in a",
        "denied entry",
        "refused admission",
        "published an article",
        "started a business",
        "detained for",
        "kept in custody",
        "can he ",
        "can she ",
        "can they ",
        "his right",
        "her right",
        "their right",
        "advise ",
        "is this valid",
        "is this legal",
        "is this constitutional",
        "violated",
        "what remedy",
        "which right",
    )
    # Extract capitalized character/company names (excluding common question starters and legal terms)
    ignore_caps = {
        "What", "Which", "Who", "Where", "When", "Why", "How", "Explain", "Describe", "Discuss",
        "Article", "Section", "Part", "Chapter", "Schedule", "Clause", "Constitution", "India",
        "Indian", "Supreme", "Court", "High", "State", "Union", "Parliament", "President",
        "Governor", "Fundamental", "Rights", "Duties", "Directive", "Principles", "Under",
        "Can", "Does", "Is", "Are", "If", "In", "The", "A", "An", "Suppose", "Imagine", "IRAC",
        "Act", "Law", "Bill", "Rule", "Issue", "Application", "Conclusion", "Police", "Government",
    }
    potential_actors: list[str] = []
    for m in re.finditer(r"\b([A-Z][a-z]{2,15})\b", q_clean):
        w = m.group(1)
        if w not in ignore_caps and w not in potential_actors:
            potential_actors.append(w)

    is_story = bool(
        any(marker in q_low for marker in story_markers)
        or (len(sentences) >= 2 and len(words) >= 16)
        or (len(potential_actors) >= 1 and len(words) >= 12)
    )

    return {
        "wants_irac": wants_irac,
        "is_story": is_story,
        "actors": potential_actors[:3],
    }


def rewrite_query(query: str, strategy: str = "auto") -> QueryRewriteResult:
    """
    General Query Rewriter (`prompt_eng.rewrite_query`):
    Transforms conversational, vague, or narrative user queries into high-recall search queries.
    Strategies: `"auto"`, `"hyde"`, `"step_back"`, `"sub_queries"`.
    """
    q_clean = re.sub(r"\s+", " ", (query or "").strip())
    if not q_clean:
        return QueryRewriteResult("")

    intent = detect_query_intent(q_clean)
    # Remove conversational filler prefixes
    stripped = re.sub(
        r"^(?:please\s+|can you\s+|could you\s+|tell me\s+|i want to know\s+|explain to me\s+|in irac format\s*[:,-]?\s*)+",
        "",
        q_clean,
        flags=re.IGNORECASE,
    ).strip()

    # Extract core content terms
    stop = {
        "the", "is", "are", "was", "were", "be", "to", "of", "and", "a", "in", "that", "have",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
        "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all",
        "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
        "go", "me", "when", "make", "can", "like", "no", "just", "him", "know", "take", "into",
    }
    key_terms = [
        w for w in re.findall(r"[a-zA-Z0-9§]+", stripped)
        if w.lower() not in stop and len(w) >= 3
    ]

    strat = (strategy or "auto").lower().strip()
    if strat == "hyde":
        rewritten = f"{stripped} — Definition, key rules, core principles, and governing provisions regarding {' '.join(key_terms[:10])}."
    elif strat == "step_back":
        rewritten = f"{stripped} (Core concepts and foundational rules: {' '.join(key_terms[:8])})"
    else:
        rewritten = f"{stripped} ({' '.join(key_terms[:10])})" if key_terms else stripped

    sub_q = [s.strip() for s in re.split(r"[?;]|\band\b", stripped) if len(s.strip()) > 6]
    if not sub_q:
        sub_q = [stripped]

    return QueryRewriteResult(
        rewritten=rewritten,
        original=q_clean,
        is_story=intent["is_story"],
        wants_irac=intent["wants_irac"],
        detected_concepts=key_terms[:8],
        actors=intent["actors"],
        sub_queries=sub_q[:4],
    )


def rewrite_law_query(query: str, strategy: str = "auto") -> QueryRewriteResult:
    """
    Specialized Legal & Constitutional Query Rewriter (`prompt_eng.rewrite_law_query`):
    - Translates everyday language and **Story / Scenario / Hypothetical Questions** into exact
      Constitutional Articles, Parts, Legal Doctrines, and Statutory terminology.
    - Detects whether the user asked a Story Question (`is_story=True`) or requested IRAC (`wants_irac=True`).
    """
    q_clean = re.sub(r"\s+", " ", (query or "").strip())
    if not q_clean:
        return QueryRewriteResult("")

    intent = detect_query_intent(q_clean)
    q_low = q_clean.lower()

    matched_expansions: list[str] = []
    detected_concepts: list[str] = []

    for triggers, expansion in LEGAL_SCENARIO_MAPPINGS:
        for trig in triggers:
            if trig in q_low:
                if expansion not in matched_expansions:
                    matched_expansions.append(expansion)
                if trig not in detected_concepts:
                    detected_concepts.append(trig)

    # Also normalize shorthand like "art 21" -> "Article 21", "sec 14" -> "Section 14"
    normalized_q = re.sub(r"\bart\.?\s*(\d+[a-zA-Z]?)\b", r"Article \1", q_clean, flags=re.IGNORECASE)
    normalized_q = re.sub(r"\bsec\.?\s*(\d+[a-zA-Z]?)\b", r"Section \1", normalized_q, flags=re.IGNORECASE)

    if matched_expansions:
        expansion_str = " | ".join(matched_expansions[:3])
        rewritten = f"{normalized_q} — [Legal & Constitutional Concepts: {expansion_str}]"
    else:
        base_rw = rewrite_query(normalized_q, strategy=strategy)
        rewritten = f"{normalized_q} — [Statutory & Constitutional Provisions: {' '.join(base_rw.detected_concepts)}]"
        detected_concepts = base_rw.detected_concepts

    sub_queries = [normalized_q]
    for exp in matched_expansions[:2]:
        sub_queries.append(exp)

    return QueryRewriteResult(
        rewritten=rewritten,
        original=q_clean,
        is_story=intent["is_story"],
        wants_irac=intent["wants_irac"],
        detected_concepts=detected_concepts,
        actors=intent["actors"],
        sub_queries=sub_queries,
    )


@dataclass
class BuiltPrompt:
    """Structured prompt package containing `system_prompt`, `user_prompt`, and `full_prompt`."""

    system_prompt: str
    user_prompt: str
    technique: str
    context_text: str
    query: str
    rewritten_query: str = ""
    is_story: bool = False
    wants_irac: bool = False
    actors: list[str] = field(default_factory=list)

    @property
    def full_prompt(self) -> str:
        return f"SYSTEM INSTRUCTIONS:\n{self.system_prompt}\n\n{self.user_prompt}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["full_prompt"] = self.full_prompt
        return d


def list_techniques() -> list[str]:
    """Return all available prompt engineering techniques."""
    return list(TECHNIQUE_INSTRUCTIONS.keys())


def register_technique(name: str, instruction: str) -> None:
    """Register a custom prompt engineering technique."""
    TECHNIQUE_INSTRUCTIONS[name.lower().strip()] = instruction.strip()


def register_template(name: str, instruction: str) -> None:
    """Alias for `register_technique`."""
    register_technique(name, instruction)


def _format_context(context_items: list[SearchResult] | list[Chunk] | list[str]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(context_items, start=1):
        if isinstance(item, SearchResult):
            chk = item.chunk
            sec_label = f" | {chk.section_path}" if chk.section_path else ""
            header = f"[Chunk {idx} | Page {chk.page_number}{sec_label} | Score: {item.score:.2f}]"
            defs = chk.metadata.get("definitions_in_scope") or []
            def_note = ("\nDefined Terms in Scope: " + "; ".join(defs)) if defs else ""
            blocks.append(f"{header}\n{chk.text}{def_note}")
        elif isinstance(item, Chunk):
            sec_label = f" | {item.section_path}" if item.section_path else ""
            header = f"[Chunk {idx} | Page {item.page_number}{sec_label}]"
            blocks.append(f"{header}\n{item.text}")
        else:
            blocks.append(f"[Chunk {idx}]\n{str(item)}")
    return "\n\n---\n\n".join(blocks)


def _extract_items_from_obj(obj: Any) -> list[Any] | None:
    """Extract a list of chunks/results if `obj` is a list, VectorStore (`db`), or ParsedDocument (`doc`)."""
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if hasattr(obj, "_chunks") and isinstance(getattr(obj, "_chunks"), list):
        return list(getattr(obj, "_chunks"))
    if hasattr(obj, "clauses") and getattr(obj, "clauses"):
        return [c.text for c in getattr(obj, "clauses")]
    if hasattr(obj, "text") and not isinstance(obj, str):
        return [str(getattr(obj, "text"))]
    return None


def _resolve_query_and_context(
    first_arg: Any,
    second_arg: Any,
    query_kw: str | None,
    context_kw: Any,
) -> tuple[str, list[SearchResult] | list[Chunk] | list[str]]:
    if context_kw is not None and query_kw is not None:
        extracted = _extract_items_from_obj(context_kw)
        return str(query_kw), (extracted if extracted is not None else [str(context_kw)])

    first_items = _extract_items_from_obj(first_arg)
    if first_items is not None:
        if isinstance(second_arg, str):
            q = second_arg
        elif query_kw is not None:
            q = query_kw
        else:
            q = "Explain the key provisions in the provided context."
        return q, first_items

    q = str(first_arg) if first_arg is not None else (query_kw or "Explain the provided context.")
    second_items = _extract_items_from_obj(second_arg)
    if second_items is not None:
        ctx = second_items
    elif context_kw is not None:
        ext_kw = _extract_items_from_obj(context_kw)
        ctx = ext_kw if ext_kw is not None else [str(context_kw)]
    elif isinstance(second_arg, str):
        ctx = [second_arg]
    else:
        ctx = []
    return q, ctx


def build(
    query_or_context: str | list[SearchResult] | list[Chunk] | list[str] | None = None,
    context_items: list[SearchResult] | list[Chunk] | list[str] | str | None = None,
    technique: str = "standard",
    system_prompt: str | None = None,
    user_template: str | None = None,
    *,
    query: str | None = None,
    context: list[SearchResult] | list[Chunk] | list[str] | None = None,
    max_chunks: int = 15,
    is_law: bool = False,
) -> BuiltPrompt:
    """
    Build a complete System + User prompt for any general or legal RAG task.
    - Generates natural, flowing sentences by default.
    - Automatically switches to IRAC formatting ONLY if the user writes 'IRAC' in their query!
    - Automatically rewrites queries & detects story/scenario questions.
    """
    resolved_query, resolved_ctx = _resolve_query_and_context(
        query_or_context, context_items, query, context
    )
    if max_chunks and len(resolved_ctx) > max_chunks:
        resolved_ctx = resolved_ctx[:max_chunks]

    rw = rewrite_law_query(resolved_query) if is_law else rewrite_query(resolved_query)
    tech_key = technique.lower().strip()

    # If the user explicitly asked for IRAC in their query, use strict_irac instructions
    if rw.wants_irac:
        tech_instruction = TECHNIQUE_INSTRUCTIONS["strict_irac"]
    else:
        tech_instruction = TECHNIQUE_INSTRUCTIONS.get(tech_key, technique)

    if rw.is_story:
        actors_note = f" (involving {', '.join(rw.actors)})" if rw.actors else ""
        tech_instruction += (
            f"\n- STORY / SCENARIO DETECTED{actors_note}: Directly address the story scenario and explain how "
            "the governing provisions apply to the people and events described."
        )

    sys_p = (system_prompt or (DEFAULT_LAW_SYSTEM_PROMPT if is_law else DEFAULT_GENERAL_SYSTEM_PROMPT)).strip()
    ctx_str = _format_context(resolved_ctx)

    if user_template:
        user_p = user_template.format(
            context=ctx_str,
            query=resolved_query,
            technique=tech_instruction,
        )
    else:
        user_p = (
            f"### RETRIEVED CONTEXT\n{ctx_str}\n\n"
            f"### ANSWER STYLE & INSTRUCTIONS ({tech_key.upper()})\n{tech_instruction}\n\n"
            f"### USER QUESTION\n{resolved_query}\n"
            f"### REWRITTEN SEARCH INTENT\n{rw.rewritten}\n\n"
            f"### RESPONSE\n"
        )

    return BuiltPrompt(
        system_prompt=sys_p,
        user_prompt=user_p,
        technique=tech_key,
        context_text=ctx_str,
        query=resolved_query,
        rewritten_query=str(rw),
        is_story=rw.is_story,
        wants_irac=rw.wants_irac,
        actors=rw.actors,
    )


def build_law(
    query_or_context: str | list[SearchResult] | list[Chunk] | list[str] | None = None,
    context_items: list[SearchResult] | list[Chunk] | list[str] | str | None = None,
    technique: str = "natural_law",
    system_prompt: str | None = None,
    user_template: str | None = None,
    *,
    query: str | None = None,
    context: list[SearchResult] | list[Chunk] | list[str] | None = None,
    max_chunks: int = 15,
) -> BuiltPrompt:
    """
    Specialized Law Prompt Builder:
    - Generates natural, fluent legal sentences by default (`technique="natural_law"`).
    - Automatically switches to IRAC if the user writes `"IRAC"` in their query!
    - Includes Legal Story / Scenario analysis & Legal Query Rewriting.
    """
    return build(
        query_or_context=query_or_context,
        context_items=context_items,
        technique=technique,
        system_prompt=system_prompt or DEFAULT_LAW_SYSTEM_PROMPT,
        user_template=user_template,
        query=query,
        context=context,
        max_chunks=max_chunks,
        is_law=True,
    )


__all__ = [
    "BuiltPrompt",
    "QueryRewriteResult",
    "build",
    "build_law",
    "rewrite_query",
    "rewrite_law_query",
    "detect_query_intent",
    "list_techniques",
    "register_technique",
    "register_template",
    "DEFAULT_GENERAL_SYSTEM_PROMPT",
    "DEFAULT_LAW_SYSTEM_PROMPT",
]
