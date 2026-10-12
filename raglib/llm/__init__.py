"""
`raglib.llm` — Multi-Runtime LLM Execution Engine with Smart Model Name Resolution.

Supports ANY model name (including human-friendly names like "gemini 3.8 flash", "claude sonnet 5.5", "gpt-4o", "llama3.2")
and automatically detects or routes to:
  - "gemini"     (Google Gemini API — e.g. "gemini 3.8 flash", "gemini-2.5-pro", "gemini-2.5-flash")
  - "anthropic"  (Anthropic Claude API — e.g. "claude sonnet 5.5", "claude-3-7-sonnet-latest", "claude opus 4")
  - "openai"     (OpenAI API — e.g. "gpt-4o", "gpt-4.1", "o3-mini")
  - "groq"       (Groq Cloud API — e.g. "llama-3.3-70b-versatile", "mixtral-8x7b")
  - "openrouter" (OpenRouter API — any model slug like "anthropic/claude-sonnet-5.5")
  - "ollama"     (Local Ollama runtime at http://localhost:11434 — e.g. "llama3.2", "deepseek-r1", "qwen2.5")
  - "extractive" (Offline zero-LLM grounded context synthesizer)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

from raglib.prompt_eng import BuiltPrompt

AVAILABLE_LLM_PROVIDERS = (
    "auto",
    "ollama",
    "gemini",
    "anthropic",
    "claude",
    "openai",
    "groq",
    "openrouter",
    "extractive",
)

# Friendly alias table for common human-written model names (while also passing through ANY custom/future model name!)
FRIENDLY_MODEL_ALIASES: dict[str, str] = {
    "gemini 3.8 flash": "gemini-3.8-flash",
    "gemini 3.8 pro": "gemini-3.8-pro",
    "gemini 2.5 flash": "gemini-2.5-flash",
    "gemini 2.5 pro": "gemini-2.5-pro",
    "gemini 2.0 flash": "gemini-2.0-flash",
    "claude sonnet 5.5": "claude-sonnet-5-5",
    "claude 5.5 sonnet": "claude-sonnet-5-5",
    "claude sonnet 4.5": "claude-sonnet-4-5",
    "claude 3.7 sonnet": "claude-3-7-sonnet-latest",
    "claude 3.5 sonnet": "claude-3-5-sonnet-latest",
    "claude opus 4": "claude-opus-4-0",
    "gpt 4o": "gpt-4o",
    "gpt 4o mini": "gpt-4o-mini",
    "gpt 4.1": "gpt-4.1",
    "gpt 4.1 mini": "gpt-4.1-mini",
    "o3 mini": "o3-mini",
    "llama 3.2": "llama3.2",
    "llama 3.3": "llama-3.3-70b-versatile",
    "deepseek r1": "deepseek-r1",
    "qwen 2.5": "qwen2.5",
}


def list_llm_providers() -> list[str]:
    return list(AVAILABLE_LLM_PROVIDERS)


def resolve_model_and_provider(
    provider: str = "auto",
    model: str | None = None,
) -> tuple[str, str]:
    """
    Smart resolver that allows users to pass human-readable model names like:
      - `model="gemini 3.8 flash"` -> ("gemini", "gemini-3.8-flash")
      - `model="claude sonnet 5.5"` -> ("anthropic", "claude-sonnet-5-5")
      - `model="gpt-4o"` -> ("openai", "gpt-4o")
      - `model="llama3.2"` -> ("ollama", "llama3.2")
    Or even pass the model name directly inside the `provider` argument!
    """
    prov = (provider or "auto").lower().strip()
    raw_model = (model or "").strip()

    # If the user passed a model name directly inside `provider` (e.g., llm_provider="gemini 3.8 flash")
    if prov not in AVAILABLE_LLM_PROVIDERS and not raw_model:
        raw_model = provider.strip()
        prov = "auto"

    if prov == "claude":
        prov = "anthropic"

    # Normalize human-readable spaces to valid API slug
    low_model = re.sub(r"\s+", " ", raw_model.lower())
    if low_model in FRIENDLY_MODEL_ALIASES:
        norm_model = FRIENDLY_MODEL_ALIASES[low_model]
    elif " " in raw_model:
        # Generic space-to-hyphen slug normalization for any new model name: e.g. "gemini 4.0 ultra" -> "gemini-4.0-ultra"
        norm_model = re.sub(r"\s+", "-", raw_model.strip().lower())
    else:
        norm_model = raw_model

    # Auto-detect provider from model name when provider is "auto" or mismatching
    if prov == "auto" and norm_model:
        m_low = norm_model.lower()
        if "/" in m_low:
            prov = "openrouter"
        elif m_low.startswith("gemini"):
            prov = "gemini"
        elif m_low.startswith("claude"):
            prov = "anthropic"
        elif m_low.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
            prov = "openai"
        elif "versatile" in m_low or "groq" in m_low:
            prov = "groq"
        else:
            prov = "ollama"
    elif prov == "auto":
        prov = "extractive"

    if prov not in AVAILABLE_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Choose from {AVAILABLE_LLM_PROVIDERS} "
            f"or pass a model name like 'gemini 3.8 flash' or 'claude sonnet 5.5'."
        )

    # Default model per provider if user only specified provider
    if not norm_model:
        defaults = {
            "gemini": "gemini-2.5-flash",
            "anthropic": "claude-sonnet-4-5",
            "openai": "gpt-4o-mini",
            "groq": "llama-3.3-70b-versatile",
            "openrouter": "google/gemini-2.5-flash",
            "ollama": "llama3.2",
            "extractive": "extractive",
        }
        norm_model = defaults.get(prov, "llama3.2")

    return prov, norm_model


class LLMResult(str):
    """
    String-compatible LLM output that works both as a plain `str` (`print(answer)`)
    and as an object with `.text`, `.answer`, `.provider`, and `.model` attributes (`print(answer.text)`).
    """

    provider: str
    model: str

    def __new__(cls, content: str, provider: str = "extractive", model: str = "extractive") -> LLMResult:
        obj = super().__new__(cls, content)
        obj.provider = provider
        obj.model = model
        return obj

    @property
    def text(self) -> str:
        return str(self)

    @property
    def answer(self) -> str:
        return str(self)

    @property
    def content(self) -> str:
        return str(self)


def generate(
    prompt: BuiltPrompt | str,
    provider: str = "auto",
    model: str | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.2,
    fallback_to_extractive: bool = True,
    query: str | None = None,
) -> LLMResult:
    """
    Execute an LLM call using any provider or human-readable model name
    (e.g., `model="gemini 3.8 flash"`, `model="claude sonnet 5.5"`, `model="gpt-4o"`, `model="llama3.2"`).
    Returns an `LLMResult` (which is a `str` AND supports `.text` / `.answer`).
    """
    prov, resolved_model = resolve_model_and_provider(provider=provider, model=model)

    if isinstance(prompt, BuiltPrompt):
        sys_p = system_prompt or prompt.system_prompt
        user_p = prompt.user_prompt
        ctx_text = prompt.context_text
        query_text = query or prompt.query
        if query:
            user_p = user_p.replace(f"### USER QUESTION\n{prompt.query}", f"### USER QUESTION\n{query}")
    else:
        sys_p = system_prompt or "You are a helpful AI assistant."
        user_p = str(prompt) if not query else f"{prompt}\n\nQuestion: {query}"
        ctx_text = str(prompt)
        query_text = query or ""


    technique = prompt.technique if isinstance(prompt, BuiltPrompt) else "standard"

    if prov == "extractive":
        return LLMResult(
            _synthesize_extractive(query_text, ctx_text, sys_p, technique=technique),
            provider=prov,
            model=resolved_model,
        )

    try:
        if prov == "ollama":
            out = _call_ollama(
                system_prompt=sys_p,
                user_prompt=user_p,
                model=resolved_model,
                base_url=base_url or "http://localhost:11434",
                temperature=temperature,
            )
            return LLMResult(out, provider=prov, model=resolved_model)
        if prov == "gemini":
            out = _call_gemini(
                system_prompt=sys_p,
                user_prompt=user_p,
                model=resolved_model,
                api_key=api_key,
                temperature=temperature,
            )
            return LLMResult(out, provider=prov, model=resolved_model)
        if prov == "anthropic":
            out = _call_anthropic(
                system_prompt=sys_p,
                user_prompt=user_p,
                model=resolved_model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
            )
            return LLMResult(out, provider=prov, model=resolved_model)
        if prov in ("openai", "groq", "openrouter"):
            default_url = (
                "https://api.groq.com/openai/v1"
                if prov == "groq"
                else ("https://openrouter.ai/api/v1" if prov == "openrouter" else "https://api.openai.com/v1")
            )
            out = _call_openai_compatible(
                system_prompt=sys_p,
                user_prompt=user_p,
                model=resolved_model,
                api_key=api_key,
                base_url=base_url or default_url,
                temperature=temperature,
            )
            return LLMResult(out, provider=prov, model=resolved_model)
    except Exception as exc:
        if fallback_to_extractive:
            return LLMResult(
                _synthesize_extractive(query_text, ctx_text, sys_p, technique=technique),
                provider=prov,
                model=resolved_model,
            )
        raise

    return LLMResult(
        _synthesize_extractive(query_text, ctx_text, sys_p, technique=technique),
        provider=prov,
        model=resolved_model,
    )


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str,
    base_url: str,
    temperature: float,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return str(data.get("message", {}).get("content", "")).strip()


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None,
    temperature: float,
) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Gemini API key missing. Pass `api_key='...'` or set GEMINI_API_KEY.")
    model_path = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        return ""


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not key:
        raise ValueError("Anthropic/Claude API key missing. Pass `api_key='...'` or set ANTHROPIC_API_KEY.")
    url = f"{(base_url or 'https://api.anthropic.com/v1').rstrip('/')}/messages"
    payload = {
        "model": model,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        content_blocks = data.get("content", [])
        return "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text").strip()


def _call_openai_compatible(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None,
    base_url: str,
    temperature: float,
) -> str:
    key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not key:
        raise ValueError("API key missing. Pass `api_key='...'` or set OPENAI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY.")
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", "")).strip()
        return ""


def _parse_chunk_block(raw_block: str, strip_irac_headers: bool = True) -> dict[str, str]:
    """Parse `[Chunk X | Page Y | Section | Score: Z]` header and clean body text into natural prose."""
    lines = [ln.rstrip() for ln in raw_block.strip().splitlines()]
    page = "1"
    section = ""
    body_lines: list[str] = []

    for idx, ln in enumerate(lines):
        if idx == 0 and ln.startswith("[Chunk ") and ln.endswith("]"):
            inner = ln[1:-1]
            parts = [p.strip() for p in inner.split("|")]
            for p in parts:
                if p.lower().startswith("page "):
                    page = p.split(" ", 1)[1].strip()
                elif not p.lower().startswith(("chunk ", "score:")):
                    section = p
            continue
        # Strip duplicate bracketed section header if present on the first body line
        if not body_lines and ln.startswith("[") and ln.endswith("]") and len(ln) < 140:
            if not section:
                section = ln[1:-1].strip()
            continue
        if strip_irac_headers:
            # Remove embedded IRAC markdown headings or horizontal rules from chunk body so output flows as natural sentences
            if re.match(r"^\s*(?:-?\s*)?#{1,4}\s*\d*\.*\s*(?:ISSUE|RULE|APPLICATION|CONCLUSION)\b", ln, flags=re.IGNORECASE):
                continue
            if re.match(r"^\s*---+\s*$", ln):
                continue
        body_lines.append(ln)

    body = "\n".join(body_lines).strip()
    # Clean mid-sentence leading fragment from sliding_window chunks (starts with lowercase word)
    if body and body[0].islower():
        first_break = re.search(r"(?:[.;:]\s+|\n+)([A-Z0-9#(])", body)
        if first_break:
            body = body[first_break.start(1) :].strip()

    # Try to extract a more specific [Part ... > Article ...] citation from body if section is generic
    if not section or section in ("Document Provision", "legal_clause", "semantic", "sliding_window", "recursive"):
        m_cite = re.search(r"\[((?:Part|Article|Section|Schedule)[^\]]+)\]", body, flags=re.IGNORECASE)
        if m_cite:
            section = m_cite.group(1).strip()

    return {
        "page": page,
        "section": section or "Document Provision",
        "body": body,
    }


def _clean_prose_text(raw_text: str) -> str:
    """Convert markdown bullets/blockquotes inside a chunk into clean, readable prose sentences."""
    cleaned_lines: list[str] = []
    for ln in raw_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        # Remove leading bullet or blockquote markers so sentences flow naturally
        s = re.sub(r"^(?:[*•\-]\s+|\d+\.\s+|>+\s*)+", "", s).strip()
        s = re.sub(r":\s*>\s*", ": ", s)
        if s:
            cleaned_lines.append(s)
    joined = " ".join(cleaned_lines)
    return re.sub(r"\s+", " ", joined).strip()


def _synthesize_extractive(
    query: str,
    context_text: str,
    system_prompt: str,
    technique: str = "standard",
) -> str:
    """
    Synthesize a fluent, natural, sentence-based response (like ChatGPT) by default.
    - Directly answers Story / Scenario / Hypothetical questions by applying the law to the story characters/facts.
    - ONLY formats into 4-part IRAC headings (`1. Issue`, `2. Rule`, `3. Application`, `4. Conclusion`)
      when the user explicitly asks for IRAC in their query (or sets `technique="strict_irac"`).
    """
    if not context_text.strip():
        return "I could not find relevant passages in the document to answer your question."

    from raglib.prompt_eng import detect_query_intent

    intent = detect_query_intent(query)
    wants_irac = intent["wants_irac"] or (technique.lower().strip() == "strict_irac")
    is_story = intent["is_story"]
    actors = intent["actors"]

    raw_blocks = [b.strip() for b in context_text.split("\n\n---\n\n") if b.strip()]
    parsed_items = [_parse_chunk_block(b, strip_irac_headers=not wants_irac) for b in raw_blocks]
    parsed_items = [it for it in parsed_items if it["body"]]

    if not parsed_items:
        return "I could not find relevant passages in the document to answer your question."

    # Deduplicate near-identical chunk bodies
    unique_items: list[dict[str, str]] = []
    seen_prefixes: set[str] = set()
    for it in parsed_items:
        norm_key = re.sub(r"\s+", " ", _clean_prose_text(it["body"])[:110].lower())
        if norm_key and norm_key not in seen_prefixes:
            seen_prefixes.add(norm_key)
            unique_items.append(it)

    top_items = unique_items[:3]
    primary = top_items[0]
    primary_prose = _clean_prose_text(primary["body"])
    primary_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", primary_prose) if s.strip()]
    lead_paragraph = " ".join(primary_sentences[:3]).strip()

    # ─── MODE A: User Explicitly Asked for IRAC in Query ("... in IRAC format") ───
    if wants_irac:
        q_display = re.sub(r"\b(?:in\s+)?irac(?:\s+format)?\b[:,-]?", "", query, flags=re.IGNORECASE).strip() or query
        out_lines: list[str] = [
            f"### 1. Issue",
            f"Whether and how **`[{primary['section']}]`** *(Page {primary['page']})* governs the question: *{q_display}*.",
            "",
            "### 2. Rule",
        ]
        for item in top_items:
            prose = _clean_prose_text(item["body"])
            if len(prose) > 450:
                prose = prose[:450].rsplit(" ", 1)[0] + "..."
            out_lines.append(f"* Under **`[{item['section']}]`** *(Page {item['page']})*: {prose}")
        out_lines.append("")
        out_lines.append("### 3. Application")
        if is_story and actors:
            out_lines.append(
                f"Applying **`[{primary['section']}]`** to the scenario involving **{', '.join(actors)}**, {lead_paragraph}"
            )
        else:
            out_lines.append(f"Applying these governing provisions to your question, {lead_paragraph}")
        out_lines.append("")
        cited_list = ", ".join(dict.fromkeys(f"**`[{it['section']}]`** (Page {it['page']})" for it in top_items))
        out_lines.append("### 4. Conclusion")
        out_lines.append(f"Therefore, under {cited_list}, {primary_sentences[0] if primary_sentences else lead_paragraph}")
        return "\n".join(out_lines).strip()

    # ─── MODE B: Story / Scenario / Hypothetical Case Question (Natural Sentences) ───
    if is_story:
        actor_phrase = f" involving **{', '.join(actors)}**" if actors else ""
        paragraphs: list[str] = []

        paragraphs.append(
            f"In this scenario{actor_phrase}, the situation is directly governed by **`[{primary['section']}]`** *(Page {primary['page']})*. "
            f"{lead_paragraph}"
        )

        if len(top_items) > 1:
            sec_item = top_items[1]
            sec_prose = _clean_prose_text(sec_item["body"])
            sec_sents = " ".join(re.split(r"(?<=[.!?])\s+", sec_prose)[:2]).strip()
            if sec_sents and sec_sents.lower() != lead_paragraph.lower():
                paragraphs.append(
                    f"Additionally, under **`[{sec_item['section']}]`** *(Page {sec_item['page']})*, {sec_sents}"
                )

        # Check for any proviso/exception sentence across retrieved chunks
        for item in top_items:
            for sent in re.split(r"(?<=[.;])\s+", _clean_prose_text(item["body"])):
                s_clean = sent.strip()
                if any(
                    kw in s_clean.lower()
                    for kw in ("provided that", "notwithstanding", "nothing in", "subject to", "reasonable restriction", "unless")
                ):
                    if len(s_clean) > 25 and s_clean not in paragraphs[0]:
                        paragraphs.append(
                            f"However, it is important to note the qualification under **`[{item['section']}]`**, which states that {s_clean[0].lower() + s_clean[1:] if len(s_clean) > 1 else s_clean}"
                        )
                        break

        cited_summary = ", ".join(dict.fromkeys(f"**`[{it['section']}]`**" for it in top_items))
        paragraphs.append(
            f"Accordingly, applying {cited_summary} to the facts of this case{actor_phrase}, the rights and legal position are protected and determined strictly by these constitutional and statutory safeguards."
        )
        return "\n\n".join(paragraphs).strip()

    # ─── MODE C: Default Natural Sentence & Paragraph Answer (No IRAC Headings!) ───
    paragraphs = []
    if primary["section"] and primary["section"] != "Document Provision":
        paragraphs.append(
            f"Under **`[{primary['section']}]`** *(Page {primary['page']})*, {lead_paragraph}"
        )
    else:
        paragraphs.append(f"{lead_paragraph} *(Page {primary['page']})*")

    for extra_item in top_items[1:]:
        extra_prose = _clean_prose_text(extra_item["body"])
        extra_sents = " ".join(re.split(r"(?<=[.!?])\s+", extra_prose)[:2]).strip()
        if extra_sents and extra_sents.lower() not in lead_paragraph.lower():
            if extra_item["section"] and extra_item["section"] != primary["section"]:
                paragraphs.append(
                    f"Furthermore, **`[{extra_item['section']}]`** *(Page {extra_item['page']})* provides that {extra_sents[0].lower() + extra_sents[1:] if len(extra_sents) > 1 and extra_sents[0].isupper() and not extra_sents.startswith(('Article', 'Section', 'Part', 'I ')) else extra_sents}"
                )
            else:
                paragraphs.append(extra_sents)

    return "\n\n".join(paragraphs).strip()


__all__ = [
    "LLMResult",
    "generate",
    "resolve_model_and_provider",
    "list_llm_providers",
    "AVAILABLE_LLM_PROVIDERS",
    "FRIENDLY_MODEL_ALIASES",
]


