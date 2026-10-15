"""
`ragspark.rag` — Production General & Specialized Law RAG Pipeline Engine + Experimentation Suite.

Supports:
  - Single or Combined Chunking Strategies (`chunk_strategy="recursive"` or `["legal_clause", "semantic"]`)
  - Selectable Embedding Provider (`embed_provider`) + Embedding Strategy (`embed_strategy="legal_domain_boost"`)
  - Selectable Retrieval Technique (`retrieval_strategy="hybrid"`, `"semantic"`, `"bm25"`, `"rrf"`, `"mmr"`, `"rerank"`, `"multi_query"`)
  - Selectable VectorDB (`vectordb="chroma"` default, `"pinecone"`, `"mongodb"`, `"qdrant"`, `"memory"`)
  - Selectable Prompt Engineering Technique + Custom `system_prompt`
  - Selectable LLM Runtime (`"gemini 3.8 flash"`, `"claude sonnet 5.5"`, `"gpt 4o"`, `"llama 3.2"`, `"extractive"`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from ragspark import chunker as chunker_mod
from ragspark import embedder as embedder_mod
from ragspark import llm as llm_mod
from ragspark import parser as parser_mod
from ragspark import prompt_eng as prompt_mod
from ragspark import vectordb as vectordb_mod
from ragspark.chunker import Chunk
from ragspark.parser import ParsedDocument
from ragspark.prompt_eng import BuiltPrompt
from ragspark.vectordb import SearchResult, VectorStore


def _extract_sentence_evidence(query: str, text: str) -> str:
    """Extract the most relevant sentence from text matching query terms."""
    if not text:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    if not sentences:
        clean = text.strip()
        return clean[:180] + ("..." if len(clean) > 180 else "")

    q_words = set(re.findall(r"\w+", query.lower()))
    if not q_words:
        return sentences[0]

    best_s = sentences[0]
    best_score = -1
    for s in sentences:
        s_words = set(re.findall(r"\w+", s.lower()))
        score = len(q_words & s_words)
        if score > best_score:
            best_score = score
            best_s = s
    return best_s


@dataclass
class RAGResponse:
    """Complete result from `pipe.ask(...)`, `rag.ask(...)`, or `rag.ask_law(...)`."""

    query: str
    answer: str
    sources: list[SearchResult]
    built_prompt: BuiltPrompt
    mode: str = "general"
    llm_provider: str = "extractive"
    vectordb_provider: str = "chroma"
    retrieval_strategy: str = "hybrid"
    rewritten_query: str = ""
    is_story: bool = False
    wants_irac: bool = False
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Alias for `answer` so both `resp.answer` and `resp.text` work."""
        return self.answer

    @property
    def citations(self) -> list[dict[str, Any]]:
        """Return structured citations with source document, page, and exact evidence sentence."""
        cites: list[dict[str, Any]] = []
        for s in self.sources:
            evidence = _extract_sentence_evidence(self.query, s.chunk.text)
            cites.append(
                {
                    "source": str(s.chunk.source or ""),
                    "page": int(s.chunk.page_number) if s.chunk.page_number else 1,
                    "section": str(s.chunk.section_path or ""),
                    "evidence": evidence,
                    "score": round(float(s.score), 4),
                    "strategy": s.retrieval_strategy,
                }
            )
        return cites

    @property
    def evidence(self) -> list[str]:
        """Return list of key evidence quotes directly supporting the answer."""
        return [c["evidence"] for c in self.citations if c["evidence"]]

    def format_citations(self) -> str:
        """Format sources and supporting evidence into a clean, human-readable markdown block."""
        if not self.sources:
            return "No citations available."
        lines = ["### Verified Sources & Evidence:"]
        for idx, c in enumerate(self.citations, start=1):
            source_name = Path(c["source"]).name if c["source"] else "Document"
            page_info = f", Page {c['page']}" if c["page"] and c["page"] > 0 else ""
            lines.append(f"{idx}. **{source_name}**{page_info} (Relevance Score: {c['score']})")
            if c["evidence"]:
                lines.append(f"   > \"{c['evidence']}\"")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "is_story": self.is_story,
            "wants_irac": self.wants_irac,
            "language": self.language,
            "answer": self.answer,
            "mode": self.mode,
            "llm_provider": self.llm_provider,
            "vectordb_provider": self.vectordb_provider,
            "retrieval_strategy": self.retrieval_strategy,
            "sources": [s.to_dict() for s in self.sources],
            "citations": self.citations,
            "prompt": self.built_prompt.to_dict(),
            "metadata": self.metadata,
        }


class RAGPipeline:
    """
    Stateful, reusable Production RAG Pipeline created via `rag.create(...)` or `rag.create_law(...)`.
    """

    def __init__(
        self,
        document: ParsedDocument,
        chunks: list[Chunk],
        store: VectorStore,
        mode: str = "general",
        embed_provider: str = "fast_local",
        embed_strategy: str = "dense",
        embed_model: str | None = None,
        embed_api_key: str | None = None,
        embed_dimension: int = 384,
        retrieval_strategy: str = "hybrid",
        llm_provider: str = "extractive",
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        prompt_technique: str = "standard",
        system_prompt: str | None = None,
        rewrite_queries: bool = True,
        language: str | None = None,
    ) -> None:
        self.document = document
        self.chunks = chunks
        self.store = store
        self.mode = mode
        self.embed_provider = embed_provider
        self.embed_strategy = embed_strategy
        self.embed_model = embed_model
        self.embed_api_key = embed_api_key
        self.embed_dimension = embed_dimension
        self.retrieval_strategy = retrieval_strategy
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.prompt_technique = prompt_technique
        self.system_prompt = system_prompt
        self.rewrite_queries = rewrite_queries
        self.language = language
        self.history: list[dict[str, str]] = []

    def rewrite_query(self, query: str) -> prompt_mod.QueryRewriteResult:
        """Rewrite a query using Law Query Rewriting (if mode=='law') or General Query Rewriting."""
        if self.mode == "law":
            return prompt_mod.rewrite_law_query(query)
        return prompt_mod.rewrite_query(query)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_strategy: str | None = None,
        chunk_strategy_filter: str | list[str] | None = None,
        rewrite: bool | None = None,
        rerank: bool = False,
        hybrid_alpha: float = 0.65,
    ) -> list[SearchResult]:
        """
        Retrieve top_k chunks using the selected retrieval strategy ('hybrid', 'semantic', 'bm25', 'rrf', 'mmr', 'rerank', 'multi_query').
        Automatically applies Query Rewriting when `rewrite=True`.
        If `rerank=True`, applies cross-encoder/flash re-ranking to push the best chunks to the top.
        """
        use_rewrite = self.rewrite_queries if rewrite is None else rewrite
        search_query = str(self.rewrite_query(query)) if use_rewrite else query

        q_vec = embedder_mod.embed_query(
            search_query,
            provider=self.embed_provider,
            strategy=self.embed_strategy,
            model=self.embed_model,
            api_key=self.embed_api_key,
            dimension=self.embed_dimension,
        )
        ret_strat = retrieval_strategy or self.retrieval_strategy
        fetch_k = top_k * 2 if rerank else top_k
        hits = self.store.search(
            query_embedding=q_vec,
            query_text=search_query,
            top_k=fetch_k,
            strategy=ret_strat,
            hybrid_alpha=hybrid_alpha,
            chunk_strategy_filter=chunk_strategy_filter,
        )
        if rerank:
            from ragspark import reranker as reranker_mod
            hits = reranker_mod.rerank(search_query, hits, top_k=top_k)
        return hits[:top_k]

    def ask(
        self,
        query: str,
        top_k: int = 5,
        retrieval_strategy: str | None = None,
        hybrid: bool = False,
        hybrid_alpha: float = 0.65,
        rerank: bool = False,
        chunk_strategy_filter: str | list[str] | None = None,
        language: str | None = None,
        prompt_technique: str | None = None,
        technique: str | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
        llm_provider: str | None = None,
        provider: str | None = None,
        llm_model: str | None = None,
        model: str | None = None,
        llm_api_key: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        rewrite: bool | None = None,
    ) -> RAGResponse:
        """
        1. Automatically rewrites query
        2. Retrieves relevant chunks (with optional Hybrid Search and Flash Re-ranking)
        3. Builds adaptive prompt (with optional Multilingual target language support)
        4. Generates grounded answer via LLM runtime
        """
        ret_strat = "hybrid" if hybrid else (retrieval_strategy or self.retrieval_strategy)
        rw_info = self.rewrite_query(query)
        hits = self.retrieve(
            query,
            top_k=top_k,
            retrieval_strategy=ret_strat,
            hybrid_alpha=hybrid_alpha,
            rerank=rerank,
            chunk_strategy_filter=chunk_strategy_filter,
            rewrite=rewrite,
        )
        chosen_tech = prompt_technique or technique or prompt or self.prompt_technique
        chosen_sys = system_prompt or self.system_prompt
        chosen_model = llm_model or model or self.llm_model
        explicit_prov = llm_provider or provider
        raw_prov = explicit_prov or (
            "auto" if (chosen_model and self.llm_provider == "extractive") else self.llm_provider
        )
        resolved_prov, resolved_model = llm_mod.resolve_model_and_provider(
            provider=raw_prov,
            model=chosen_model,
        )
        chosen_key = llm_api_key or api_key or self.llm_api_key
        chosen_url = base_url or self.llm_base_url
        chosen_lang = language or self.language

        if self.mode == "law":
            built = prompt_mod.build_law(
                query=query,
                context_items=hits,
                technique=chosen_tech or "natural_law",
                system_prompt=chosen_sys,
                user_template=user_template,
                language=chosen_lang,
            )
        else:
            built = prompt_mod.build(
                query=query,
                context_items=hits,
                technique=chosen_tech or "standard",
                system_prompt=chosen_sys,
                user_template=user_template,
                language=chosen_lang,
            )

        answer = llm_mod.generate(
            prompt=built,
            provider=resolved_prov,
            model=resolved_model,
            system_prompt=built.system_prompt,
            api_key=chosen_key,
            base_url=chosen_url,
        )

        return RAGResponse(
            query=query,
            answer=str(answer),
            sources=hits,
            built_prompt=built,
            mode=self.mode,
            llm_provider=f"{resolved_prov}:{resolved_model}",
            vectordb_provider=self.store.provider,
            retrieval_strategy=ret_strat,
            rewritten_query=str(rw_info),
            is_story=rw_info.is_story,
            wants_irac=rw_info.wants_irac,
            language=chosen_lang,
            metadata={
                "doc_type": self.document.doc_type,
                "num_chunks_indexed": self.store.count(),
                "indexed_chunk_strategies": self.store.list_indexed_strategies(),
                "technique": built.technique,
                "embed_strategy": self.embed_strategy,
                "retrieval_strategy": ret_strat,
                "resolved_model": resolved_model,
                "rewritten_query": str(rw_info),
                "is_story": rw_info.is_story,
                "wants_irac": rw_info.wants_irac,
                "detected_concepts": rw_info.detected_concepts,
                "language": chosen_lang,
            },
        )

    def chat(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
        hybrid: bool = False,
        language: str | None = None,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Multi-Turn Conversational Q&A.
        Maintains conversation history and contextually grounds follow-up questions.
        Supports Multilingual answering (`language="Hindi"`, `"Spanish"`, etc.).
        """
        search_query = query
        if self.history:
            last_user_turns = [m["content"] for m in self.history if m["role"] == "user"]
            if last_user_turns:
                last_user = last_user_turns[-1]
                q_lower = query.lower()
                # Check for pronoun or continuation follow-ups
                if any(p in q_lower for p in (" it", " this", " that", " he", " she", " they", " its", " his", " her", " also", " what about")):
                    search_query = f"{last_user} {query}"

        res = self.ask(
            query=search_query,
            top_k=top_k,
            rerank=rerank,
            hybrid=hybrid,
            language=language or self.language,
            **kwargs,
        )
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": res.answer})
        return res

    def clear_history(self) -> None:
        """Clear multi-turn conversational chat history."""
        self.history.clear()


    def compare_retrieval(
        self,
        query: str,
        strategies: list[str] | None = None,
        top_k: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run an instant side-by-side retrieval experiment across multiple retrieval techniques."""
        strats = strategies or ["semantic", "bm25", "hybrid", "rrf", "mmr", "rerank"]
        report: dict[str, list[dict[str, Any]]] = {}
        for st in strats:
            hits = self.retrieve(query, top_k=top_k, retrieval_strategy=st)
            report[st] = [
                {
                    "rank": h.rank,
                    "score": round(h.score, 4),
                    "chunk_strategy": h.chunk.strategy,
                    "section_path": h.chunk.section_path,
                    "preview": h.chunk.text[:140],
                }
                for h in hits
            ]
        return report


def _resolve_source_doc(source: str | Path | ParsedDocument, force_law: bool = False) -> ParsedDocument:
    if isinstance(source, ParsedDocument):
        return source
    if isinstance(source, Path):
        return parser_mod.parse_law(source) if force_law else parser_mod.parse(source, mode="auto")
    if isinstance(source, str) and "\n" not in source and len(source) < 260:
        cand = Path(source)
        if cand.exists():
            return parser_mod.parse_law(cand) if force_law else parser_mod.parse(cand, mode="auto")
        if cand.suffix.lower() in (".pdf", ".txt", ".md", ".markdown"):
            raise FileNotFoundError(f"File not found: {cand}")
    return parser_mod.parse_text(str(source), mode="law" if force_law else "auto")


def create(
    source: str | Path | ParsedDocument,
    chunk_strategy: str | list[str] = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    overlap: int | None = None,
    combination_mode: str = "ensemble",
    embed_provider: str = "fast_local",
    embed_strategy: str = "dense",
    embed_model: str | None = None,
    embed_api_key: str | None = None,
    embed_dimension: int = 384,
    retrieval_strategy: str = "hybrid",
    vectordb: str = "chroma",
    backend: str | None = None,
    collection_name: str = "ragspark_general",
    persist_directory: str | Path | None = None,
    prompt_technique: str = "standard",
    prompt: str | None = None,
    technique: str | None = None,
    system_prompt: str | None = None,
    llm_provider: str = "extractive",
    provider: str | None = None,
    llm_model: str | None = None,
    model: str | None = None,
    llm_api_key: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    rewrite_queries: bool = True,
    language: str | None = None,
    **vectordb_kwargs: Any,
) -> RAGPipeline:
    """
    Create a full RAG pipeline for ANY general document, PDF, text file, or string.
    Supports single or combined chunking strategies (`chunk_strategy=["recursive", "semantic"]`),
    customizable embedding strategies (`embed_strategy`), customizable retrieval strategies (`retrieval_strategy`),
    prompt engineering (`prompt_technique` & `system_prompt`), LLM (`llm_provider`, `llm_model`, `llm_api_key`),
    and Multilingual Q&A (`language="Hindi"`, `"Spanish"`, `"French"`, `"auto"`).
    """
    chosen_tech = prompt or technique or prompt_technique
    chosen_llm_prov = provider or llm_provider
    chosen_llm_model = model or llm_model
    chosen_llm_key = llm_api_key or api_key
    chosen_embed_key = embed_api_key or api_key
    chosen_vdb = backend or vectordb
    eff_overlap = overlap if overlap is not None else chunk_overlap

    doc = _resolve_source_doc(source, force_law=False)
    chunks = chunker_mod.chunk(
        doc,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=eff_overlap,
        combination_mode=combination_mode,
    )
    embeddings = embedder_mod.embed(
        chunks,
        provider=embed_provider,
        strategy=embed_strategy,
        model=embed_model,
        api_key=chosen_embed_key,
        dimension=embed_dimension,
    )
    store = vectordb_mod.connect(
        provider=chosen_vdb,
        collection_name=collection_name,
        persist_directory=persist_directory,
        api_key=api_key,
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        embed_model=embed_model,
        embed_api_key=chosen_embed_key,
        embed_dimension=embed_dimension,
        **vectordb_kwargs,
    )
    store.add(chunks, embeddings)

    return RAGPipeline(
        document=doc,
        chunks=chunks,
        store=store,
        mode="general",
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        embed_model=embed_model,
        embed_api_key=chosen_embed_key,
        embed_dimension=embed_dimension,
        retrieval_strategy=retrieval_strategy,
        llm_provider=chosen_llm_prov,
        llm_model=chosen_llm_model,
        llm_api_key=chosen_llm_key,
        llm_base_url=base_url,
        prompt_technique=chosen_tech,
        system_prompt=system_prompt,
        rewrite_queries=rewrite_queries,
        language=language,
    )


def create_law(
    source: str | Path | ParsedDocument,
    chunk_strategy: str | list[str] = "legal_clause",
    chunk_size: int = 900,
    chunk_overlap: int = 100,
    overlap: int | None = None,
    combination_mode: str = "ensemble",
    embed_provider: str = "fast_local",
    embed_strategy: str = "legal_domain_boost",
    embed_model: str | None = None,
    embed_api_key: str | None = None,
    embed_dimension: int = 384,
    retrieval_strategy: str = "rerank",
    vectordb: str = "chroma",
    backend: str | None = None,
    collection_name: str = "ragspark_law",
    persist_directory: str | Path | None = None,
    prompt_technique: str = "natural_law",
    prompt: str | None = None,
    technique: str | None = None,
    system_prompt: str | None = None,
    llm_provider: str = "extractive",
    provider: str | None = None,
    llm_model: str | None = None,
    model: str | None = None,
    llm_api_key: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    rewrite_queries: bool = True,
    language: str | None = None,
    **vectordb_kwargs: Any,
) -> RAGPipeline:
    """
    Specialized One-Command Law RAG Pipeline:
    1. Parses with Legal Regex Intelligence (`parser.parse_law`)
    2. Chunks by statutory/contractual hierarchy (`chunk_strategy="legal_clause"` or combined `["legal_clause", "semantic"]`)
    3. Embeds with `embed_strategy="legal_domain_boost"`
    4. Indexes into VectorDB (`vectordb="chroma"` by default) and retrieves with 2-stage legal reranking (`retrieval_strategy="rerank"`)
    5. Automatically rewrites Legal & Story Queries (`rewrite_law_query`) to map stories/scenarios to exact Articles/Sections
    6. Generates natural, fluent legal sentences by default, and switches to IRAC only when the user writes 'IRAC' in their query!
    7. Supports Multilingual answers (`language="Hindi"`, `"Spanish"`, `"French"`, `"auto"`).
    """
    chosen_tech = prompt or technique or prompt_technique
    chosen_llm_prov = provider or llm_provider
    chosen_llm_model = model or llm_model
    chosen_llm_key = llm_api_key or api_key
    chosen_embed_key = embed_api_key or api_key
    chosen_vdb = backend or vectordb
    eff_overlap = overlap if overlap is not None else chunk_overlap

    doc = _resolve_source_doc(source, force_law=True)
    if chunk_strategy == "legal_clause":
        chunks = chunker_mod.chunk_law(doc, chunk_size=chunk_size, chunk_overlap=eff_overlap)
    else:
        chunks = chunker_mod.chunk(
            doc,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=eff_overlap,
            combination_mode=combination_mode,
        )

    embeddings = embedder_mod.embed(
        chunks,
        provider=embed_provider,
        strategy=embed_strategy,
        model=embed_model,
        api_key=chosen_embed_key,
        dimension=embed_dimension,
    )
    store = vectordb_mod.connect(
        provider=chosen_vdb,
        collection_name=collection_name,
        persist_directory=persist_directory,
        api_key=api_key,
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        embed_model=embed_model,
        embed_api_key=chosen_embed_key,
        embed_dimension=embed_dimension,
        **vectordb_kwargs,
    )
    store.add(chunks, embeddings)

    return RAGPipeline(
        document=doc,
        chunks=chunks,
        store=store,
        mode="law",
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        embed_model=embed_model,
        embed_api_key=chosen_embed_key,
        embed_dimension=embed_dimension,
        retrieval_strategy=retrieval_strategy,
        llm_provider=chosen_llm_prov,
        llm_model=chosen_llm_model,
        llm_api_key=chosen_llm_key,
        llm_base_url=base_url,
        prompt_technique=chosen_tech,
        system_prompt=system_prompt,
        rewrite_queries=rewrite_queries,
        language=language,
    )


def ask(
    source: str | Path | ParsedDocument,
    query: str,
    top_k: int = 5,
    hybrid: bool = False,
    hybrid_alpha: float = 0.65,
    rerank: bool = False,
    chunk_strategy: str | list[str] = "recursive",
    embed_provider: str = "fast_local",
    embed_strategy: str = "dense",
    retrieval_strategy: str = "hybrid",
    vectordb: str = "chroma",
    prompt_technique: str = "standard",
    system_prompt: str | None = None,
    llm_provider: str = "extractive",
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    language: str | None = None,
) -> RAGResponse:
    """One-line general RAG execution with optional Hybrid Search, Re-ranking, and Multilingual answers."""
    pipe = create(
        source=source,
        chunk_strategy=chunk_strategy,
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        retrieval_strategy=retrieval_strategy,
        vectordb=vectordb,
        prompt_technique=prompt_technique,
        system_prompt=system_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key or api_key,
        base_url=base_url,
        language=language,
    )
    return pipe.ask(
        query=query,
        top_k=top_k,
        hybrid=hybrid,
        hybrid_alpha=hybrid_alpha,
        rerank=rerank,
        language=language,
    )


def chat(
    source: str | Path | ParsedDocument,
    query: str,
    top_k: int = 5,
    hybrid: bool = False,
    rerank: bool = False,
    language: str | None = None,
    **kwargs: Any,
) -> RAGResponse:
    """Convenience multi-turn conversational chat function with optional multilingual support."""
    pipe = create(source=source, language=language, **kwargs)
    return pipe.chat(query=query, top_k=top_k, hybrid=hybrid, rerank=rerank, language=language)


def ask_law(
    source: str | Path | ParsedDocument,
    query: str,
    top_k: int = 5,
    chunk_strategy: str | list[str] = "legal_clause",
    embed_provider: str = "fast_local",
    embed_strategy: str = "legal_domain_boost",
    retrieval_strategy: str = "rerank",
    vectordb: str = "chroma",
    prompt_technique: str = "natural_law",
    system_prompt: str | None = None,
    llm_provider: str = "extractive",
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    language: str | None = None,
) -> RAGResponse:
    """Specialized One-Line Law RAG execution with optional multilingual support."""
    pipe = create_law(
        source=source,
        chunk_strategy=chunk_strategy,
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        retrieval_strategy=retrieval_strategy,
        vectordb=vectordb,
        prompt_technique=prompt_technique,
        system_prompt=system_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key or api_key,
        base_url=base_url,
        language=language,
    )
    return pipe.ask(query=query, top_k=top_k, language=language)


def load(
    collection_name: str = "ragspark_collection",
    persist_directory: str | Path | None = None,
    mode: str = "general",
    embed_provider: str = "fast_local",
    embed_strategy: str = "dense",
    embed_model: str | None = None,
    embed_api_key: str | None = None,
    embed_dimension: int = 384,
    retrieval_strategy: str = "hybrid",
    prompt_technique: str | None = None,
    system_prompt: str | None = None,
    llm_provider: str = "extractive",
    provider: str | None = None,
    llm_model: str | None = None,
    model: str | None = None,
    llm_api_key: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    rewrite_queries: bool = True,
    language: str | None = None,
    **vectordb_kwargs: Any,
) -> RAGPipeline:
    """
    Instant 1-Line Pipeline Loader (`bot = rag.load()`):
    Reloads your already-chunked & embedded VectorDB from disk (`.ragspark_store`) in <0.05s
    so you NEVER have to re-run Document Parsing, Chunking, or Embedding cells!
    """
    store = vectordb_mod.load(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        embed_model=embed_model,
        embed_api_key=embed_api_key or api_key,
        embed_dimension=embed_dimension,
        **vectordb_kwargs,
    )
    chunks = list(store._chunks)
    src_name = chunks[0].source if chunks else collection_name
    doc_type = str(chunks[0].metadata.get("doc_type", "unstructured")) if chunks else "unstructured"
    placeholder_doc = ParsedDocument(
        source=src_name,
        doc_type=doc_type,
        mode_used=mode,
        engine_used="disk_cache",
        raw_text="",
        text="\n\n".join(c.text for c in chunks[:20]),
        pages=[],
    )
    return RAGPipeline(
        document=placeholder_doc,
        chunks=chunks,
        store=store,
        mode=mode,
        embed_provider=embed_provider,
        embed_strategy=embed_strategy,
        embed_model=embed_model,
        embed_api_key=embed_api_key or api_key,
        embed_dimension=embed_dimension,
        retrieval_strategy=retrieval_strategy,
        llm_provider=provider or llm_provider,
        llm_model=model or llm_model,
        llm_api_key=llm_api_key or api_key,
        llm_base_url=base_url,
        prompt_technique=prompt_technique or ("natural_law" if mode == "law" else "standard"),
        system_prompt=system_prompt,
        rewrite_queries=rewrite_queries,
        language=language,
    )


rewrite_query = prompt_mod.rewrite_query
rewrite_law_query = prompt_mod.rewrite_law_query


__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "create",
    "create_law",
    "load",
    "ask",
    "chat",
    "ask_law",
    "rewrite_query",
    "rewrite_law_query",
]


