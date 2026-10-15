"""CLI entrypoint for `ragspark` (parse, ask, ask-law)."""

from __future__ import annotations

import argparse
import json
import sys

from ragspark import parser, rag


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        prog="ragspark",
        description="ragspark — Unified One-Command Industry RAG Pipeline Suite",
    )
    sub = cli.add_subparsers(dest="command", required=True)

    # 1. parse command
    parse_cmd = sub.add_parser("parse", help="Parse any PDF, DOCX, CSV, JSON, HTML, or text file")
    parse_cmd.add_argument("source", help="Path to input file")
    parse_cmd.add_argument(
        "--mode",
        choices=["auto", "law", "unstructured"],
        default="auto",
        help="Parsing mode: 'auto' (default), 'law' (structural hierarchy), or 'unstructured'",
    )
    parse_cmd.add_argument("--out", "-o", help="Optional output path (.json or .md)")
    parse_cmd.add_argument(
        "--tables",
        choices=["none", "fast", "deep"],
        default="none",
        help="Table extraction mode: 'none' (default, blazing fast), 'fast' (PyMuPDF), or 'deep' (pdfplumber)",
    )

    # 2. ask command (General RAG)
    ask_cmd = sub.add_parser("ask", help="Run a one-command RAG query on any file")
    ask_cmd.add_argument("source", help="Path to PDF, DOCX, CSV, JSON, or text file")
    ask_cmd.add_argument("query", help="Question to ask")
    ask_cmd.add_argument("--strategy", default="recursive", help="Chunking strategy (recursive, sentence, token, markdown, semantic)")
    ask_cmd.add_argument("--vectordb", default="chroma", help="VectorDB backend (chroma, pinecone, mongodb, qdrant, memory)")
    ask_cmd.add_argument("--hybrid", action="store_true", help="Enable hybrid BM25 + vector search")
    ask_cmd.add_argument("--rerank", action="store_true", help="Enable cross-encoder flash re-ranking")
    ask_cmd.add_argument("--citations", action="store_true", help="Print exact sentence evidence and citations")
    ask_cmd.add_argument("--technique", default="standard", help="Prompt engineering technique (standard, cot, few_shot, citation_strict, concise)")
    ask_cmd.add_argument("--system-prompt", default=None, help="Custom system instruction for the LLM")
    ask_cmd.add_argument("--llm", default="extractive", help="LLM runtime (ollama, gemini, openai, groq, extractive)")
    ask_cmd.add_argument("--model", default=None, help="LLM model name")

    # 3. ask-law command (Specialized Structural Law RAG)
    law_cmd = sub.add_parser("ask-law", help="Run a specialized Law RAG query on a legal PDF/file")
    law_cmd.add_argument("source", help="Path to Legal PDF, TXT, or MD file")
    law_cmd.add_argument("query", help="Legal question to ask")
    law_cmd.add_argument("--vectordb", default="chroma", help="VectorDB backend (default: chroma)")
    law_cmd.add_argument("--technique", default="natural_law", help="Prompt engineering technique (default: natural_law)")
    law_cmd.add_argument("--system-prompt", default=None, help="Custom system instruction for the LLM")
    law_cmd.add_argument("--llm", default="extractive", help="LLM runtime (ollama, gemini, openai, groq, extractive)")
    law_cmd.add_argument("--model", default=None, help="LLM model name")

    # 4. ui command (Launch Chatbot & Experimentation Studio)
    ui_cmd = sub.add_parser("ui", help="Launch the ragspark Studio Chatbot Web UI")
    ui_cmd.add_argument("--port", type=int, default=8501, help="Port to run UI server on (default: 8501)")
    ui_cmd.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    ui_cmd.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")

    args = cli.parse_args(argv)

    if args.command == "ui":
        from ragspark import ui

        ui.launch(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0

    if args.command == "parse":
        tbl_arg: bool | str = False if args.tables == "none" else args.tables
        doc = parser.parse(source=args.source, mode=args.mode, extract_tables=tbl_arg)
        if args.out:
            saved_path = doc.save(args.out)
            print(f"[ragspark] Saved parsed output to: {saved_path}")
            print(json.dumps(doc.summary(), indent=2))
        else:
            print(json.dumps(doc.summary(), indent=2))
            print("\n" + "=" * 60 + "\n")
            print(doc.to_markdown()[:2500])
        return 0

    if args.command == "ask":
        res = rag.ask(
            source=args.source,
            query=args.query,
            chunk_strategy=args.strategy,
            vectordb=args.vectordb,
            hybrid=args.hybrid,
            rerank=args.rerank,
            prompt_technique=args.technique,
            system_prompt=args.system_prompt,
            llm_provider=args.llm,
            llm_model=args.model,
        )
        print(res.answer)
        if args.citations:
            print("\n" + res.format_citations())
        return 0

    if args.command == "ask-law":
        res = rag.ask_law(
            source=args.source,
            query=args.query,
            vectordb=args.vectordb,
            prompt_technique=args.technique,
            system_prompt=args.system_prompt,
            llm_provider=args.llm,
            llm_model=args.model,
        )
        print(res.answer)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
