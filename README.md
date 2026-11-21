# ⚡ `ragspark` — Unified One-Command Industry RAG Pipeline Library

[![PyPI version](https://img.shields.io/badge/pypi-v0.4.0-blue.svg)](https://pypi.org/project/ragspark/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`ragspark`** is an enterprise-ready, developer-friendly Python library that collapses the entire Retrieval-Augmented Generation (RAG) stack into **clean 1-line commands**. From multi-format ingestion with 4-layer OCR fallbacks and 15+ chunking strategies, to hybrid BM25+vector search, flash re-ranking, multilingual support, multi-turn chat memory, and sentence-level evidence citations.

Works out of the box across any industry domain: Finance, Tech, Healthcare, E-Commerce, Legal, and Customer Support.

---

## 📦 Installation

```bash
pip install ragspark
```

To install with all vector database, OCR, and LLM backends:
```bash
pip install "ragspark[full]"
```

For real local semantic embeddings (quantized ONNX on CPU, no API key):
```bash
pip install "ragspark[onnx]"
```

---

## 🚀 1-Minute Quickstart

```python
import ragspark

bot = ragspark.pipeline("annual_report.pdf", llm="groq")   # file, folder, text, or a list of them
print(bot.ask("What were the key drivers of Q4 revenue growth?"))
```

`llm=` accepts `"groq"`, `"gemini"`, `"openai"`, `"anthropic"`, `"ollama"` or a model name; API keys are
read from the environment (`GROQ_API_KEY`, `GEMINI_API_KEY`, ...). Leave it out for offline extractive answers.
`print(res.format_citations())` shows the supporting evidence.

### v0.4.0 defaults

| Concern | Default behaviour |
| :--- | :--- |
| Chunking | `chunk_size=None` chunks by natural boundaries (Part/Article/Section headings, Markdown headings, functions/classes, tables, paragraphs). Blocks over 2,500 chars are subdivided. No strategy ever cuts a word in half. |
| Noise | Table-of-contents and gazette boilerplate are dropped before indexing. |
| Embeddings | `embed_provider="auto"`: local ONNX (`ragspark[onnx]`), then Gemini/OpenAI if a key is set, then pure BM25. It never crashes. `fast_local` hashing is no longer chosen automatically. |
| Storage | In memory. Nothing is written unless you pass `persist_dir=` (SQLite, WAL, one transaction per `add()`). In-memory stores past 10,000 chunks spill to the OS temp dir. |
| Answers | Markdown heading and bullets, with no "Based on the text..." / "According to Page 1..." filler. |
| Chat | `bot.chat()` passes the recent turns to the LLM; `bot.respond(q, history)` is the stateless form. |
| UI | `ui.launch(bot)` serves a ChatGPT-style page with a conversation sidebar. If the port is busy, it uses a free one. |

```python
bot = ragspark.pipeline("contracts/", persist_dir="contract_index")      # build once, saved to disk
bot = ragspark.rag.load(persist_dir="contract_index")                    # reload later, no re-embedding
```

---

## ⚡ Core Capabilities & Feature Matrix

| Capability | What It Does | Code Example |
| :--- | :--- | :--- |
| **15 Chunking Strategies** | Recursive, semantic, table-aware, code-aware, contextual, adaptive, etc. | `chunker.chunk(doc, strategy="table_aware")` |
| **Multi-Strategy Combinations**| Combine multiple chunkers (multi-resolution, ensemble, chain, interleave) | `chunker.combine([c1, c2], mode="multi_resolution")` |
| **Multilingual Engine** | One-parameter language switching with automatic script detection | `bot.ask("What is the refund policy?", language="Hindi")` |
| **Hybrid Retrieval** | BM25 keyword matching fused with dense vector semantic search (RRF) | `bot.ask(q, hybrid=True, alpha=0.65)` |
| **Flash Cross-Encoder Re-Ranker**| Re-score top candidate passages for maximum relevance before LLM generation | `bot.ask(q, rerank=True, reranker_top_k=3)` |
| **Multi-Turn Chat Memory** | Stateful conversation tracking with sliding context memory | `bot.chat("What was our profit?"); bot.chat("How does that compare to 2024?")` |
| **Sentence-Level Citations** | Fine-grained passage verification with exact page, chunk ID, and snippet | `print(res.format_citations())` |
| **4-Layer OCR Fallback** | Native text $\to$ RapidOCR $\to$ PyTesseract $\to$ Gemini Vision AI | `parser.parse("scanned_receipt.pdf")` |
| **Instant Disk Reload** | Opt-in SQLite index (`persist_dir=`); reload without re-parsing or re-embedding | `bot = ragspark.rag.load(persist_dir="idx")` |
| **Built-in Chat UI** | ChatGPT-style page on a built-in threaded HTTP server (no Streamlit), or inline in Jupyter/Colab | `ui.launch(bot)` or `ui.notebook_chat(bot)` |

---

## 🛠 Deep Dive: Feature Highlights

### 1. 15 Advanced Chunking Strategies & Combinations

No single chunking strategy works for all data types. `ragspark` provides 15 specialized strategies and multi-strategy combiners:

```python
from ragspark import chunker, parser

doc = parser.parse("technical_spec.pdf")

# Specialized Strategies
table_chunks = chunker.chunk(doc, strategy="table_aware")     # Preserves Markdown/HTML table integrity
code_chunks  = chunker.chunk(doc, strategy="code_aware")      # Preserves AST functions & classes
ctx_chunks   = chunker.chunk(doc, strategy="contextual")      # Prepends document-level contextual headers
adapt_chunks = chunker.chunk(doc, strategy="adaptive")        # Dynamically splits based on document density

# Multi-Strategy Combinations
# 1. Multi-Resolution (Hierarchical parent-child indexing for precise search + broad LLM context)
multi_res = chunker.combine([doc], mode="multi_resolution", small_chunk=150, large_chunk=600)

# 2. Ensemble (Union of semantic + recursive chunkers with deduplication)
ensemble_chunks = chunker.combine([table_chunks, code_chunks], mode="ensemble")

# 3. Interleave (Round-robin interleaving across multiple sources)
interleaved = chunker.combine([table_chunks, code_chunks], mode="interleave")
```

### 2. Multilingual RAG in 1 Parameter

Deliver localized responses in any language without rebuilding prompt pipelines. `ragspark` detects Unicode scripts and injects native language directives:

```python
# Force output language explicitly
res = bot.ask("Summarize the main compliance requirements", language="Spanish")
print(res.answer)

res = bot.ask("Summarize the main compliance requirements", language="Hindi")
print(res.answer)

# Or detect input question language automatically
res = bot.ask("Comment fonctionne le processus d'onboarding?", language="auto")
print(res.answer)
```

### 3. Hybrid BM25 + Dense Vector Search & Flash Re-Ranker

Avoid semantic vector drift on exact acronyms, serial numbers, and entities with Reciprocal Rank Fusion (RRF) and Flash Re-ranking:

```python
# Hybrid Search + Cross-Encoder Flash Re-ranking in 1 call:
res = bot.ask(
    "What is error code E-8492 in Section 4.2?",
    hybrid=True,          # BM25 + Dense vector search
    alpha=0.65,           # Balance between semantic (0.65) and keyword (0.35)
    rerank=True,          # Cross-encoder passage re-ranking
    reranker_top_k=3,     # Send only the top 3 verified passages to the LLM
)
print(res.answer)
```

### 4. Stateful Multi-Turn Conversational Chat

Build chatbots that remember previous turns and resolve conversational pronouns:

```python
# Turn 1
ans1 = bot.chat("Who is the CEO of Acme Corp?")
print(ans1.answer)

# Turn 2 (Resolves 'he' based on conversation history)
ans2 = bot.chat("When did he join the company?")
print(ans2.answer)

# Reset conversation history anytime
bot.clear_history()
```

### 5. Sentence-Level Evidence Citations

Guarantee explainability and prevent hallucination with verifiable sentence-level citations:

```python
res = bot.ask("What are the quarterly milestone deliverables?")

# Formatted citations string
print(res.format_citations())
# Output:
# [1] annual_report.pdf (Page 12, Chunk 84): "Milestone deliverable for Q3 entails API v2 release..."
# [2] annual_report.pdf (Page 15, Chunk 97): "Q4 milestones target global deployment across 4 regions..."

# Structured citation inspection
for cite in res.citations:
    print(f"Source: {cite['source']} (Page {cite['page']}) -> {cite['sentence']}")
```

### 6. Interactive Studio UI (Jupyter & Web)

Test and demo your RAG pipelines with zero frontend boilerplate:

```python
from ragspark import ui

# 1. Inside Jupyter Notebook / Google Colab (Interactive widget):
ui.notebook_chat(bot)

# 2. Or launch the standalone web browser app:
ui.launch(port=8501)
```

---

## 💻 Command Line Interface (CLI)

`ragspark` comes equipped with a powerful CLI:

```bash
# Launch interactive web studio
ragspark ui --port 8501

# Ingest and parse any document to clean JSON
ragspark parse report.pdf --mode auto --out parsed.json

# Instant terminal Q&A on any document
ragspark ask system_manual.pdf "How do I configure the API token?"
```

---

## 📑 Modular Architecture

If you prefer building custom pipelines step-by-step, all modular components are decoupled:

```python
from ragspark import parser, chunker, vectordb, prompt_eng, llm, reranker

# Step 1: Ingestion
doc = parser.parse("document.pdf")

# Step 2: Chunking
chunks = chunker.chunk(doc, strategy="table_aware", chunk_size=500)

# Step 3: Vector Storage
db = vectordb.store(chunks, collection_name="knowledge_base")   # in memory; add persist_dir="kb" to save

# Step 4: Hybrid Retrieval
hits = db.hybrid_search("financial forecast", top_k=10, alpha=0.7)

# Step 5: Flash Re-ranking
ranker = reranker.Reranker()
top_hits = ranker.rerank("financial forecast", hits, top_k=3)

# Step 6: Prompt Engineering
prompt = prompt_eng.build(top_hits, query="financial forecast", technique="chain_of_thought")

# Step 7: Generation
answer = llm.generate(prompt, provider="gemini", model="gemini 3.8 flash")
print(answer.text)
```

---

## 📄 License

MIT License. Built for production-grade Retrieval-Augmented Generation across all domains.
