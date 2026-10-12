# 📘 `ragspark` / `raglib` (v0.3.0) — Complete Official Reference Guide

---

## 🌟 1. About `ragspark` (`raglib`)

**`ragspark`** (also importable as **`raglib`**) is an industry-grade, modular, laptop-friendly Python **Retrieval-Augmented Generation (RAG)** library built for **both general documents** (books, research papers, manuals, notes, PDFs) **and specialized legal documents** (the Constitution of India, Statutes, Acts, Judgments, and Contracts).

### Core Highlights:
* **🔁 Dual Compatible Imports:** Both `from ragspark import ...` and `from raglib import ...` work identically out of the box!
* **⚡ Ultra-Fast PDF Parsing (`< 0.5s`):** Uses C-level **PyMuPDF (`fitz`)** block extraction so 30–400+ page PDFs parse in milliseconds without freezing laptop CPU or RAM.
* **🗣️ Natural Sentence Answers by Default (Conditional IRAC):** Generates fluent, ChatGPT-like sentence and paragraph explanations by default—and **automatically switches to 4-part IRAC format** (`Issue, Rule, Application, Conclusion`) **only when the user writes `"in IRAC"` in their query**.
* **📖 Legal Story & Scenario Question Solver:** Understands real-life hypothetical stories (e.g., *"Ramesh was arrested and kept in lockup for 3 days without a magistrate..."*), identifies the characters (`actors`), maps the story to exact Constitutional Articles (`Article 22`, `Article 21`, `Article 19`), and answers the story directly in natural sentences.
* **🔄 Built-In Query Rewriting (General + Law-Specialized):** Includes `rewrite_query()` for general RAG and `rewrite_law_query()` for translating everyday/story questions into statutory and Constitutional terminology.
* **✂️ 10 Chunking Strategies + Combinations:** Single (`"recursive"`, `"legal_clause"`, `"semantic"`, etc.) or combined (`["legal_clause", "semantic"]` ensemble or `"legal_clause>sentence"` chain), supporting both `chunk_overlap=` and `overlap=`.
* **🗄️ 1-Line Vector Indexing + 7 Retrieval Techniques:** Index and search in one line via `vectordb.store(chunks)` and `db.search("query")`, or customize across `"semantic"`, `"bm25"`, `"hybrid"`, `"rrf"`, `"mmr"`, `"rerank"`, and `"multi_query"`.
* **💬 Clean, Customizable ChatGPT-Style UI (Web & Jupyter):** Renders rich Markdown, blockquotes, and citation pills in a clean Question & Answer chat window (`ui.launch(bot)` or `ui.notebook_chat(bot)`).

---

## ⚡ 2. Quick Overview — All Simple Commands `ragspark` / `raglib` Can Do

```python
# 0. Import all 8 modules (via `ragspark` or `raglib`)
from ragspark import parser, chunker, embedder, vectordb, prompt_eng, llm, rag, ui

# ─── A. SIMPLE GENERAL RAG COMMANDS (For Any PDF, Text, or Document) ───
doc      = parser.parse("notes.pdf")                                                  # 1. Parse PDF/Text
chunks   = chunker.chunk(doc, strategy="recursive", chunk_size=700, overlap=80)       # 2. Chunk text
vectors  = embedder.embed(chunks, provider="fast_local", strategy="dense")            # 3. Embed chunks
gen_rw   = prompt_eng.rewrite_query("How does photosynthesis work?", strategy="hyde") # 4. Rewrite general query
q_vec    = embedder.embed_query(gen_rw, provider="fast_local", strategy="dense")      # 5. Embed query
db       = vectordb.store(chunks, backend="chroma", collection_name="my_docs")        # 6a. 1-Line Auto-Embed & Store
hits     = db.search("How does photosynthesis work?", strategy="hybrid", top_k=3)     # 6b. 1-Line Auto-Embed Search
prompt   = prompt_eng.build("How does photosynthesis work?", hits)                    # 7. Build Prompt
answer   = llm.generate(prompt, provider="extractive", model="extractive")            # 8. Generate Answer
bot      = rag.create("notes.pdf", chunk_strategy="recursive", vectordb="chroma")     # 9. 1-Line General RAG
res      = bot.ask("Summarize the key points.")                                       # 9b. Ask General Bot
server   = ui.launch(bot, title="My Study Bot", theme="dark", background=True)        # 10a. Launch Clean Web UI
nb_srv   = ui.notebook_chat(bot, title="My Study Bot")                                # 10b. Launch Jupyter/Colab UI

# ─── B. SPECIALIZED LAW RAG COMMANDS (For Constitution, Acts, Judgments & Contracts) ───
law_doc  = parser.parse_law("constitution_of_india.pdf")                              # 1. Legal Regex PDF Parser
law_chks = chunker.chunk_law(law_doc, chunk_size=700, overlap=100)                    # 2. Legal Hierarchy Chunker
law_vecs = embedder.embed(law_chks, strategy="legal_domain_boost")                    # 3. Legal Domain-Boost Embed
law_rw   = prompt_eng.rewrite_law_query("Ramesh was arrested without warrant...")     # 4. Law & Story Query Rewriter
law_hits = db.search(q_vec, query_text=law_rw, strategy="rerank", top_k=3)            # 5. 2-Stage Legal Reranker
law_pmpt = prompt_eng.build_law(law_rw.original, law_hits, technique="natural_law")   # 6. Adaptive Law Prompt
law_bot  = rag.create_law("constitution_of_india.pdf", retrieval_strategy="rerank")   # 7. 1-Line Law RAG Pipeline
ans_sent = law_bot.ask("What is Article 21?")                                         # 8a. Natural Sentence Answer
ans_stry = law_bot.ask("Ramesh was detained for 3 days without a magistrate. Advise.")# 8b. Story Scenario Answer
ans_irac = law_bot.ask("Explain Article 19 in IRAC format")                           # 8c. Auto IRAC Answer
```

---

## 🛠️ 3. Part 1 — General / Simple RAG Commands (`Command` • `Details` • `How It Works`)

Use these commands for **any general document** (books, research papers, manuals, notes, PDFs, or text files).

---

### 📄 1. `parser` — General Document Parsing
**Import:** `from ragspark import parser` *(or `from raglib import parser`)*

| Command | Details (Function & Outputs) | How It Works |
| :--- | :--- | :--- |
| `doc = parser.parse("file.pdf")` | Parses any PDF, `.txt`, `.md` file, or inline string into a `ParsedDocument`.<br>**Outputs:** `doc.total_pages`, `doc.text`, `doc.pages`, `doc.headings`, `doc.tables`, `doc.to_markdown()`. | Uses C-level **PyMuPDF (`fitz`)** block extraction (`<0.5s` for 30+ pages), removes repeating headers/footers, and unwraps broken lines. |
| `doc = parser.parse_unstructured("file.pdf")` | Explicitly runs general layout & paragraph extraction without legal regexes. | Extracts clean reading-order text blocks and optional tables (`extract_tables="fast"` or `"deep"`). |
| `doc = parser.parse_text(raw_str)` | Parses a raw Python string directly into a `ParsedDocument`. | Cleans whitespace, normalizes line breaks, and wraps text into a standard `ParsedDocument`. |
| `docs = parser.parse_dir("./pdfs")` | Batch-parses all matching PDFs in a folder into `list[ParsedDocument]`. | Iterates through matching files in the directory and parses each with `parser.parse()`. |

---

### ✂️ 2. `chunker` — General Text Chunking (10 Strategies + Combinations)
**Import:** `from ragspark import chunker` *(or `from raglib import chunker`)*

| Command | Details (Function & Strategies) | How It Works |
| :--- | :--- | :--- |
| `chunks = chunker.chunk(doc, strategy="recursive", chunk_size=700, overlap=80)` | Splits `doc` using a single strategy (accepts `overlap=` or `chunk_overlap=`).<br>**General Strategies:** `"recursive"`, `"fixed"`, `"sentence"`, `"token"`, `"paragraph"`, `"markdown"`, `"semantic"`, `"sliding_window"`, `"parent_child"`. | Splits text while preserving sentence/paragraph boundaries and attaches `.page_number`, `.strategy`, and `.metadata` to each `Chunk`. |
| `chunks = chunker.chunk(doc, strategy=["recursive", "semantic"])` | **Ensemble Combination:** Runs multiple strategies together and indexes them side-by-side. | Executes each strategy in the list, tags every chunk with its `chunk.strategy`, and deduplicates identical text spans. |
| `chunks = chunker.chunk(doc, strategy="markdown>sentence")` | **Chain Combination (`>`):** Coarse-to-fine multi-stage chunking. | Splits first by the outer strategy (`markdown`), then subdivides large sections using the inner strategy (`sentence`). |
| `chunker.list_strategies()` | Returns all 10 supported chunking strategy names. | Lists built-in strategies available for single or combined indexing. |

---

### 🔢 3. `embedder` — Multi-Provider & Strategy Embeddings
**Import:** `from ragspark import embedder` *(or `from raglib import embedder`)*

| Command | Details (Providers & Strategies) | How It Works |
| :--- | :--- | :--- |
| `vectors = embedder.embed(chunks, provider="fast_local", strategy="dense")` | Embeds chunks into `list[list[float]]`.<br>**Providers:** `"fast_local"` (offline default), `"ollama"`, `"gemini"`, `"openai"`, `"sentence_transformers"`.<br>**Strategies:** `"dense"`, `"hybrid_sparse_dense"`, `"context_enriched"`. | Enriches text based on `strategy` and computes $L_2$-normalized vectors (384-d by default). |
| `q_vec = embedder.embed_query("Your question?", provider="fast_local")` | Embeds a single search query into `list[float]`. | Applies the same provider and strategy transformation to the query string for accurate cosine similarity matching. |
| `embedder.register_custom("my_emb", fn)` | Registers a custom Python embedding function `fn(texts, dim) -> list[list[float]]`. | Allows plugging in any custom or proprietary embedding model into `embedder.embed()`. |

---

### 🗄️ 4. `vectordb` — ChromaDB Default, 1-Line `store()`, & 7 Retrieval Techniques
**Import:** `from ragspark import vectordb` *(or `from raglib import vectordb`)*

| Command | Details (Databases & Retrieval Techniques) | How It Works |
| :--- | :--- | :--- |
| `db = vectordb.store(chunks, backend="chroma", collection_name="my_docs")` | **1-Line Vector Indexer:** Connects to the database, automatically embeds `chunks` if `embeddings` is omitted, stores them, and returns `VectorStore`. | Supports `backend=` or `provider=` across `"chroma"` (default), `"pinecone"`, `"mongodb"`, `"qdrant"`, and `"memory"`. |
| `db = vectordb.connect("chroma", collection_name="docs")` | Connects to an existing or new VectorDB collection. | Initializes ChromaDB (in-memory or persistent on disk via `persist_directory="./chroma_db"`) or connects to external cloud vector stores. |
| `db.add(chunks, vectors)` *(or `db.add(chunks)`)* | Stores chunks, embeddings, page numbers, and `chunk_strategy` tags in the database. | Upserts documents, vectors, and metadata into the active collection (auto-embedding if `vectors` is omitted). |
| `hits = db.search("Your question?", strategy="hybrid", top_k=5)`<br>*(or `db.search(q_vec, query_text=q, ...)`)* | Searches the DB using a direct query string OR pre-computed query vector across **7 Retrieval Techniques**:<br>1. `"semantic"` (dense cosine)<br>2. `"bm25"` / `"keyword"` (exact lexical)<br>3. `"hybrid"` (dense + BM25)<br>4. `"rrf"` (Reciprocal Rank Fusion)<br>5. `"mmr"` (Maximal Marginal Relevance)<br>6. `"rerank"` (2-stage cross-feature reranker)<br>7. `"multi_query"` (query expansion + RRF). | Auto-embeds string queries using the store's configured embedding strategy, computes candidate scores (with optional `chunk_strategy_filter`), and returns ranked `SearchResult` objects (`hit.chunk.text`, `hit.score`). |

---

### 🔄 5. `prompt_eng` — General Query Rewriting & Prompt Engineering
**Import:** `from ragspark import prompt_eng` *(or `from raglib import prompt_eng`)*

| Command | Details (Parameters & Techniques) | How It Works |
| :--- | :--- | :--- |
| `rw = prompt_eng.rewrite_query("Can you tell me...", strategy="auto")` | **General Query Rewriter:** Cleans and expands conversational queries.<br>**Strategies:** `"auto"`, `"hyde"` (Hypothetical Document Embeddings), `"step_back"`, `"sub_queries"`. | Strips conversational filler, extracts core domain keywords, generates sub-queries (`rw.sub_queries`), and detects whether the user asked a story (`rw.is_story`) or requested IRAC (`rw.wants_irac`). |
| `prompt = prompt_eng.build(hits, query=rw.rewritten_query, technique="standard")` | Builds a grounded `BuiltPrompt` from retrieved `hits` (accepts `(query, hits)` or `(hits, query=...)`).<br>**Techniques:** `"standard"`, `"cot"`, `"citation_strict"`, `"few_shot"`, `"concise"`. | Formats retrieved chunks with `[Page X]` citations and instructs the LLM to answer in natural, flowing sentences (switching to IRAC only if the user's query asks for IRAC). |

---

### 🧠 6. `llm` — Smart Multi-Runtime LLM Answer Generator
**Import:** `from ragspark import llm` *(or `from raglib import llm`)*

| Command | Details (Supported Runtimes & Models) | How It Works |
| :--- | :--- | :--- |
| `answer = llm.generate(prompt, provider="extractive", model="extractive")` | **Offline Mode:** Generates natural, fluent sentence answers without any API key.<br>**Output:** `answer` (works with both `print(answer)` and `print(answer.text)`). | Strips chunk debug headers, cleans prose, and synthesizes natural flowing sentences citing `[Section]` and `(Page X)`. Automatically formats in IRAC if `"irac"` is in the query. |
| `answer = llm.generate(prompt, provider="gemini", model="gemini 3.8 flash", api_key="...")` | **Cloud API / Local Ollama Mode:**<br>**Providers:** `"gemini"`, `"anthropic"` (`"claude sonnet 5.5"`), `"openai"` (`"gpt 4o"`), `"groq"`, `"openrouter"`, `"ollama"` (`"llama 3.2"`). | Resolves human-friendly model names (`"gemini 3.8 flash"` $\rightarrow$ `gemini-3.8-flash`), calls the provider's API or local Ollama server, and falls back safely if offline. |

---

### 🚀 7. `rag` — All-in-One General RAG Pipeline
**Import:** `from ragspark import rag` *(or `from raglib import rag`)*

| Command | Details | How It Works |
| :--- | :--- | :--- |
| `bot = rag.create("notes.pdf", chunk_strategy="recursive", retrieval_strategy="hybrid", vectordb="chroma", llm_provider="extractive", llm_model="extractive", llm_api_key="...")` | Builds the complete general RAG pipeline in a single command (`rewrite_queries=True` by default). | Automatically runs `parser.parse` $\rightarrow$ `chunker.chunk` $\rightarrow$ `embedder.embed` $\rightarrow$ `vectordb.connect` and returns a stateful `RAGPipeline` (`bot`). |
| `res = bot.ask("Summarize the document.")` | Rewrites the query, retrieves top chunks, and generates a natural sentence answer (`res.answer`, `res.rewritten_query`, `res.sources`). | Runs automatic query rewriting + hybrid retrieval + adaptive prompt building + LLM generation. |
| `report = bot.compare_retrieval("Your question?", top_k=3)` | Side-by-side retrieval benchmark across `"semantic"`, `"bm25"`, `"hybrid"`, `"rrf"`, `"mmr"`, and `"rerank"`. | Runs all 6 retrieval strategies on the same query and returns a comparison dictionary of top hits and scores. |

---

### 💬 8. `ui` — Clean, Customizable ChatGPT-Style Chatbot UI (Web & Notebook)
**Import:** `from ragspark import ui` *(or `from raglib import ui`)*

| Command | Details (Customization Options) | How It Works |
| :--- | :--- | :--- |
| `server = ui.launch(bot, background=True)` | Launches the clean web Chatbot UI at `http://127.0.0.1:8501` in the background. | Starts a zero-dependency local web server showing **only** the Question & Answer chat window + bottom message bar. |
| `server = ui.notebook_chat(bot, height=620)` | Renders the interactive Chatbot UI directly inside Jupyter Notebook / JupyterLab / Google Colab. | Starts the background UI server and embeds it inline via an `IPython.display.IFrame`. |
| `server = ui.launch(bot, title="...", bot_name="...", theme="emerald", accent_color="#10b981", bg_color="#061411", card_color="#0d2620", text_color="#ecfdf5", font_size="15px", show_sources=True, allow_ui_edit=True, background=True)` | Full UI customization via Python parameters + live **`🎨 Edit UI`** button in the browser header. | Renders bot answers with full **ChatGPT-style Markdown formatting** (headings, bold text, blockquotes, bullet lists, and inline `[Article]` citation pills). |

---
---

## ⚖️ 4. Part 2 — Specialized Law RAG Commands (`Command` • `Details` • `How It Works`)

Use these **Law-Specific Commands** when working with the **Constitution of India**, **Acts & Statutes (IPC/BNS, CrPC/BNSS, Contract Act)**, **Supreme Court / High Court Judgments**, or **Contracts**.

---

### ⚖️ 1. Specialized Legal Commands Reference Table

| Command | Details (What It Does & Outputs) | How It Works (Legal Intelligence Under the Hood) |
| :--- | :--- | :--- |
| **`doc = parser.parse_law("constitution_of_india.pdf")`** | Parses legal PDFs using the Legal Regex Engine.<br>**Outputs:** `doc.total_pages`, `doc.clauses` (or `doc.articles`), `doc.citations`, `doc.definitions`, `doc.obligations`, `doc.legal_metadata`. | Single-pass $O(N)$ regex state machine (`raglib/parser/legal_regex.py`) extracts **Parts (`PART I`–`XXII`, `PART IVA`)**, **Articles (`14.`, `21A.`)**, **Sections**, **Schedules (`FIRST`–`TWELFTH`)**, Provisos (`Provided that`), Case Citations (`AIR`, `SCC`), and Defined Terms. |
| **`chunks = chunker.chunk_law(doc, chunk_size=700, overlap=100)`** | Splits strictly along legal boundaries (`PART`, `CHAPTER`, `ARTICLE`, `SECTION`) instead of cutting mid-clause. | Stamps every chunk with its exact legal hierarchy breadcrumb header (e.g., `[PART III > ARTICLE 21 — Protection of life and personal liberty]`) so context is never lost. |
| **`vecs = embedder.embed(chunks, strategy="legal_domain_boost")`** | Computes legal domain-boosted vector embeddings. | Upweights statutory identifiers (`Article 14`, `Article 19(1)(a)`, `Article 21A`, `Section 302`), section paths, and binding modal verbs (`shall`, `notwithstanding`, `provided that`). |
| **`rw = prompt_eng.rewrite_law_query(query)`**<br>*(or `rag.rewrite_law_query(query)`)* | **Specialized Law & Story Query Rewriter:**<br>**Outputs:** `rw.rewritten`, `rw.is_story`, `rw.wants_irac`, `rw.actors` (e.g. `["Ramesh"]`), `rw.detected_concepts`. | Maps everyday language & **real-life story scenarios** (e.g., *arrest without magistrate for 3 days* $\rightarrow$ `Article 22, Article 21`; *12-year-old child in factory* $\rightarrow$ `Article 24, Article 23`; *journalist speech* $\rightarrow$ `Article 19(1)(a), 19(2)`) into exact Constitutional Articles & statutory terms. |
| **`hits = db.search(q_vec, query_text=rw, strategy="rerank")`** | **2-Stage Legal Statutory Reranker** over ChromaDB. | Stage 1 pulls a wide candidate pool via Hybrid search; Stage 2 boosts chunks whose `section_path` or heading matches the exact Article/Section numbers and legal phrases in the rewritten query. |
| **`prompt = prompt_eng.build_law(query, hits, system_prompt="...")`** | Builds an **Adaptive Legal Prompt** (`technique="natural_law"` by default). | Instructs the LLM to write in **fluent, natural sentences** citing `[Part > Article]` inline, directly solving **Story/Scenario questions**, and **switching to IRAC only if the user writes `"IRAC"` in their query**. |
| **`bot = rag.create_law("constitution_of_india.pdf", ...)`** | **1-Command Law RAG Pipeline** combining all legal stages (`parse_law` + `chunk_law` + `legal_domain_boost` + `rerank` + `rewrite_law_query` + adaptive sentence/story/IRAC generation). | Returns a ready-to-use legal `RAGPipeline` (`bot`) that answers direct legal questions, hypothetical story questions, and on-demand IRAC questions. |
| **`res = rag.ask_law("constitution_of_india.pdf", "Question?")`** | One-line instant legal question & answer. | Builds the law pipeline and immediately returns `RAGResponse` (`res.answer`). |

---

### ⚖️ 2. Complete Copy-Paste Law Demo (Normal Sentences + Story Questions + On-Demand IRAC)

```python
from ragspark import parser, chunker, embedder, vectordb, prompt_eng, llm, rag, ui

# 1. Create the Specialized Law Pipeline (with Prompt, LLM Provider, Model & API Key)
bot = rag.create_law(
    "constitution_of_india.pdf",
    chunk_strategy=["legal_clause", "semantic"],   # Combine Legal Hierarchy + Semantic chunking
    embed_provider="fast_local",                   # "fast_local" (offline), "ollama", "gemini", "openai"
    embed_strategy="legal_domain_boost",           # Boosts Article/Section numbers & legal terms
    retrieval_strategy="rerank",                   # 2-stage legal statutory reranker
    vectordb="chroma",                             # Default ChromaDB
    prompt_technique="natural_law",                # Natural sentences by default; auto-IRAC when asked!
    system_prompt="""You are a Constitutional Law Bench Senior Counsel of the Supreme Court of India.
    Answer questions in clear, natural sentences strictly based on the Constitution of India context.
    Always cite the exact Part, Article, Clause, Sub-clause, Provisos, and Schedules.""",
    llm_provider="extractive",                     # "extractive" (offline), "ollama", "gemini", "openai", "anthropic"
    llm_model="extractive",                        # Or "gemini 3.8 flash", "claude sonnet 5.5", "gpt 4o", "llama 3.2"
    llm_api_key="YOUR_API_KEY_HERE"                # Optional when using "extractive" or local "ollama"
)

# ─── A. Ask a Direct Legal Question (Generates Fluent Natural Sentences) ───
res1 = bot.ask("What is Article 19(1)(a) of the Constitution?")
print("Direct Answer:\n", res1.answer)

# ─── B. Ask a Story / Hypothetical Scenario Question (Law Story Solver + Query Rewriter) ───
story_q = (
    "Ramesh, a journalist in Delhi, published an article criticizing a policy. The police arrested him "
    "and kept him in custody for 3 days without producing him before a magistrate. Which rights of Ramesh are violated?"
)
res2 = bot.ask(story_q)
print("\nRewritten Law Query:", res2.rewritten_query)
print("\nStory Answer:\n", res2.answer)

# ─── C. Ask in IRAC Format (Automatically Triggers 4-Part IRAC Because 'IRAC' Is in the Query!) ───
res3 = bot.ask("Explain freedom of speech under Article 19 in IRAC format")
print("\nIRAC Answer:\n", res3.answer)

# ─── D. Launch the Clean, Customizable ChatGPT-Style Chatbot UI ───
server = ui.launch(
    bot,
    title="🇮🇳 Constitution of India AI",
    subtitle="Ask direct questions, real-life story scenarios, or ask 'in IRAC format'",
    bot_name="Samvidhan Bot",
    theme="emerald",
    accent_color="#10b981",
    show_sources=True,
    allow_ui_edit=True,
    background=True
)
```
