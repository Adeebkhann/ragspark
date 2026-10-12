# ⚡ `ragspark` — Unified One-Command RAG & Legal AI Pipeline Library

**`ragspark`** is an all-in-one, student- and developer-friendly Python library that turns every stage of a Retrieval-Augmented Generation (RAG) system into a **single one-line command**—from smart PDF parsing and multi-strategy chunking to vector indexing, automatic query rewriting, LLM grounding, and a built-in ChatGPT-style Web & Jupyter UI.

> 💡 **Backwards-Compatible Imports:** Both `from ragspark import ...` and `from raglib import ...` work out of the box!

---

## 📦 Installation

```bash
pip install ragspark
```

To install with all optional vector database, embedding, and LLM providers:
```bash
pip install "ragspark[full]"
```

---

## 🚀 1-Line Quickstart (General RAG)

```python
from ragspark import parser, chunker, vectordb, prompt_eng, llm, rag, ui

# 1. Parse any PDF, Markdown, or inline string
doc = parser.parse("notes.pdf")

# 2. Chunk with 15+ strategies (recursive, semantic, sentence, token, hybrid...)
chunks = chunker.chunk(doc, strategy="recursive", chunk_size=500, overlap=80)

# 3. Store & search in ChromaDB, Pinecone, Qdrant, MongoDB, or In-Memory
db = vectordb.store(chunks, backend="chroma", collection_name="my_docs")
hits = db.search("What are the key takeaways?", top_k=5)

# 4. Rewrite conversational queries & build grounded prompts
rewritten = prompt_eng.rewrite_query("tell me about attention")
prompt = prompt_eng.build(hits, query=rewritten.rewritten_query, technique="standard")

# 5. Generate natural, fluent sentence answers (Ollama, Gemini, OpenAI, Groq, or offline Extractive)
answer = llm.generate(prompt, provider="extractive")
print(answer.text)
```

### Or Run the Entire Pipeline in 1 Line:
```python
from ragspark import rag

bot = rag.create("notes.pdf", llm_provider="gemini", llm_api_key="YOUR_KEY")
print(bot.ask("Explain the main architecture in simple sentences.").answer)
```

---

## ⚖️ Specialized Law & Constitutional RAG Engine

`ragspark` includes a dedicated legal intelligence pipeline for Statutes, Constitutions, Judgments, and Contracts:
- **Legal Hierarchy Parser & Chunker**: Preserves `Part > Chapter > Article > Clause > Proviso` boundaries without splitting mid-clause.
- **Story / Scenario Legal Solver**: Automatically detects real-world story/hypothetical questions (e.g., *"Ramesh was arrested by police and not produced before a magistrate for 4 days..."*) and maps the facts to exact Constitutional Articles (`Article 22(2)`, `Article 21`, `Article 32`).
- **Natural Sentences by Default + On-Demand IRAC**: Generates clear, fluent legal prose by default, and automatically switches to structured **IRAC (`1. ISSUE`, `2. RULE`, `3. APPLICATION`, `4. CONCLUSION`)** whenever the user includes `"IRAC"` in their query.

```python
from ragspark import rag

law_bot = rag.create_law("constitution_of_india.pdf")

# 1. Story / Scenario Question -> Natural Legal Prose + Exact Article Citations
res = law_bot.ask("Ramesh ran a newspaper and the government banned it for criticizing a minister. Advise Ramesh.")
print(res.answer)

# 2. On-Demand IRAC Mode -> Triggered simply by mentioning "IRAC" in the query
irac_res = law_bot.ask("Explain Article 19(1)(a) and its reasonable restrictions in IRAC format")
print(irac_res.answer)
```

---

## 💬 Built-In ChatGPT-Style Studio UI (Jupyter & Web)

Launch an interactive ChatGPT-style Q&A Studio inside JupyterLab/Colab or as a standalone web server:

```python
from ragspark import ui

# Inside Jupyter Notebook / Google Colab:
ui.notebook_chat(bot)

# Or launch the full standalone browser app at http://127.0.0.1:8501:
ui.launch()
```

### CLI Commands
```bash
ragspark ui --port 8501
ragspark parse sample.pdf --mode auto --out parsed.json
ragspark ask sample.pdf "Summarize the key findings"
ragspark ask-law constitution.pdf "Explain Article 21"
```

---

## 📄 License

MIT License. See [`RAGSPARK_DOCUMENTATION.md`](RAGSPARK_DOCUMENTATION.md) for the complete command reference.
