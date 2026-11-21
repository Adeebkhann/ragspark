<!-- AGENTS.md - RAGSpark Project Invariants & Pitfalls -->
# RAGSpark Agent Instructions

## Environment & Tooling
- **Command Execution on Windows:** Always run Python tools using `uv run <cmd>` or system `py -m <cmd>`. Never invoke `.venv/Scripts/python.exe` directly (Windows Application Control policies block unsigned binaries in local user virtualenvs).
- **Test Command:** Execute tests with: `uv run pytest -v` (or `py -m pytest -v`).
- **Packaging:** Build distributions using `uv build` (or `py -m build`). Check metadata with `twine check dist/*`.

## Architecture Invariants
- **Multi-Format Parsing:** `parser.parse()` accepts PDF, DOCX, TXT, Markdown, CSV, JSON, and HTML. Do not bypass `extract_pdf_hybrid()` with manual `read_text()`.
- **VectorStore Persistence:** In `VectorStore.add()`, never re-serialize the whole database on every file in loops. Use `auto_save=False` during bulk operations and call `db.save()` once at completion.
- **LLM Error Transparency:** When `fallback_to_extractive=False`, never swallow HTTP errors. Always populate `.error` on `LLMResult` for diagnostic tracing.
- **Model Resolution:** Google Gemini must default to `gemini-3.8-flash` (or `gemini-2.5-flash`). Groq defaults to `llama-3.3-70b-versatile` or `qwen/qwen3.8-27b`.
- **UI Engine:** `ragspark.ui` is a standalone multi-threaded `ThreadingHTTPServer` (`ui.launch(bot, background=False)`). Never claim Streamlit dependency.

## Security
- Never commit or log API keys (`gsk_...`, `AIzaSy...`, `sk-...`, `pypi-...`). Always read from environment variables (`os.environ`).
