"""
Comprehensive unit tests for core RAGSpark modules.
Uses standard library unittest so it runs everywhere with zero dependencies.
"""

import os
import random
import re
import socket
import tempfile
import unittest
import urllib.request
from pathlib import Path

import ragspark
from ragspark import parser, chunker, embedder, vectordb, prompt_eng, llm, rag, ui

_SAVED_ENV: dict[str, str | None] = {}


def setUpModule() -> None:
    # Keep the suite offline and deterministic even when API keys are set on this machine.
    _SAVED_ENV["RAGSPARK_EMBED_PROVIDER"] = os.environ.get("RAGSPARK_EMBED_PROVIDER")
    os.environ["RAGSPARK_EMBED_PROVIDER"] = "lexical"


def tearDownModule() -> None:
    old = _SAVED_ENV.get("RAGSPARK_EMBED_PROVIDER")
    if old is None:
        os.environ.pop("RAGSPARK_EMBED_PROVIDER", None)
    else:
        os.environ["RAGSPARK_EMBED_PROVIDER"] = old


def _assert_no_word_cuts(test: unittest.TestCase, source: str, chunks: list) -> None:
    """Every whitespace-delimited token in every chunk must be a whole token of the source."""
    vocab = set(source.split())
    for c in chunks:
        text = c.text if hasattr(c, "text") else c
        for tok in text.split():
            test.assertIn(tok, vocab, f"word cut: {tok!r} in chunk {text[:60]!r}")


LEGAL_TEXT = """TABLE OF CONTENTS
Article 1. Name and territory ........ 1
Article 2. Admission of new States ........ 2
Article 21. Protection of life ........ 9

THE GAZETTE OF INDIA
EXTRAORDINARY
PUBLISHED BY AUTHORITY

PART I
THE UNION AND ITS TERRITORY

Article 1. Name and territory of the Union
India, that is Bharat, shall be a Union of States. The States and the territories thereof shall be as specified in the First Schedule.

Article 2. Admission or establishment of new States
Parliament may by law admit into the Union, or establish, new States on such terms and conditions as it thinks fit.

PART III
FUNDAMENTAL RIGHTS

Article 21. Protection of life and personal liberty
No person shall be deprived of his life or personal liberty except according to procedure established by law.
"""


class TestRAGSparkCore(unittest.TestCase):

    def test_version(self):
        self.assertEqual(ragspark.__version__, "0.4.0")

    def test_parser_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)

            # Markdown
            md_file = d / "test.md"
            md_file.write_text("# Title\n\nContent paragraph.", encoding="utf-8")
            doc_md = parser.parse(md_file)
            self.assertIn("Content paragraph", doc_md.text)
            self.assertEqual(doc_md.total_pages, 1)

            # JSON
            json_file = d / "test.json"
            json_file.write_text('{"name": "ragspark", "version": "0.3.1"}', encoding="utf-8")
            doc_json = parser.parse(json_file)
            self.assertIn("ragspark", doc_json.text)

            # CSV
            csv_file = d / "test.csv"
            csv_file.write_text("col1,col2\nval1,val2\n", encoding="utf-8")
            doc_csv = parser.parse(csv_file)
            self.assertIn("val1", doc_csv.text)

    def test_chunker_strategies_and_validation(self):
        text = "Democracy is a system of government. Citizens elect their leaders. Fundamental rights are guaranteed."
        doc = parser.parse_text(text)

        # Recursive
        chks_rec = chunker.chunk(doc, strategy="recursive", chunk_size=50, overlap=10)
        self.assertGreaterEqual(len(chks_rec), 1)

        # Overlap validation
        with self.assertRaises(ValueError):
            chunker.chunk(doc, chunk_size=50, overlap=60)

        with self.assertRaises(ValueError):
            chunker.chunk(doc, chunk_size=0, overlap=0)

    def test_vectordb_operations_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            text = "Article 21 guarantees protection of life and personal liberty."
            doc = parser.parse_text(text)
            chunks = chunker.chunk(doc, strategy="recursive", chunk_size=100, overlap=10)

            store = vectordb.connect(
                provider="chroma",
                collection_name="test_col",
                persist_directory=tmpdir
            )
            self.assertEqual(len(store), 0)

            # Test add with auto_save
            count = store.add(chunks, auto_save=True)
            self.assertEqual(count, len(chunks))
            self.assertEqual(len(store), len(chunks))
            self.assertEqual(store.count(), len(chunks))

            # Test search
            hits = store.search("Article 21", strategy="hybrid", top_k=2)
            self.assertGreaterEqual(len(hits), 1)
            self.assertIn("Article 21", hits[0].chunk.text)

    def test_prompt_eng_and_mock_llm(self):
        text = "The Constitution of India was adopted on 26 November 1949."
        doc = parser.parse_text(text)
        chunks = chunker.chunk(doc, strategy="recursive", chunk_size=100, overlap=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = vectordb.store(chunks, backend="chroma", persist_directory=tmpdir)
            hits = store.search("Constitution", top_k=1)

            prompt = prompt_eng.build(hits, query="When was the constitution adopted?")
            self.assertIn("USER QUESTION", prompt.user_prompt)
            self.assertIn("RETRIEVED CONTEXT", prompt.user_prompt)

            # Test extractive fallback
            ans = llm.generate(prompt, query="When was the constitution adopted?", provider="extractive")
            self.assertGreater(len(str(ans)), 0)
            self.assertEqual(ans.provider, "extractive")


class TestV040Invariants(unittest.TestCase):
    """v0.4.0 overhaul invariants (CLAUDE_CODE_MASTER_PROMPT.md)."""

    # -- Invariants 1-3: chunking ------------------------------------------------

    def test_no_word_slicing_in_any_strategy(self):
        rnd = random.Random(7)
        words = ["jurisdiction", "notwithstanding", "tribunal", "provision", "of", "the", "and", "Parliament."]
        blob = " ".join(rnd.choice(words) for _ in range(500))
        sizes = {"token": 60}
        for strat in ("fixed", "recursive", "sentence", "semantic", "sliding_window", "paragraph", "token", "adaptive", "natural", "micro_sentence"):
            size = sizes.get(strat, 120)
            chunks = chunker.chunk(blob, strategy=strat, chunk_size=size, chunk_overlap=size // 4)
            self.assertTrue(chunks, strat)
            _assert_no_word_cuts(self, blob, chunks)

    def test_safe_split_cuts_only_at_whitespace(self):
        text = "Alpha beta gamma. Delta epsilon zeta! Eta theta iota? " * 40
        pieces = chunker.safe_split(text, 50, overlap=12)
        self.assertTrue(all(len(p) <= 50 for p in pieces))
        _assert_no_word_cuts(self, text, pieces)
        self.assertTrue(all(p.endswith((".", "!", "?")) for p in pieces[:-1]), "sentence ends preferred")
        # A single token longer than the limit is kept whole, never cut.
        self.assertEqual(chunker.safe_split("a" * 80 + " tail", 20), ["a" * 80, "tail"])

    def test_chunk_size_none_uses_natural_boundaries(self):
        chunks = chunker.chunk(parser.parse_text(LEGAL_TEXT))
        self.assertTrue(all(c.strategy == "natural" for c in chunks))
        paths = [c.section_path for c in chunks]
        self.assertTrue(any(p.startswith("PART I > Article 1") for p in paths), paths)
        self.assertTrue(any("Article 21" in p and p.startswith("PART III") for p in paths), paths)
        self.assertEqual(len([p for p in paths if "Article" in p]), 3)

    def test_toc_and_gazette_noise_filtered(self):
        joined = " ".join(c.text for c in chunker.chunk(parser.parse_text(LEGAL_TEXT)))
        self.assertNotIn("........", joined)
        self.assertNotIn("PUBLISHED BY AUTHORITY", joined)
        self.assertIn("procedure established by law", joined)

    def test_natural_ceiling_subdivides_huge_sections(self):
        blob = " ".join(["schedule entry"] * 4000)  # ~60k chars, no punctuation or newlines
        chunks = chunker.chunk(blob)
        self.assertGreater(len(chunks), 20)
        self.assertLessEqual(max(len(c.text) for c in chunks), chunker.NATURAL_CEILING)
        _assert_no_word_cuts(self, blob, chunks)

    def test_ceiling_guard_applies_to_every_strategy(self):
        one_sentence = " ".join(["word"] * 3000)  # one 15k-char "sentence"
        for strat in ("sentence", "semantic", "news_chunk"):
            chunks = chunker.chunk(one_sentence, strategy=strat, chunk_size=500)
            self.assertLessEqual(max(len(c.text) for c in chunks), max(chunker.NATURAL_CEILING, 1000), strat)

    def test_code_split_by_functions_and_classes(self):
        code = "import os\n\ndef alpha(x):\n    return x + 1\n\nclass Beta:\n    def run(self):\n        return alpha(2)\n\ndef gamma():\n    return os.getcwd()\n"
        chunks = chunker.chunk(parser.parse_text(code))
        self.assertEqual([c.section_path for c in chunks], ["def alpha(x)", "class Beta", "def gamma()"])
        self.assertIn("    def run(self):", chunks[1].text, "indentation preserved, method stays with its class")

    def test_natural_micro_sentence_ensemble(self):
        long_para = " ".join(f"Clause {i} requires notice within {i} days." for i in range(1, 40))
        doc = parser.parse_text(f"# Notices\n\n{long_para}")
        chunks = chunker.chunk(doc, strategy=["natural", "micro_sentence"])
        natural = {c.id for c in chunks if c.strategy == "natural"}
        micro = [c for c in chunks if c.strategy == "micro_sentence"]
        self.assertTrue(natural and micro)
        self.assertTrue(all(m.metadata["parent_id"] in natural for m in micro))
        self.assertTrue(all(len(m.text) <= 320 for m in micro))
        _assert_no_word_cuts(self, doc.text, micro)

    def test_default_overlap_adapts_to_small_chunk_size(self):
        chunks = chunker.chunk("Short text. " * 30, strategy="recursive", chunk_size=50)
        self.assertTrue(all(len(c.text) <= 50 for c in chunks))

    # -- Invariant 4: storage --------------------------------------------------------

    def test_default_store_is_memory_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                db = vectordb.store(chunker.chunk(LEGAL_TEXT))
                db.search("personal liberty")
                rag.create(LEGAL_TEXT).ask("What does Article 21 protect?")
                self.assertEqual(db.provider, "memory")
                self.assertEqual(os.listdir(tmp), [], "nothing may be written to the working directory")
            finally:
                os.chdir(cwd)

    def test_sqlite_persistence_single_transaction_and_reload(self):
        chunks = chunker.chunk(LEGAL_TEXT, strategy="recursive", chunk_size=200)
        with tempfile.TemporaryDirectory() as tmp:
            db = vectordb.connect(persist_dir=tmp, collection_name="law")
            calls = []
            original = db._sqlite.upsert
            db._sqlite.upsert = lambda rows, meta=None: (calls.append(len(rows)), original(rows, meta))[1]
            db.add(chunks)
            self.assertEqual(calls, [len(chunks)], "one transaction for the whole batch")

            db.add(chunks[:2], auto_save=False)
            db.add(chunks[2:4], auto_save=False)
            self.assertEqual(len(calls), 1, "auto_save=False defers writes")
            db.save()
            self.assertEqual(len(calls), 2, "save() flushes once")

            self.assertEqual(sorted(os.listdir(tmp)), ["ragspark.db"])
            reloaded = vectordb.load(persist_dir=tmp, collection_name="law")
            self.assertEqual(len(reloaded), len(chunks))
            self.assertEqual(reloaded.embed_provider, "lexical")
            self.assertIn("Article 21", reloaded.search("Article 21 personal liberty", top_k=1)[0].chunk.text)

    def test_large_memory_store_spills_to_tempdir(self):
        old = vectordb.SPILL_THRESHOLD
        vectordb.SPILL_THRESHOLD = 5
        try:
            chunks = chunker.chunk(" ".join(f"Item {i} is stored safely." for i in range(40)), strategy="sentence", chunk_size=40)
            vecs = [[float(i % 3), 1.0, float(i % 5)] for i in range(len(chunks))]
            db = vectordb.connect()
            db.add(chunks, vecs)
            self.assertTrue(db.is_spilled)
            self.assertEqual(db._embeddings, [])
            spill_path = db._spill.path
            self.assertTrue(str(spill_path).startswith(tempfile.gettempdir()))
            self.assertTrue(spill_path.exists())
            self.assertEqual(len(db.search(vecs[3], query_text="Item 3", top_k=3)), 3)
            db.close()
            self.assertFalse(spill_path.exists(), "spill file removed on close")
        finally:
            vectordb.SPILL_THRESHOLD = old

    # -- Invariant 5: embeddings -------------------------------------------------------

    def test_embedder_fallback_ladder_never_crashes(self):
        def broken(texts, dim):
            raise ConnectionError("offline")

        embedder.register_custom("broken_cloud", broken)
        os.environ["RAGSPARK_EMBED_PROVIDER"] = "broken_cloud"
        try:
            vecs, used = embedder.embed_with_provider(["hello world"])
            self.assertEqual(used, "lexical")
            self.assertEqual(vecs, [[]])
            db = vectordb.store(chunker.chunk(LEGAL_TEXT, strategy="recursive", chunk_size=200))
            hits = db.search("personal liberty", strategy="semantic", top_k=1)
            self.assertEqual(hits[0].retrieval_strategy, "bm25", "semantic degrades to BM25 without vectors")
            self.assertIn("liberty", hits[0].chunk.text)
        finally:
            os.environ["RAGSPARK_EMBED_PROVIDER"] = "lexical"

    def test_auto_is_default_and_fast_local_never_chosen(self):
        self.assertNotIn("fast_local", embedder.auto_chain())
        self.assertEqual(vectordb.VectorStore().embed_provider, "auto")

    # -- Invariant 6: answer structure -------------------------------------------------

    def test_prompt_forbids_filler_and_requires_markdown(self):
        p = prompt_eng.build(["Article 21 protects life and liberty."], query="What does Article 21 protect?")
        self.assertIn("NEVER open with filler", p.system_prompt)
        self.assertIn("Based on the text", p.system_prompt)
        self.assertIn("bullet", p.system_prompt)

    def test_clean_answer_strips_filler_intros(self):
        self.assertEqual(llm.clean_answer("Based on the provided context, the term is 5 years."), "The term is 5 years.")
        self.assertEqual(llm.clean_answer("According to Page 1, notice is required."), "Notice is required.")
        self.assertEqual(llm.clean_answer("### Notice\n- 30 days"), "### Notice\n- 30 days")

    def test_extractive_answer_is_structured_markdown(self):
        bot = rag.create(LEGAL_TEXT)
        ans = bot.ask("What does Article 21 protect?").answer
        self.assertTrue(ans.startswith("### "), ans)
        self.assertNotRegex(ans.splitlines()[2], r"^(Based on|According to|Under \*\*)")

    # -- RAG pipeline -----------------------------------------------------------------

    def test_pipeline_three_line_entry_point(self):
        bot = ragspark.pipeline(LEGAL_TEXT)
        res = bot.ask("What does Article 21 protect?")
        self.assertIn("liberty", res.answer.lower())
        self.assertEqual(bot.store.provider, "memory")

    def test_pipeline_multiple_files_with_persist_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "docs"
            src.mkdir()
            (src / "a.md").write_text("# Refunds\n\nRefunds are issued within 14 days of purchase.", encoding="utf-8")
            (src / "b.txt").write_text("Shipping takes 3 to 5 business days.", encoding="utf-8")
            bot = ragspark.pipeline(src, persist_dir=Path(tmp) / "index")
            self.assertEqual(len(bot.document.metadata["sources"]), 2)
            self.assertIn("14 days", bot.ask("How long do refunds take?").answer)
            self.assertTrue((Path(tmp) / "index" / "ragspark.db").exists())

    def test_chat_memory_reaches_the_prompt(self):
        bot = rag.create(LEGAL_TEXT)
        bot.chat("What does Article 21 protect?")
        res = bot.chat("What are its exceptions?")
        self.assertIn("CONVERSATION HISTORY", res.built_prompt.user_prompt)
        self.assertIn("User: What does Article 21 protect?", res.built_prompt.user_prompt)
        self.assertEqual(res.metadata["history_turns_used"], 1)
        self.assertEqual(len(bot.history), 4)
        bot.clear_history()
        self.assertNotIn("CONVERSATION HISTORY", bot.chat("Hello").built_prompt.user_prompt)

    def test_respond_is_stateless(self):
        bot = rag.create(LEGAL_TEXT)
        bot.respond("What about it?", history=[{"role": "user", "content": "Article 21"}])
        self.assertEqual(bot.history, [])

    # -- Invariant 7 + UI ----------------------------------------------------------

    def test_ui_html_has_chatgpt_theme_sidebar_and_new_chat(self):
        page = ui._render_chatbot_html({"title": "Law <Bot>"})
        for needle in ("#000000", "#171717", 'class="flame"', 'id="sidebar"', 'id="newChatBtn"', "toggleSidebar", "typeOut"):
            self.assertIn(needle, page)
        self.assertIn("Law &lt;Bot&gt;", page)
        self.assertNotIn("Law <Bot>", page)

    def test_ui_uses_ephemeral_port_when_default_is_taken(self):
        bot = rag.create(LEGAL_TEXT)
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        busy_port = blocker.getsockname()[1]
        server = ui.launch(bot, port=busy_port, open_browser=False, background=True)
        try:
            port = server.server_address[1]
            self.assertNotEqual(port, busy_port)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r:
                self.assertIn(b"ok", r.read())
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/ask",
                data=b'{"query": "What are its exceptions?", "history": [{"role": "user", "content": "Article 21"}]}',
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                body = r.read().decode("utf-8")
            self.assertIn('"answer"', body)
            self.assertIn('"history_turns_used": 1', body)
        finally:
            server.shutdown()
            server.server_close()
            blocker.close()


if __name__ == "__main__":
    unittest.main()
