"""
High-Speed Multi-Engine PDF & Document Extractor for raglib.parser.
Optimized for Laptop Performance:
  1. PyMuPDF (pymupdf) C-level block extraction (`page.get_text("blocks")`) — 100+ pages in <0.3s
  2. Native PyMuPDF C-level table finder (`page.find_tables()`) when `extract_tables="fast"`
  3. Optional `pdfplumber` deep table extractor only when `extract_tables="deep"`
  4. `pypdf` pure-Python fallback
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raglib.parser.models import TableBlock


def _rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Convert table headers and rows into a clean GitHub-Flavored Markdown table."""
    if not headers and not rows:
        return ""
    cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    if cols == 0:
        return ""

    norm_headers = [str(h or "").replace("\n", " ").strip() for h in headers]
    if len(norm_headers) < cols:
        norm_headers.extend([""] * (cols - len(norm_headers)))

    lines = [
        "| " + " | ".join(norm_headers) + " |",
        "| " + " | ".join(["---"] * cols) + " |",
    ]
    for r in rows:
        norm_row = [str(cell or "").replace("\n", " ").strip() for cell in r[:cols]]
        if len(norm_row) < cols:
            norm_row.extend([""] * (cols - len(norm_row)))
        lines.append("| " + " | ".join(norm_row) + " |")
    return "\n".join(lines)


def _ocr_scanned_page(page: Any, ocr_mode: bool | str = "auto", api_key: str | None = None) -> tuple[str, str]:
    """
    Smart Per-Page Hybrid OCR Fallback (only triggered on scanned/image pages where digital text is empty):
      1. PyMuPDF Native C-Level Tesseract OCR (`page.get_textpage_ocr()`)
      2. RapidOCR ONNX CPU (`rapidocr_onnxruntime` — zero external binary needed)
      3. `pytesseract` + PIL Image fallback
      4. Gemini Vision OCR (`gemini-2.5-flash` multimodal transcription when `api_key` or `GEMINI_API_KEY` is set)
    Returns `(extracted_text, ocr_engine_used)`.
    """
    import base64
    import json
    import os
    import urllib.request

    mode_str = "auto" if ocr_mode is True else str(ocr_mode).lower().strip()

    # 1. PyMuPDF Native C-Level OCR (fastest when Tesseract tessdata is available on system)
    if mode_str in ("auto", "tesseract", "pymupdf_ocr"):
        try:
            tp = page.get_textpage_ocr(flags=0, dpi=200, full=True)
            ocr_txt = (page.get_text("text", textpage=tp) or "").strip()
            if ocr_txt:
                return ocr_txt, "pymupdf_tesseract_ocr"
        except Exception:
            pass

    # Render page to PNG bytes at 200 DPI for RapidOCR / Pytesseract / Gemini Vision
    try:
        pix = page.get_pixmap(dpi=200)
        png_bytes: bytes = pix.tobytes("png")
    except Exception:
        return "", ""

    # 2. RapidOCR (Pure Python + ONNX Runtime on CPU — no system binary required)
    if mode_str in ("auto", "rapidocr"):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

            engine = RapidOCR()
            result, _ = engine(png_bytes)
            if result:
                lines = [str(item[1]).strip() for item in result if len(item) >= 2 and str(item[1]).strip()]
                if lines:
                    return "\n".join(lines), "rapidocr_onnx"
        except Exception:
            pass

    # 3. Pytesseract + Pillow fallback
    if mode_str in ("auto", "tesseract", "pytesseract"):
        try:
            import io
            from PIL import Image  # type: ignore[import-not-found]
            import pytesseract  # type: ignore[import-not-found]

            img = Image.open(io.BytesIO(png_bytes))
            tess_txt = (pytesseract.image_to_string(img) or "").strip()
            if tess_txt:
                return tess_txt, "pytesseract_ocr"
        except Exception:
            pass

    # 4. Gemini Vision Multimodal OCR (Transcribes scanned pages, stamps, & tables via Gemini Flash Vision)
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key and mode_str in ("auto", "gemini", "gemini_vision", "vision"):
        try:
            b64_img = base64.b64encode(png_bytes).decode("ascii")
            url = (
                f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:generateContent?key={key}"
                if key.startswith("AQ.")
                else f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
            )
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Transcribe all text from this scanned document page accurately, preserving headings, clauses, and tables."},
                            {"inlineData": {"mimeType": "image/png", "data": b64_img}},
                        ],
                    }
                ]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cands = data.get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    vis_txt = "".join(p.get("text", "") for p in parts).strip()
                    if vis_txt:
                        return vis_txt, "gemini_vision_ocr"
        except Exception:
            pass

    return "", ""


def extract_with_pymupdf_fast(
    pdf_path: Path,
    extract_tables: bool | str = False,
    ocr: bool | str = "auto",
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[TableBlock]], dict[str, Any]]:
    """
    Ultra-fast C-level PyMuPDF extraction using `get_text("blocks")` with Smart Per-Page Hybrid OCR fallback
    for scanned/image-only pages (`len(raw_text) < 20`).
    """
    import pymupdf as fitz

    pages_data: list[dict[str, Any]] = []
    tables_by_page: dict[int, list[TableBlock]] = {}
    doc_meta: dict[str, Any] = {}
    ocr_engines_triggered: list[str] = []

    with fitz.open(str(pdf_path)) as doc:
        raw_meta = doc.metadata or {}
        doc_meta = {
            "title": raw_meta.get("title") or pdf_path.stem,
            "author": raw_meta.get("author") or "",
            "subject": raw_meta.get("subject") or "",
            "producer": raw_meta.get("producer") or "",
            "page_count": len(doc),
        }

        for page_idx, page in enumerate(doc, start=1):
            # C-level block extraction: returns list of (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks") or []
            text_chunks: list[str] = []
            headings: list[str] = []

            for b in blocks:
                if len(b) >= 7 and b[6] == 0:  # text block
                    b_text = (b[4] or "").strip()
                    if not b_text:
                        continue
                    text_chunks.append(b_text)
                    # Fast heuristic heading detection without expensive span dicts:
                    # Single-line short uppercase or title-case block
                    if "\n" not in b_text and 3 <= len(b_text) <= 100:
                        if b_text.isupper() or b_text.startswith(("PART ", "CHAPTER ", "SECTION ", "ARTICLE ", "CLAUSE ")):
                            if b_text not in headings:
                                headings.append(b_text)

            raw_text = "\n\n".join(text_chunks)
            # If the page has almost no embedded digital text (scanned image PDF page) or ocr="force", run Per-Page Hybrid OCR!
            if ocr and (ocr == "force" or len(raw_text.strip()) < 20):
                ocr_text, ocr_used = _ocr_scanned_page(page, ocr_mode=ocr, api_key=api_key)
                if ocr_text:
                    raw_text = ocr_text
                    if ocr_used and ocr_used not in ocr_engines_triggered:
                        ocr_engines_triggered.append(ocr_used)

            pages_data.append(
                {
                    "page_number": page_idx,
                    "raw_text": raw_text,
                    "headings": headings,
                }
            )

            # Fast C-level table extraction via PyMuPDF only when requested
            if extract_tables in (True, "fast") and hasattr(page, "find_tables"):
                try:
                    # Quick pre-check: only run table finder if page has vector drawings/lines
                    if len(page.get_drawings()) >= 4:
                        found = page.find_tables()
                        page_tables: list[TableBlock] = []
                        for tbl in getattr(found, "tables", []):
                            extracted = tbl.extract() or []
                            cleaned_rows = [
                                [(c or "").strip() for c in row]
                                for row in extracted
                                if row and any((c or "").strip() for c in row)
                            ]
                            if len(cleaned_rows) >= 2:
                                headers = cleaned_rows[0]
                                body = cleaned_rows[1:]
                                page_tables.append(
                                    TableBlock(
                                        page_number=page_idx,
                                        headers=headers,
                                        rows=body,
                                        markdown=_rows_to_markdown(headers, body),
                                    )
                                )
                        if page_tables:
                            tables_by_page[page_idx] = page_tables
                except Exception:
                    pass

        if ocr_engines_triggered:
            doc_meta["ocr_engines_used"] = ocr_engines_triggered

    return pages_data, tables_by_page, doc_meta


def extract_tables_pdfplumber(pdf_path: Path) -> dict[int, list[TableBlock]]:
    """Optional deep table extractor using pdfplumber (only run when extract_tables='deep')."""
    tables_by_page: dict[int, list[TableBlock]] = {}
    try:
        import pdfplumber
    except ImportError:
        return tables_by_page

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                raw_tables = page.extract_tables() or []
                page_tables: list[TableBlock] = []
                for tbl in raw_tables:
                    cleaned_rows = [
                        [(cell or "").strip() for cell in row]
                        for row in tbl
                        if row and any((c or "").strip() for c in row)
                    ]
                    if len(cleaned_rows) >= 2:
                        headers = cleaned_rows[0]
                        body = cleaned_rows[1:]
                        md = _rows_to_markdown(headers, body)
                        page_tables.append(
                            TableBlock(
                                page_number=idx,
                                headers=headers,
                                rows=body,
                                markdown=md,
                            )
                        )
                if page_tables:
                    tables_by_page[idx] = page_tables
    except Exception:
        pass

    return tables_by_page


def extract_with_pypdf(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fallback PDF extraction using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages_data: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        pages_data.append(
            {
                "page_number": idx,
                "raw_text": raw_text,
                "headings": [],
            }
        )
    doc_meta = {
        "title": pdf_path.stem,
        "page_count": len(reader.pages),
    }
    return pages_data, doc_meta


def extract_pdf_hybrid(
    pdf_path: str | Path,
    extract_tables: bool | str = False,
    ocr: bool | str = "auto",
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[TableBlock]], dict[str, Any], str]:
    """
    Master high-speed PDF & document extractor:
    - Defaults to `extract_tables=False` (or `"fast"` using PyMuPDF C-level table detection on pages with drawings)
      so large PDFs parse in milliseconds without freezing laptops.
    - Automatically runs Per-Page Hybrid OCR (`ocr="auto"`) ONLY on scanned/image-only pages (`len(raw_text) < 20`).
    - Pass `extract_tables="deep"` to force full pdfplumber scanning.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() in (".txt", ".md", ".markdown"):
        content = path.read_text(encoding="utf-8")
        pages_data = [
            {
                "page_number": 1,
                "raw_text": content,
                "headings": [
                    ln.lstrip("# ").strip()
                    for ln in content.splitlines()
                    if ln.strip().startswith("#")
                ],
            }
        ]
        return pages_data, {}, {"title": path.stem, "page_count": 1}, "text_reader"

    engines_used: list[str] = []
    tables_by_page: dict[int, list[TableBlock]] = {}

    try:
        pages_data, tables_by_page, doc_meta = extract_with_pymupdf_fast(
            path, extract_tables=extract_tables, ocr=ocr, api_key=api_key
        )
        engines_used.append("pymupdf_fast")
        if doc_meta.get("ocr_engines_used"):
            engines_used.extend(doc_meta["ocr_engines_used"])
    except Exception:
        pages_data, doc_meta = extract_with_pypdf(path)
        engines_used.append("pypdf")

    if extract_tables == "deep":
        tables_by_page = extract_tables_pdfplumber(path)
        if tables_by_page:
            engines_used.append("pdfplumber")

    return pages_data, tables_by_page, doc_meta, "+".join(engines_used)
