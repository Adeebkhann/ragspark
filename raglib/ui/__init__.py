"""
`raglib.ui` — Clean, Minimal, Fully Customizable Chatbot UI for Student Demos & Production.

Shows ONLY the clean Chat Header, Question & Answer messages, and the Message Input Bar at the bottom!
100% customizable via Python arguments (`title`, `subtitle`, `bot_name`, `welcome_message`, `theme`,
`accent_color`, `bg_color`, `show_sources`, `custom_css`) OR live in the browser via the optional Customize drawer.

Usage:
    from raglib import ui

    # Pass your RAG pipeline directly into the Chatbot UI:
    ui.launch(pipe, title="Constitution of India Chatbot", theme="dark")
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from raglib import rag
from raglib.rag import RAGPipeline

DEFAULT_DEMO_LAW_TEXT = """IN THE SUPREME COURT OF INDIA
PART III - FUNDAMENTAL RIGHTS
Article 14. Equality before law.- The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.
Article 19. Protection of certain rights regarding freedom of speech, etc.- (1) All citizens shall have the right (a) to freedom of speech and expression; (b) to assemble peaceably and without arms; (c) to form associations or unions.
Article 21. Protection of life and personal liberty.- No person shall be deprived of his life or personal liberty except according to procedure established by law.
Article 21A. Right to education.- The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine.
Article 32. Remedies for enforcement of rights conferred by this Part.- (1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed."""

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0f172a",
        "card": "#1e293b",
        "bot_bubble": "#1e293b",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "border": "#334155",
        "accent": "#6366f1",
    },
    "light": {
        "bg": "#f8fafc",
        "card": "#ffffff",
        "bot_bubble": "#ffffff",
        "text": "#0f172a",
        "muted": "#64748b",
        "border": "#e2e8f0",
        "accent": "#4f46e5",
    },
    "emerald": {
        "bg": "#061814",
        "card": "#0d2922",
        "bot_bubble": "#0d2922",
        "text": "#ecfdf5",
        "muted": "#6ee7b7",
        "border": "#115e59",
        "accent": "#10b981",
    },
    "midnight": {
        "bg": "#09090b",
        "card": "#18181b",
        "bot_bubble": "#18181b",
        "text": "#fafafa",
        "muted": "#a1a1aa",
        "border": "#27272a",
        "accent": "#3b82f6",
    },
}


def _render_chatbot_html(config: dict[str, Any]) -> str:
    theme_name = str(config.get("theme", "dark")).lower()
    palette = dict(THEMES.get(theme_name, THEMES["dark"]))
    if config.get("bg_color"):
        palette["bg"] = config["bg_color"]
    if config.get("card_color"):
        palette["card"] = config["card_color"]
        palette["bot_bubble"] = config["card_color"]
    if config.get("text_color"):
        palette["text"] = config["text_color"]
    if config.get("accent_color"):
        palette["accent"] = config["accent_color"]

    title = config.get("title", "raglib Chatbot")
    subtitle = config.get("subtitle", "Powered by raglib RAG Pipeline")
    bot_name = config.get("bot_name", "AI Assistant")
    welcome_message = config.get("welcome_message", "Hello! Ask me any question about your document.")
    placeholder = config.get("placeholder", "Type your question here and press Enter...")
    show_sources = "true" if config.get("show_sources", False) else "false"
    allow_ui_edit = config.get("allow_ui_edit", True)
    custom_css = config.get("custom_css", "")

    customize_btn_html = (
        '<button class="icon-btn" onclick="toggleSettings()" title="Customize UI">🎨 Edit UI</button>'
        if allow_ui_edit
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: {palette["bg"]};
    --card: {palette["card"]};
    --bot-bubble: {palette["bot_bubble"]};
    --text: {palette["text"]};
    --muted: {palette["muted"]};
    --border: {palette["border"]};
    --accent: {palette["accent"]};
    --font-size: {config.get("font_size", "15px")};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .chat-container {{
    width: 100%;
    max-width: 820px;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  .header {{
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .header-titles h1 {{ font-size: 19px; font-weight: 700; }}
  .header-titles p {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .header-actions {{ display: flex; gap: 8px; }}
  .icon-btn {{
    background: var(--card);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    cursor: pointer;
  }}
  .icon-btn:hover {{ border-color: var(--accent); }}
  .messages {{
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .msg-row {{ display: flex; flex-direction: column; max-width: 86%; }}
  .msg-row.user {{ align-self: flex-end; align-items: flex-end; }}
  .msg-row.bot {{ align-self: flex-start; align-items: flex-start; width: 100%; max-width: 86%; }}
  .sender-label {{ font-size: 11px; color: var(--muted); margin-bottom: 5px; padding: 0 4px; font-weight: 600; letter-spacing: 0.02em; }}
  .bubble {{
    padding: 16px 20px;
    border-radius: 16px;
    line-height: 1.68;
    font-size: var(--font-size);
  }}
  .msg-row.user .bubble {{
    background: var(--accent);
    color: #ffffff;
    border-bottom-right-radius: 4px;
    white-space: pre-wrap;
    padding: 12px 16px;
  }}
  .msg-row.bot .bubble {{
    background: var(--bot-bubble);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-left-radius: 4px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.14);
  }}
  /* ChatGPT-style Rich Markdown Typography inside Bot Bubbles */
  .md-content p {{ margin: 0 0 12px 0; }}
  .md-content p:last-child {{ margin-bottom: 0; }}
  .md-content h1, .md-content h2, .md-content h3, .md-content h4 {{
    margin: 16px 0 8px 0;
    font-weight: 700;
    line-height: 1.35;
    color: var(--text);
  }}
  .md-content h1:first-child, .md-content h2:first-child, .md-content h3:first-child {{ margin-top: 0; }}
  .md-content h3 {{
    font-size: 1.06em;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border);
  }}
  .md-content ul, .md-content ol {{
    margin: 8px 0 12px 0;
    padding-left: 22px;
  }}
  .md-content li {{ margin-bottom: 6px; }}
  .md-content blockquote {{
    margin: 10px 0 14px 0;
    padding: 10px 14px;
    border-left: 3px solid var(--accent);
    background: rgba(255, 255, 255, 0.03);
    border-radius: 0 8px 8px 0;
    font-style: italic;
    color: var(--text);
  }}
  .md-content code {{
    background: rgba(99, 102, 241, 0.14);
    border: 1px solid var(--border);
    color: var(--accent);
    padding: 2px 7px;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.88em;
    font-weight: 600;
  }}
  .md-content hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 14px 0;
  }}
  .sources {{
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    font-size: 11.5px;
    color: var(--muted);
  }}
  .source-pill {{
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border);
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    color: var(--muted);
  }}
  .input-bar {{
    padding: 16px 24px 22px;
    display: flex;
    gap: 10px;
  }}
  .input-wrapper {{
    flex: 1;
    display: flex;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 6px 6px 6px 16px;
    align-items: center;
  }}
  .input-wrapper:focus-within {{ border-color: var(--accent); }}
  .input-wrapper input {{
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    font-size: 15px;
    outline: none;
  }}
  .send-btn {{
    background: var(--accent);
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
  }}
  .send-btn:hover {{ opacity: 0.92; }}
  /* Optional clean UI customizer modal */
  .settings-drawer {{
    display: none;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
    font-size: 12px;
  }}
  .settings-drawer.open {{ display: flex; }}
  .settings-drawer label {{ display: flex; align-items: center; gap: 6px; color: var(--muted); }}
  .settings-drawer input[type="text"], .settings-drawer select {{
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
  }}
  {custom_css}
</style>
</head>
<body>
  <div class="chat-container">
    <header class="header">
      <div class="header-titles">
        <h1 id="ui_title">{title}</h1>
        <p id="ui_subtitle">{subtitle}</p>
      </div>
      <div class="header-actions">
        <button class="icon-btn" onclick="clearChat()">🗑️ Clear</button>
        {customize_btn_html}
      </div>
    </header>

    <div class="settings-drawer" id="settings_drawer">
      <label>Title: <input type="text" id="cfg_title" value="{title}" oninput="applyLiveUI()" /></label>
      <label>Bot Name: <input type="text" id="cfg_bot" value="{bot_name}" oninput="applyLiveUI()" /></label>
      <label>Theme:
        <select id="cfg_theme" onchange="applyThemePreset(this.value)">
          <option value="dark">Dark</option>
          <option value="light">Light</option>
          <option value="emerald">Emerald</option>
          <option value="midnight">Midnight</option>
        </select>
      </label>
      <label>Accent: <input type="color" id="cfg_accent" value="{palette["accent"]}" oninput="applyLiveUI()" /></label>
      <label>Bg: <input type="color" id="cfg_bg" value="{palette["bg"]}" oninput="applyLiveUI()" /></label>
      <label><input type="checkbox" id="cfg_sources" {"checked" if config.get("show_sources") else ""} /> Show Sources</label>
    </div>

    <div class="messages" id="messages">
      <div class="msg-row bot">
        <span class="sender-label" id="first_bot_label">{bot_name}</span>
        <div class="bubble"><div class="md-content">{welcome_message}</div></div>
      </div>
    </div>

    <div class="input-bar">
      <div class="input-wrapper">
        <input id="query_input" placeholder="{placeholder}" onkeydown="if(event.key==='Enter') sendMessage()" autofocus />
        <button class="send-btn" onclick="sendMessage()">Send</button>
      </div>
    </div>
  </div>

<script>
  let botName = {json.dumps(bot_name)};
  let showSources = {show_sources};
  const themes = {json.dumps(THEMES)};

  function escapeHtml(str) {{
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }}

  function formatInline(text) {{
    let s = escapeHtml(text);
    // Inline code / bracket badges
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold + Italic
    s = s.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
    s = s.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
    // Highlight standalone [Part ... > Article ...] legal citations if not already inside <code>
    s = s.replace(/(?<!<code>)\\[(Part\\s+[^\\]]+|Article\\s+[^\\]]+|Section\\s+[^\\]]+)\\]/gi, '<code>[$1]</code>');
    return s;
  }}

  function renderMarkdown(md) {{
    if (!md) return '';
    // Strip any legacy "- [Chunk X | Page Y | Score: Z]" lines if present
    const cleaned = String(md).replace(/^\\s*-?\\s*\\[Chunk\\s+\\d+[^\\]]*\\]\\s*$/gm, '');
    const lines = cleaned.split(/\\r?\\n/);
    let html = [];
    let inList = false;
    let listType = 'ul';
    let inQuote = false;
    let quoteLines = [];

    function flushList() {{
      if (inList) {{
        html.push(`</${{listType}}>`);
        inList = false;
      }}
    }}
    function flushQuote() {{
      if (inQuote) {{
        html.push(`<blockquote>${{quoteLines.join('<br>')}}</blockquote>`);
        inQuote = false;
        quoteLines = [];
      }}
    }}

    for (let rawLine of lines) {{
      let line = rawLine.trim();
      // Clean accidental "- ### Heading" prefix
      line = line.replace(/^-\\s+(#{1,4}\\s+)/, '$1');

      if (!line) {{
        flushQuote();
        flushList();
        continue;
      }}
      if (/^---+$/.test(line)) {{
        flushQuote();
        flushList();
        html.push('<hr>');
        continue;
      }}
      let hMatch = line.match(/^(#{1,4})\\s+(.*)$/);
      if (hMatch) {{
        flushQuote();
        flushList();
        let lvl = Math.min(4, hMatch[1].length);
        html.push(`<h${{lvl}}>${{formatInline(hMatch[2])}}</h${{lvl}}>`);
        continue;
      }}
      if (line.startsWith('>')) {{
        flushList();
        inQuote = true;
        quoteLines.push(formatInline(line.replace(/^>\\s?/, '')));
        continue;
      }} else {{
        flushQuote();
      }}
      let ulMatch = line.match(/^[*\\-]\\s+(.*)$/);
      if (ulMatch) {{
        let body = ulMatch[1].trim();
        // If a bullet contains a blockquote marker ">", split label and quote
        let qIdx = body.indexOf('> *');
        if (qIdx === -1) qIdx = body.indexOf('> "');
        if (!inList || listType !== 'ul') {{
          flushList();
          inList = true;
          listType = 'ul';
          html.push('<ul>');
        }}
        if (qIdx !== -1) {{
          let headPart = body.slice(0, qIdx).trim();
          let quotePart = body.slice(qIdx + 1).trim();
          html.push(`<li>${{formatInline(headPart)}}<blockquote>${{formatInline(quotePart)}}</blockquote></li>`);
        }} else {{
          html.push(`<li>${{formatInline(body)}}</li>`);
        }}
        continue;
      }}
      let olMatch = line.match(/^\\d+\\.\\s+(.*)$/);
      if (olMatch) {{
        if (!inList || listType !== 'ol') {{
          flushList();
          inList = true;
          listType = 'ol';
          html.push('<ol>');
        }}
        html.push(`<li>${{formatInline(olMatch[1])}}</li>`);
        continue;
      }}
      flushList();
      html.push(`<p>${{formatInline(line)}}</p>`);
    }}
    flushQuote();
    flushList();
    return `<div class="md-content">${{html.join('')}}</div>`;
  }}

  function toggleSettings() {{
    document.getElementById('settings_drawer').classList.toggle('open');
  }}

  function applyThemePreset(name) {{
    const t = themes[name];
    if (!t) return;
    document.documentElement.style.setProperty('--bg', t.bg);
    document.documentElement.style.setProperty('--card', t.card);
    document.documentElement.style.setProperty('--bot-bubble', t.bot_bubble);
    document.documentElement.style.setProperty('--text', t.text);
    document.documentElement.style.setProperty('--muted', t.muted);
    document.documentElement.style.setProperty('--border', t.border);
    document.documentElement.style.setProperty('--accent', t.accent);
    document.getElementById('cfg_accent').value = t.accent;
    document.getElementById('cfg_bg').value = t.bg;
  }}

  function applyLiveUI() {{
    document.getElementById('ui_title').textContent = document.getElementById('cfg_title').value;
    botName = document.getElementById('cfg_bot').value || 'AI';
    document.documentElement.style.setProperty('--accent', document.getElementById('cfg_accent').value);
    document.documentElement.style.setProperty('--bg', document.getElementById('cfg_bg').value);
    showSources = document.getElementById('cfg_sources').checked;
  }}

  function clearChat() {{
    const box = document.getElementById('messages');
    box.innerHTML = '';
  }}

  async function sendMessage() {{
    const input = document.getElementById('query_input');
    const q = input.value.trim();
    if (!q) return;
    input.value = '';

    const box = document.getElementById('messages');

    const uRow = document.createElement('div');
    uRow.className = 'msg-row user';
    uRow.innerHTML = `<span class="sender-label">You</span><div class="bubble"></div>`;
    uRow.querySelector('.bubble').textContent = q;
    box.appendChild(uRow);

    const bRow = document.createElement('div');
    bRow.className = 'msg-row bot';
    bRow.innerHTML = `<span class="sender-label">${{botName}}</span><div class="bubble"><div class="md-content"><em>Thinking...</em></div></div>`;
    box.appendChild(bRow);
    box.scrollTop = box.scrollHeight;

    try {{
      const res = await fetch('/api/ask', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ query: q }})
      }});
      const data = await res.json();
      const bubble = bRow.querySelector('.bubble');
      if (data.error) {{
        bubble.textContent = 'Error: ' + data.error;
        return;
      }}
      bubble.innerHTML = renderMarkdown(data.answer);

      const sToggle = document.getElementById('cfg_sources');
      const displaySrc = sToggle ? sToggle.checked : showSources;
      if (displaySrc && data.sources && data.sources.length > 0) {{
        const sDiv = document.createElement('div');
        sDiv.className = 'sources';
        sDiv.innerHTML = '<span>📚 Sources:</span>' + data.sources.slice(0, 4).map(
          s => `<span class="source-pill">Page ${{s.chunk.page_number}} • ${{escapeHtml(s.chunk.section_path || s.chunk.strategy)}}</span>`
        ).join('');
        bubble.appendChild(sDiv);
      }}
      box.scrollTop = box.scrollHeight;
    }} catch (err) {{
      bRow.querySelector('.bubble').textContent = 'Error: ' + err;
    }}
  }}
</script>
</body>
</html>
"""



def _find_available_port(preferred_port: int) -> int:
    for p in range(preferred_port, preferred_port + 25):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return preferred_port


def launch(
    pipeline_or_source: RAGPipeline | str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = True,
    background: bool = True,
    # Full UI Customization Options:
    title: str = "raglib Chatbot",
    subtitle: str = "Ask any question about your document",
    bot_name: str = "AI Assistant",
    welcome_message: str = "Hi! Ask me anything about the document.",
    placeholder: str = "Ask a question and press Enter...",
    theme: str = "dark",
    accent_color: str | None = None,
    bg_color: str | None = None,
    card_color: str | None = None,
    text_color: str | None = None,
    font_size: str = "15px",
    show_sources: bool = False,
    allow_ui_edit: bool = True,
    custom_css: str = "",
) -> HTTPServer:
    """
    Launch a clean, minimal, fully customizable Question & Answer Chatbot UI.

    Args:
        pipeline_or_source: A `RAGPipeline` (created via `rag.create` or `rag.create_law`), a PDF path, or None.
        title: Header title shown at the top of the chatbot.
        subtitle: Subtitle shown below the title.
        bot_name: Display name of the chatbot above each answer bubble.
        welcome_message: Opening greeting shown in the chat window.
        placeholder: Input bar placeholder text.
        theme: Preset color theme: `"dark"`, `"light"`, `"emerald"`, or `"midnight"`.
        accent_color: Optional hex color (e.g. `"#ff5722"`) for buttons and user chat bubbles.
        bg_color: Optional hex color for the page background.
        card_color: Optional hex color for bot bubbles and input bar.
        text_color: Optional hex color for text.
        font_size: Message text size (e.g. `"15px"`, `"16px"`).
        show_sources: Whether to display source page/section citations under each answer.
        allow_ui_edit: Whether to show the `🎨 Edit UI` button in the header for live visual editing.
        custom_css: Optional custom CSS string to inject into the page.
    """
    if isinstance(pipeline_or_source, RAGPipeline):
        active_pipe = pipeline_or_source
    elif pipeline_or_source is not None:
        active_pipe = rag.create_law(pipeline_or_source)
    else:
        active_pipe = rag.create_law(DEFAULT_DEMO_LAW_TEXT)

    ui_config = {
        "title": title,
        "subtitle": subtitle,
        "bot_name": bot_name,
        "welcome_message": welcome_message,
        "placeholder": placeholder,
        "theme": theme,
        "accent_color": accent_color,
        "bg_color": bg_color,
        "card_color": card_color,
        "text_color": text_color,
        "font_size": font_size,
        "show_sources": show_sources,
        "allow_ui_edit": allow_ui_edit,
        "custom_css": custom_css,
    }
    rendered_html = _render_chatbot_html(ui_config)

    class _CleanChatHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(rendered_html.encode("utf-8"))

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                query = body.get("query", "").strip()
                res = active_pipe.ask(query)
                out = json.dumps(res.to_dict(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(out)
            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(err)

    actual_port = _find_available_port(port)
    server = HTTPServer((host, actual_port), _CleanChatHandler)
    url = f"http://{host}:{actual_port}"
    print(f"🚀 Clean Chatbot UI running at: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
    return server


def notebook_chat(
    pipeline_or_source: RAGPipeline | str | Path | None = None,
    title: str = "raglib Studio Chatbot",
    subtitle: str = "Interactive Notebook RAG Assistant",
    bot_name: str = "AI Assistant",
    welcome_message: str = "Hi! Ask me anything about the document.",
    theme: str = "dark",
    height: int = 620,
    port: int = 8501,
    **kwargs: Any,
) -> Any:
    """
    Launch an interactive ChatGPT-style RAG Chatbot directly inside Jupyter Notebook / JupyterLab / Google Colab,
    or return the running background server if not in an IPython environment.
    """
    server = launch(
        pipeline_or_source=pipeline_or_source,
        port=port,
        open_browser=False,
        background=True,
        title=title,
        subtitle=subtitle,
        bot_name=bot_name,
        welcome_message=welcome_message,
        theme=theme,
        **kwargs,
    )
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}"
    try:
        from IPython.display import IFrame, display  # type: ignore[import-not-found]

        display(IFrame(src=url, width="100%", height=height))
    except Exception:
        pass
    return server


UI_HTML = _render_chatbot_html({"title": "raglib Studio Chatbot"})

__all__ = ["launch", "notebook_chat", "UI_HTML", "THEMES"]
