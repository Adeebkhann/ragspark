"""
`ragspark.ui` — ChatGPT-style chatbot UI served by a standalone multi-threaded `ThreadingHTTPServer`.

    from ragspark import ui
    ui.launch(bot, background=False)            # blocks; Ctrl+C to stop
    server = ui.launch(bot)                     # background thread; server.shutdown() to stop

Features: ChatGPT dark palette, RagSpark flame header, collapsible conversation sidebar (kept in the
browser's localStorage), "New chat", per-conversation multi-turn memory, typing-effect answers,
optional source pills. If the requested port is taken, the OS assigns a free ephemeral port.
No Streamlit or other UI dependency.
"""

from __future__ import annotations

import html
import json
import os
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ragspark import rag
from ragspark.rag import RAGPipeline

DEFAULT_DEMO_TEXT = """RAGSPARK ENTERPRISE KNOWLEDGE BASE
Chapter 1: Platform Architecture & Operations
Section 1.1: System Overview - RagSpark is an all-in-one unified Retrieval-Augmented Generation (RAG) framework designed for multi-format document ingestion, hybrid vector/keyword search, and accurate grounded generation across any domain.
Section 1.2: Privacy & Security - All vector embeddings and parsed document chunks are kept in memory unless you pass persist_dir. Zero proprietary data leaves your environment without your explicit configuration.
Section 1.3: Scalability - RagSpark supports multi-threading, dynamic batching, and sub-50ms hybrid retrieval across enterprise-scale document repositories.
Section 2: Support & SLA - Standard enterprise response times are guaranteed within 1 hour for critical incidents, with automated health checks running continuously."""

THEMES: dict[str, dict[str, str]] = {
    # ChatGPT exact dark palette (per modern ChatGPT desktop/web interface).
    "chatgpt": {
        "bg": "#000000",          # Pure pitch black main canvas
        "sidebar": "#171717",     # Dark graphite sidebar
        "card": "#212121",        # Subtle elevated card
        "bot_bubble": "transparent", # Seamless stream like real ChatGPT
        "user_bubble": "#212121", # User pill
        "input": "#212121",       # Pill composer bar
        "text": "#ececf1",        # Crisp off-white text
        "muted": "#8e8ea0",       # Muted gray captions & icons
        "border": "#2f2f2f",      # Very faint border lines
        "accent": "#2563eb",      # ChatGPT modern blue action button
    },
    "dark": {
        "bg": "#0f172a",
        "sidebar": "#0b1222",
        "card": "#1e293b",
        "bot_bubble": "#1e293b",
        "user_bubble": "#1e293b",
        "input": "#1e293b",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "border": "#334155",
        "accent": "#6366f1",
    },
    "light": {
        "bg": "#f7f7f8",
        "sidebar": "#ececf1",
        "card": "#ffffff",
        "bot_bubble": "#ffffff",
        "user_bubble": "#ffffff",
        "input": "#ffffff",
        "text": "#202123",
        "muted": "#6e6e80",
        "border": "#d9d9e3",
        "accent": "#10a37f",
    },
    "emerald": {
        "bg": "#061814",
        "sidebar": "#04110e",
        "card": "#0d2922",
        "bot_bubble": "#0d2922",
        "user_bubble": "#0d2922",
        "input": "#0d2922",
        "text": "#ecfdf5",
        "muted": "#6ee7b7",
        "border": "#115e59",
        "accent": "#10b981",
    },
    "midnight": {
        "bg": "#09090b",
        "sidebar": "#050506",
        "card": "#18181b",
        "bot_bubble": "#18181b",
        "user_bubble": "#18181b",
        "input": "#18181b",
        "text": "#fafafa",
        "muted": "#a1a1aa",
        "border": "#27272a",
        "accent": "#3b82f6",
    },
}

FLAME_SVG = (
    '<svg class="flame" viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">'
    '<defs><linearGradient id="rsFlame" x1="0" y1="1" x2="0" y2="0">'
    '<stop offset="0" stop-color="#f97316"/><stop offset="0.55" stop-color="#f59e0b"/>'
    '<stop offset="1" stop-color="#fde047"/></linearGradient></defs>'
    '<path fill="url(#rsFlame)" d="M16 2c1.2 4.3-1.1 6.6-3.2 9C10.6 13.5 8 16.3 8 20.6 8 25.8 11.6 30 16 30'
    's8-4.2 8-9.4c0-3.6-1.8-6.4-3.6-8.3.3 2.4-.4 4.3-2.1 5.2.6-4.9-.6-10.3-2.3-15.5z"/>'
    '<path fill="#fff7ed" opacity=".85" d="M16.2 17.5c.5 2-.5 3-1.4 4.1-.8 1-1.6 2-1.6 3.4 0 1.9 1.3 3.2 2.8 3.2'
    's2.9-1.3 2.9-3.2c0-2.6-1.2-5-2.7-7.5z"/></svg>'
)

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  :root {
    --bg: __BG__;
    --sidebar: __SIDEBAR__;
    --card: __CARD__;
    --bot-bubble: __BOT__;
    --user-bubble: __USER__;
    --input: __INPUT__;
    --text: __TEXT__;
    --muted: __MUTED__;
    --border: __BORDER__;
    --accent: __ACCENT__;
    --font-size: __FONT_SIZE__;
    --sidebar-w: 260px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: "Söhne", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    overflow: hidden;
  }
  button { font: inherit; color: inherit; cursor: pointer; }

  /* Sidebar */
  .sidebar {
    width: var(--sidebar-w);
    flex: 0 0 var(--sidebar-w);
    background: var(--sidebar);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 12px;
    gap: 12px;
    transition: margin-left .2s ease;
    z-index: 20;
  }
  body.sidebar-collapsed .sidebar { margin-left: calc(-1 * var(--sidebar-w)); }
  .new-chat {
    display: flex; align-items: center; gap: 10px;
    width: 100%;
    padding: 10px 14px;
    background: #212121;
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 14px;
    font-weight: 500;
    text-align: left;
    transition: background .15s ease;
  }
  .new-chat:hover { background: #2f2f2f; }
  .history-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); padding: 8px 10px 2px; }
  .conv-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
  .conv {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    font-size: 13.5px;
    background: transparent;
    border: none;
    text-align: left;
    width: 100%;
    color: var(--text);
  }
  .conv:hover { background: #212121; }
  .conv.active { background: #212121; font-weight: 500; }
  .conv .conv-title { flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .conv .conv-del { opacity: 0; background: none; border: none; color: var(--muted); padding: 0 4px; font-size: 16px; line-height: 1; }
  .conv:hover .conv-del, .conv.active .conv-del { opacity: 1; }
  .conv .conv-del:hover { color: #ef4444; }
  .sidebar-foot { font-size: 11.5px; color: var(--muted); padding: 12px 10px 4px; border-top: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }

  /* Main column */
  .main { flex: 1; min-width: 0; display: flex; flex-direction: column; height: 100vh; position: relative; background: var(--bg); }
  .topbar {
    display: flex; align-items: center; gap: 14px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .icon-btn {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 7px;
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--muted);
  }
  .icon-btn:hover { color: var(--text); background: #212121; }
  .brand { display: flex; align-items: center; gap: 12px; min-width: 0; flex: 1; }
  .brand h1 { font-size: 17px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .brand p { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .toggle-src { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); white-space: nowrap; cursor: pointer; }

  .messages { flex: 1; overflow-y: auto; padding: 24px 20px 12px; }
  .thread { max-width: 760px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
  .msg { display: flex; gap: 14px; padding: 12px 0; border-radius: 0; line-height: 1.75; font-size: var(--font-size); }
  .msg.user {
    align-self: flex-end;
    background: #212121;
    border: 1px solid #2f2f2f;
    border-radius: 20px;
    padding: 10px 18px;
    max-width: 80%;
  }
  .msg.user .avatar { display: none; }
  .msg.bot { background: transparent; width: 100%; border-bottom: 1px solid rgba(255,255,255,0.04); padding-bottom: 20px; }
  .avatar {
    flex: 0 0 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
  }
  .msg.bot .avatar { background: #212121; border: 1px solid #2f2f2f; }
  .msg.bot .avatar .flame { width: 18px; height: 18px; }
  .body { flex: 1; min-width: 0; overflow-wrap: anywhere; }
  .msg.user .body { white-space: pre-wrap; font-size: 14.5px; }
  .empty { text-align: center; margin-top: 24vh; color: var(--muted); }
  .empty .flame { width: 50px; height: 50px; margin-bottom: 14px; }
  .empty h2 { color: #f3f4f6; font-size: 28px; margin: 0 0 8px; font-weight: 500; letter-spacing: -0.02em; }
  .empty p { font-size: 14.5px; color: #8e8ea0; }

  .md-content p { margin: 0 0 12px; }
  .md-content p:last-child { margin-bottom: 0; }
  .md-content h1, .md-content h2, .md-content h3, .md-content h4 { margin: 16px 0 8px; font-weight: 600; line-height: 1.35; color: #f9fafb; }
  .md-content h1:first-child, .md-content h2:first-child, .md-content h3:first-child, .md-content h4:first-child { margin-top: 0; }
  .md-content h3 { font-size: 1.1em; }
  .md-content ul, .md-content ol { margin: 8px 0 12px; padding-left: 22px; }
  .md-content li { margin-bottom: 6px; }
  .md-content blockquote { margin: 10px 0 14px; padding: 10px 16px; border-left: 3px solid var(--accent); background: rgba(255,255,255,.03); border-radius: 0 8px 8px 0; }
  .md-content code { background: rgba(255,255,255,.08); padding: 2px 6px; border-radius: 5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .88em; }
  .md-content pre { background: #171717; border: 1px solid #2f2f2f; color: #e5e7eb; padding: 14px; border-radius: 10px; overflow-x: auto; margin: 10px 0; }
  .md-content pre code { background: none; padding: 0; }
  .md-content hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
  .cursor { display: inline-block; width: 8px; height: 1.05em; vertical-align: text-bottom; background: #ececf1; margin-left: 2px; animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .typing { display: inline-flex; gap: 5px; padding: 6px 0; }
  .typing span { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); animation: bounce 1.2s infinite ease-in-out; }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce { 0%, 80%, 100% { opacity: .3; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
  .sources { margin-top: 14px; padding-top: 10px; border-top: 1px dashed var(--border); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12px; color: var(--muted); }
  .source-pill { background: #171717; border: 1px solid #2f2f2f; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; }
  .error { color: #fca5a5; }

  /* Pill-Shaped Composer Bar matching Screenshot */
  .composer { padding: 12px 20px 24px; background: var(--bg); }
  .composer form {
    max-width: 760px; margin: 0 auto;
    display: flex; align-items: center; gap: 10px;
    background: #1b1b1b;
    border: 1px solid #2a2a2a;
    border-radius: 999px;
    padding: 8px 10px 8px 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,.35);
    transition: border-color .15s ease;
  }
  .composer form:focus-within { border-color: #404040; }
  .composer textarea {
    flex: 1; resize: none; border: none; outline: none; background: transparent; color: var(--text);
    font: inherit; font-size: 15px; line-height: 1.4; max-height: 140px; padding: 4px 0;
  }
  .send-btn {
    background: #2563eb; color: #fff; border: none; border-radius: 50%;
    width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center;
    transition: opacity .15s ease, transform .1s ease;
  }
  .send-btn:hover { background: #1d4ed8; }
  .send-btn:disabled { opacity: .25; cursor: not-allowed; }
  .hint { max-width: 760px; margin: 8px auto 0; text-align: center; font-size: 11.5px; color: #71717a; }

  @media (max-width: 768px) {
    .sidebar { position: fixed; top: 0; bottom: 0; left: 0; box-shadow: 4px 0 24px rgba(0,0,0,.35); }
    .messages { padding: 16px 12px 8px; }
    .msg { padding: 14px; gap: 10px; }
    .toggle-src span { display: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    .sidebar { transition: none; }
    .cursor, .typing span { animation: none; }
  }
  __CUSTOM_CSS__
</style>
</head>
<body>
  <aside class="sidebar" id="sidebar" aria-label="Conversation history">
    <button class="new-chat" id="newChatBtn" type="button">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>
      New chat
    </button>
    <div class="history-label">Conversations</div>
    <nav class="conv-list" id="convList"></nav>
    <div class="sidebar-foot">RagSpark __VERSION__</div>
  </aside>

  <main class="main">
    <header class="topbar">
      <button class="icon-btn" id="toggleSidebar" type="button" aria-label="Toggle sidebar" title="Toggle sidebar">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></svg>
      </button>
      <div class="brand">
        __FLAME__
        <div style="min-width:0">
          <h1 id="ui_title">__TITLE__</h1>
          <p id="ui_subtitle">__SUBTITLE__</p>
        </div>
      </div>
      <label class="toggle-src"><input type="checkbox" id="cfg_sources" __SOURCES_CHECKED__> <span>Show sources</span></label>
    </header>

    <section class="messages" id="messages" aria-live="polite">
      <div class="thread" id="thread"></div>
    </section>

    <footer class="composer">
      <form id="composer">
        <textarea id="query_input" rows="1" placeholder="__PLACEHOLDER__" aria-label="Message"></textarea>
        <button class="send-btn" id="sendBtn" type="submit" aria-label="Send">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
        </button>
      </form>
      <p class="hint">Answers are generated from your documents. Enter to send, Shift+Enter for a new line.</p>
    </footer>
  </main>

<script>
(function () {
  const BOT_NAME = __BOT_NAME_JSON__;
  const WELCOME = __WELCOME_JSON__;
  const FLAME = __FLAME_JSON__;
  const STORE_KEY = 'ragspark.conversations.v1';
  const REDUCED_MOTION = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const el = (id) => document.getElementById(id);
  const thread = el('thread');
  const scroller = el('messages');
  const input = el('query_input');
  const sendBtn = el('sendBtn');
  let busy = false;

  function load() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '[]'); } catch (e) { return []; }
  }
  function save() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state.conversations.slice(0, 50))); } catch (e) {}
  }
  function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

  const state = { conversations: load(), activeId: null };

  function active() { return state.conversations.find(c => c.id === state.activeId); }

  function newChat() {
    const cur = active();
    if (cur && cur.messages.length === 0) { render(); input.focus(); return; }
    const conv = { id: uid(), title: 'New chat', messages: [] };
    state.conversations.unshift(conv);
    state.activeId = conv.id;
    save(); render(); input.focus();
    if (window.innerWidth <= 768) document.body.classList.add('sidebar-collapsed');
  }

  function escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function formatInline(text) {
    let s = escapeHtml(text);
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
    return s;
  }
  function renderMarkdown(md) {
    if (!md) return '';
    const lines = String(md).replace(/^\s*-?\s*\[Chunk\s+\d+[^\]]*\]\s*$/gm, '').split(/\r?\n/);
    const out = [];
    let list = null, quote = [], code = null;
    const flushList = () => { if (list) { out.push('</' + list + '>'); list = null; } };
    const flushQuote = () => { if (quote.length) { out.push('<blockquote>' + quote.join('<br>') + '</blockquote>'); quote = []; } };
    for (const raw of lines) {
      if (code !== null) {
        if (/^\s*```/.test(raw)) { out.push('<pre><code>' + escapeHtml(code.join('\n')) + '</code></pre>'); code = null; }
        else code.push(raw);
        continue;
      }
      if (/^\s*```/.test(raw)) { flushQuote(); flushList(); code = []; continue; }
      const line = raw.trim().replace(/^-\s+(#{1,4}\s+)/, '$1');
      if (!line) { flushQuote(); flushList(); continue; }
      if (/^---+$/.test(line)) { flushQuote(); flushList(); out.push('<hr>'); continue; }
      let m = line.match(/^(#{1,4})\s+(.*)$/);
      if (m) { flushQuote(); flushList(); const l = m[1].length; out.push('<h' + l + '>' + formatInline(m[2]) + '</h' + l + '>'); continue; }
      if (line.startsWith('>')) { flushList(); quote.push(formatInline(line.replace(/^>\s?/, ''))); continue; }
      flushQuote();
      m = line.match(/^[*\-•]\s+(.*)$/);
      if (m) { if (list !== 'ul') { flushList(); list = 'ul'; out.push('<ul>'); } out.push('<li>' + formatInline(m[1]) + '</li>'); continue; }
      m = line.match(/^\d+[.)]\s+(.*)$/);
      if (m) { if (list !== 'ol') { flushList(); list = 'ol'; out.push('<ol>'); } out.push('<li>' + formatInline(m[1]) + '</li>'); continue; }
      flushList();
      out.push('<p>' + formatInline(line) + '</p>');
    }
    if (code !== null) out.push('<pre><code>' + escapeHtml(code.join('\n')) + '</code></pre>');
    flushQuote(); flushList();
    return '<div class="md-content">' + out.join('') + '</div>';
  }

  function sourcesHtml(sources) {
    if (!el('cfg_sources').checked || !sources || !sources.length) return '';
    const pills = sources.slice(0, 4).map(s => {
      const name = (s.source || '').split(/[\\/]/).pop();
      const sec = s.section ? ' • ' + s.section : '';
      const label = (name && name !== 'inline' ? name + ' • ' : '') + 'p. ' + (s.page || 1) + sec;
      return '<span class="source-pill" title="' + escapeHtml(s.evidence || '') + '">' + escapeHtml(label.slice(0, 80)) + '</span>';
    }).join('');
    return '<div class="sources"><span>Sources:</span>' + pills + '</div>';
  }

  function messageNode(role) {
    const row = document.createElement('div');
    row.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
    const av = document.createElement('div');
    av.className = 'avatar';
    if (role === 'user') av.textContent = 'You'.slice(0, 1); else av.innerHTML = FLAME;
    av.setAttribute('aria-label', role === 'user' ? 'You' : BOT_NAME);
    const body = document.createElement('div');
    body.className = 'body';
    row.append(av, body);
    return { row, body };
  }

  function renderSidebar() {
    const list = el('convList');
    list.innerHTML = '';
    for (const c of state.conversations) {
      const item = document.createElement('div');
      item.className = 'conv' + (c.id === state.activeId ? ' active' : '');
      item.setAttribute('role', 'button');
      item.tabIndex = 0;
      const title = document.createElement('span');
      title.className = 'conv-title';
      title.textContent = c.title;
      const del = document.createElement('button');
      del.className = 'conv-del';
      del.type = 'button';
      del.title = 'Delete conversation';
      del.setAttribute('aria-label', 'Delete conversation');
      del.textContent = '×';
      del.onclick = (ev) => {
        ev.stopPropagation();
        state.conversations = state.conversations.filter(x => x.id !== c.id);
        if (state.activeId === c.id) state.activeId = state.conversations.length ? state.conversations[0].id : null;
        save(); render();
      };
      const open = () => { if (busy) return; state.activeId = c.id; render(); };
      item.onclick = open;
      item.onkeydown = (ev) => { if (ev.key === 'Enter') open(); };
      item.append(title, del);
      list.appendChild(item);
    }
  }

  function renderThread() {
    thread.innerHTML = '';
    const conv = active();
    if (!conv || conv.messages.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.innerHTML = FLAME + '<h2>' + escapeHtml(BOT_NAME) + '</h2>';
      const p = document.createElement('p');
      p.textContent = WELCOME;
      empty.appendChild(p);
      thread.appendChild(empty);
      return;
    }
    for (const m of conv.messages) {
      const { row, body } = messageNode(m.role);
      if (m.role === 'user') body.textContent = m.content;
      else body.innerHTML = (m.error ? '<p class="error">' + escapeHtml(m.content) + '</p>' : renderMarkdown(m.content)) + sourcesHtml(m.sources);
      thread.appendChild(row);
    }
    scroller.scrollTop = scroller.scrollHeight;
  }

  function render() { renderSidebar(); renderThread(); }

  function typeOut(body, text, sources) {
    return new Promise(resolve => {
      if (REDUCED_MOTION || !text) { body.innerHTML = renderMarkdown(text) + sourcesHtml(sources); resolve(); return; }
      const tokens = text.match(/\S+\s*|\s+/g) || [text];
      let i = 0;
      const step = Math.max(1, Math.ceil(tokens.length / 160));
      (function tick() {
        i = Math.min(tokens.length, i + step);
        body.innerHTML = renderMarkdown(tokens.slice(0, i).join('')).replace(/<\/(p|li|h\d)>(?!.*<\/(p|li|h\d)>)/, '<span class="cursor"></span></$1>');
        scroller.scrollTop = scroller.scrollHeight;
        if (i < tokens.length) setTimeout(tick, 18);
        else { body.innerHTML = renderMarkdown(text) + sourcesHtml(sources); scroller.scrollTop = scroller.scrollHeight; resolve(); }
      })();
    });
  }

  async function send(query) {
    if (busy) return;
    if (!active()) newChat();
    const conv = active();
    const history = conv.messages.filter(m => !m.error).slice(-12).map(m => ({ role: m.role, content: m.content }));
    conv.messages.push({ role: 'user', content: query });
    if (conv.title === 'New chat') conv.title = query.length > 42 ? query.slice(0, 42).replace(/\s+\S*$/, '') + '…' : query;
    save(); render();

    busy = true; sendBtn.disabled = true;
    const { row, body } = messageNode('assistant');
    body.innerHTML = '<div class="typing" aria-label="Thinking"><span></span><span></span><span></span></div>';
    thread.appendChild(row);
    scroller.scrollTop = scroller.scrollHeight;

    let msg;
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history })
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || ('HTTP ' + res.status));
      msg = { role: 'assistant', content: data.answer || '', sources: data.citations || [] };
      await typeOut(body, msg.content, msg.sources);
    } catch (err) {
      msg = { role: 'assistant', content: 'Error: ' + (err && err.message ? err.message : err), error: true };
      body.innerHTML = '<p class="error">' + escapeHtml(msg.content) + '</p>';
    }
    conv.messages.push(msg);
    save();
    busy = false; sendBtn.disabled = false;
    input.focus();
  }

  el('composer').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const q = input.value.trim();
    if (!q || busy) return;
    input.value = '';
    input.style.height = 'auto';
    send(q);
  });
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); el('composer').requestSubmit(); }
  });
  input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(200, input.scrollHeight) + 'px'; });
  el('newChatBtn').addEventListener('click', () => { if (!busy) newChat(); });
  el('toggleSidebar').addEventListener('click', () => document.body.classList.toggle('sidebar-collapsed'));
  el('cfg_sources').addEventListener('change', () => { if (!busy) renderThread(); });

  if (window.innerWidth <= 768) document.body.classList.add('sidebar-collapsed');
  state.activeId = state.conversations.length ? state.conversations[0].id : null;
  if (!state.activeId) newChat(); else render();
  input.focus();
})();
</script>
</body>
</html>
"""


def _palette(config: dict[str, Any]) -> dict[str, str]:
    theme_name = str(config.get("theme", "chatgpt")).lower()
    palette = dict(THEMES.get(theme_name, THEMES["chatgpt"]))
    if config.get("bg_color"):
        palette["bg"] = config["bg_color"]
        palette["sidebar"] = config["bg_color"]
    if config.get("card_color"):
        palette["card"] = palette["bot_bubble"] = palette["user_bubble"] = config["card_color"]
    if config.get("text_color"):
        palette["text"] = config["text_color"]
    if config.get("accent_color"):
        palette["accent"] = config["accent_color"]
    return palette


def _css_value(value: Any) -> str:
    """Keep user-supplied colours/sizes from breaking out of the CSS declaration."""
    return "".join(ch for ch in str(value) if ch.isalnum() or ch in "#%.,() -")[:64]


def _render_chatbot_html(config: dict[str, Any]) -> str:
    from ragspark import __version__

    p = _palette(config)
    esc = lambda key, default: html.escape(str(config.get(key) or default), quote=True)  # noqa: E731
    replacements = {
        "__TITLE__": esc("title", "RagSpark Chat"),
        "__SUBTITLE__": esc("subtitle", "Ask anything about your documents"),
        "__PLACEHOLDER__": esc("placeholder", "Message RagSpark..."),
        "__BG__": _css_value(p["bg"]),
        "__SIDEBAR__": _css_value(p["sidebar"]),
        "__CARD__": _css_value(p["card"]),
        "__BOT__": _css_value(p["bot_bubble"]),
        "__USER__": _css_value(p["user_bubble"]),
        "__INPUT__": _css_value(p["input"]),
        "__TEXT__": _css_value(p["text"]),
        "__MUTED__": _css_value(p["muted"]),
        "__BORDER__": _css_value(p["border"]),
        "__ACCENT__": _css_value(p["accent"]),
        "__FONT_SIZE__": _css_value(config.get("font_size", "15px")),
        "__SOURCES_CHECKED__": "checked" if config.get("show_sources") else "",
        "__VERSION__": html.escape(f"v{__version__}"),
        "__FLAME__": FLAME_SVG,
        "__BOT_NAME_JSON__": json.dumps(str(config.get("bot_name") or "RagSpark")).replace("<", "\\u003c"),
        "__WELCOME_JSON__": json.dumps(str(config.get("welcome_message") or "How can I help you today?")).replace("<", "\\u003c"),
        "__FLAME_JSON__": json.dumps(FLAME_SVG),
        "__CUSTOM_CSS__": str(config.get("custom_css", "")).replace("</", "<\\/"),
    }
    out = _TEMPLATE
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


class _RagSparkServer(ThreadingHTTPServer):
    """Threaded HTTP server that refuses to share a port (so a busy port is detected on Windows too)."""

    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _bind_server(host: str, port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    """Bind `port`; if it is taken, let the OS pick a free ephemeral port instead."""
    try:
        return _RagSparkServer((host, port), handler)
    except OSError:
        return _RagSparkServer((host, 0), handler)


def _find_available_port(preferred_port: int, host: str = "127.0.0.1") -> int:
    """Return `preferred_port` if it can be bound, else a free ephemeral port chosen by the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind((host, preferred_port))
            return preferred_port
        except OSError:
            s.bind((host, 0))
            return s.getsockname()[1]


def _make_handler(pipe: RAGPipeline, page: bytes) -> type[BaseHTTPRequestHandler]:
    class _ChatHandler(BaseHTTPRequestHandler):
        server_version = "RagSpark"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/api/ask":
                self._json(404, {"error": "not found"})
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 2 * 1024 * 1024)
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                self._json(400, {"error": "Request body must be JSON."})
                return
            query = str(body.get("query", "")).strip()
            if not query:
                self._json(400, {"error": "Empty question."})
                return
            history = body.get("history") if isinstance(body.get("history"), list) else []
            try:
                res = pipe.respond(query, history=[m for m in history if isinstance(m, dict)])
                self._json(
                    200,
                    {
                        "answer": res.answer,
                        "citations": res.citations,
                        "sources": [s.to_dict() for s in res.sources],
                        "llm_provider": res.llm_provider,
                        "metadata": {k: v for k, v in res.metadata.items() if k != "detected_concepts"},
                    },
                )
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return _ChatHandler


def launch(
    pipeline_or_source: RAGPipeline | str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = True,
    background: bool = True,
    title: str = "RagSpark Chat",
    subtitle: str = "Ask anything about your documents",
    bot_name: str = "RagSpark",
    welcome_message: str = "How can I help you today?",
    placeholder: str = "Message RagSpark...",
    theme: str = "chatgpt",
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
    Launch the ChatGPT-style chatbot UI for a `RAGPipeline` (or a path / text to build one from).

    Args:
        pipeline_or_source: A `RAGPipeline`, a document path, raw text, or None for a demo.
        port: Preferred port. If it is busy, a free ephemeral port is used (see the printed URL
            or `server.server_address[1]`).
        background: True runs the server in a daemon thread and returns it; False blocks.
        theme: `"chatgpt"` (default), `"dark"`, `"light"`, `"emerald"`, or `"midnight"`.
        accent_color / bg_color / card_color / text_color / font_size: palette overrides.
        show_sources: Initial state of the "Show sources" toggle.
        allow_ui_edit: Kept for backward compatibility (the new UI has no edit drawer).
        custom_css: Extra CSS appended to the page.
    """
    if isinstance(pipeline_or_source, RAGPipeline):
        active_pipe = pipeline_or_source
    elif pipeline_or_source is not None:
        active_pipe = rag.create(pipeline_or_source)
    else:
        active_pipe = rag.create(DEFAULT_DEMO_TEXT)

    page = _render_chatbot_html(
        {
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
            "custom_css": custom_css,
        }
    ).encode("utf-8")

    server = _bind_server(host, port, _make_handler(active_pipe, page))
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    note = "" if actual_port == port or port == 0 else f" (port {port} was busy)"
    print(f"RagSpark chat UI running at {url}{note}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if background:
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        return server

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server


def notebook_chat(
    pipeline_or_source: RAGPipeline | str | Path | None = None,
    title: str = "RagSpark Chat",
    subtitle: str = "Interactive notebook RAG assistant",
    bot_name: str = "RagSpark",
    welcome_message: str = "How can I help you today?",
    theme: str = "chatgpt",
    height: int = 620,
    port: int = 8501,
    **kwargs: Any,
) -> Any:
    """Launch the chat UI in the background and embed it in Jupyter / JupyterLab / Colab (returns the server)."""
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
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        from IPython.display import IFrame, display  # type: ignore[import-not-found]

        display(IFrame(src=url, width="100%", height=height))
    except Exception:
        pass
    return server


def __getattr__(name: str) -> Any:
    # Rendered lazily: building it at import time would import `ragspark.__version__` circularly.
    if name == "UI_HTML":
        return _render_chatbot_html({"title": "RagSpark Chat"})
    raise AttributeError(name)


__all__ = ["launch", "notebook_chat", "UI_HTML", "THEMES", "FLAME_SVG"]
