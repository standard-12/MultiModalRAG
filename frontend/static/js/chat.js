/**
 * chat.js — Recall frontend logic
 *
 * Modules (logical sections):
 *   1. DOM references
 *   2. State
 *   3. API service layer
 *   4. UI helpers (text formatting, HTML escaping)
 *   5. Message rendering (user, assistant, typing, sources, pipeline details)
 *   6. Send flow
 *   7. Upload flow
 *   8. Status / document polling
 *   9. Input event bindings
 *  10. Boot
 */

'use strict';

/* ═══════════════════════════════════════════════════════════ 1. DOM refs ════ */
const dom = {
  messagesContainer: document.getElementById('messagesContainer'),
  messagesViewport:  document.getElementById('messagesViewport'),
  queryInput:        document.getElementById('queryInput'),
  sendBtn:           document.getElementById('sendBtn'),
  statText:          document.getElementById('statText'),
  statImages:        document.getElementById('statImages'),
  statAudio:         document.getElementById('statAudio'),
  docList:           document.getElementById('docList'),
  dropZone:          document.getElementById('dropZone'),
  fileInput:         document.getElementById('fileInput'),
  uploadStatus:      document.getElementById('uploadStatus'),
};

/* ══════════════════════════════════════════════════════════════ 2. State ════ */
let isBusy = false;

/* ══════════════════════════════════════════════════════ 3. API service ══════ */
const API = {
  async status() {
    const res = await fetch('/api/status');
    return res.json();
  },
  async documents() {
    const res = await fetch('/api/documents');
    return res.json();
  },
  async chatStream(query) {
    return fetch('/api/chat/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query }),
    });
  },
  async upload(file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    return { ok: res.ok, data: await res.json() };
  },
};

/* ════════════════════════════════════════════════ 4. UI Helpers / Formatters ═ */

/** Escape HTML special characters. */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Minimal markdown-like formatting (bold, italic, inline code, newlines). */
function formatMarkdown(text) {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g,  '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,      '<em>$1</em>')
    .replace(/`(.+?)`/g,        '<code>$1</code>')
    .replace(/\n/g,             '<br>');
}

/** Scroll the message viewport to the bottom. */
function scrollToBottom() {
  dom.messagesViewport.scrollTop = dom.messagesViewport.scrollHeight;
}

/* ═══════════════════════════════════════════════════ 5. Message rendering ═══ */

/**
 * Append a message row to the conversation.
 * @param {'user'|'assistant'} role
 * @param {string} htmlContent  — inner HTML for the bubble
 * @returns {HTMLElement}  the bubble element
 */
function appendMessage(role, htmlContent) {
  const isUser = role === 'user';

  const avatarIcon = isUser
    ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
    : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="20,2 4,11 4,29 20,38 36,29 36,11"/><circle cx="20" cy="20" r="5" fill="currentColor"/></svg>`;

  const wrap = document.createElement('div');
  wrap.className = `message message--${role}`;
  wrap.innerHTML = `
    <div class="message-avatar">${avatarIcon}</div>
    <div class="message-body">
      <div class="message-bubble">${htmlContent}</div>
    </div>`;

  dom.messagesContainer.appendChild(wrap);
  scrollToBottom();
  return wrap.querySelector('.message-bubble');
}

/** Append a typing indicator and return the bubble element for later replacement. */
function appendTypingIndicator() {
  return appendMessage(
    'assistant',
    '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>'
  );
}

/* ── Sources ──────────────────────────────────────────────────────────────── */
function renderSources(sources, bubble) {
  if (!sources?.length) return;

  // Source chips
  const chips = sources.map(s => {
    const typeClass = s.type === 'image'
      ? 'source-chip-type--img'
      : s.type === 'audio'
        ? 'source-chip-type--audio'
        : 'source-chip-type--txt';
    const typeLabel = s.type === 'image' ? 'IMG' : s.type === 'audio' ? 'AUD' : 'TXT';
    const extra = s.type === 'audio' && s.duration
      ? ` <span class="source-chip-score">${s.duration}s</span>`
      : s.type !== 'audio'
        ? `<span class="source-chip-score">${s.score}</span>`
        : '';
    return `<span class="source-chip" title="${escHtml(s.file)}">
      <span class="source-chip-type ${typeClass}">${typeLabel}</span>
      ${escHtml(s.title)}
      ${extra}
    </span>`;
  }).join('');

  const strip = document.createElement('div');
  strip.className = 'sources-strip';
  strip.innerHTML = chips;
  bubble.appendChild(strip);

  // Thumbnails for image sources
  const imgSources = sources.filter(s => s.type === 'image' && s.thumbnail);
  if (imgSources.length) {
    const thumbs = document.createElement('div');
    thumbs.className = 'source-thumbnails';
    imgSources.forEach(s => {
      const img = document.createElement('img');
      img.src       = s.thumbnail;
      img.alt       = s.title;
      img.className = 'source-thumbnail';
      thumbs.appendChild(img);
    });
    bubble.appendChild(thumbs);
  }

  scrollToBottom();
}

/* ── Pipeline details (expandable) ───────────────────────────────────────── */
function renderPipelineDetails(details, bubble) {
  const { hyde_document, expanded_queries, rrf_results } = details;

  const maxScore = rrf_results?.length
    ? Math.max(...rrf_results.map(r => r.rrf_score))
    : 1;

  const rrfRows = (rrf_results || []).map(r => {
    const pct      = maxScore > 0 ? (r.rrf_score / maxScore) * 100 : 0;
    const modClass = r.modality === 'image' ? 'rrf-modality--img' : 'rrf-modality--text';
    const modLabel = r.modality === 'image' ? 'IMG' : 'TXT';
    return `<div class="rrf-row">
      <span class="rrf-modality ${modClass}">${modLabel}</span>
      <span class="rrf-title" title="${escHtml(r.file)}">${escHtml(r.title || r.file)}</span>
      <div class="rrf-bar-track"><div class="rrf-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="rrf-score">${r.rrf_score}</span>
    </div>`;
  }).join('');

  const queryItems = (expanded_queries || []).map((q, i) =>
    `<li class="${i === 0 ? 'pd-query--original' : ''}">${escHtml(q)}</li>`
  ).join('');

  const details_el = document.createElement('details');
  details_el.className = 'pipeline-details';
  details_el.innerHTML = `
    <summary>Pipeline Details</summary>
    <div class="pd-body">
      <div>
        <div class="pd-section-label">HyDE Document</div>
        <div class="pd-hyde-text">${escHtml(hyde_document || '—')}</div>
      </div>
      <div>
        <div class="pd-section-label">Expanded Queries (${(expanded_queries || []).length})</div>
        <ol class="pd-queries">${queryItems}</ol>
      </div>
      <div>
        <div class="pd-section-label">RRF Scores</div>
        <div class="pd-rrf-list">${rrfRows || '<span class="pd-empty">No results</span>'}</div>
      </div>
    </div>`;

  bubble.appendChild(details_el);
  scrollToBottom();
}

/* ══════════════════════════════════════════════════════════ 6. Send flow ════ */
async function sendMessage() {
  const query = dom.queryInput.value.trim();
  if (!query || isBusy) return;

  isBusy = true;
  dom.sendBtn.disabled = true;
  dom.queryInput.value = '';
  autoResizeTextarea();

  appendMessage('user', escHtml(query));
  const assistantBubble = appendTypingIndicator();

  try {
    const res = await API.chatStream(query);
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();

    let buffer          = '';
    let isFirstToken    = true;
    let pendingSources  = null;
    let pendingDetails  = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE events (separated by double newlines)
      const events = buffer.split('\n\n');
      buffer = events.pop(); // keep any partial trailing event

      for (const rawEvent of events) {
        const line = rawEvent.trim();
        if (!line.startsWith('data:')) continue;

        const payload = JSON.parse(line.slice(5).trim());

        if (payload.type === 'pipeline_details') {
          pendingDetails = payload;
        }

        if (payload.type === 'token') {
          if (isFirstToken) {
            assistantBubble.innerHTML = ''; // clear typing indicator
            isFirstToken = false;
          }
          const span = document.createElement('span');
          span.innerHTML = formatMarkdown(payload.content);
          assistantBubble.appendChild(span);
          scrollToBottom();
        }

        if (payload.type === 'sources') {
          pendingSources = payload.sources;
        }

        if (payload.type === 'done') {
          if (pendingSources)  renderSources(pendingSources, assistantBubble);
          if (pendingDetails)  renderPipelineDetails(pendingDetails, assistantBubble);
        }
      }
    }
  } catch (err) {
    assistantBubble.innerHTML = `<span style="color:var(--color-error)">Error: ${escHtml(err.message)}</span>`;
  } finally {
    isBusy = false;
    dom.sendBtn.disabled = false;
    dom.queryInput.focus();
  }
}

/* ══════════════════════════════════════════════════════════ 7. Upload flow ══ */
function setUploadFeedback(message, variant /* 'loading'|'ok'|'err' */) {
  dom.uploadStatus.textContent = message;
  dom.uploadStatus.className   = `upload-feedback upload-feedback--${variant}`;
  // 'hidden' class is removed by setting className above, so nothing extra needed
}

async function uploadFile(file) {
  setUploadFeedback(`Uploading ${file.name}…`, 'loading');

  try {
    const { ok, data } = await API.upload(file);

    if (!ok) {
      setUploadFeedback(data.error || 'Upload failed', 'err');
    } else {
      setUploadFeedback(data.message || 'File indexed successfully', 'ok');
      await Promise.all([refreshStatus(), refreshDocumentList()]);
    }
  } catch (err) {
    setUploadFeedback(`Network error: ${err.message}`, 'err');
  }

  setTimeout(() => dom.uploadStatus.classList.add('hidden'), 5000);
}

/* ════════════════════════════════════════════════ 8. Status / doc polling ═══ */
async function refreshStatus() {
  try {
    const data = await API.status();

    dom.statText.textContent    = data.stats?.text_chunks  ?? '—';
    dom.statImages.textContent  = data.stats?.images       ?? '—';
    if (dom.statAudio) dom.statAudio.textContent = data.stats?.audio_chunks ?? '—';
  } catch { /* statistics stay unchanged if the server is unavailable */ }
}

async function refreshDocumentList() {
  try {
    const data = await API.documents();
    const all  = [
      ...(data.documents || []).map(f => ({ name: f, kind: 'doc' })),
      ...(data.images    || []).map(f => ({ name: f, kind: 'img' })),
      ...(data.audio     || []).map(f => ({ name: f, kind: 'audio' })),
    ];

    dom.docList.innerHTML = all.length
      ? all.map(f => `
          <li class="file-item">
            <span class="file-badge file-badge--${f.kind}">${
              f.kind === 'doc' ? 'DOC' : f.kind === 'img' ? 'IMG' : 'AUD'
            }</span>
            ${escHtml(f.name)}
          </li>`).join('')
      : '<li class="file-item file-item--empty">No files indexed yet</li>';
  } catch { /* silent */ }
}

/* ═══════════════════════════════════════════════ 9. Input event bindings ════ */

/** Auto-expand textarea height to fit content. */
function autoResizeTextarea() {
  dom.queryInput.style.height = 'auto';
  dom.queryInput.style.height = Math.min(dom.queryInput.scrollHeight, 150) + 'px';
}

dom.queryInput.addEventListener('input', autoResizeTextarea);
dom.queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
dom.sendBtn.addEventListener('click', sendMessage);

/* File upload bindings */
dom.fileInput.addEventListener('change', () => {
  if (dom.fileInput.files[0]) uploadFile(dom.fileInput.files[0]);
  dom.fileInput.value = '';
});

dom.dropZone.addEventListener('click', () => dom.fileInput.click());

dom.dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dom.dropZone.classList.add('drop-zone--active');
});
dom.dropZone.addEventListener('dragleave', () => {
  dom.dropZone.classList.remove('drop-zone--active');
});

dom.dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dom.dropZone.classList.remove('drop-zone--active');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

/* ═══════════════════════════════════════════════════════════════ 10. Boot ═══ */
(async () => {
  await Promise.all([refreshStatus(), refreshDocumentList()]);
  setInterval(refreshStatus, 30_000);
})();
