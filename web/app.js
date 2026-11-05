// ---------- Theme ----------
(function initTheme() {
  const stored = (() => { try { return localStorage.getItem('tc-theme'); } catch (e) { return null; } })();
  if (stored === 'light' || stored === 'dark') document.documentElement.setAttribute('data-theme', stored);

  const toggle = document.getElementById('themeToggle');
  toggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('tc-theme', next); } catch (e) {}
  });
})();

// ---------- Tab navigation ----------
const views = {
  home: document.getElementById('view-home'),
  chat: document.getElementById('view-chat'),
  faq: document.getElementById('view-faq'),
};
const tabBtns = document.querySelectorAll('.tab-btn');

function goTo(name, anchorId) {
  Object.entries(views).forEach(([key, el]) => el.classList.toggle('active', key === name));
  tabBtns.forEach(b => {
    const isActive = b.dataset.nav === name;
    b.classList.toggle('active', isActive);
    if (b.getAttribute('role') === 'tab') b.setAttribute('aria-selected', String(isActive));
  });
  const anchor = anchorId && document.getElementById(anchorId);
  if (anchor) {
    anchor.scrollIntoView({ block: 'start', behavior: 'instant' in window ? 'instant' : 'auto' });
  } else {
    window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
  }
}

document.querySelectorAll('[data-nav]').forEach(el => {
  el.addEventListener('click', () => goTo(el.dataset.nav, el.dataset.anchor));
});

// ---------- FAQ accordion ----------
document.querySelectorAll('.faq-item').forEach(item => {
  const trigger = item.querySelector('.faq-q');
  trigger.addEventListener('click', () => {
    const wasOpen = item.classList.contains('open');
    document.querySelectorAll('.faq-item.open').forEach(i => {
      i.classList.remove('open');
      i.querySelector('.faq-q').setAttribute('aria-expanded', 'false');
    });
    if (!wasOpen) {
      item.classList.add('open');
      trigger.setAttribute('aria-expanded', 'true');
    }
  });
});

// ---------- Live backend ----------
const API_ENDPOINT = '/api/v1/fact-check/stream';

// What runs next once the named pipeline step has finished.
const NEXT_STAGE = {
  ingest: 'Breaking the claim into checkable parts',
  route_and_decompose: 'Searching the medical library',
  hybrid_retrieve_and_rerank: 'Weighing the evidence',
  evidence_and_calibration: 'Checking for safety risks',
  assess_risk: 'Writing the answer',
};

const VERDICT_LABEL = {
  SUPPORTED: 'Supported by evidence',
  MOSTLY_SUPPORTED: 'Mostly supported',
  SUPPORTED_WITH_CAVEATS: 'Supported, with caveats',
  MIXED_EVIDENCE: 'Mixed evidence',
  MISLEADING: 'Misleading',
  MOSTLY_FALSE: 'Mostly false',
  FALSE: 'False',
  INSUFFICIENT_EVIDENCE: 'Insufficient evidence',
  WITHHELD: 'Verdict withheld',
  NOT_VERIFIABLE: 'Not verifiable',
  NOT_A_FACT_CHECK: 'Not a fact-check',
};

const VERDICT_BADGE_CLASS = {
  SUPPORTED: 'v-supported',
  MOSTLY_SUPPORTED: 'v-supported',
  SUPPORTED_WITH_CAVEATS: 'v-caveats',
  MIXED_EVIDENCE: 'v-mixed',
  MISLEADING: 'v-misleading',
  MOSTLY_FALSE: 'v-false',
  FALSE: 'v-false',
  INSUFFICIENT_EVIDENCE: 'v-insufficient',
  WITHHELD: 'v-misleading',
  NOT_VERIFIABLE: 'v-notverifiable',
  NOT_A_FACT_CHECK: 'v-notfactcheck',
};

async function fetchVerdict(text, onStage) {
  let res;
  try {
    res = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, input_type: 'text', language: 'auto' }),
    });
  } catch (networkErr) {
    throw new Error('Could not reach the verification engine. Is the TruthCheck server running?');
  }
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail || ''; } catch (e) { /* body not JSON */ }
    throw new Error(detail || `The server returned an error (HTTP ${res.status}).`);
  }
  // One JSON object per line: {stage} as each step ends, then {result} or {error}.
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.error) throw new Error(event.error);
      if (event.result) return event.result;
      if (event.stage && NEXT_STAGE[event.stage]) onStage(NEXT_STAGE[event.stage]);
    }
    if (done) throw new Error('The server stopped before finishing this check. Please try again.');
  }
}

// ---------- Chat rendering ----------
const chatLog = document.getElementById('chatLog');
const composerForm = document.getElementById('composerForm');
const composerInput = document.getElementById('composerInput');
const composerSend = document.getElementById('composerSend');
const suggestionRow = document.getElementById('suggestionRow');

function scrollLogToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-avatar">You</div>
    <div class="msg-bubble">${escapeHtml(text)}</div>
  `;
  chatLog.appendChild(row);
  scrollLogToBottom();
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

const BOT_AVATAR_SVG = '<svg viewBox="0 0 24 24" fill="none" width="14" height="14"><path d="M12 2L4 5.5V11C4 16.2 7.4 20.9 12 22C16.6 20.9 20 16.2 20 11V5.5L12 2Z" stroke="white" stroke-width="1.8" stroke-linejoin="round"/></svg>';

function addThinkingIndicator() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = `
    <div class="msg-avatar">${BOT_AVATAR_SVG}</div>
    <div class="msg-bubble"><span class="typing-dots"><span></span><span></span><span></span></span><span class="stage-label" role="status">Reading the claim</span></div>
  `;
  chatLog.appendChild(row);
  scrollLogToBottom();
  return row;
}

function badgeLabelForConfidence(v) {
  return Math.round((v || 0) * 100) + '%';
}

function renderVerdictCard(data) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  const verdict = data.verdict || 'INSUFFICIENT_EVIDENCE';
  const badgeClass = VERDICT_BADGE_CLASS[verdict] || 'v-notverifiable';
  const label = VERDICT_LABEL[verdict] || verdict;
  const explanation = escapeHtml(data.explanation || 'No explanation was returned for this claim.');

  if (verdict === 'NOT_A_FACT_CHECK') {
    row.innerHTML = `
      <div class="msg-avatar">${BOT_AVATAR_SVG}</div>
      <div class="verdict-result">
        <div class="vr-head">
          <span class="vr-badge ${badgeClass}">${escapeHtml(label)}</span>
          <span class="vr-conf">routed before retrieval</span>
        </div>
        <div class="vr-body">
          <p class="short" style="margin:0 0 14px;">${explanation}</p>
          <p class="why" style="margin:0;"><b>Please consult a qualified clinician or registered dietitian</b> about your specific case — this is general information, not personalized medical advice.</p>
        </div>
      </div>
    `;
    chatLog.appendChild(row);
    scrollLogToBottom();
    return;
  }

  const safetyHtml = data.safety_notice ? `<div class="vr-safety" role="alert">${escapeHtml(data.safety_notice)}</div>` : '';

  if (verdict === 'WITHHELD') {
    row.innerHTML = `
      <div class="msg-avatar">${BOT_AVATAR_SVG}</div>
      <div class="verdict-result">
        <div class="vr-head">
          <span class="vr-badge ${badgeClass}">${escapeHtml(label)}</span>
        </div>
        <div class="vr-body">${safetyHtml}</div>
      </div>
    `;
    chatLog.appendChild(row);
    scrollLogToBottom();
    return;
  }

  const details = Array.isArray(data.evidence_details) ? data.evidence_details.filter(d => d && d.passage) : [];
  const evidenceHtml = details.length
    ? details.map(d => `<div class="evidence-item"><span class="src">${escapeHtml(d.passage)}</span>${d.source_file ? `<div class="evidence-source">Source: ${/^https?:\/\//.test(d.source_url || '') ? `<a href="${escapeHtml(d.source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(new URL(d.source_url).hostname)}</a>` : escapeHtml(d.source_file)}${d.source_tier ? ` · ${escapeHtml(d.source_tier)}` : ''}</div>` : ''}</div>`).join('')
    : `<div class="evidence-item"><span class="src" style="color:var(--text-faint);">No passages were retrieved for this claim.</span></div>`;

  row.innerHTML = `
    <div class="msg-avatar">${BOT_AVATAR_SVG}</div>
    <div class="verdict-result">
      <div class="vr-head">
        <span class="vr-badge ${badgeClass}">${escapeHtml(label)}</span>
        <button type="button" class="vr-explain">what does this mean?</button>
      </div>
      <div class="vr-body">
        ${safetyHtml}
        <p class="short">${explanation}</p>
        ${verdict === 'INSUFFICIENT_EVIDENCE' || verdict === 'MIXED_EVIDENCE' ? '' : `<p class="why">confidence ${badgeLabelForConfidence(data.confidence)}</p>`}
        ${verdict === 'MIXED_EVIDENCE' || (verdict !== 'INSUFFICIENT_EVIDENCE' && data.confidence < 0.6) ? `<div class="vr-note">Treat this with caution: confidence is low or the evidence is mixed.</div>` : ''}
        <div class="vr-section-label">Evidence retrieved</div>
        <div class="evidence-list">${evidenceHtml}</div>
      </div>
    </div>
  `;
  chatLog.appendChild(row);
  const explainBtn = row.querySelector('.vr-explain');
  if (explainBtn) explainBtn.addEventListener('click', () => goTo('home', 'verdict-taxonomy'));
  scrollLogToBottom();
}

function renderError(message) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = `
    <div class="msg-avatar">${BOT_AVATAR_SVG}</div>
    <div class="verdict-result">
      <div class="vr-head">
        <span class="vr-badge v-false">Couldn't verify this claim</span>
      </div>
      <div class="vr-body">
        <p class="why" style="margin:0;">${escapeHtml(message)}</p>
      </div>
    </div>
  `;
  chatLog.appendChild(row);
  scrollLogToBottom();
}

async function handleSubmit(text) {
  if (!text.trim()) return;
  composerInput.value = '';
  composerSend.disabled = true;

  addUserMessage(text);
  const indicator = addThinkingIndicator();

  try {
    const data = await fetchVerdict(text, label => {
      indicator.querySelector('.stage-label').textContent = label;
      scrollLogToBottom();
    });
    indicator.remove();
    renderVerdictCard(data);
  } catch (err) {
    indicator.remove();
    renderError(err.message || 'Something went wrong while checking this claim. Please try again.');
  }

  composerSend.disabled = false;
  composerInput.focus();
}

composerForm.addEventListener('submit', (e) => {
  e.preventDefault();
  handleSubmit(composerInput.value);
});

document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => handleSubmit(chip.dataset.q));
});

// Seed with a welcome message
(function seed() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 24 24" fill="none" width="14" height="14"><path d="M12 2L4 5.5V11C4 16.2 7.4 20.9 12 22C16.6 20.9 20 16.2 20 11V5.5L12 2Z" stroke="white" stroke-width="1.8" stroke-linejoin="round"/></svg>
    </div>
    <div class="msg-bubble">Bhej dijiye koi health claim — Hinglish, Hindi ya English, jaise bhi aaya ho. Main ise decompose karke, evidence retrieve karke, ek verdict citations ke saath dunga.</div>
  `;
  chatLog.appendChild(row);
})();
