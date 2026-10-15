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

// ---------- Chat mode ----------
let currentMode = 'standard';
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
  });
});

// ---------- Live backend ----------
const API_ENDPOINT = '/api/v1/fact-check';

const VERDICT_LABEL = {
  SUPPORTED: 'Supported by evidence',
  MOSTLY_SUPPORTED: 'Mostly supported',
  SUPPORTED_WITH_CAVEATS: 'Supported, with caveats',
  MIXED_EVIDENCE: 'Mixed evidence',
  MISLEADING: 'Misleading',
  MOSTLY_FALSE: 'Mostly false',
  FALSE: 'False',
  INSUFFICIENT_EVIDENCE: 'Insufficient evidence',
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
  NOT_VERIFIABLE: 'v-notverifiable',
  NOT_A_FACT_CHECK: 'v-notfactcheck',
};

async function fetchVerdict(text, mode) {
  let res;
  try {
    res = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, input_type: 'text', language: 'auto', mode }),
    });
  } catch (networkErr) {
    throw new Error('Could not reach the verification engine. Is the TruthCheck server running?');
  }
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail || ''; } catch (e) { /* body not JSON */ }
    throw new Error(detail || `The server returned an error (HTTP ${res.status}).`);
  }
  return res.json();
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
    <div class="msg-bubble"><span class="typing-dots"><span></span><span></span><span></span></span></div>
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

  const evidence = Array.isArray(data.evidence) ? data.evidence.filter(Boolean) : [];
  const evidenceHtml = evidence.length
    ? evidence.map(e => `<div class="evidence-item"><span class="src">${escapeHtml(typeof e === 'string' ? e : e.src || JSON.stringify(e))}</span></div>`).join('')
    : `<div class="evidence-item"><span class="src" style="color:var(--text-faint);">No passages were retrieved for this claim.</span></div>`;

  row.innerHTML = `
    <div class="msg-avatar">${BOT_AVATAR_SVG}</div>
    <div class="verdict-result">
      <div class="vr-head">
        <span class="vr-badge ${badgeClass}">${escapeHtml(label)}</span>
        <button type="button" class="vr-explain">what does this mean?</button>
      </div>
      <div class="vr-body">
        <p class="short">${explanation}</p>
        <p class="why">confidence ${badgeLabelForConfidence(data.confidence)} · mode ${escapeHtml(currentMode)}</p>
        ${data.needs_human_review ? `<div class="vr-note">Flagged for human review — confidence is low or the evidence set is mixed. This verdict is provisional until reviewed.</div>` : ''}
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
    const data = await fetchVerdict(text, currentMode);
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
    <div class="msg-bubble">Bhej dijiye koi health claim — Hinglish, Hindi ya English, jaise bhi aaya ho. Main ise decompose karke, evidence retrieve karke, ek calibrated verdict citations ke saath dunga.</div>
  `;
  chatLog.appendChild(row);
})();
