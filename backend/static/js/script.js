// ── Examples ──────────────────────────────────────────────────────────────
const examples = {
  spam: "URGENT! You've won $5000! Click here NOW to claim your prize before it expires: http://fakescam.win/claim",
  ham:  "Hey, are we still on for lunch tomorrow at 1pm? Let me know if anything changes!"
};

// ── Character counter ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadStats();

  const input = document.getElementById('sms-input');
  const hint  = document.getElementById('char-hint');

  input.addEventListener('input', () => {
    hint.textContent = `${input.value.length} character${input.value.length !== 1 ? 's' : ''}`;
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      verifyMessage();
    }
  });
});

// ── Clear ──────────────────────────────────────────────────────────────────
function clearAll() {
  document.getElementById('sms-input').value = '';
  document.getElementById('char-hint').textContent = '0 characters';
  document.getElementById('results').style.display  = 'none';
  document.getElementById('loading').style.display  = 'none';
}

// ── Try example ────────────────────────────────────────────────────────────
function tryExample(type) {
  const input = document.getElementById('sms-input');
  input.value = examples[type];
  document.getElementById('char-hint').textContent =
    `${input.value.length} characters`;
  setTimeout(verifyMessage, 250);
}

// ── Main verify ────────────────────────────────────────────────────────────
async function verifyMessage() {
  const message = document.getElementById('sms-input').value.trim();
  if (!message) {
    shakeInput();
    return;
  }

  document.getElementById('loading').style.display = 'flex';
  document.getElementById('results').style.display  = 'none';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    const data = await res.json();

    document.getElementById('loading').style.display = 'none';

    if (data.success) {
      displayResults(data);
    } else {
      showError(data.error || 'Unknown error');
    }
  } catch (err) {
    document.getElementById('loading').style.display = 'none';
    showError(err.message);
  }
}

// ── Display results ────────────────────────────────────────────────────────
function displayResults(data) {
  const wrap   = document.getElementById('results');
  const header = document.getElementById('result-header');

  // Verdict
  const icon  = document.getElementById('verdict-icon');
  const label = document.getElementById('verdict-label');
  const sub   = document.getElementById('verdict-sub');

  if (data.is_spam) {
    icon.textContent  = '🚨';
    label.textContent = 'SPAM DETECTED';
    label.style.color = 'var(--red)';
    sub.textContent   = 'This message shows signs of spam';
    header.className  = 'result-header is-spam';
  } else {
    icon.textContent  = '✅';
    label.textContent = 'SAFE MESSAGE';
    label.style.color = 'var(--green)';
    sub.textContent   = 'This message appears legitimate';
    header.className  = 'result-header is-ham';
  }

  // Confidence bars
  const hamPct  = data.confidence.ham;
  const spamPct = data.confidence.spam;

  document.getElementById('ham-percent').textContent  = hamPct  + '%';
  document.getElementById('spam-percent').textContent = spamPct + '%';

  // Animate bars after a tiny tick
  requestAnimationFrame(() => {
    document.getElementById('ham-bar').style.width  = hamPct  + '%';
    document.getElementById('spam-bar').style.width = spamPct + '%';
  });

  // Analysed message
  document.getElementById('analyzed-message').textContent = data.message;

  // Keywords
  const analysisBox = document.getElementById('analysis-box');
  const pillsEl     = document.getElementById('important-words');

  if (data.analysis && data.analysis.important_words.length > 0) {
    pillsEl.innerHTML = data.analysis.important_words
      .map(w => `<span class="kw-pill">${w}</span>`)
      .join('');
    analysisBox.style.display = 'block';
  } else {
    analysisBox.style.display = 'none';
  }

  // Reveal
  wrap.style.display = 'flex';
  wrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Stats ──────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res  = await fetch('/stats');
    const data = await res.json();
    if (data.success) {
      document.getElementById('best-model').textContent    = data.best_model;
      document.getElementById('best-accuracy').textContent = data.best_accuracy + '%';
      document.getElementById('total-messages').textContent = data.dataset_info.total_messages.toLocaleString();
    }
  } catch (e) {
    console.error('Stats error:', e);
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function shakeInput() {
  const el = document.getElementById('sms-input');
  el.style.animation = 'none';
  el.offsetHeight; // reflow
  el.style.animation = 'shake 0.4s ease';
  el.focus();
}

function showError(msg) {
  console.error(msg);
  // You could render a toast here; for now a simple alert
  alert('Error: ' + msg);
}