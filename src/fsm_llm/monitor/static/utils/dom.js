// FSM-LLM Monitor — DOM Utilities
// Escaping, toast, status messages, and helpers.

/** HTML-entity escape for safe insertion. */
export function esc(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** Shorthand for getElementById. */
export function $(id) {
    return document.getElementById(id);
}

export function showError(elementId, msg) {
    const el = $(elementId);
    if (el) el.innerHTML = `<span class="error-message">${esc(msg)}</span>`;
}

export function showStatus(elementId, msg, color) {
    const el = $(elementId);
    if (el) el.innerHTML = msg ? `<span class="status-msg status-${color}">${esc(msg)}</span>` : '';
}

export function showToast(msg, type) {
    document.querySelector('.toast')?.remove();
    const toast = document.createElement('div');
    toast.className = `toast toast-${type === 'success' ? 'success' : 'error'}`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('toast-visible'));
    setTimeout(() => {
        toast.classList.remove('toast-visible');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
    // Mirror the text into the persistent polite live region (templates/index.html)
    // so screen readers announce it; a freshly inserted node is often not read.
    const live = $('toast-live');
    if (live) {
        live.textContent = '';
        setTimeout(() => { live.textContent = String(msg ?? ''); }, 50);
    }
}

/** Reduce a server-provided enum-like value to a safe CSS class token. */
export function safeClass(value, fallback = 'unknown') {
    const s = String(value ?? '').toLowerCase().replace(/[^a-z0-9_-]/g, '');
    return s || fallback;
}

const _LOG_LEVELS = ['trace', 'debug', 'info', 'success', 'warning', 'error', 'critical'];

/** Whitelisted lowercase log/event level for use in class attributes. */
export function levelClass(level) {
    const l = String(level ?? '').toLowerCase();
    return _LOG_LEVELS.includes(l) ? l : 'info';
}

export function statusBadge(status) {
    const s = status == null || status === '' ? 'unknown' : String(status);
    return `<span class="badge badge-${safeClass(s)}">${esc(s.toUpperCase())}</span>`;
}

// --- Dialog focus management ---

const _returnFocus = new Map();

function _firstFocusable(el) {
    const cands = el.querySelectorAll('button, [href], input:not([type="hidden"]), select, textarea, [tabindex]:not([tabindex="-1"])');
    for (const c of cands) {
        if (!c.disabled && c.getClientRects().length > 0) return c;
    }
    return null;
}

function _isHidden(el) {
    return el.style.display === 'none' || getComputedStyle(el).display === 'none';
}

/**
 * Show a modal/drawer element and move focus into it. Remembers the element
 * that had focus so closeDialog() can restore it. No-op on focus when the
 * element is already visible.
 */
export function openDialog(el, display = 'flex', focusTarget = null) {
    if (!el) return;
    const wasHidden = _isHidden(el);
    el.style.display = display;
    if (!wasHidden) return;
    const active = document.activeElement;
    _returnFocus.set(el, active && active !== document.body ? active : null);
    const target = focusTarget || _firstFocusable(el);
    if (target && typeof target.focus === 'function') target.focus();
}

/** Hide a dialog opened with openDialog() and restore the previous focus. */
export function closeDialog(el) {
    if (!el || _isHidden(el)) return;
    el.style.display = 'none';
    const prev = _returnFocus.get(el);
    _returnFocus.delete(el);
    if (prev && document.contains(prev) && typeof prev.focus === 'function') prev.focus();
}

export function renderResultBanner(success) {
    const cls = success ? 'success' : 'failure';
    const text = success ? 'Agent completed successfully' : 'Agent failed';
    return `<div class="result-banner ${cls}">${text}</div>`;
}

/** Render an object as key-value HTML pairs. */
export function renderLLMData(obj) {
    if (!obj || typeof obj !== 'object') return '<span class="text-dim">No data</span>';
    let html = '';
    for (const k of Object.keys(obj)) {
        const v = obj[k];
        if (v == null) continue;
        const display = typeof v === 'object'
            ? `<pre>${esc(JSON.stringify(v, null, 2))}</pre>`
            : esc(String(v));
        html += `<div class="llm-kv"><span class="llm-key">${esc(k)}:</span> ${display}</div>`;
    }
    return html || '<span class="text-dim">Empty</span>';
}

/** Highlight search query matches within text. */
export function highlightText(text, query) {
    if (!query) return esc(text);
    const escaped = esc(text);
    const escapedQuery = esc(query);
    const re = new RegExp(`(${escapedQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return escaped.replace(re, '<span class="search-highlight">$1</span>');
}

/** Quick hash for change detection on instance lists. */
export function hashInstances(items) {
    let h = items.length;
    for (const item of items) {
        const s = `${item.instance_id}:${item.status}`;
        for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    }
    return h;
}

/** Parse numeric value from form input. */
export function numVal(id, fallback) {
    const v = parseFloat($(id)?.value);
    return Number.isFinite(v) ? v : fallback;
}

/** Parse integer value from form input. */
export function intVal(id, fallback) {
    const v = parseInt($(id)?.value, 10);
    return Number.isFinite(v) ? v : fallback;
}

/** Copy text to clipboard with fallback. */
export async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.cssText = 'position:fixed;opacity:0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        return true;
    }
}
