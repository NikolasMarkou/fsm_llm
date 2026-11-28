// FSM-LLM Monitor — DOM Utilities
// Escaping, toast, status messages, and helpers.
import { JSON_VIEW_CLASS } from './json_highlight.js';

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

/** Markup for JSON text in a `tag` (the text escaped); `highlightJsonIn(root)` colours it after `root.innerHTML = html`. */
export function jsonViewHtml(text, tag = 'span') {
    return `<${tag} class="${JSON_VIEW_CLASS}">${esc(text)}</${tag}>`;
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

/** Status text of a launch that made the instance but failed to start its first conversation (never "Launch failed"). */
export function launchedNotStartedText(data, e) {
    return `Launched: ${data.label || data.instance_id}, but starting the first conversation failed: ${e.message}`;
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
function safeClass(value, fallback = 'unknown') {
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
const _open = [];  // the dialogs and drawers opened with openDialog, the top one last

function _focusables(el) {
    const cands = el.querySelectorAll('button, summary, [href], input:not([type="hidden"]), select, textarea, [tabindex]:not([tabindex="-1"])');
    return [...cands].filter(c => !c.disabled && c.getClientRects().length > 0);
}

function _firstFocusable(el) {
    return _focusables(el)[0] ?? null;
}

// DECISION plan-2026-11-27T201735-9283c1fd/D-007: Tab and Shift+Tab stay inside the TOP open dialog (O4).
// One module stack and one document keydown listener that exists only while the stack is not empty; closeDialog
// is the one place that pops. Do NOT trap with `inert` on the page behind alone (no browser proof, older engines
// lack it), do NOT leave the listener after the last close, do NOT let a closed dialog trap (pop it in closeDialog
// even when it is already hidden), do NOT trap the bottom dialog of a stack (the API key modal opens over the
// launch modal on a 401), do NOT preventDefault when the browser's own Tab move stays inside, do NOT decide "outside" by
// membership in the focusable list (a focusable the selector misses, inside the dialog, is not outside: Tab moves on by itself).
function _trap(e) {
    if (e.key !== 'Tab') return;
    const top = _open[_open.length - 1];
    const list = _focusables(top);
    if (!list.length) { e.preventDefault(); top.setAttribute('tabindex', '-1'); top.focus(); return; }
    const outside = !top.contains(document.activeElement);
    const at = list.indexOf(document.activeElement);
    const to = e.shiftKey ? (outside || at === 0 ? list[list.length - 1] : null) : (outside || at === list.length - 1 ? list[0] : null);
    if (to) { e.preventDefault(); to.focus(); }
}

function _pushOpen(el) {
    const at = _open.indexOf(el);
    if (at >= 0) _open.splice(at, 1);
    else if (!_open.length) document.addEventListener('keydown', _trap);
    _open.push(el);
}

function _popOpen(el) {
    const at = _open.indexOf(el);
    if (at < 0) return;
    _open.splice(at, 1);
    if (!_open.length) document.removeEventListener('keydown', _trap);
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
    _pushOpen(el);
    const active = document.activeElement;
    _returnFocus.set(el, active && active !== document.body ? active : null);
    const target = focusTarget || _firstFocusable(el);
    if (target && typeof target.focus === 'function') target.focus();
}

/** Hide a dialog opened with openDialog() and restore the previous focus. */
export function closeDialog(el) {
    if (!el) return;
    _popOpen(el);
    if (_isHidden(el)) return;
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
            ? jsonViewHtml(JSON.stringify(v, null, 2), 'pre')
            : esc(String(v));
        html += `<div class="llm-kv"><span class="llm-key">${esc(k)}:</span> ${display}</div>`;
    }
    return html || '<span class="text-dim">Empty</span>';
}

/** Highlight search query matches within text (matched on the raw text, so a
 *  query like "amp" never lands inside an escaped "&amp;"). */
export function highlightText(text, query) {
    if (!query) return esc(text);
    const re = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    // split() with a capture group puts the matches at the odd indexes.
    return String(text ?? '').split(re)
        .map((part, i) => (i % 2 ? `<span class="search-highlight">${esc(part)}</span>` : esc(part)))
        .join('');
}

/** Prev/next pager in *el*; the buttons carry data-action `<action>-prev` and `-next`. */
export function renderPagination(el, page, totalPages, action) {
    if (!el) return;
    if (totalPages <= 1) {
        el.style.display = 'none';
        return;
    }
    el.style.display = 'flex';
    el.innerHTML =
        `<button class="btn btn-sm" data-action="${action}-prev"${page === 0 ? ' disabled' : ''}>&laquo; Prev</button>` +
        `<span class="pagination-info">Page ${page + 1} of ${totalPages}</span>` +
        `<button class="btn btn-sm" data-action="${action}-next"${page >= totalPages - 1 ? ' disabled' : ''}>Next &raquo;</button>`;
}

/** Show the instance detail in the Control drawer (hide the conversation sub-view). */
export function showDrawerDetailView() {
    const backBtn = $('ctrl-drawer-back');
    const drawerContent = $('ctrl-drawer-content');
    const convWrapper = $('conv-detail-wrapper');
    const eventsWrapper = $('ctrl-drawer-events-wrapper');
    if (backBtn) backBtn.style.display = 'none';
    if (drawerContent) drawerContent.style.display = 'block';
    if (convWrapper) convWrapper.style.display = 'none';
    if (eventsWrapper) eventsWrapper.style.display = 'block';
}

/** Quick hash for change detection on instance lists. */
export function hashInstances(items) {
    let h = items.length;
    for (const item of items) {
        const s = `${item.instance_id}:${item.status}:${item.conversation_count}:${item.active_workflows}`;
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

/**
 * Fill a select from agent type rows in groups `[{label, rows}]` (a falsy label means no optgroup).
 * Options are built with createElement and textContent only, never markup from server data (pinned by
 * test_agent_selects.py::test_the_scripts_set_option_text_with_textcontent_only: do not use innerHTML here). Keeps the
 * current value when it is still offered, else `preferred`, else the first option; no rows at all
 * leaves one disabled option reading `emptyText`.
 */
export function fillSelect(sel, groups, emptyText, preferred) {
    const nodes = [];
    const ids = [];
    for (const { label, rows } of groups) {
        if (!rows.length) continue;
        const group = label ? document.createElement('optgroup') : null;
        if (group) group.label = label;
        for (const row of rows) {
            const opt = document.createElement('option');
            opt.value = row.id;
            opt.textContent = row.name || row.id;
            ids.push(row.id);
            if (group) group.appendChild(opt); else nodes.push(opt);
        }
        if (group) nodes.push(group);
    }
    if (!nodes.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.disabled = true;
        opt.textContent = emptyText;
        nodes.push(opt);
    }
    const keep = [sel.value, preferred].find(v => ids.includes(v));
    sel.replaceChildren(...nodes);
    if (keep) sel.value = keep;
    else if (!ids.length) sel.value = '';
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
