// FSM-LLM Monitor — Logs Page

import { state, scheduleRefresh } from '../services/state.js';
import { fetchJson } from '../services/api.js';
import { $, esc, highlightText, showToast, levelClass } from '../utils/dom.js';
import { formatTime } from '../utils/format.js';

// --- State ---
let _logPaused = false;
let _logBuffer = [];
let _logFollowing = true;
let _logErrorCount = 0;
let _logPillCounts = {};

// De-duplication of records across WS pushes, periodic syncs and full
// refreshes. Key = timestamp + level + source line + message; bounded FIFO.
const _SEEN_CAP = 10000;
let _seenKeys = new Set();
let _seenOrder = [];

function _logKey(r) {
    return `${r.timestamp}|${r.level}|${r.module}:${r.line}|${r.message}`;
}

function _markSeen(key) {
    _seenKeys.add(key);
    _seenOrder.push(key);
    if (_seenOrder.length > _SEEN_CAP) {
        const drop = _seenOrder.splice(0, _seenOrder.length - _SEEN_CAP);
        for (const k of drop) _seenKeys.delete(k);
    }
}

/** Keep only records not seen before, marking them seen. */
function _takeUnseen(logs) {
    const fresh = [];
    for (const r of logs) {
        const k = _logKey(r);
        if (_seenKeys.has(k)) continue;
        _markSeen(k);
        fresh.push(r);
    }
    return fresh;
}


export function isLogPaused() {
    return _logPaused;
}

// --- Pill Toggles ---

export function toggleLogPill(btn) {
    btn.classList.toggle('active');
    btn.setAttribute('aria-pressed', btn.classList.contains('active'));
    refreshLogs();
}

function getActiveLogLevels() {
    const levels = [];
    document.querySelectorAll('#log-pills .log-pill.active').forEach(p => {
        levels.push(p.getAttribute('data-level'));
    });
    return levels;
}

function getMinLogLevel() {
    const order = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
    const active = getActiveLogLevels();
    for (const level of order) {
        if (active.includes(level)) return level;
    }
    return 'INFO';
}

function _updateLogPillCounts(newLogs, reset) {
    if (reset) _logPillCounts = {};
    for (const r of newLogs) {
        _logPillCounts[r.level] = (_logPillCounts[r.level] || 0) + 1;
    }
    document.querySelectorAll('#log-pills .log-pill').forEach(pill => {
        const level = pill.getAttribute('data-level');
        const label = level.charAt(0) + level.slice(1).toLowerCase();
        const count = _logPillCounts[level] || 0;
        pill.textContent = count > 0 ? `${label} (${count})` : label;
    });
}

// --- Search ---

export function onLogSearchInput() {
    scheduleRefresh('log-search', refreshLogs, 300);
}

// --- Log Entry HTML ---

function _logEntryHtml(r, filter) {
    const ts = formatTime(r.timestamp);
    const levelLower = levelClass(r.level);
    const dotHtml = `<span class="log-level-dot ${levelLower}"></span>`;
    const conv = r.conversation_id ? ` [${r.conversation_id}]` : '';
    const msgText = `${r.module}:${r.line}${conv} ${r.message}`;
    const msgHtml = filter ? highlightText(msgText, filter) : esc(msgText);
    const entryClass = (levelLower === 'error' || levelLower === 'critical') ? ' error' : '';
    return `<div class="entry${entryClass}"><span class="ts log-${levelLower}">${esc(ts)}</span><span class="type log-type-col log-${levelLower}">${dotHtml}${esc(r.level)}</span><span class="msg text-dim">${msgHtml}</span></div>`;
}

// --- Auto-scroll ---

function _isNearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
}

function _scrollToBottom(el) {
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
}

export function updateJumpButton() {
    const btn = $('log-jump-btn');
    if (btn) btn.classList.toggle('visible', !_logFollowing);
}

export function logJumpToLatest() {
    const stream = $('log-stream');
    if (stream) {
        _scrollToBottom(stream);
        _logFollowing = true;
        updateJumpButton();
    }
}

// Expose for onscroll inline (will be called from delegated handler)
export function onLogScroll() {
    const stream = $('log-stream');
    if (stream) {
        _logFollowing = _isNearBottom(stream);
        updateJumpButton();
    }
}

// --- Live / Paused ---

function _updatePauseButton() {
    const btn = $('log-pause-btn');
    if (!btn) return;
    btn.setAttribute('aria-pressed', _logPaused ? 'true' : 'false');
    if (_logPaused) {
        const count = _logBuffer.length;
        btn.textContent = count > 0 ? `Resume (${count} pending)` : 'Resume';
        btn.classList.add('paused');
    } else {
        btn.textContent = 'Live';
        btn.classList.remove('paused');
    }
}

export function toggleLogPause() {
    _logPaused = !_logPaused;
    const pending = _logBuffer;
    _logBuffer = [];
    _updatePauseButton();
    if (!_logPaused && pending.length > 0 && state.currentPage === 'logs') {
        _renderLogs(pending);
    }
}

// --- Clear ---

export function clearLogs() {
    const stream = $('log-stream');
    if (stream) stream.innerHTML = '';
    const statsEl = $('log-stats');
    if (statsEl) statsEl.textContent = '0 entries';
    _logBuffer = [];
    _updatePauseButton();
    _logPillCounts = {};
    _updateLogPillCounts([], true);
    // Cleared records stay marked as seen so the periodic sync does not
    // bring them back; the error badge restarts from zero.
    _logErrorCount = 0;
    _updateLogSidebarBadge();
}

// --- Incremental Append (from WebSocket / periodic sync) ---

// Cap on the paused-mode buffer to prevent unbounded growth during a long
// paused high-volume run (frontend finding F-06).
const _LOG_BUFFER_CAP = 5000;

/** Render already de-duplicated records (newest first) into the stream. */
function _renderLogs(logs) {
    const stream = $('log-stream');
    if (!stream || !logs.length) return;

    const activeLevels = getActiveLogLevels();
    const filter = $('log-filter')?.value.trim().toLowerCase();

    stream.querySelector('.empty-state')?.remove();

    const wasFollowing = _isNearBottom(stream);
    let html = '';

    for (let i = logs.length - 1; i >= 0; i--) {
        const r = logs[i];
        if (!activeLevels.includes(r.level)) continue;
        if (filter && !String(r.message ?? '').toLowerCase().includes(filter)) continue;
        html += _logEntryHtml(r, filter);
    }

    if (html) {
        stream.insertAdjacentHTML('beforeend', html);
        while (stream.children.length > 1000) {
            stream.removeChild(stream.firstChild);
        }
    }

    if (wasFollowing && state.autoScrollLogs !== false) {
        _scrollToBottom(stream);
        _logFollowing = true;
    } else {
        _logFollowing = _isNearBottom(stream);
    }
    updateJumpButton();

    const statsEl = $('log-stats');
    if (statsEl) statsEl.textContent = `${stream.children.length} entries`;
}

export function appendLogs(logs) {
    if (!logs?.length) return;
    const fresh = _takeUnseen(logs);
    if (!fresh.length) return;

    // Counters track records as received, once each, whatever the view state.
    _updateLogPillCounts(fresh);
    for (const log of fresh) {
        if (log.level === 'ERROR' || log.level === 'CRITICAL') _logErrorCount++;
    }
    _updateLogSidebarBadge();

    if (_logPaused) {
        for (const log of fresh) _logBuffer.push(log);
        if (_logBuffer.length > _LOG_BUFFER_CAP) {
            _logBuffer.splice(0, _logBuffer.length - _LOG_BUFFER_CAP);
        }
        _updatePauseButton();
        return;
    }

    // Not visible: the Logs page rebuilds from the server's bounded log deque
    // on navigation (finding F-02).
    if (state.currentPage !== 'logs') return;

    _renderLogs(fresh);
}

// --- Sidebar Error Badge ---
// One meaning only: the number of ERROR/CRITICAL log lines this page has
// received (not the server's error-event metric).

function _updateLogSidebarBadge() {
    const badge = $('log-error-badge');
    if (!badge) return;
    if (_logErrorCount > 0) {
        badge.textContent = _logErrorCount > 99 ? '99+' : String(_logErrorCount);
        const label = `${_logErrorCount} error or critical log line${_logErrorCount === 1 ? '' : 's'} received`;
        badge.title = label;
        badge.setAttribute('aria-label', label);
        badge.style.display = 'inline-flex';
    } else {
        badge.style.display = 'none';
    }
}

// --- Full Refresh ---

/** Page-show hook: rebuild the stream unless the user paused it. */
export function onShowLogs() {
    const stream = $('log-stream');
    if (_logPaused && stream && stream.children.length > 0) return;
    refreshLogs();
}

/**
 * Periodic catch-up: fetch the latest records and append only unseen ones.
 * Never wipes the stream; while paused, new records go to the pause buffer.
 */
export async function syncLogs() {
    try {
        const logs = await fetchJson(`/api/logs?limit=500&level=${encodeURIComponent(getMinLogLevel())}`);
        appendLogs(logs);
    } catch (e) {
        console.error('syncLogs:', e);
    }
}

export async function refreshLogs() {
    const activeLevels = getActiveLogLevels();
    const minLevel = getMinLogLevel();
    const filter = $('log-filter')?.value.trim().toLowerCase();

    try {
        let logs = await fetchJson(`/api/logs?limit=500&level=${encodeURIComponent(minLevel)}`);

        // The rebuilt view reflects the server's latest state, so anything held
        // in the pause buffer is either shown now or already evicted server side.
        // Records not seen before still count toward the error badge once.
        for (const r of _takeUnseen(logs)) {
            if (r.level === 'ERROR' || r.level === 'CRITICAL') _logErrorCount++;
        }
        _updateLogSidebarBadge();
        _logBuffer = [];
        _updatePauseButton();

        _updateLogPillCounts(logs, true);
        logs = logs.filter(r => activeLevels.includes(r.level));
        if (filter) logs = logs.filter(r => String(r.message ?? '').toLowerCase().includes(filter));

        const stream = $('log-stream');
        if (!stream) return;
        $('log-empty')?.remove();
        logs.reverse();

        let html = '';
        if (logs.length === 0) {
            html = `<div class="empty-state">`
                + `<div class="empty-title">No log entries</div>`
                + `<div class="empty-hint">Logs appear here when FSM conversations, agents, or workflows are active.<br>`
                + `Launch an instance from the <strong>Dashboard</strong> or use the <strong>Builder</strong> to generate activity.<br>`
                + `Check that your log level filter includes the levels you expect (currently: ${esc(activeLevels.join(', '))}).</div>`
                + `</div>`;
        }
        for (const log of logs) html += _logEntryHtml(log, filter);
        const prevTop = stream.scrollTop;
        stream.innerHTML = html;
        const statsEl = $('log-stats');
        if (statsEl) statsEl.textContent = `${logs.length} entries`;

        if (state.autoScrollLogs !== false) {
            stream.scrollTop = stream.scrollHeight;
            _logFollowing = true;
        } else {
            stream.scrollTop = prevTop;
            _logFollowing = _isNearBottom(stream);
        }
        updateJumpButton();
    } catch (e) {
        console.error('refreshLogs:', e);
        showToast(`Failed to load logs: ${e.message}`, 'error');
    }
}
