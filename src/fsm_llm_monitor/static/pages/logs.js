// FSM-LLM Monitor — Logs Page

import { state } from '../services/state.js';
import { fetchJson } from '../services/api.js';
import { $, esc, highlightText } from '../utils/dom.js';
import { formatTime } from '../utils/format.js';

// --- State ---
let _logPaused = false;
let _logBuffer = [];
let _logFollowing = true;
let _logSearchTimer = null;
let _logErrorCount = 0;
let _logPillCounts = {};
let _logScrollListenerAttached = false;

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
        pill.textContent = count > 0 ? label + ' (' + count + ')' : label;
    });
}

// --- Search ---

export function onLogSearchInput() {
    if (_logSearchTimer) clearTimeout(_logSearchTimer);
    _logSearchTimer = setTimeout(refreshLogs, 300);
}

// --- Log Entry HTML ---

function _logEntryHtml(r, filter) {
    const ts = formatTime(r.timestamp);
    const levelLower = r.level.toLowerCase();
    const dotHtml = '<span class="log-level-dot ' + levelLower + '"></span>';
    const conv = r.conversation_id ? ' [' + r.conversation_id + ']' : '';
    const msgText = r.module + ':' + r.line + conv + ' ' + r.message;
    const msgHtml = filter ? highlightText(msgText, filter) : esc(msgText);
    const entryClass = (levelLower === 'error' || levelLower === 'critical') ? ' error' : '';
    return '<div class="entry' + entryClass + '"><span class="ts log-' + levelLower + '">' + ts + '</span><span class="type log-type-col log-' + levelLower + '">' + dotHtml + r.level + '</span><span class="msg text-dim">' + msgHtml + '</span></div>';
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
    if (_logPaused) {
        const count = _logBuffer.length;
        btn.textContent = count > 0 ? 'Resume (' + count + ' pending)' : 'Resume';
        btn.classList.add('paused');
    } else {
        btn.textContent = 'Live';
        btn.classList.remove('paused');
    }
}

export function toggleLogPause() {
    _logPaused = !_logPaused;
    _updatePauseButton();
    if (!_logPaused && _logBuffer.length > 0) {
        appendLogs(_logBuffer);
        _logBuffer = [];
    }
}

// --- Clear ---

export function clearLogs() {
    const stream = $('log-stream');
    if (stream) stream.innerHTML = '';
    $('log-stats').textContent = '0 entries';
    _logPillCounts = {};
    _updateLogPillCounts([], true);
}

// --- Incremental Append (from WebSocket) ---

export function appendLogs(logs) {
    if (!logs?.length) return;
    if (state.currentPage !== 'logs') return;

    if (_logPaused) {
        for (const log of logs) _logBuffer.push(log);
        _updatePauseButton();
        return;
    }

    const activeLevels = getActiveLogLevels();
    const filter = $('log-filter')?.value.trim().toLowerCase();
    const stream = $('log-stream');
    if (!stream) return;

    stream.querySelector('.empty-state')?.remove();

    const wasFollowing = _isNearBottom(stream);
    let html = '';

    for (let i = logs.length - 1; i >= 0; i--) {
        const r = logs[i];
        if (!activeLevels.includes(r.level)) continue;
        if (filter && !r.message.toLowerCase().includes(filter)) continue;
        html += _logEntryHtml(r, filter);
    }

    if (html) {
        stream.insertAdjacentHTML('beforeend', html);
        const overflow = stream.children.length - 1000;
        if (overflow > 0) {
            const range = document.createRange();
            range.setStartBefore(stream.firstChild);
            range.setEndAfter(stream.children[overflow - 1]);
            range.deleteContents();
        }
    }

    if (wasFollowing) {
        _scrollToBottom(stream);
        _logFollowing = true;
    }
    updateJumpButton();

    const statsEl = $('log-stats');
    if (statsEl) statsEl.textContent = stream.children.length + ' entries';

    _updateLogPillCounts(logs);

    for (const log of logs) {
        if (log.level === 'ERROR' || log.level === 'CRITICAL') _logErrorCount++;
    }
    _updateLogSidebarBadge();
}

// --- Sidebar Error Badge ---

function _updateLogSidebarBadge() {
    const badge = $('log-error-badge');
    if (!badge) return;
    if (_logErrorCount > 0) {
        badge.textContent = _logErrorCount > 99 ? '99+' : _logErrorCount;
        badge.style.display = 'inline-flex';
    } else {
        badge.style.display = 'none';
    }
}

export function updateLogErrorBadge(metrics) {
    if (metrics?.total_errors !== undefined) {
        _logErrorCount = metrics.total_errors;
        _updateLogSidebarBadge();
    }
}

// --- Scroll Listener ---

function _attachLogScrollListener() {
    if (_logScrollListenerAttached) return;
    const stream = $('log-stream');
    if (!stream) return;
    stream.addEventListener('scroll', () => {
        _logFollowing = _isNearBottom(stream);
        updateJumpButton();
    });
    _logScrollListenerAttached = true;
}

// --- Full Refresh ---

export async function refreshLogs() {
    const activeLevels = getActiveLogLevels();
    const minLevel = getMinLogLevel();
    const filter = $('log-filter')?.value.trim().toLowerCase();

    _attachLogScrollListener();

    const hiddenSelect = $('log-level');
    if (hiddenSelect) hiddenSelect.value = minLevel;

    try {
        let logs = await fetchJson('/api/logs?limit=500&level=' + encodeURIComponent(minLevel));

        _updateLogPillCounts(logs, true);
        logs = logs.filter(r => activeLevels.includes(r.level));
        if (filter) logs = logs.filter(r => r.message.toLowerCase().includes(filter));

        const stream = $('log-stream');
        $('log-empty')?.remove();
        logs.reverse();

        let html = '';
        if (logs.length === 0) {
            html = '<div class="empty-state">'
                + '<div class="empty-title">No log entries</div>'
                + '<div class="empty-hint">Logs appear here when FSM conversations, agents, or workflows are active.<br>'
                + 'Launch an instance from the <strong>Dashboard</strong> or use the <strong>Builder</strong> to generate activity.<br>'
                + 'Check that your log level filter includes the levels you expect (currently: ' + esc(activeLevels.join(', ')) + ').</div>'
                + '</div>';
        }
        for (const log of logs) html += _logEntryHtml(log, filter);
        stream.innerHTML = html;
        $('log-stats').textContent = logs.length + ' entries';

        stream.scrollTop = stream.scrollHeight;
        _logFollowing = true;
        updateJumpButton();
    } catch (e) {
        console.error('refreshLogs:', e);
    }
}
