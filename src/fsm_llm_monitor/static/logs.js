// FSM-LLM Monitor — Logs Page

'use strict';

// --- State ---

var _logPaused = false;
var _logBuffer = [];    // buffered logs when paused
var _logFollowing = true; // auto-scroll active
var _logSearchTimer = null;
var _logErrorCount = 0;
var _logPillCounts = {};  // cumulative per-level counts

// --- Pill Toggles ---

function toggleLogPill(btn) {
    btn.classList.toggle('active');
    btn.setAttribute('aria-pressed', btn.classList.contains('active'));
    refreshLogs();
}

function getActiveLogLevels() {
    var levels = [];
    var pills = document.querySelectorAll('#log-pills .log-pill.active');
    pills.forEach(function(p) { levels.push(p.getAttribute('data-level')); });
    return levels;
}

function getMinLogLevel() {
    var order = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
    var active = getActiveLogLevels();
    for (var i = 0; i < order.length; i++) {
        if (active.indexOf(order[i]) !== -1) return order[i];
    }
    return 'INFO';
}

function _updateLogPillCounts(newLogs, reset) {
    if (reset) _logPillCounts = {};
    for (var i = 0; i < newLogs.length; i++) {
        var lv = newLogs[i].level;
        _logPillCounts[lv] = (_logPillCounts[lv] || 0) + 1;
    }
    document.querySelectorAll('#log-pills .log-pill').forEach(function(pill) {
        var level = pill.getAttribute('data-level');
        var label = level.charAt(0) + level.slice(1).toLowerCase();
        var count = _logPillCounts[level] || 0;
        pill.textContent = count > 0 ? label + ' (' + count + ')' : label;
    });
}

// --- Search ---

function _onLogSearchInput() {
    if (_logSearchTimer) clearTimeout(_logSearchTimer);
    _logSearchTimer = setTimeout(refreshLogs, 300);
}

// --- Log Entry HTML ---

function _logEntryHtml(r, filter) {
    var ts = formatTime(r.timestamp);
    var levelLower = r.level.toLowerCase();
    var levelClass = 'log-' + levelLower;
    var dotHtml = '<span class="log-level-dot ' + levelLower + '"></span>';
    var conv = r.conversation_id ? ' [' + r.conversation_id + ']' : '';
    var msgText = r.module + ':' + r.line + conv + ' ' + r.message;
    var msgHtml = filter ? highlightText(msgText, filter) : esc(msgText);
    var entryClass = (levelLower === 'error' || levelLower === 'critical') ? ' error' : '';
    return '<div class="entry' + entryClass + '"><span class="ts ' + levelClass + '">' + ts + '</span><span class="type log-type-col ' + levelClass + '">' + dotHtml + r.level + '</span><span class="msg text-dim">' + msgHtml + '</span></div>';
}

// --- Auto-scroll ---

function _isNearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
}

function _scrollToBottom(el) {
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
}

function _updateJumpButton() {
    var btn = document.getElementById('log-jump-btn');
    if (btn) {
        if (_logFollowing) {
            btn.classList.remove('visible');
        } else {
            btn.classList.add('visible');
        }
    }
}

function logJumpToLatest() {
    var stream = document.getElementById('log-stream');
    if (stream) {
        _scrollToBottom(stream);
        _logFollowing = true;
        _updateJumpButton();
    }
}

// --- Live / Paused ---

function _updatePauseButton() {
    var btn = document.getElementById('log-pause-btn');
    if (!btn) return;
    if (_logPaused) {
        var count = _logBuffer.length;
        btn.textContent = count > 0 ? 'Resume (' + count + ' pending)' : 'Resume';
        btn.classList.add('paused');
    } else {
        btn.textContent = 'Live';
        btn.classList.remove('paused');
    }
}

function toggleLogPause() {
    _logPaused = !_logPaused;
    _updatePauseButton();
    if (!_logPaused && _logBuffer.length > 0) {
        // Flush buffered logs
        appendLogs(_logBuffer);
        _logBuffer = [];
    }
}

// --- Clear ---

function clearLogs() {
    var stream = document.getElementById('log-stream');
    if (stream) stream.innerHTML = '';
    document.getElementById('log-stats').textContent = '0 entries';
    _logPillCounts = {};
    _updateLogPillCounts([], true);
}

// --- Incremental Append (called from WebSocket) ---

function appendLogs(logs) {
    if (!logs || logs.length === 0) return;
    if (App.currentPage !== 'logs') return;

    if (_logPaused) {
        for (var b = 0; b < logs.length; b++) _logBuffer.push(logs[b]);
        _updatePauseButton();
        return;
    }

    var activeLevels = getActiveLogLevels();
    var filter = document.getElementById('log-filter').value.trim().toLowerCase();
    var stream = document.getElementById('log-stream');
    if (!stream) return;

    // Remove empty-state placeholder if present
    var emptyEl = stream.querySelector('.empty-state');
    if (emptyEl) emptyEl.remove();

    var wasFollowing = _isNearBottom(stream);
    var html = '';
    var added = 0;

    // Logs from WS arrive newest-first; we want to append newest at bottom
    for (var i = logs.length - 1; i >= 0; i--) {
        var r = logs[i];
        if (activeLevels.indexOf(r.level) === -1) continue;
        if (filter && r.message.toLowerCase().indexOf(filter) === -1) continue;
        html += _logEntryHtml(r, filter);
        added++;
    }

    if (html) {
        stream.insertAdjacentHTML('beforeend', html);
        // Cap displayed entries at 1000 — batch removal for performance
        var overflow = stream.children.length - 1000;
        if (overflow > 0) {
            var range = document.createRange();
            range.setStartBefore(stream.firstChild);
            range.setEndAfter(stream.children[overflow - 1]);
            range.deleteContents();
        }
    }

    if (wasFollowing) {
        _scrollToBottom(stream);
        _logFollowing = true;
    }
    _updateJumpButton();

    // Update entry count
    var statsEl = document.getElementById('log-stats');
    if (statsEl) statsEl.textContent = stream.children.length + ' entries';

    // Update pill counts with new data
    _updateLogPillCounts(logs);

    // Track errors for sidebar badge
    for (var j = 0; j < logs.length; j++) {
        if (logs[j].level === 'ERROR' || logs[j].level === 'CRITICAL') _logErrorCount++;
    }
    _updateLogSidebarBadge();
}

// --- Sidebar Error Badge ---

function _updateLogSidebarBadge() {
    var badge = document.getElementById('log-error-badge');
    if (!badge) return;
    if (_logErrorCount > 0) {
        badge.textContent = _logErrorCount > 99 ? '99+' : _logErrorCount;
        badge.style.display = 'inline-flex';
    } else {
        badge.style.display = 'none';
    }
}

function updateLogErrorBadge(metrics) {
    // Called from websocket.js with metrics data
    if (metrics && metrics.total_errors !== undefined) {
        _logErrorCount = metrics.total_errors;
        _updateLogSidebarBadge();
    }
}

// --- Scroll Event Listener ---

// Detect manual scroll to update auto-follow state.
// Deferred to first call of refreshLogs() to ensure the element exists.
var _logScrollListenerAttached = false;

function _attachLogScrollListener() {
    if (_logScrollListenerAttached) return;
    var stream = document.getElementById('log-stream');
    if (!stream) return;
    stream.addEventListener('scroll', function() {
        _logFollowing = _isNearBottom(stream);
        _updateJumpButton();
    });
    _logScrollListenerAttached = true;
}

// --- Full Refresh (on page load / filter change) ---

async function refreshLogs() {
    var activeLevels = getActiveLogLevels();
    var minLevel = getMinLogLevel();
    var filter = document.getElementById('log-filter').value.trim().toLowerCase();

    _attachLogScrollListener();

    // Keep hidden select in sync for settings compat
    var hiddenSelect = document.getElementById('log-level');
    if (hiddenSelect) hiddenSelect.value = minLevel;

    try {
        var logs = await fetchJson('/api/logs?limit=500&level=' + encodeURIComponent(minLevel));

        // Reset pill counts on full refresh and recount
        _updateLogPillCounts(logs, true);

        // Filter by active levels
        logs = logs.filter(function(r) {
            return activeLevels.indexOf(r.level) !== -1;
        });

        if (filter) {
            logs = logs.filter(function(r) { return r.message.toLowerCase().indexOf(filter) !== -1; });
        }
        var stream = document.getElementById('log-stream');
        var logEmpty = document.getElementById('log-empty');
        if (logEmpty) logEmpty.remove();
        logs.reverse();
        var html = '';
        if (logs.length === 0) {
            html = '<div class="empty-state">'
                + '<div class="empty-title">No log entries</div>'
                + '<div class="empty-hint">Logs appear here when FSM conversations, agents, or workflows are active.<br>'
                + 'Launch an instance from the <strong>Dashboard</strong> or use the <strong>Builder</strong> to generate activity.<br>'
                + 'Check that your log level filter includes the levels you expect (currently: ' + esc(activeLevels.join(', ')) + ').</div>'
                + '</div>';
        }
        for (var i = 0; i < logs.length; i++) {
            html += _logEntryHtml(logs[i], filter);
        }
        stream.innerHTML = html;
        document.getElementById('log-stats').textContent = logs.length + ' entries';

        // Scroll to bottom on full refresh (instant, not smooth)
        stream.scrollTop = stream.scrollHeight;
        _logFollowing = true;
        _updateJumpButton();
    } catch (e) {
        console.error('refreshLogs:', e);
    }
}
