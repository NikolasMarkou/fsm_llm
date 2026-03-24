// FSM-LLM Monitor — Logs Page

'use strict';

// --- State ---

var _logPaused = false;
var _logBuffer = [];    // buffered logs when paused
var _logFollowing = true; // auto-scroll active
var _logSearchTimer = null;
var _logErrorCount = 0;

// --- Pill Toggles ---

function toggleLogPill(btn) {
    btn.classList.toggle('active');
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

function _updateLogPillCounts(allLogs) {
    var counts = {};
    for (var i = 0; i < allLogs.length; i++) {
        var lv = allLogs[i].level;
        counts[lv] = (counts[lv] || 0) + 1;
    }
    document.querySelectorAll('#log-pills .log-pill').forEach(function(pill) {
        var level = pill.getAttribute('data-level');
        var label = level.charAt(0) + level.slice(1).toLowerCase();
        var count = counts[level] || 0;
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
    el.scrollTop = el.scrollHeight;
}

function _updateJumpButton() {
    var btn = document.getElementById('log-jump-btn');
    if (btn) btn.style.display = _logFollowing ? 'none' : 'flex';
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

function toggleLogPause() {
    _logPaused = !_logPaused;
    var btn = document.getElementById('log-pause-btn');
    if (btn) {
        btn.textContent = _logPaused ? 'Resume' : 'Live';
        btn.classList.toggle('paused', _logPaused);
    }
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
}

// --- Incremental Append (called from WebSocket) ---

function appendLogs(logs) {
    if (!logs || logs.length === 0) return;
    if (App.currentPage !== 'logs') return;

    if (_logPaused) {
        for (var b = 0; b < logs.length; b++) _logBuffer.push(logs[b]);
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
        // Cap displayed entries at 1000
        while (stream.children.length > 1000) stream.removeChild(stream.firstChild);
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

// --- Full Refresh (on page load / filter change) ---

async function refreshLogs() {
    var activeLevels = getActiveLogLevels();
    var minLevel = getMinLogLevel();
    var filter = document.getElementById('log-filter').value.trim().toLowerCase();

    // Keep hidden select in sync for settings compat
    var hiddenSelect = document.getElementById('log-level');
    if (hiddenSelect) hiddenSelect.value = minLevel;

    try {
        var resp = await fetch('/api/logs?limit=500&level=' + encodeURIComponent(minLevel));
        var logs = await resp.json();

        // Update pill counts before level filtering
        _updateLogPillCounts(logs);

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
            html = '<div class="empty-state"><div class="empty-hint">No log entries matching filter</div></div>';
        }
        for (var i = 0; i < logs.length; i++) {
            html += _logEntryHtml(logs[i], filter);
        }
        stream.innerHTML = html;
        document.getElementById('log-stats').textContent = logs.length + ' entries';

        // Scroll to bottom on full refresh
        _scrollToBottom(stream);
        _logFollowing = true;
        _updateJumpButton();
    } catch (e) {
        console.error('refreshLogs:', e);
    }
}
