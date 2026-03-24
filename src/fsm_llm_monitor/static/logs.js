// FSM-LLM Monitor — Logs Page

'use strict';

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
    // Return the lowest active level for the API call
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

var _logSearchTimer = null;

function _onLogSearchInput() {
    if (_logSearchTimer) clearTimeout(_logSearchTimer);
    _logSearchTimer = setTimeout(refreshLogs, 300);
}

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
            var r = logs[i];
            var ts = formatTime(r.timestamp);
            var levelLower = r.level.toLowerCase();
            var levelClass = 'log-' + levelLower;
            var dotHtml = '<span class="log-level-dot ' + levelLower + '"></span>';
            var conv = r.conversation_id ? ' [' + r.conversation_id + ']' : '';
            var msgText = r.module + ':' + r.line + conv + ' ' + r.message;
            var msgHtml = filter ? highlightText(msgText, filter) : esc(msgText);
            var entryClass = (levelLower === 'error' || levelLower === 'critical') ? ' error' : '';
            html += '<div class="entry' + entryClass + '"><span class="ts ' + levelClass + '">' + ts + '</span><span class="type log-type-col ' + levelClass + '">' + dotHtml + r.level + '</span><span class="msg text-dim">' + msgHtml + '</span></div>';
        }
        stream.innerHTML = html;
        document.getElementById('log-stats').textContent = logs.length + ' entries';
    } catch (e) {
        console.error('refreshLogs:', e);
    }
}
