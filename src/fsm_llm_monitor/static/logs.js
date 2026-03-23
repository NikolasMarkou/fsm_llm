// FSM-LLM Monitor — Logs Page

'use strict';

async function refreshLogs() {
    var level = document.getElementById('log-level').value;
    var filter = document.getElementById('log-filter').value.trim().toLowerCase();
    try {
        var resp = await fetch('/api/logs?limit=500&level=' + encodeURIComponent(level));
        var logs = await resp.json();
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
            var levelClass = 'log-' + r.level.toLowerCase();
            var conv = r.conversation_id ? ' [' + r.conversation_id + ']' : '';
            html += '<div class="entry"><span class="ts ' + levelClass + '">' + ts + '</span><span class="type log-type-col ' + levelClass + '">' + r.level + '</span><span class="msg text-dim">' + esc(r.module) + ':' + r.line + conv + ' ' + esc(r.message) + '</span></div>';
        }
        stream.innerHTML = html;
        document.getElementById('log-stats').textContent = 'Total: ' + logs.length + ' | Level: >= ' + level;
    } catch (e) {
        console.error('refreshLogs:', e);
    }
}
