// FSM-LLM Monitor — Settings Page

'use strict';

async function loadSettings() {
    try {
        var resp = await fetch('/api/config');
        var cfg = await resp.json();
        document.getElementById('set-refresh').value = cfg.refresh_interval;
        document.getElementById('set-max-events').value = cfg.max_events;
        document.getElementById('set-max-logs').value = cfg.max_log_lines;
        document.getElementById('set-level').value = cfg.log_level;
    } catch (e) {
        console.error('loadSettings config:', e);
    }
    try {
        var resp = await fetch('/api/info');
        var info = await resp.json();
        var el = document.getElementById('sys-info');
        var html = '';
        for (var k in info) {
            html += '<span class="key">' + esc(k.replace(/_/g, ' ')) + ':</span><span class="val">' + esc(info[k]) + '</span>';
        }
        el.innerHTML = html;
        document.getElementById('version-info').textContent = 'v' + info.monitor_version;
    } catch (e) {
        console.error('loadSettings info:', e);
    }
}

async function saveSettings() {
    try {
        await fetch('/api/config', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                refresh_interval: numVal('set-refresh', 1.0),
                max_events: intVal('set-max-events', 1000),
                max_log_lines: intVal('set-max-logs', 5000),
                log_level: document.getElementById('set-level').value,
                show_internal_keys: false,
                auto_scroll_logs: true,
            }),
        });
        showStatus('conn-status', 'Settings saved', 'success');
    } catch (e) {
        showError('conn-status', 'Save failed');
        console.error('saveSettings:', e);
    }
}

function resetSettings() {
    document.getElementById('set-refresh').value = '1.0';
    document.getElementById('set-max-events').value = '1000';
    document.getElementById('set-max-logs').value = '5000';
    document.getElementById('set-level').value = 'INFO';
}
