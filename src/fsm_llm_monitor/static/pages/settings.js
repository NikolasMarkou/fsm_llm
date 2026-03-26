// FSM-LLM Monitor — Settings Page

import { fetchJson } from '../services/api.js';
import { $, esc, numVal, intVal, showError, showStatus } from '../utils/dom.js';

export async function loadSettings() {
    try {
        const cfg = await fetchJson('/api/config');
        $('set-refresh').value = cfg.refresh_interval;
        $('set-max-events').value = cfg.max_events;
        $('set-max-logs').value = cfg.max_log_lines;
        $('set-level').value = cfg.log_level;
        const internalKeysEl = $('set-internal-keys');
        if (internalKeysEl) internalKeysEl.checked = cfg.show_internal_keys || false;
        const autoScrollEl = $('set-auto-scroll');
        if (autoScrollEl) autoScrollEl.checked = cfg.auto_scroll_logs !== false;
    } catch (e) {
        console.error('loadSettings config:', e);
    }
    try {
        const info = await fetchJson('/api/info');
        const el = $('sys-info');
        let html = '';
        for (const k in info) {
            html += '<span class="key">' + esc(k.replace(/_/g, ' ')) + ':</span><span class="val">' + esc(info[k]) + '</span>';
        }
        el.innerHTML = html;
        $('version-info').textContent = 'v' + info.monitor_version;
        const footerEl = $('footer-version');
        if (footerEl) footerEl.textContent = 'FSM-LLM Monitor v' + info.monitor_version;
    } catch (e) {
        console.error('loadSettings info:', e);
    }
}

export async function saveSettings() {
    try {
        await fetchJson('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                refresh_interval: numVal('set-refresh', 1.0),
                max_events: intVal('set-max-events', 1000),
                max_log_lines: intVal('set-max-logs', 5000),
                log_level: $('set-level').value,
                show_internal_keys: $('set-internal-keys')?.checked ?? false,
                auto_scroll_logs: $('set-auto-scroll')?.checked ?? true,
            }),
        });
        showStatus('conn-status', 'Settings saved', 'success');
        const panel = $('settings-panel');
        if (panel) {
            panel.classList.add('animate-save');
            setTimeout(() => panel.classList.remove('animate-save'), 600);
        }
    } catch (e) {
        showError('conn-status', 'Save failed');
        console.error('saveSettings:', e);
    }
}

export function resetSettings() {
    $('set-refresh').value = '1.0';
    $('set-max-events').value = '1000';
    $('set-max-logs').value = '5000';
    $('set-level').value = 'INFO';
    const internalKeysEl = $('set-internal-keys');
    if (internalKeysEl) internalKeysEl.checked = false;
    const autoScrollEl = $('set-auto-scroll');
    if (autoScrollEl) autoScrollEl.checked = true;
}
