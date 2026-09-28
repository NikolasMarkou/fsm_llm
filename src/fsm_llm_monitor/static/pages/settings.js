// FSM-LLM Monitor — Settings Page

import { state } from '../services/state.js';
import { fetchJson } from '../services/api.js';
import { getApiKey, setApiKey, clearApiKey, checkAuthRequired } from '../services/auth.js';
import { $, esc, showError, showStatus, showToast } from '../utils/dom.js';

// Server-side MonitorConfig bounds (the server answers 422 outside them).
const LIMITS = {
    refresh_interval: { min: 0.5, max: 60, label: 'Refresh interval', integer: false },
    max_events: { min: 10, max: 100000, label: 'Max events', integer: true },
    max_log_lines: { min: 10, max: 100000, label: 'Max log lines', integer: true },
};

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
        state.autoScrollLogs = cfg.auto_scroll_logs !== false;
    } catch (e) {
        console.error('loadSettings config:', e);
        showToast(`Failed to load settings: ${e.message}`, 'error');
    }
    try {
        const info = await fetchJson('/api/info');
        const el = $('sys-info');
        let html = '';
        for (const k in info) {
            html += `<span class="key">${esc(k.replace(/_/g, ' '))}:</span><span class="val">${esc(info[k])}</span>`;
        }
        el.innerHTML = html;
        $('version-info').textContent = `v${info.monitor_version}`;
        const footerEl = $('footer-version');
        if (footerEl) footerEl.textContent = `FSM-LLM Monitor v${info.monitor_version}`;
    } catch (e) {
        console.error('loadSettings info:', e);
        showToast('Failed to load system info', 'error');
    }
    _renderApiKeyState();
}

/** Read and validate one numeric field; returns {value} or {error}. */
function _readNumber(id, key) {
    const lim = LIMITS[key];
    const raw = $(id)?.value;
    const v = Number(raw);
    if (raw === '' || raw == null || !Number.isFinite(v)) {
        return { error: `${lim.label} must be a number.` };
    }
    if (lim.integer && !Number.isInteger(v)) {
        return { error: `${lim.label} must be a whole number.` };
    }
    if (v < lim.min || v > lim.max) {
        return { error: `${lim.label} must be between ${lim.min} and ${lim.max} (got ${v}).` };
    }
    return { value: v };
}

export async function saveSettings() {
    const fields = [
        ['set-refresh', 'refresh_interval'],
        ['set-max-events', 'max_events'],
        ['set-max-logs', 'max_log_lines'],
    ];
    const body = {};
    const errors = [];
    for (const [id, key] of fields) {
        const r = _readNumber(id, key);
        if (r.error) {
            errors.push(r.error);
            $(id)?.setAttribute('aria-invalid', 'true');
        } else {
            body[key] = r.value;
            $(id)?.removeAttribute('aria-invalid');
        }
    }
    if (errors.length) {
        showError('settings-status', errors.join(' '));
        return;
    }
    body.log_level = $('set-level').value;
    body.show_internal_keys = $('set-internal-keys')?.checked ?? false;
    body.auto_scroll_logs = $('set-auto-scroll')?.checked ?? true;

    try {
        await fetchJson('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        state.autoScrollLogs = body.auto_scroll_logs;
        showStatus('settings-status', 'Settings saved', 'success');
        const panel = $('settings-panel');
        if (panel) {
            panel.classList.add('animate-save');
            setTimeout(() => panel.classList.remove('animate-save'), 600);
        }
    } catch (e) {
        showError('settings-status', `Save failed: ${e.message}`);
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
    for (const id of ['set-refresh', 'set-max-events', 'set-max-logs']) $(id)?.removeAttribute('aria-invalid');
    showStatus('settings-status', '');
}

// --- API key (stored in sessionStorage only; see services/auth.js) ---

async function _renderApiKeyState() {
    const el = $('api-key-state');
    if (!el) return;
    const hasKey = !!getApiKey();
    const required = await checkAuthRequired();
    let text = hasKey ? 'A key is stored for this browser tab session.' : 'No key stored.';
    if (required === true) text += ' The server requires a key.';
    else if (required === false) text += ' The server does not require a key.';
    el.textContent = text;
}

export function saveApiKeySetting() {
    const input = $('set-api-key');
    const value = input?.value.trim() || '';
    if (!value) {
        showError('api-key-status', 'Enter a key to save, or use Clear.');
        return;
    }
    setApiKey(value);
    if (input) input.value = '';
    showStatus('api-key-status', 'API key saved for this session', 'success');
    _renderApiKeyState();
}

export function clearApiKeySetting() {
    clearApiKey();
    const input = $('set-api-key');
    if (input) input.value = '';
    showStatus('api-key-status', 'API key cleared', 'success');
    _renderApiKeyState();
}
