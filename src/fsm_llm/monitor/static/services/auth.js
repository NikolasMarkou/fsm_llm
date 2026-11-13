// FSM-LLM Monitor: API Key Handling
// Stores the optional monitor API key in sessionStorage (never localStorage,
// never in URLs, never logged) and drives the key-entry modal
// (#apikey-modal in templates/index.html).

import { $, openDialog, closeDialog } from '../utils/dom.js';

const STORAGE_KEY = 'fsmMonitorApiKey';

// Fallback when sessionStorage is unavailable (privacy mode, sandboxed frame).
let _memKey = null;
// Shared pending prompt so concurrent 401s open one modal and all retry.
let _pending = null;
// Set when the user cancels the modal: background 401s stop re-opening it
// until a key is saved (Settings or a forced prompt such as WS 4401).
let _dismissed = false;
let _authRequired = null;
const _listeners = new Set();

export function getApiKey() {
    try {
        return sessionStorage.getItem(STORAGE_KEY) || null;
    } catch {
        return _memKey;
    }
}

function _notify() {
    for (const fn of _listeners) {
        try { fn(); } catch (e) { console.error('api key listener failed'); }
    }
}

export function setApiKey(key) {
    const k = String(key ?? '').trim();
    if (!k) { clearApiKey(); return; }
    try { sessionStorage.setItem(STORAGE_KEY, k); } catch { _memKey = k; }
    _dismissed = false;
    _notify();
}

export function clearApiKey() {
    try { sessionStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
    _memKey = null;
    _notify();
}

/** Subscribe to key changes (save or clear). Returns an unsubscribe function. */
export function onApiKeyChange(fn) {
    _listeners.add(fn);
    return () => _listeners.delete(fn);
}

/**
 * Ask the server whether an API key is configured (GET /api/auth is never
 * gated). Returns true/false, or null when the server does not answer.
 */
export async function checkAuthRequired() {
    try {
        const resp = await fetch('/api/auth');
        if (!resp.ok) return null;
        const data = await resp.json();
        _authRequired = !!data?.auth_required;
        return _authRequired;
    } catch {
        return null;
    }
}

export function isApiKeyModalOpen() {
    return _pending !== null;
}

/**
 * Show the key-entry modal. Resolves true once the user saves a key, false on
 * cancel. `force` opens it even after the user dismissed an earlier prompt.
 */
export function requestApiKey({ force = false, message = '' } = {}) {
    if (_pending) return _pending.promise;
    if (!force && _dismissed) return Promise.resolve(false);
    const modal = $('apikey-modal');
    if (!modal) return Promise.resolve(false);

    let resolve;
    const promise = new Promise(r => { resolve = r; });
    _pending = { promise, resolve };

    const msgEl = $('apikey-modal-msg');
    if (msgEl) {
        msgEl.textContent = message
            || 'This monitor requires an API key. It is kept in this browser tab\'s session storage only.';
    }
    const status = $('apikey-status');
    if (status) status.textContent = '';
    const input = $('apikey-input');
    if (input) input.value = '';
    openDialog(modal, 'flex', input);
    return promise;
}

function _finish(result) {
    const p = _pending;
    _pending = null;
    closeDialog($('apikey-modal'));
    p?.resolve(result);
}

/** Save handler for the modal (data-action="apikey-save", Enter in the input). */
export function submitApiKeyModal() {
    const input = $('apikey-input');
    const value = input?.value.trim() || '';
    if (!value) {
        const status = $('apikey-status');
        if (status) status.textContent = 'Enter a key, or Cancel.';
        input?.focus();
        return;
    }
    if (input) input.value = '';
    setApiKey(value);
    _finish(true);
}

/** Cancel handler for the modal (button, backdrop, Escape). */
export function cancelApiKeyModal() {
    if (!_pending) { closeDialog($('apikey-modal')); return; }
    _dismissed = true;
    _finish(false);
}
