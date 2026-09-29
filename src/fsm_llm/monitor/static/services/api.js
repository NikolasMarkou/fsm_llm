// FSM-LLM Monitor — HTTP Client
// Thin fetch wrapper with JSON error handling and the optional API key header.

import { getApiKey, requestApiKey } from './auth.js';

function _withKey(opts) {
    const o = { ...(opts || {}) };
    const headers = new Headers(o.headers || {});
    const key = getApiKey();
    if (key) headers.set('X-API-Key', key);
    else headers.delete('X-API-Key');
    o.headers = headers;
    return o;
}

/**
 * Turn a FastAPI error `detail` into readable text. 422 bodies carry a list
 * of {loc, msg, type, ...}; other errors carry a string.
 */
export function formatErrorDetail(detail, fallback) {
    if (detail == null || detail === '') return fallback || 'Request failed';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        const parts = detail.map(d => {
            if (d && typeof d === 'object') {
                const loc = Array.isArray(d.loc)
                    ? d.loc.filter(p => p !== 'body' && p !== 'query' && p !== 'path').join('.')
                    : '';
                const msg = d.msg || d.message || JSON.stringify(d);
                return loc ? `${loc}: ${msg}` : msg;
            }
            return String(d);
        }).filter(Boolean);
        return parts.length ? parts.join('; ') : (fallback || 'Request failed');
    }
    if (typeof detail === 'object') {
        return detail.msg || detail.message || JSON.stringify(detail);
    }
    return String(detail);
}

export async function fetchJson(url, opts, _retried = false) {
    const resp = await fetch(url, _withKey(opts));
    if (!resp.ok) {
        let body;
        try { body = await resp.json(); } catch { body = null; }
        const detail = formatErrorDetail(body?.detail, resp.statusText || `HTTP ${resp.status}`);
        if (resp.status === 401 && !_retried) {
            // Prompt once (shared by concurrent requests) and retry this call
            // a single time with the saved key.
            const saved = await requestApiKey({
                message: `The server rejected the request (${detail}). Enter the monitor API key to continue.`,
            });
            if (saved) return fetchJson(url, opts, true);
        }
        const err = new Error(detail);
        err.status = resp.status;
        throw err;
    }
    return resp.json();
}

export function postJson(url, data) {
    return fetchJson(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
}
