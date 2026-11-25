// FSM-LLM Monitor — HTTP Client
// Thin fetch wrapper with JSON error handling and the optional API key header.

import { getApiKey, requestApiKey } from './auth.js';
import { state } from './state.js';
import { showToast } from '../utils/dom.js';

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
function formatErrorDetail(detail, fallback) {
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

export function postJson(url, data, opts) {
    return fetchJson(url, {
        ...opts,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
}

/**
 * The rows of GET /api/agent/types (`id`, `name`, `launchable`, `needs_tools`,
 * `takes_tools`, `tool_names`, `drawable`), fetched once and kept on `state.agentTypes`;
 * callers that ask while the fetch is in flight share it. A failed fetch shows one toast,
 * caches nothing (the next call retries) and gives [].
 */
export function loadAgentTypes() {
    if (state.agentTypes) return Promise.resolve(state.agentTypes);
    _agentTypesLoad ??= _fetchAgentTypes().finally(() => { _agentTypesLoad = null; });
    return _agentTypesLoad;
}

let _agentTypesLoad = null;

async function _fetchAgentTypes() {
    try {
        const data = await fetchJson('/api/agent/types');
        state.agentTypes = (Array.isArray(data?.agent_types) ? data.agent_types : [])
            .filter(r => r && typeof r.id === 'string');
        return state.agentTypes;
    } catch (e) {
        showToast(`Agent types failed to load: ${e.message || e}`, 'error');
        return [];
    }
}

/** The loaded row of an agent type id, or undefined (not loaded, or not a listed id). */
export function agentRow(id) {
    return (state.agentTypes || []).find(r => r.id === id);
}

/** True when launching `id` needs a stub tool typed by the user (a needs_tools row with no tool names of its own). */
export function toolsRequired(id) {
    const row = agentRow(id);
    return !!row?.needs_tools && !row.tool_names?.length;
}
