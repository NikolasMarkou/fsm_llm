// FSM-LLM Monitor — WebSocket Connection Manager
// Handles connection, first-message auth, reconnection with exponential
// backoff, and message dispatch.

import { state, scheduleRefresh, WS_MAX_DELAY } from './state.js';
import { getApiKey, requestApiKey, onApiKeyChange } from './auth.js';
import { hashInstances, $ } from '../utils/dom.js';

// Close code the server uses when the first-message auth fails.
const WS_AUTH_FAILED = 4401;

let _lastWsInstancesHash = 0;
let _dispatch = {};
let _reconnectTimer = null;
let _authFailed = false;

/** Register page-level handlers for WebSocket message dispatch. */
export function registerHandlers(handlers) {
    _dispatch = { ..._dispatch, ...handlers };
}

function _setStatus(text, cls, connected) {
    const statusEl = $('ws-status');
    const dotEl = $('ws-dot');
    if (statusEl) { statusEl.textContent = text; statusEl.className = cls; }
    if (dotEl) dotEl.classList.toggle('connected', connected);
    const connEl = $('conn-status');
    if (connEl) connEl.textContent = text;
}

// After a 4401 close auto-reconnect stops; saving a key resumes it.
onApiKeyChange(() => {
    if (_authFailed && getApiKey()) {
        _authFailed = false;
        state.wsRetryDelay = 3000;
        connectWS();
    }
});

export function connectWS() {
    if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
    // Detach the previous socket's handlers before replacing it so a late
    // onclose from the old socket can't start a second reconnect chain
    // (frontend finding F-04).
    if (state.ws) {
        const old = state.ws;
        old.onopen = old.onmessage = old.onclose = old.onerror = null;
        if (old.readyState === WebSocket.OPEN || old.readyState === WebSocket.CONNECTING) {
            try { old.close(); } catch { /* ignore */ }
        }
    }
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/ws`);
    state.ws = ws;

    ws.onopen = () => {
        // The server expects the auth message first (empty key when none is set).
        try {
            ws.send(JSON.stringify({ type: 'auth', api_key: getApiKey() || '' }));
        } catch (e) {
            console.error('WS auth send failed');
        }
        _setStatus('Connected', 'ws-label', true);
        state.wsRetryDelay = 3000;
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'metrics') _dispatch.updateMetrics?.(data.data);
            if (data.events) _dispatch.updateEvents?.(data.events);

            if (data.instances) {
                const iHash = hashInstances(data.instances);
                if (iHash !== _lastWsInstancesHash) {
                    _lastWsInstancesHash = iHash;
                    state.instances = data.instances;
                    _dispatch.renderInstanceGrid?.();
                    if (state.currentPage === 'control') _dispatch.renderUnifiedTable?.();
                }
            }

            if (data.agent_updates) {
                state.agentUpdates = data.agent_updates;
                _dispatch.updateRunningAgents?.(data.agent_updates);
            }

            if (data.workflow_updates) {
                state.workflowUpdates = data.workflow_updates;
                // Refresh activity table when workflow status changes
                if (state.currentPage === 'dashboard') {
                    scheduleRefresh('dash-activity-wf', () => _dispatch.refreshActivityTable?.(), 3000);
                }
                // Refresh detail panel if viewing a workflow. The callback reads
                // the selection when it fires, so a timer scheduled for instance
                // A never paints into B's drawer.
                if (state.currentPage === 'control' && state.selectedDetailType === 'workflow' && state.selectedDetailId) {
                    scheduleRefresh('ctrl-detail-wf', () => {
                        if (state.selectedDetailType === 'workflow' && state.selectedDetailId) {
                            _dispatch.refreshDetailPanel?.(state.selectedDetailId, 'workflow');
                        }
                    }, 2000);
                }
            }

            if (data.logs?.length > 0) _dispatch.appendLogs?.(data.logs);

            if (data.dashboard_config) _dispatch.dashboardConfigChanged?.(data.dashboard_config);

            if (data.events?.length > 0) {
                const hasActivityEvent = data.events.some(e => {
                    const t = e.event_type;
                    return t === 'conversation_start' || t === 'conversation_end'
                        || t === 'state_transition' || t === 'post_processing'
                        || t === 'agent_started' || t === 'agent_completed' || t === 'agent_failed'
                        || t === 'workflow_started' || t === 'workflow_completed' || t === 'workflow_cancelled';
                });
                if (hasActivityEvent) {
                    if (state.currentPage === 'dashboard') {
                        scheduleRefresh('dash-activity', () => _dispatch.refreshActivityTable?.(), 3000);
                    }
                    if (state.currentPage === 'control') {
                        if (state.selectedConvId) {
                            scheduleRefresh('conv-detail', () => {
                                if (state.selectedConvId) _dispatch.showConversationDetail?.(state.selectedConvId);
                            }, 2000);
                        }
                        if (state.selectedDetailId && state.selectedDetailType) {
                            scheduleRefresh('ctrl-detail', () => {
                                if (state.selectedDetailId && state.selectedDetailType) {
                                    _dispatch.refreshDetailPanel?.(state.selectedDetailId, state.selectedDetailType);
                                }
                            }, 2000);
                        }
                    }
                }
            }
        } catch (e) {
            console.error('WS message parse error:', e);
        }
    };

    ws.onclose = (ev) => {
        if (state.ws !== ws) return;
        if (ev?.code === WS_AUTH_FAILED) {
            // Wrong or missing key: stop the reconnect loop and ask for a key.
            // The onApiKeyChange listener reconnects once a key is saved.
            _authFailed = true;
            _setStatus('API key required', 'ws-label', false);
            requestApiKey({
                force: true,
                message: 'The live connection was refused: missing or invalid API key.',
            });
            return;
        }
        _setStatus('Reconnecting...', 'ws-label blink', false);
        _reconnectTimer = setTimeout(connectWS, state.wsRetryDelay + Math.random() * 1000);
        state.wsRetryDelay = Math.min(state.wsRetryDelay * 2, WS_MAX_DELAY);
    };

    ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
}
