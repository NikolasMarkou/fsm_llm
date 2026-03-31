// FSM-LLM Monitor — WebSocket Connection Manager
// Handles connection, reconnection with exponential backoff, and message dispatch.

import { state, scheduleRefresh, WS_MAX_DELAY } from './state.js';
import { hashInstances, $ } from '../utils/dom.js';

let _lastWsInstancesHash = '';
let _dispatch = {};

/** Register page-level handlers for WebSocket message dispatch. */
export function registerHandlers(handlers) {
    _dispatch = { ..._dispatch, ...handlers };
}

export function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(proto + '//' + location.host + '/ws');

    state.ws.onopen = () => {
        const statusEl = $('ws-status');
        const dotEl = $('ws-dot');
        if (statusEl) { statusEl.textContent = 'Connected'; statusEl.className = 'ws-label'; }
        if (dotEl) dotEl.classList.add('connected');
        state.wsRetryDelay = 3000;
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'metrics') _dispatch.updateMetrics?.(data.data);
            if (data.events) _dispatch.updateEvents?.(data.events);

            if (data.instances) {
                const iHash = data.instances.length + ':' + data.instances.map(i => i.instance_id + ':' + i.status).join(',');
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
                // Refresh detail panel if viewing a workflow
                if (state.currentPage === 'control' && state.selectedDetailType === 'workflow' && state.selectedDetailId) {
                    const detailId = state.selectedDetailId;
                    scheduleRefresh('ctrl-detail-wf', () => _dispatch.refreshDetailPanel?.(detailId, 'workflow'), 2000);
                }
            }

            if (data.logs?.length > 0) _dispatch.appendLogs?.(data.logs);

            if (data.type === 'metrics' && data.data) _dispatch.updateLogErrorBadge?.(data.data);

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
                            const convId = state.selectedConvId;
                            scheduleRefresh('conv-detail', () => _dispatch.showConversationDetail?.(convId), 2000);
                        }
                        if (state.selectedDetailId && state.selectedDetailType) {
                            const detailId = state.selectedDetailId;
                            const detailType = state.selectedDetailType;
                            scheduleRefresh('ctrl-detail', () => _dispatch.refreshDetailPanel?.(detailId, detailType), 2000);
                        }
                    }
                }
            }
        } catch (e) {
            console.error('WS message parse error:', e);
        }
    };

    state.ws.onclose = () => {
        const statusEl = $('ws-status');
        const dotEl = $('ws-dot');
        if (statusEl) { statusEl.textContent = 'Reconnecting...'; statusEl.className = 'ws-label blink'; }
        if (dotEl) dotEl.classList.remove('connected');
        setTimeout(connectWS, state.wsRetryDelay + Math.random() * 1000);
        state.wsRetryDelay = Math.min(state.wsRetryDelay * 2, WS_MAX_DELAY);
    };

    state.ws.onerror = () => { state.ws.close(); };
}
