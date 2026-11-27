// FSM-LLM Monitor — shared UI state
// Single source of truth: one plain object every module reads and writes.

export const WS_MAX_DELAY = 30000;
export const DEFAULT_AGENT_TYPE = 'react';

export const state = {
    ws: null,
    currentPage: 'dashboard',
    presets: null,
    agentTypes: null,
    wsRetryDelay: 3000,
    // Advisory: permissive until /api/capabilities answers; the server refuses with a 400 text.
    capabilities: { fsm: true, workflows: true, agents: true },
    instances: [],
    selectedConvId: null,
    selectedConvInstanceId: null,
    selectedDetailId: null,
    selectedDetailType: null,
    detailPollTimer: null,
    refreshTimers: {},
    autoScrollLogs: true,
    _lastContextData: null,
};

/** Debounced refresh: only schedule once per key until it fires. */
export function scheduleRefresh(key, fn, delayMs) {
    if (state.refreshTimers[key]) return;
    state.refreshTimers[key] = setTimeout(() => {
        state.refreshTimers[key] = null;
        fn();
    }, delayMs);
}

/** Cancel pending scheduleRefresh timers for the given keys. */
export function cancelRefresh(...keys) {
    for (const key of keys) {
        if (state.refreshTimers[key]) {
            clearTimeout(state.refreshTimers[key]);
            state.refreshTimers[key] = null;
        }
    }
}
