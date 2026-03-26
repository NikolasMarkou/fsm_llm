// FSM-LLM Monitor — Proxy-Based Reactive State Manager
// Single source of truth. Emits 'statechange' events on mutation.

export const WS_MAX_DELAY = 30000;
export const TOOL_BASED_AGENTS = ['ReactAgent', 'ReflexionAgent', 'PlanExecuteAgent', 'REWOOAgent', 'ADaPTAgent'];

const _target = {
    ws: null,
    currentPage: 'dashboard',
    presets: null,
    wsRetryDelay: 3000,
    capabilities: { fsm: true, workflows: false, agents: false },
    instances: [],
    selectedConvId: null,
    selectedConvInstanceId: null,
    selectedDetailId: null,
    selectedDetailType: null,
    detailPollTimer: null,
    agentUpdates: {},
    refreshTimers: {},
    stubToolCount: 0,
    _lastContextData: null,
};

const _bus = new EventTarget();

export const state = new Proxy(_target, {
    set(target, prop, value) {
        const old = target[prop];
        target[prop] = value;
        if (old !== value) {
            _bus.dispatchEvent(new CustomEvent('statechange', {
                detail: { prop, value, old },
            }));
        }
        return true;
    },
});

/** Subscribe to state changes. Returns an unsubscribe function. */
export function onChange(fn) {
    const handler = (e) => fn(e.detail);
    _bus.addEventListener('statechange', handler);
    return () => _bus.removeEventListener('statechange', handler);
}

/** Debounced refresh: only schedule once per key until it fires. */
export function scheduleRefresh(key, fn, delayMs) {
    if (state.refreshTimers[key]) return;
    state.refreshTimers[key] = setTimeout(() => {
        state.refreshTimers[key] = null;
        fn();
    }, delayMs);
}
