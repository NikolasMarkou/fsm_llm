// FSM-LLM Monitor — Shared State
// All shared mutable state lives here in the App namespace.

'use strict';

var App = {
    ws: null,
    currentPage: 'dashboard',
    presets: null,
    wsRetryDelay: 3000,
    WS_MAX_DELAY: 30000,
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
    TOOL_BASED_AGENTS: ['ReactAgent', 'ReflexionAgent', 'PlanExecuteAgent', 'REWOOAgent', 'ADaPTAgent'],
    _lastContextData: null
};
