// FSM-LLM Monitor — WebSocket Connection

'use strict';

function connectWS() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    App.ws = new WebSocket(proto + '//' + location.host + '/ws');
    App.ws.onopen = function() {
        var statusEl = document.getElementById('ws-status');
        var dotEl = document.getElementById('ws-dot');
        if (statusEl) { statusEl.textContent = 'Connected'; statusEl.className = 'ws-label'; }
        if (dotEl) dotEl.classList.add('connected');
        App.wsRetryDelay = 3000;
    };
    App.ws.onmessage = function(event) {
        try {
            var data = JSON.parse(event.data);
            if (data.type === 'metrics') updateMetrics(data.data);
            if (data.events) updateEvents(data.events);
            if (data.instances) {
                App.instances = data.instances;
                renderInstanceGrid();
                if (App.currentPage === 'control') {
                    renderUnifiedTable();
                }
            }
            if (data.agent_updates) {
                App.agentUpdates = data.agent_updates;
                updateRunningAgents(data.agent_updates);
            }
            if (data.events && data.events.length > 0) {
                var hasConvEvent = data.events.some(function(e) {
                    var t = e.event_type;
                    return t === 'conversation_start' || t === 'conversation_end'
                        || t === 'state_transition' || t === 'post_processing';
                });
                if (hasConvEvent) {
                    if (App.currentPage === 'dashboard') scheduleRefresh('dash-conv', refreshConversationTable, 3000);
                    if (App.currentPage === 'conversations') {
                        scheduleRefresh('conv-list', refreshConversations, 2000);
                        if (App.selectedConvId) {
                            var convId = App.selectedConvId;
                            scheduleRefresh('conv-detail', function() { showConversationDetail(convId); }, 2000);
                        }
                    }
                    if (App.currentPage === 'control' && App.selectedDetailId && App.selectedDetailType === 'fsm') {
                        var detailId = App.selectedDetailId;
                        scheduleRefresh('ctrl-detail', function() { refreshDetailPanel(detailId, 'fsm'); }, 2000);
                    }
                }
            }
        } catch (e) {
            console.error('WS message parse error:', e);
        }
    };
    App.ws.onclose = function() {
        var statusEl = document.getElementById('ws-status');
        var dotEl = document.getElementById('ws-dot');
        if (statusEl) { statusEl.textContent = 'Reconnecting...'; statusEl.className = 'ws-label blink'; }
        if (dotEl) dotEl.classList.remove('connected');
        setTimeout(connectWS, App.wsRetryDelay);
        App.wsRetryDelay = Math.min(App.wsRetryDelay * 2, App.WS_MAX_DELAY);
    };
    App.ws.onerror = function() { App.ws.close(); };
}
