// FSM-LLM Monitor — WebSocket Connection

'use strict';

var _lastWsInstancesHash = '';

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
                // Quick hash to avoid re-processing identical instance lists
                var iHash = data.instances.length + ':' + data.instances.map(function(i) { return i.instance_id + ':' + i.status; }).join(',');
                if (iHash !== _lastWsInstancesHash) {
                    _lastWsInstancesHash = iHash;
                    App.instances = data.instances;
                    renderInstanceGrid();
                    if (App.currentPage === 'control') {
                        renderUnifiedTable();
                    }
                }
            }
            if (data.agent_updates) {
                App.agentUpdates = data.agent_updates;
                updateRunningAgents(data.agent_updates);
            }
            // Push logs to logs page via incremental append
            if (data.logs && data.logs.length > 0) {
                appendLogs(data.logs);
            }
            // Update error badge on sidebar from metrics
            if (data.type === 'metrics' && data.data) {
                updateLogErrorBadge(data.data);
            }
            if (data.events && data.events.length > 0) {
                var hasConvEvent = data.events.some(function(e) {
                    var t = e.event_type;
                    return t === 'conversation_start' || t === 'conversation_end'
                        || t === 'state_transition' || t === 'post_processing';
                });
                if (hasConvEvent) {
                    if (App.currentPage === 'dashboard') scheduleRefresh('dash-conv', refreshConversationTable, 3000);
                    if (App.currentPage === 'control') {
                        // Refresh conversation detail in drawer if viewing one
                        if (App.selectedConvId) {
                            var convId = App.selectedConvId;
                            scheduleRefresh('conv-detail', function() { showConversationDetail(convId); }, 2000);
                        }
                        // Refresh FSM detail (conv list) in drawer
                        if (App.selectedDetailId && App.selectedDetailType === 'fsm') {
                            var detailId = App.selectedDetailId;
                            scheduleRefresh('ctrl-detail', function() { refreshDetailPanel(detailId, 'fsm'); }, 2000);
                        }
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
