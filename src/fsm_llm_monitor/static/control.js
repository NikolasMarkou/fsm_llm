// FSM-LLM Monitor — Control Center Page

'use strict';

async function refreshControlCenter() {
    await refreshInstances();
    renderControlFSMs();
    renderControlWorkflows();
    renderControlAgents();
    if (App.selectedDetailId) {
        refreshDetailPanel(App.selectedDetailId, App.selectedDetailType);
    }
}

function closeDetail(panelId) {
    document.getElementById(panelId).style.display = 'none';
    App.selectedDetailId = null;
    App.selectedDetailType = null;
    if (App.detailPollTimer) { clearInterval(App.detailPollTimer); App.detailPollTimer = null; }
    document.querySelectorAll('tr.clickable-row.selected').forEach(function(r) { r.classList.remove('selected'); });
}

function navigateToInstance(instanceId, instanceType) {
    showPage('control');
    var tabMap = { fsm: 'ctrl-tab-fsm', workflow: 'ctrl-tab-workflows', agent: 'ctrl-tab-agents' };
    var tabId = tabMap[instanceType];
    if (tabId) {
        var tabBtns = document.querySelectorAll('#page-control .tab-bar .tab');
        tabBtns.forEach(function(btn) {
            if (btn.getAttribute('data-tab') === tabId) {
                switchTab(tabId, btn);
            }
        });
    }
    setTimeout(function() { selectInstance(instanceId, instanceType); }, 100);
}

function selectInstance(instanceId, type) {
    document.querySelectorAll('tr.clickable-row.selected').forEach(function(r) { r.classList.remove('selected'); });
    var row = document.querySelector('tr[data-instance-id="' + instanceId + '"]');
    if (row) row.classList.add('selected');

    App.selectedDetailId = instanceId;
    App.selectedDetailType = type;

    var panels = { fsm: 'ctrl-fsm-detail', workflow: 'ctrl-wf-detail', agent: 'ctrl-agent-detail' };
    for (var t in panels) {
        var p = document.getElementById(panels[t]);
        if (p) p.style.display = (t === type) ? 'block' : 'none';
    }

    refreshDetailPanel(instanceId, type);

    if (App.detailPollTimer) clearInterval(App.detailPollTimer);
    App.detailPollTimer = setInterval(function() {
        if (App.selectedDetailId && App.currentPage === 'control') {
            refreshDetailPanel(App.selectedDetailId, App.selectedDetailType);
        }
    }, 2000);
}

async function refreshDetailPanel(instanceId, type) {
    if (type === 'fsm') await renderFSMDetail(instanceId);
    else if (type === 'workflow') await renderWorkflowDetail(instanceId);
    else if (type === 'agent') await renderAgentDetail(instanceId);
    await refreshDetailEvents(instanceId, type);
}

async function refreshDetailEvents(instanceId, type) {
    var eventsIds = { fsm: 'ctrl-fsm-events', workflow: 'ctrl-wf-events', agent: 'ctrl-agent-events' };
    var logEl = document.getElementById(eventsIds[type]);
    if (!logEl) return;
    try {
        var resp = await fetch('/api/instances/' + encodeURIComponent(instanceId) + '/events?limit=50');
        var events = await resp.json();
        if (events.length === 0) {
            logEl.innerHTML = '<div class="empty-state"><div class="empty-hint">No events yet...</div></div>';
            return;
        }
        var html = '';
        for (var i = 0; i < events.length; i++) {
            var e = events[i];
            var ts = formatTime(e.timestamp);
            var level = (e.level || 'INFO').toLowerCase();
            html += '<div class="entry ' + level + '">';
            html += '<span class="ts">' + ts + '</span>';
            html += '<span class="type">' + esc(e.event_type) + '</span>';
            html += '<span class="msg">' + esc(e.message) + '</span>';
            html += '</div>';
        }
        logEl.innerHTML = html;
    } catch (e) {
        console.error('refreshDetailEvents:', e);
    }
}

// --- FSM Detail ---

async function renderFSMDetail(instanceId) {
    var titleEl = document.getElementById('ctrl-fsm-detail-title');
    var contentEl = document.getElementById('ctrl-fsm-detail-content');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    titleEl.textContent = inst.label || instanceId;

    try {
        var resp = await fetch('/api/fsm/' + encodeURIComponent(instanceId) + '/conversations');
        var convs = await resp.json();

        var html = '<div class="kv detail-kv">';
        html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
        html += '<span class="key">Source:</span><span class="val">' + esc(inst.source || 'custom') + '</span>';
        html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
        html += '</div>';

        html += '<div class="panel-title">CONVERSATIONS (' + convs.length + ')</div>';
        if (convs.length === 0 || (convs.length === 1 && convs[0].error)) {
            html += '<div class="empty-state"><div class="empty-hint">No active conversations.</div></div>';
            if (inst.status === 'running') {
                html += '<button class="btn btn-primary btn-sm mt-4" onclick="startConversationOn(\'' + esc(instanceId) + '\')">START CONVERSATION</button>';
            }
        } else {
            for (var i = 0; i < convs.length; i++) {
                var c = convs[i];
                if (c.error) continue;
                html += '<div class="conv-card" onclick="goToConversation(\'' + esc(instanceId) + '\',\'' + esc(c.conversation_id) + '\')">';
                html += '<div class="conv-info">';
                html += '<span class="mono-id">' + esc(c.conversation_id.substring(0, 12)) + '</span>';
                html += '<span class="conv-state">' + esc(c.current_state) + '</span>';
                html += '<span class="text-dim">' + (c.message_history ? c.message_history.length : 0) + ' msgs</span>';
                html += '</div>';
                html += '<div>' + (c.is_terminal ? statusBadge('ended') : statusBadge('active')) + '</div>';
                html += '</div>';
            }
            if (inst.status === 'running') {
                html += '<button class="btn btn-sm mt-4" onclick="startConversationOn(\'' + esc(instanceId) + '\')">+ NEW CONVERSATION</button>';
            }
        }
        contentEl.innerHTML = html;
    } catch (e) {
        contentEl.innerHTML = '<span class="error-message">Failed to load FSM detail</span>';
    }
}

function goToConversation(instanceId, convId) {
    App.selectedConvInstanceId = instanceId;
    showPage('conversations');
    setTimeout(function() { showConversationDetail(convId); }, 300);
}

// --- Workflow Detail ---

async function renderWorkflowDetail(instanceId) {
    var titleEl = document.getElementById('ctrl-wf-detail-title');
    var contentEl = document.getElementById('ctrl-wf-detail-content');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    titleEl.textContent = inst.label || instanceId;

    var html = '<div class="kv detail-kv">';
    html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
    html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
    html += '<span class="key">Active Workflows:</span><span class="val">' + (inst.active_workflows || 0) + '</span>';
    html += '</div>';

    try {
        var resp = await fetch('/api/workflow/' + encodeURIComponent(instanceId) + '/instances');
        var wfInstances = await resp.json();

        html += '<div class="panel-title">WORKFLOW INSTANCES (' + wfInstances.length + ')</div>';
        if (wfInstances.length === 0) {
            html += '<div class="empty-state"><div class="empty-hint">No workflow instances.</div></div>';
        } else {
            for (var i = 0; i < wfInstances.length; i++) {
                var wf = wfInstances[i];
                if (wf.error) continue;
                var wfId = wf.workflow_instance_id || '';
                var wfStatus = wf.status || 'unknown';
                html += '<div class="conv-card">';
                html += '<div class="conv-info">';
                html += '<span class="mono-id">' + esc(wfId.substring(0, 12)) + '</span>';
                if (wf.current_step) {
                    html += '<span class="conv-state">' + esc(wf.current_step) + '</span>';
                }
                if (wf.created_at) {
                    html += '<span class="text-dim">' + formatTime(wf.created_at) + '</span>';
                }
                html += '</div>';
                html += '<div>' + statusBadge(wfStatus) + '</div>';
                html += '</div>';
            }
        }
    } catch (e) {
        html += '<div class="panel-title">WORKFLOW INSTANCES</div>';
        html += '<div class="empty-state"><div class="empty-hint">Could not load workflow instances.</div></div>';
    }

    contentEl.innerHTML = html;
}

// --- Agent Detail ---

async function renderAgentDetail(instanceId) {
    var titleEl = document.getElementById('ctrl-agent-detail-title');
    var contentEl = document.getElementById('ctrl-agent-detail-content');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    titleEl.textContent = inst.label || instanceId;

    try {
        var resp = await fetch('/api/agent/' + encodeURIComponent(instanceId) + '/status');
        var data = await resp.json();
        if (data.error && !data.status) {
            contentEl.innerHTML = '<span class="error-message">' + esc(data.error) + '</span>';
            return;
        }

        var html = '<div class="kv detail-kv">';
        html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
        html += '<span class="key">Agent Type:</span><span class="val">' + esc(data.agent_type || '') + '</span>';
        html += '<span class="key">Status:</span><span class="val">' + statusBadge(data.status) + '</span>';
        html += '<span class="key">Task:</span><span class="val word-break">' + esc(data.task || '') + '</span>';
        if (data.total_iterations !== undefined) {
            html += '<span class="key">Iterations:</span><span class="val">' + data.total_iterations + '</span>';
        }
        html += '</div>';

        if (data.status === 'running') {
            var iterCount = data.iteration_count || 0;
            var maxIter = 10;
            var pct = Math.min(Math.round((iterCount / maxIter) * 100), 95);
            var stateLabel = data.current_state || 'initializing';
            var stateClass = stateLabel === 'think' ? 'state-think' : stateLabel === 'act' ? 'state-act' : stateLabel === 'conclude' ? 'state-conclude' : 'state-default';

            html += '<div class="agent-progress">';
            html += '<div class="agent-progress-header">';
            html += '<span>Iteration <b>' + iterCount + '</b></span>';
            html += '<span class="' + stateClass + '">' + esc(stateLabel.toUpperCase()) + '</span>';
            html += '</div>';
            html += '<div class="progress-bar"><div class="progress-fill" style="width:' + pct + '%;"></div></div>';
            if (data.last_tool_call) {
                html += '<div class="agent-tool-info">Last tool: ' + esc(data.last_tool_call) + '</div>';
            }
            html += '</div>';
        }

        if (data.answer) {
            html += '<div class="panel-title">ANSWER</div>';
            html += '<div class="event-log event-log-compact">';
            html += '<div class="entry"><span class="msg pre-wrap">' + esc(data.answer) + '</span></div>';
            html += '</div>';
        }

        if (data.error) {
            html += '<div class="panel-title">ERROR</div>';
            html += '<div class="error-message">' + esc(data.error) + '</div>';
        }

        if (data.status !== 'running') {
            await _renderAgentTrace(instanceId, html, contentEl, data);
            return;
        }

        if (data.tools_used && data.tools_used.length > 0) {
            html += _renderToolCalls(data.tools_used);
        }

        if (data.success !== undefined && data.status !== 'running') {
            html += renderResultBanner(data.success);
        }

        contentEl.innerHTML = html;
    } catch (e) {
        contentEl.innerHTML = '<span class="error-message">Failed to load agent detail</span>';
    }
}

function _renderToolCalls(toolsUsed) {
    var html = '<div class="panel-title">TOOL CALLS (' + toolsUsed.length + ')</div>';
    for (var i = 0; i < toolsUsed.length; i++) {
        var tc = toolsUsed[i];
        html += '<div class="trace-step step-act">';
        html += '<div class="step-header"><span class="step-label">' + esc(tc.tool_name) + '</span></div>';
        html += '<div class="step-body d-block">' + esc(JSON.stringify(tc.parameters || {}, null, 1)) + '</div>';
        html += '</div>';
    }
    return html;
}

function toggleAllTraceSteps(expand) {
    document.querySelectorAll('.trace-step .step-body').forEach(function(el) {
        el.style.display = expand ? 'block' : 'none';
    });
}

async function _renderAgentTrace(instanceId, html, contentEl, statusData) {
    try {
        var resp = await fetch('/api/agent/' + encodeURIComponent(instanceId) + '/result');
        var result = await resp.json();
        if (result.trace_steps && result.trace_steps.length > 0) {
            html += '<div class="panel-title panel-title-flex">EXECUTION TRACE (' + result.trace_steps.length + ' steps)';
            html += '<div class="flex-row-gap-4">';
            html += '<button class="btn btn-sm" onclick="event.stopPropagation();toggleAllTraceSteps(true)">EXPAND ALL</button>';
            html += '<button class="btn btn-sm" onclick="event.stopPropagation();toggleAllTraceSteps(false)">COLLAPSE ALL</button>';
            html += '</div></div>';
            var iteration = 0;
            for (var i = 0; i < result.trace_steps.length; i++) {
                var step = result.trace_steps[i];
                var state = step.state || '';
                if (state === 'think') iteration++;

                var stepClass = 'step-' + state;
                var stateColorClass = state === 'think' ? 'state-think' : state === 'act' ? 'state-act' : state === 'conclude' ? 'state-conclude' : 'state-default';
                var stepIcon = state === 'think' ? '&#9679;' : state === 'act' ? '&#9654;' : state === 'conclude' ? '&#10003;' : '&#8226;';

                html += '<div class="trace-step ' + stepClass + '" onclick="this.querySelector(\'.step-body\').style.display=this.querySelector(\'.step-body\').style.display===\'none\'?\'block\':\'none\'">';
                html += '<div class="step-header">';
                html += '<span class="' + stateColorClass + '">' + stepIcon + ' ' + esc(state.toUpperCase());
                if (state === 'think') html += ' #' + iteration;
                if (step.tool_name) html += ' &mdash; ' + esc(step.tool_name);
                html += '</span></div>';
                html += '<div class="step-body">';
                if (step.reasoning) html += '<div><b>Reasoning:</b> ' + esc(step.reasoning) + '</div>';
                if (step.tool_input) html += '<div><b>Input:</b> ' + esc(step.tool_input) + '</div>';
                if (step.tool_result) html += '<div><b>Result:</b> ' + esc(step.tool_result) + '</div>';
                html += '</div></div>';
            }
        } else if (statusData.tools_used && statusData.tools_used.length > 0) {
            html += _renderToolCalls(statusData.tools_used);
        }
    } catch (e) {
        // Trace fetch failed, skip
    }

    if (statusData.success !== undefined) {
        html += renderResultBanner(statusData.success);
    }

    contentEl.innerHTML = html;
}

function updateRunningAgents(updates) {
    if (App.selectedDetailType === 'agent' && App.selectedDetailId && updates[App.selectedDetailId]) {
        renderAgentDetail(App.selectedDetailId);
    }
}

// --- Control Center Table Renderers ---

function renderControlFSMs() {
    var body = document.getElementById('ctrl-fsm-body');
    var empty = document.getElementById('ctrl-fsm-empty');
    var fsms = App.instances.filter(function(i) { return i.instance_type === 'fsm'; });
    if (fsms.length === 0) {
        body.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';
    var rows = '';
    for (var i = 0; i < fsms.length; i++) {
        var f = fsms[i];
        var sel = (App.selectedDetailId === f.instance_id) ? ' selected' : '';
        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(f.instance_id) + '" onclick="selectInstance(\'' + esc(f.instance_id) + '\',\'fsm\')">';
        rows += '<td>' + esc(f.label || f.instance_id) + '</td>';
        rows += '<td class="cell-truncate">' + esc(f.source || 'custom') + '</td>';
        rows += '<td>' + (f.conversation_count || 0) + '</td>';
        rows += '<td>' + statusBadge(f.status) + '</td>';
        rows += '<td>';
        if (f.status === 'running') {
            rows += '<button class="btn btn-sm" onclick="event.stopPropagation();startConversationOn(\'' + esc(f.instance_id) + '\')">+ CONV</button> ';
        }
        rows += '<button class="btn btn-sm btn-danger" onclick="event.stopPropagation();destroyInstance(\'' + esc(f.instance_id) + '\')">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;
}

function renderControlWorkflows() {
    var body = document.getElementById('ctrl-wf-body');
    var empty = document.getElementById('ctrl-wf-empty');
    var wfs = App.instances.filter(function(i) { return i.instance_type === 'workflow'; });
    if (wfs.length === 0) {
        body.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';
    var rows = '';
    for (var i = 0; i < wfs.length; i++) {
        var w = wfs[i];
        var sel = (App.selectedDetailId === w.instance_id) ? ' selected' : '';
        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(w.instance_id) + '" onclick="selectInstance(\'' + esc(w.instance_id) + '\',\'workflow\')">';
        rows += '<td>' + esc(w.label || w.instance_id) + '</td>';
        rows += '<td>' + statusBadge(w.status) + '</td>';
        rows += '<td>' + (w.active_workflows || 0) + '</td>';
        rows += '<td>';
        rows += '<button class="btn btn-sm btn-danger" onclick="event.stopPropagation();destroyInstance(\'' + esc(w.instance_id) + '\')">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;
}

function renderControlAgents() {
    var body = document.getElementById('ctrl-agent-body');
    var empty = document.getElementById('ctrl-agent-empty');
    var agents = App.instances.filter(function(i) { return i.instance_type === 'agent'; });
    if (agents.length === 0) {
        body.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';
    var rows = '';
    for (var i = 0; i < agents.length; i++) {
        var a = agents[i];
        var sel = (App.selectedDetailId === a.instance_id) ? ' selected' : '';
        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(a.instance_id) + '" onclick="selectInstance(\'' + esc(a.instance_id) + '\',\'agent\')">';
        rows += '<td>' + esc(a.label || a.instance_id) + '</td>';
        rows += '<td>' + esc(a.agent_type || '') + '</td>';
        rows += '<td class="cell-truncate">' + esc(a.task || '') + '</td>';
        rows += '<td>' + statusBadge(a.status) + '</td>';
        rows += '<td>';
        if (a.status === 'running') {
            rows += '<button class="btn btn-sm btn-warning" onclick="event.stopPropagation();cancelAgent(\'' + esc(a.instance_id) + '\')">CANCEL</button> ';
        }
        rows += '<button class="btn btn-sm btn-danger" onclick="event.stopPropagation();destroyInstance(\'' + esc(a.instance_id) + '\')">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;
}

async function startConversationOn(instanceId) {
    try {
        var resp = await fetch('/api/fsm/' + encodeURIComponent(instanceId) + '/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_context: {} })
        });
        var data = await resp.json();
        if (data.error) {
            console.error('startConversation:', data.error);
            return;
        }
        App.selectedConvInstanceId = instanceId;
        App.selectedConvId = data.conversation_id;
        refreshInstances();
        showPage('conversations');
        setTimeout(function() {
            refreshConversations();
            if (data.conversation_id) showConversationDetail(data.conversation_id);
        }, 300);
    } catch (e) {
        console.error('startConversationOn:', e);
    }
}

async function destroyInstance(instanceId) {
    try {
        await fetch('/api/instances/' + encodeURIComponent(instanceId), { method: 'DELETE' });
        if (App.selectedDetailId === instanceId) {
            closeDetail('ctrl-fsm-detail');
            closeDetail('ctrl-wf-detail');
            closeDetail('ctrl-agent-detail');
        }
        refreshInstances();
        if (App.currentPage === 'control') refreshControlCenter();
    } catch (e) {
        console.error('destroyInstance:', e);
    }
}

async function cancelAgent(instanceId) {
    try {
        await fetch('/api/agent/' + encodeURIComponent(instanceId) + '/cancel', { method: 'POST' });
        if (App.selectedDetailId === instanceId) renderAgentDetail(instanceId);
        refreshControlCenter();
    } catch (e) {
        console.error('cancelAgent:', e);
    }
}
