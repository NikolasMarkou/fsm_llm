// FSM-LLM Monitor — Control Center Page (Unified View)

'use strict';

// --- Filter, Search & Pagination State ---

var _ctrlFilter = 'all';
var _ctrlPage = 0;
var _ctrlPerPage = 25;
var _ctrlSearch = '';
var _lastCtrlHash = '';
var _lastDrawerEventsHash = '';

function filterControlInstances(filter, btn) {
    _ctrlFilter = filter;
    _ctrlPage = 0;
    _lastCtrlHash = '';
    document.querySelectorAll('#ctrl-filter-chips .filter-chip').forEach(function(c) { c.classList.remove('active'); });
    if (btn) btn.classList.add('active');
    renderUnifiedTable();
}

function onCtrlSearchInput() {
    _ctrlSearch = (document.getElementById('ctrl-search') || {}).value || '';
    _ctrlPage = 0;
    _lastCtrlHash = '';
    renderUnifiedTable();
}

async function refreshControlCenter() {
    await refreshInstances();
    renderUnifiedTable();
    if (App.selectedDetailId) {
        refreshDetailPanel(App.selectedDetailId, App.selectedDetailType);
    }
}

function _getFilteredCtrlItems() {
    var items = App.instances;
    if (_ctrlFilter !== 'all') {
        items = items.filter(function(i) { return i.instance_type === _ctrlFilter; });
    }
    if (_ctrlSearch) {
        var q = _ctrlSearch.toLowerCase();
        items = items.filter(function(i) {
            return (i.label || i.instance_id).toLowerCase().indexOf(q) !== -1
                || i.instance_type.toLowerCase().indexOf(q) !== -1
                || (i.agent_type || '').toLowerCase().indexOf(q) !== -1;
        });
    }
    return items;
}

function renderUnifiedTable() {
    var body = document.getElementById('ctrl-unified-body');
    var empty = document.getElementById('ctrl-unified-empty');
    var paginationEl = document.getElementById('ctrl-pagination');
    var countEl = document.getElementById('ctrl-table-count');
    if (!body) return;

    var items = _getFilteredCtrlItems();

    // Quick hash — skip re-render if unchanged
    var hash = items.length + ':' + items.map(function(i) { return i.instance_id + i.status; }).join(',');
    if (hash === _lastCtrlHash) return;
    _lastCtrlHash = hash;

    // Update count
    if (countEl) countEl.textContent = items.length;

    if (items.length === 0) {
        body.innerHTML = '';
        if (empty) empty.style.display = 'block';
        if (paginationEl) paginationEl.style.display = 'none';
        return;
    }
    if (empty) empty.style.display = 'none';

    // Paginate
    var totalPages = Math.ceil(items.length / _ctrlPerPage);
    if (_ctrlPage >= totalPages) _ctrlPage = totalPages - 1;
    if (_ctrlPage < 0) _ctrlPage = 0;
    var start = _ctrlPage * _ctrlPerPage;
    var pageItems = items.slice(start, start + _ctrlPerPage);

    var rows = '';
    for (var i = 0; i < pageItems.length; i++) {
        var inst = pageItems[i];
        var sel = (App.selectedDetailId === inst.instance_id) ? ' selected' : '';
        var typeDot = '<span class="type-dot type-' + esc(inst.instance_type) + '"></span>';

        var detail = '';
        if (inst.instance_type === 'fsm') {
            detail = (inst.conversation_count || 0) + ' conv' + ((inst.conversation_count || 0) !== 1 ? 's' : '');
            if (inst.source) detail += ' &middot; ' + esc(inst.source);
        } else if (inst.instance_type === 'agent') {
            detail = esc(inst.agent_type || '');
            if (inst.task) detail += ' &middot; ' + esc((inst.task || '').substring(0, 40));
        } else if (inst.instance_type === 'workflow') {
            detail = (inst.active_workflows || 0) + ' active';
        }

        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(inst.instance_id) + '" onclick="openDrawer(\'' + esc(inst.instance_id) + '\',\'' + esc(inst.instance_type) + '\')">';
        rows += '<td>' + typeDot + esc(inst.instance_type) + '</td>';
        rows += '<td>' + esc(inst.label || inst.instance_id) + '</td>';
        rows += '<td class="cell-truncate text-dim">' + detail + '</td>';
        rows += '<td>' + statusBadge(inst.status) + '</td>';
        rows += '<td>';
        if (inst.instance_type === 'fsm' && inst.status === 'running') {
            rows += '<button class="btn btn-sm" onclick="event.stopPropagation();startConversationOn(\'' + esc(inst.instance_id) + '\')">+ Conv</button> ';
        }
        if (inst.instance_type === 'agent' && inst.status === 'running') {
            rows += '<button class="btn btn-sm btn-warning" onclick="event.stopPropagation();cancelAgent(\'' + esc(inst.instance_id) + '\')">Cancel</button> ';
        }
        rows += '<button class="btn btn-sm btn-danger" onclick="event.stopPropagation();destroyInstance(\'' + esc(inst.instance_id) + '\')">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;

    // Pagination controls
    if (paginationEl) {
        if (totalPages <= 1) {
            paginationEl.style.display = 'none';
        } else {
            paginationEl.style.display = 'flex';
            paginationEl.innerHTML = '<button class="btn btn-sm" onclick="ctrlPagePrev()"' + (_ctrlPage === 0 ? ' disabled' : '') + '>&laquo; Prev</button>'
                + '<span class="pagination-info">Page ' + (_ctrlPage + 1) + ' of ' + totalPages + '</span>'
                + '<button class="btn btn-sm" onclick="ctrlPageNext()"' + (_ctrlPage >= totalPages - 1 ? ' disabled' : '') + '>Next &raquo;</button>';
        }
    }
}

function ctrlPagePrev() {
    if (_ctrlPage > 0) { _ctrlPage--; _lastCtrlHash = ''; renderUnifiedTable(); }
}

function ctrlPageNext() {
    var items = _getFilteredCtrlItems();
    var totalPages = Math.ceil(items.length / _ctrlPerPage);
    if (_ctrlPage < totalPages - 1) { _ctrlPage++; _lastCtrlHash = ''; renderUnifiedTable(); }
}

// --- Drawer ---

function openDrawer(instanceId, type) {
    document.querySelectorAll('tr.clickable-row.selected').forEach(function(r) { r.classList.remove('selected'); });
    var row = document.querySelector('tr[data-instance-id="' + instanceId + '"]');
    if (row) row.classList.add('selected');

    App.selectedDetailId = instanceId;
    App.selectedDetailType = type;

    document.getElementById('ctrl-drawer-backdrop').style.display = 'block';
    document.getElementById('ctrl-drawer').style.display = 'block';

    _lastDrawerEventsHash = '';
    refreshDetailPanel(instanceId, type);

    if (App.detailPollTimer) clearInterval(App.detailPollTimer);
    App.detailPollTimer = setInterval(function() {
        if (App.selectedDetailId && App.currentPage === 'control') {
            refreshDetailPanel(App.selectedDetailId, App.selectedDetailType);
        }
    }, 2000);
}

function closeDrawer() {
    document.getElementById('ctrl-drawer-backdrop').style.display = 'none';
    document.getElementById('ctrl-drawer').style.display = 'none';
    App.selectedDetailId = null;
    App.selectedDetailType = null;
    App.selectedConvId = null;
    if (App.detailPollTimer) { clearInterval(App.detailPollTimer); App.detailPollTimer = null; }
    document.querySelectorAll('tr.clickable-row.selected').forEach(function(r) { r.classList.remove('selected'); });
    // Reset drawer view state
    var backBtn = document.getElementById('ctrl-drawer-back');
    var drawerContent = document.getElementById('ctrl-drawer-content');
    var convWrapper = document.getElementById('conv-detail-wrapper');
    var eventsWrapper = document.getElementById('ctrl-drawer-events-wrapper');
    if (backBtn) backBtn.style.display = 'none';
    if (drawerContent) drawerContent.style.display = 'block';
    if (convWrapper) convWrapper.style.display = 'none';
    if (eventsWrapper) eventsWrapper.style.display = 'block';
    _lastDrawerEventsHash = '';
}

function navigateToInstance(instanceId, instanceType) {
    showPage('control');
    setTimeout(function() { openDrawer(instanceId, instanceType); }, 100);
}

async function refreshDetailPanel(instanceId, type) {
    var titleEl = document.getElementById('ctrl-drawer-title');
    var contentEl = document.getElementById('ctrl-drawer-content');
    var eventsEl = document.getElementById('ctrl-drawer-events');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });

    if (titleEl && inst) titleEl.textContent = inst.label || instanceId;

    if (type === 'fsm') await renderFSMDetail(instanceId, contentEl);
    else if (type === 'workflow') await renderWorkflowDetail(instanceId, contentEl);
    else if (type === 'agent') await renderAgentDetail(instanceId, contentEl);

    await refreshDetailEvents(instanceId, eventsEl);
}

async function refreshDetailEvents(instanceId, logEl) {
    if (!logEl) return;
    try {
        var events = await fetchJson('/api/instances/' + encodeURIComponent(instanceId) + '/events?limit=50');

        // Change detection — skip re-render if events haven't changed
        var evHash = events.length + ':' + (events.length > 0 ? events[0].timestamp + events[events.length - 1].timestamp : '');
        if (evHash === _lastDrawerEventsHash) return;
        _lastDrawerEventsHash = evHash;

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

// --- FSM Detail (renders into provided container) ---

async function renderFSMDetail(instanceId, contentEl) {
    if (!contentEl) contentEl = document.getElementById('ctrl-fsm-detail-content');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    try {
        var convs = await fetchJson('/api/fsm/' + encodeURIComponent(instanceId) + '/conversations');

        var html = '<div class="kv detail-kv">';
        html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
        html += '<span class="key">Source:</span><span class="val">' + esc(inst.source || 'custom') + '</span>';
        html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
        html += '</div>';

        html += '<div class="panel-title">Conversations (' + convs.length + ')</div>';
        if (convs.length === 0 || (convs.length === 1 && convs[0].error)) {
            html += '<div class="empty-state"><div class="empty-hint">No active conversations.</div></div>';
            if (inst.status === 'running') {
                html += '<button class="btn btn-primary btn-sm mt-4" onclick="startConversationOn(\'' + esc(instanceId) + '\')">Start Conversation</button>';
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
                html += '<button class="btn btn-sm mt-4" onclick="startConversationOn(\'' + esc(instanceId) + '\')">+ New Conversation</button>';
            }
        }
        contentEl.innerHTML = html;
    } catch (e) {
        contentEl.innerHTML = '<span class="error-message">Failed to load FSM detail</span>';
    }
}

function goToConversation(instanceId, convId) {
    showConversationInDrawer(instanceId, convId);
}

// --- Workflow Detail ---

async function renderWorkflowDetail(instanceId, contentEl) {
    if (!contentEl) contentEl = document.getElementById('ctrl-wf-detail-content');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    var html = '<div class="kv detail-kv">';
    html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
    html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
    html += '<span class="key">Active Workflows:</span><span class="val">' + (inst.active_workflows || 0) + '</span>';
    html += '</div>';

    try {
        var wfInstances = await fetchJson('/api/workflow/' + encodeURIComponent(instanceId) + '/instances');

        html += '<div class="panel-title">Workflow Instances (' + wfInstances.length + ')</div>';
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
        html += '<div class="panel-title">Workflow Instances</div>';
        html += '<div class="empty-state"><div class="empty-hint">Could not load workflow instances.</div></div>';
    }

    contentEl.innerHTML = html;
}

// --- Agent Detail ---

function _captureTraceState(contentEl) {
    var expanded = [];
    if (!contentEl) return expanded;
    contentEl.querySelectorAll('.trace-step .step-body').forEach(function(el, idx) {
        if (el.style.display === 'block') expanded.push(idx);
    });
    return expanded;
}

function _restoreTraceState(contentEl, expanded) {
    if (!contentEl || !expanded.length) return;
    var bodies = contentEl.querySelectorAll('.trace-step .step-body');
    for (var i = 0; i < expanded.length; i++) {
        if (bodies[expanded[i]]) bodies[expanded[i]].style.display = 'block';
    }
}

async function renderAgentDetail(instanceId, contentEl) {
    if (!contentEl) contentEl = document.getElementById('ctrl-agent-detail-content');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    // Capture expanded trace steps before re-render
    var expandedSteps = _captureTraceState(contentEl);

    try {
        var data = await fetchJson('/api/agent/' + encodeURIComponent(instanceId) + '/status');
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
            html += '<div class="panel-title">Answer</div>';
            html += '<div class="event-log event-log-compact">';
            html += '<div class="entry"><span class="msg pre-wrap">' + esc(data.answer) + '</span></div>';
            html += '</div>';
        }

        if (data.error) {
            html += '<div class="panel-title">Error</div>';
            html += '<div class="error-message">' + esc(data.error) + '</div>';
        }

        if (data.status !== 'running') {
            await _renderAgentTrace(instanceId, html, contentEl, data, expandedSteps);
            return;
        }

        if (data.tools_used && data.tools_used.length > 0) {
            html += _renderToolCalls(data.tools_used);
        }

        if (data.success !== undefined && data.status !== 'running') {
            html += renderResultBanner(data.success);
        }

        contentEl.innerHTML = html;
        _restoreTraceState(contentEl, expandedSteps);
    } catch (e) {
        contentEl.innerHTML = '<span class="error-message">Failed to load agent detail</span>';
    }
}

function _renderToolCalls(toolsUsed) {
    var html = '<div class="panel-title">Tool Calls (' + toolsUsed.length + ')</div>';
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

async function _renderAgentTrace(instanceId, html, contentEl, statusData, expandedSteps) {
    try {
        var result = await fetchJson('/api/agent/' + encodeURIComponent(instanceId) + '/result');
        if (result.trace_steps && result.trace_steps.length > 0) {
            html += '<div class="panel-title panel-title-flex">Execution Trace (' + result.trace_steps.length + ' steps)';
            html += '<div class="flex-row-gap-4">';
            html += '<button class="btn btn-sm" onclick="event.stopPropagation();toggleAllTraceSteps(true)">Expand</button>';
            html += '<button class="btn btn-sm" onclick="event.stopPropagation();toggleAllTraceSteps(false)">Collapse</button>';
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
    _restoreTraceState(contentEl, expandedSteps || []);
}

function updateRunningAgents(updates) {
    if (App.selectedDetailType === 'agent' && App.selectedDetailId && updates[App.selectedDetailId]) {
        var contentEl = document.getElementById('ctrl-drawer-content');
        if (contentEl) renderAgentDetail(App.selectedDetailId, contentEl);
    }
}

// --- Actions ---

async function startConversationOn(instanceId) {
    try {
        var data = await fetchJson('/api/fsm/' + encodeURIComponent(instanceId) + '/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_context: {} })
        });
        refreshInstances();
        if (data.conversation_id) {
            showConversationInDrawer(instanceId, data.conversation_id);
        }
    } catch (e) {
        console.error('startConversationOn:', e);
    }
}

async function destroyInstance(instanceId) {
    if (!confirm('Destroy this instance? This cannot be undone.')) return;
    try {
        await fetchJson('/api/instances/' + encodeURIComponent(instanceId), { method: 'DELETE' });
        if (App.selectedDetailId === instanceId) {
            closeDrawer();
        }
        _lastCtrlHash = '';
        refreshInstances();
        if (App.currentPage === 'control') refreshControlCenter();
    } catch (e) {
        console.error('destroyInstance:', e);
    }
}

async function cancelAgent(instanceId) {
    if (!confirm('Cancel this agent? It will stop execution.')) return;
    try {
        await fetchJson('/api/agent/' + encodeURIComponent(instanceId) + '/cancel', { method: 'POST' });
        if (App.selectedDetailId === instanceId) {
            var contentEl = document.getElementById('ctrl-drawer-content');
            if (contentEl) renderAgentDetail(instanceId, contentEl);
        }
        refreshControlCenter();
    } catch (e) {
        console.error('cancelAgent:', e);
    }
}
