// FSM-LLM Monitor — Control Center Page (Unified View)

import { state, scheduleRefresh } from '../services/state.js';
import { fetchJson, postJson } from '../services/api.js';
import { $, esc, statusBadge, hashInstances, showToast, renderResultBanner } from '../utils/dom.js';
import { formatTime } from '../utils/format.js';
import { showConversationInDrawer } from './conversations.js';
import { refreshInstances } from './dashboard.js';

// Forward references
let _showPage;
export function setDeps(deps) { _showPage = deps.showPage; }

// --- State ---
let _ctrlFilter = 'all';
let _ctrlPage = 0;
const _ctrlPerPage = 25;
let _ctrlSearch = '';
let _lastCtrlHash = '';
let _lastDrawerEventsHash = '';

// --- Filter & Search ---

export function filterControlInstances(filter, btn) {
    _ctrlFilter = filter;
    _ctrlPage = 0;
    _lastCtrlHash = '';
    document.querySelectorAll('#ctrl-filter-chips .filter-chip').forEach(c => c.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderUnifiedTable();
}

export function onCtrlSearchInput() {
    _ctrlSearch = $('ctrl-search')?.value || '';
    _ctrlPage = 0;
    _lastCtrlHash = '';
    scheduleRefresh('ctrl-search', renderUnifiedTable, 250);
}

export async function refreshControlCenter() {
    await refreshInstances();
    renderUnifiedTable();
    if (state.selectedDetailId) {
        refreshDetailPanel(state.selectedDetailId, state.selectedDetailType);
    }
}

function _getFilteredCtrlItems() {
    let items = state.instances;
    if (_ctrlFilter !== 'all') items = items.filter(i => i.instance_type === _ctrlFilter);
    if (_ctrlSearch) {
        const q = _ctrlSearch.toLowerCase();
        items = items.filter(i =>
            (i.label || i.instance_id).toLowerCase().includes(q) ||
            i.instance_type.toLowerCase().includes(q) ||
            (i.agent_type || '').toLowerCase().includes(q)
        );
    }
    return items;
}

// --- Unified Table ---

export function renderUnifiedTable() {
    const body = $('ctrl-unified-body');
    const empty = $('ctrl-unified-empty');
    const paginationEl = $('ctrl-pagination');
    const countEl = $('ctrl-table-count');
    if (!body) return;

    const items = _getFilteredCtrlItems();
    const hash = hashInstances(items);
    if (hash === _lastCtrlHash) return;
    _lastCtrlHash = hash;

    if (countEl) countEl.textContent = items.length;
    _updateFilterChipCounts();

    if (items.length === 0) {
        body.innerHTML = '';
        if (empty) empty.style.display = 'block';
        if (paginationEl) paginationEl.style.display = 'none';
        return;
    }
    if (empty) empty.style.display = 'none';

    const totalPages = Math.ceil(items.length / _ctrlPerPage);
    _ctrlPage = Math.max(0, Math.min(_ctrlPage, totalPages - 1));
    const start = _ctrlPage * _ctrlPerPage;
    const pageItems = items.slice(start, start + _ctrlPerPage);

    let rows = '';
    for (const inst of pageItems) {
        const sel = (state.selectedDetailId === inst.instance_id) ? ' selected' : '';
        const typeDot = '<span class="type-dot type-' + esc(inst.instance_type) + '"></span>';

        let detail = '';
        if (inst.instance_type === 'fsm') {
            detail = (inst.conversation_count || 0) + ' conv' + ((inst.conversation_count || 0) !== 1 ? 's' : '');
            if (inst.source) detail += ' &middot; ' + esc(inst.source);
        } else if (inst.instance_type === 'agent') {
            detail = esc(inst.agent_type || '');
            if (inst.task) detail += ' &middot; ' + esc((inst.task || '').substring(0, 40));
        } else if (inst.instance_type === 'workflow') {
            detail = (inst.active_workflows || 0) + ' active';
        }

        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(inst.instance_id) + '" data-action="open-drawer" data-instance-type="' + esc(inst.instance_type) + '">';
        rows += '<td>' + typeDot + esc(inst.instance_type) + '</td>';
        rows += '<td>' + esc(inst.label || inst.instance_id) + '</td>';
        rows += '<td class="cell-truncate text-dim">' + detail + '</td>';
        rows += '<td>' + statusBadge(inst.status) + '</td>';
        rows += '<td>';
        if (inst.instance_type === 'fsm' && inst.status === 'running') {
            rows += '<button class="btn btn-sm" data-action="start-conv" data-instance-id="' + esc(inst.instance_id) + '">+ Conv</button> ';
        }
        if (inst.instance_type === 'agent' && inst.status === 'running') {
            rows += '<button class="btn btn-sm btn-warning" data-action="cancel-agent" data-instance-id="' + esc(inst.instance_id) + '">Cancel</button> ';
        }
        rows += '<button class="btn btn-sm btn-danger" data-action="destroy-instance" data-instance-id="' + esc(inst.instance_id) + '">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;

    if (paginationEl) {
        if (totalPages <= 1) {
            paginationEl.style.display = 'none';
        } else {
            paginationEl.style.display = 'flex';
            paginationEl.innerHTML =
                '<button class="btn btn-sm" data-action="ctrl-page-prev"' + (_ctrlPage === 0 ? ' disabled' : '') + '>&laquo; Prev</button>' +
                '<span class="pagination-info">Page ' + (_ctrlPage + 1) + ' of ' + totalPages + '</span>' +
                '<button class="btn btn-sm" data-action="ctrl-page-next"' + (_ctrlPage >= totalPages - 1 ? ' disabled' : '') + '>Next &raquo;</button>';
        }
    }
}

export function ctrlPagePrev() {
    if (_ctrlPage > 0) { _ctrlPage--; _lastCtrlHash = ''; renderUnifiedTable(); }
}

export function ctrlPageNext() {
    const totalPages = Math.ceil(_getFilteredCtrlItems().length / _ctrlPerPage);
    if (_ctrlPage < totalPages - 1) { _ctrlPage++; _lastCtrlHash = ''; renderUnifiedTable(); }
}

function _updateFilterChipCounts() {
    const counts = { all: 0, fsm: 0, workflow: 0, agent: 0 };
    for (const inst of state.instances) {
        counts.all++;
        if (counts[inst.instance_type] !== undefined) counts[inst.instance_type]++;
    }
    document.querySelectorAll('#ctrl-filter-chips .filter-chip').forEach(chip => {
        const f = chip.getAttribute('data-filter');
        const n = counts[f];
        if (n !== undefined) {
            const label = f === 'all' ? 'All' : f === 'fsm' ? 'FSMs' : f === 'workflow' ? 'Workflows' : 'Agents';
            chip.textContent = label + ' (' + n + ')';
        }
    });
}

// --- Drawer ---

export function openDrawer(instanceId, type) {
    document.querySelectorAll('tr.clickable-row.selected').forEach(r => r.classList.remove('selected'));
    const row = document.querySelector('tr[data-instance-id="' + instanceId + '"]');
    if (row) row.classList.add('selected');

    state.selectedDetailId = instanceId;
    state.selectedDetailType = type;

    $('ctrl-drawer-backdrop').style.display = 'block';
    $('ctrl-drawer').style.display = 'block';

    _lastDrawerEventsHash = '';
    refreshDetailPanel(instanceId, type);

    if (state.detailPollTimer) clearInterval(state.detailPollTimer);
    state.detailPollTimer = setInterval(() => {
        if (state.selectedDetailId && state.currentPage === 'control') {
            refreshDetailPanel(state.selectedDetailId, state.selectedDetailType);
        }
    }, 2000);
}

export function closeDrawer() {
    $('ctrl-drawer-backdrop').style.display = 'none';
    $('ctrl-drawer').style.display = 'none';
    state.selectedDetailId = null;
    state.selectedDetailType = null;
    state.selectedConvId = null;
    state._lastContextData = null;
    if (state.detailPollTimer) { clearInterval(state.detailPollTimer); state.detailPollTimer = null; }
    document.querySelectorAll('tr.clickable-row.selected').forEach(r => r.classList.remove('selected'));

    const backBtn = $('ctrl-drawer-back');
    const drawerContent = $('ctrl-drawer-content');
    const convWrapper = $('conv-detail-wrapper');
    const eventsWrapper = $('ctrl-drawer-events-wrapper');
    if (backBtn) backBtn.style.display = 'none';
    if (drawerContent) drawerContent.style.display = 'block';
    if (convWrapper) convWrapper.style.display = 'none';
    if (eventsWrapper) eventsWrapper.style.display = 'block';
    _lastDrawerEventsHash = '';
}

export function navigateToInstance(instanceId, instanceType) {
    _showPage?.('control');
    setTimeout(() => openDrawer(instanceId, instanceType), 100);
}

export async function refreshDetailPanel(instanceId, type) {
    const titleEl = $('ctrl-drawer-title');
    const contentEl = $('ctrl-drawer-content');
    const eventsEl = $('ctrl-drawer-events');
    const inst = state.instances.find(i => i.instance_id === instanceId);

    if (!inst) { closeDrawer(); return; }
    if (titleEl) titleEl.textContent = inst.label || instanceId;

    if (type === 'fsm') await renderFSMDetail(instanceId, contentEl);
    else if (type === 'workflow') await renderWorkflowDetail(instanceId, contentEl);
    else if (type === 'agent') await renderAgentDetail(instanceId, contentEl);

    await refreshDetailEvents(instanceId, eventsEl);
}

function _enrichAgentEvents(events) {
    // Convert raw FSM events to meaningful agent events
    const enriched = [];
    let iterCount = 0;
    for (const e of events) {
        if (e.event_type === 'state_transition') {
            const target = e.target_state || '';
            const source = e.source_state || '';
            if (target === 'think') {
                iterCount++;
                enriched.push({
                    ...e,
                    event_type: 'iteration',
                    message: 'Iteration ' + iterCount + ' started — thinking',
                    level: 'INFO'
                });
            } else if (target === 'act') {
                const tool = e.data?.tool_name || '';
                enriched.push({
                    ...e,
                    event_type: 'tool_call',
                    message: tool ? 'Calling tool: ' + tool : 'Executing action',
                    level: 'INFO'
                });
            } else if (target === 'conclude') {
                enriched.push({
                    ...e,
                    event_type: 'conclude',
                    message: 'Generating final answer',
                    level: 'INFO'
                });
            } else if (target && target !== source) {
                enriched.push({
                    ...e,
                    event_type: 'transition',
                    message: source + ' \u2192 ' + target,
                    level: 'INFO'
                });
            }
        } else if (e.event_type === 'conversation_start') {
            enriched.push({ ...e, event_type: 'agent_init', message: 'Agent FSM initialized' });
        } else if (e.event_type === 'conversation_end') {
            enriched.push({ ...e, event_type: 'agent_done', message: 'Agent execution completed' });
        } else if (e.event_type === 'error') {
            enriched.push(e);
        } else if (e.event_type === 'context_update') {
            enriched.push({ ...e, event_type: 'context', message: 'Context updated' });
        }
        // Skip pre_processing/post_processing — noise for agents
    }
    return enriched;
}

async function refreshDetailEvents(instanceId, logEl) {
    if (!logEl) return;
    try {
        const events = await fetchJson('/api/instances/' + encodeURIComponent(instanceId) + '/events?limit=100');
        const evHash = events.length + ':' + (events.length > 0 ? events[0].timestamp + events[events.length - 1].timestamp : '');
        if (evHash === _lastDrawerEventsHash) return;
        _lastDrawerEventsHash = evHash;

        if (events.length === 0) {
            logEl.innerHTML = '<div class="empty-state"><div class="empty-hint">No events yet...</div></div>';
            return;
        }

        // For agents, enrich events to show meaningful info
        const instType = state.selectedDetailType;
        const displayEvents = (instType === 'agent')
            ? _enrichAgentEvents(events)
            : events;

        if (displayEvents.length === 0) {
            logEl.innerHTML = '<div class="empty-state"><div class="empty-hint">Agent initializing...</div></div>';
            return;
        }

        let html = '';
        for (const e of displayEvents) {
            const level = (e.level || 'INFO').toLowerCase();
            const typeClass = e.event_type === 'iteration' ? ' transition' : e.event_type === 'tool_call' ? ' conversation' : e.event_type === 'error' ? ' error' : '';
            html += '<div class="entry ' + level + typeClass + '">';
            html += '<span class="ts">' + formatTime(e.timestamp) + '</span>';
            html += '<span class="type">' + esc(e.event_type) + '</span>';
            html += '<span class="msg">' + esc(e.message) + '</span>';
            html += '</div>';
        }
        logEl.innerHTML = html;
    } catch (e) {
        console.error('refreshDetailEvents:', e);
        showToast('Failed to load events', 'error');
    }
}

// --- FSM Detail ---

async function renderFSMDetail(instanceId, contentEl) {
    const inst = state.instances.find(i => i.instance_id === instanceId);
    if (!inst) return;
    try {
        const convs = await fetchJson('/api/fsm/' + encodeURIComponent(instanceId) + '/conversations');
        let html = '<div class="kv detail-kv">';
        html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
        html += '<span class="key">Source:</span><span class="val">' + esc(inst.source || 'custom') + '</span>';
        html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
        html += '</div>';
        html += '<div class="panel-title">Conversations (' + convs.length + ')</div>';
        if (convs.length === 0 || (convs.length === 1 && convs[0].error)) {
            html += '<div class="empty-state"><div class="empty-hint">No active conversations.</div></div>';
            if (inst.status === 'running') {
                html += '<button class="btn btn-primary btn-sm mt-4" data-action="start-conv" data-instance-id="' + esc(instanceId) + '">Start Conversation</button>';
            }
        } else {
            for (const c of convs) {
                if (c.error) continue;
                html += '<div class="conv-card" data-action="go-to-conv" data-instance-id="' + esc(instanceId) + '" data-conv-id="' + esc(c.conversation_id) + '">';
                html += '<div class="conv-info">';
                html += '<span class="mono-id">' + esc(c.conversation_id.substring(0, 12)) + '</span>';
                html += '<span class="conv-state">' + esc(c.current_state) + '</span>';
                html += '<span class="text-dim">' + (c.message_history ? c.message_history.length : 0) + ' msgs</span>';
                html += '</div>';
                html += '<div>' + (c.is_terminal ? statusBadge('ended') : statusBadge('active')) + '</div>';
                html += '</div>';
            }
            if (inst.status === 'running') {
                html += '<button class="btn btn-sm mt-4" data-action="start-conv" data-instance-id="' + esc(instanceId) + '">+ New Conversation</button>';
            }
        }
        contentEl.innerHTML = html;
    } catch {
        contentEl.innerHTML = '<span class="error-message">Failed to load FSM detail</span>';
    }
}

// --- Workflow Detail ---

async function renderWorkflowDetail(instanceId, contentEl) {
    const inst = state.instances.find(i => i.instance_id === instanceId);
    if (!inst) return;
    let html = '<div class="kv detail-kv">';
    html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
    html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
    html += '<span class="key">Active Workflows:</span><span class="val">' + (inst.active_workflows || 0) + '</span>';
    html += '</div>';
    try {
        const wfInstances = await fetchJson('/api/workflow/' + encodeURIComponent(instanceId) + '/instances');
        html += '<div class="panel-title">Workflow Instances (' + wfInstances.length + ')</div>';
        if (wfInstances.length === 0) {
            html += '<div class="empty-state"><div class="empty-hint">No workflow instances.</div></div>';
        } else {
            for (const wf of wfInstances) {
                if (wf.error) continue;
                const wfId = wf.workflow_instance_id || '';
                const wfStatus = (wf.status || 'unknown').toLowerCase();
                const isActive = wfStatus === 'running' || wfStatus === 'pending' || wfStatus === 'active';

                html += '<div class="conv-card">';
                html += '<div class="conv-info">';
                html += '<span class="mono-id">' + esc(wfId.substring(0, 12)) + '</span>';
                if (wf.current_step) html += '<span class="conv-state">' + esc(wf.current_step) + '</span>';
                html += '<div>' + statusBadge(wf.status || 'unknown') + '</div>';
                html += '</div>';

                // Action buttons for active workflows
                if (isActive) {
                    html += '<div class="flex-row-gap-4 mt-4">';
                    html += '<button class="btn btn-sm btn-primary" data-action="advance-workflow" data-instance-id="' + esc(instanceId) + '" data-wf-instance-id="' + esc(wfId) + '">Advance</button>';
                    html += '<button class="btn btn-sm btn-warning" data-action="cancel-workflow" data-instance-id="' + esc(instanceId) + '" data-wf-instance-id="' + esc(wfId) + '">Cancel</button>';
                    html += '</div>';
                }

                // Step history as conversation view
                if (wf.history?.length > 0) {
                    html += '<div class="panel-title panel-title-spaced" style="font-size:0.8rem;">Step History</div>';
                    html += '<div class="chat-container" style="max-height:300px;overflow-y:auto;">';
                    for (const step of wf.history) {
                        const hasError = step.error;
                        const borderColor = hasError ? 'var(--danger)' : 'var(--info)';
                        html += '<div class="chat-bubble assistant" style="background:var(--surface-alt,#1a1c24);border-left:3px solid ' + borderColor + ';">';
                        html += '<div class="chat-role-tag" style="color:' + borderColor + ';">' + esc(step.step_id || 'step') + '</div>';
                        if (step.message) html += '<div>' + esc(step.message) + '</div>';
                        if (step.data && typeof step.data === 'object') {
                            const dataStr = JSON.stringify(step.data, null, 1);
                            if (dataStr.length > 2) {
                                html += '<div style="margin-top:0.25rem;font-size:0.8rem;" class="text-dim">' + esc(dataStr.substring(0, 300)) + '</div>';
                            }
                        }
                        if (hasError) html += '<div class="error-message" style="margin-top:0.25rem;">' + esc(step.error) + '</div>';
                        if (step.timestamp) html += '<div class="text-dim" style="font-size:0.7rem;margin-top:0.25rem;">' + formatTime(step.timestamp) + '</div>';
                        html += '</div>';
                    }
                    html += '</div>';
                }

                // Context data display (collapsible)
                if (wf.context && Object.keys(wf.context).length > 0) {
                    const ctxKeys = Object.keys(wf.context).filter(k => !k.startsWith('_'));
                    if (ctxKeys.length > 0) {
                        html += '<details style="margin-top:0.5rem;"><summary class="panel-title" style="font-size:0.8rem;cursor:pointer;">Context Data (' + ctxKeys.length + ' keys)</summary>';
                        html += '<div class="kv">';
                        for (const k of ctxKeys) {
                            const v = wf.context[k];
                            html += '<span class="key">' + esc(k) + ':</span><span class="val">' + esc(typeof v === 'object' ? JSON.stringify(v) : String(v)) + '</span>';
                        }
                        html += '</div></details>';
                    }
                }

                // Timestamps
                if (wf.created_at || wf.updated_at) {
                    html += '<div class="text-dim" style="font-size:0.75rem;margin-top:0.5rem;">';
                    if (wf.created_at) html += 'Created: ' + formatTime(wf.created_at);
                    if (wf.updated_at) html += ' &middot; Updated: ' + formatTime(wf.updated_at);
                    html += '</div>';
                }

                html += '</div>';
            }
        }
    } catch {
        html += '<div class="panel-title">Workflow Instances</div>';
        html += '<div class="empty-state"><div class="empty-hint">Could not load workflow instances.</div></div>';
    }
    contentEl.innerHTML = html;
}

export async function advanceWorkflow(instanceId, wfInstanceId) {
    try {
        await postJson('/api/workflow/' + encodeURIComponent(instanceId) + '/advance', {
            workflow_instance_id: wfInstanceId,
            user_input: ''
        });
        showToast('Workflow advanced', 'success');
        if (state.selectedDetailId === instanceId) {
            const contentEl = $('ctrl-drawer-content');
            if (contentEl) renderWorkflowDetail(instanceId, contentEl);
        }
    } catch (e) {
        console.error('advanceWorkflow:', e);
        showToast('Failed to advance workflow: ' + e.message, 'error');
    }
}

export async function cancelWorkflow(instanceId, wfInstanceId) {
    if (!confirm('Cancel this workflow instance?')) return;
    try {
        await postJson('/api/workflow/' + encodeURIComponent(instanceId) + '/cancel', {
            workflow_instance_id: wfInstanceId,
            reason: 'Cancelled from monitor'
        });
        showToast('Workflow cancelled', 'success');
        if (state.selectedDetailId === instanceId) {
            const contentEl = $('ctrl-drawer-content');
            if (contentEl) renderWorkflowDetail(instanceId, contentEl);
        }
    } catch (e) {
        console.error('cancelWorkflow:', e);
        showToast('Failed to cancel workflow: ' + e.message, 'error');
    }
}

// --- Agent Detail ---

function _captureTraceState(contentEl) {
    const expanded = [];
    contentEl?.querySelectorAll('.trace-step .step-body').forEach((el, idx) => {
        if (el.style.display === 'block') expanded.push(idx);
    });
    return expanded;
}

function _restoreTraceState(contentEl, expanded) {
    if (!contentEl || !expanded.length) return;
    const bodies = contentEl.querySelectorAll('.trace-step .step-body');
    for (const i of expanded) {
        if (bodies[i]) bodies[i].style.display = 'block';
    }
}

function _extractTaskText(raw) {
    if (!raw) return '';
    // If it looks like JSON config, try to extract a meaningful description
    const trimmed = raw.trim();
    if (trimmed.startsWith('{')) {
        try {
            const obj = JSON.parse(trimmed);
            // Common fields in agent definitions
            if (obj.description) return obj.description;
            if (obj.task) return obj.task;
            if (obj.name) return obj.name;
        } catch { /* not valid JSON, show as-is */ }
    }
    return raw;
}

// State → display config for agent conversation rendering
const _STATE_STYLES = {
    think:    { label: 'Think',    color: 'var(--primary)', icon: '\u{1F4AD}' },
    act:      { label: 'Act',      color: 'var(--warning)', icon: '\u{26A1}' },
    observe:  { label: 'Observe',  color: 'var(--text)',    icon: '\u{1F441}' },
    conclude: { label: 'Answer',   color: 'var(--success)', icon: '\u2714' },
};

// Keys to highlight prominently (shown first, with labels)
const _KEY_DISPLAY = {
    reasoning:            'Reasoning',
    tool_name:            'Tool',
    tool_input:           'Input',
    tool_result:          'Result',
    final_answer:         'Answer',
    observations:         'Observations',
    evaluation_feedback:  'Feedback',
    reflection:           'Reflection',
    plan_steps:           'Plan',
    aggregated_answer:    'Answer',
    confidence:           'Confidence',
};

function _renderAgentConversation(log) {
    let html = '<div class="panel-title panel-title-spaced">Conversation</div>';
    html += '<div class="chat-container" style="max-height:500px;overflow-y:auto;">';
    let iteration = 0;
    for (const entry of log) {
        if (entry.type === 'transition') {
            const target = entry.target || '';
            if (target === 'think') {
                iteration++;
                html += '<div style="text-align:center;padding:0.25rem 0;font-size:0.7rem;">';
                html += '<span class="text-dim">\u2500\u2500 Iteration ' + iteration + ' \u2500\u2500</span></div>';
            }
            continue;
        }
        if (entry.type !== 'state_output' || !entry.data) continue;

        const st = entry.state || '';
        const style = _STATE_STYLES[st] || { label: st, color: 'var(--text)', icon: '\u2022' };
        const d = entry.data;

        html += '<div class="chat-bubble assistant" style="background:var(--surface-alt,#1a1c24);border-left:3px solid ' + style.color + ';">';
        html += '<div class="chat-role-tag" style="color:' + style.color + ';">' + style.icon + ' ' + esc(style.label) + '</div>';

        // Render known keys first with labels
        let rendered = new Set();
        for (const [key, label] of Object.entries(_KEY_DISPLAY)) {
            if (!d[key]) continue;
            rendered.add(key);
            const val = d[key];
            if (key === 'tool_name') {
                html += '<div><span class="text-dim">' + label + ':</span> <b>' + esc(val) + '</b>';
                if (d.tool_input) { html += ' &mdash; ' + esc(d.tool_input); rendered.add('tool_input'); }
                html += '</div>';
            } else {
                html += '<div style="margin-top:0.25rem;"><span class="text-dim">' + label + ':</span></div>';
                html += '<div style="white-space:pre-wrap;margin-left:0.5rem;">' + esc(val) + '</div>';
            }
        }

        // Render remaining keys (captures non-standard agent patterns)
        for (const [key, val] of Object.entries(d)) {
            if (rendered.has(key)) continue;
            html += '<div style="margin-top:0.25rem;"><span class="text-dim">' + esc(key) + ':</span> ' + esc(val) + '</div>';
        }

        html += '</div>';
    }
    if (log.length === 0) {
        html += '<div class="empty-state"><div class="empty-hint">Waiting for agent output...</div></div>';
    }
    html += '</div>';
    return html;
}

async function renderAgentDetail(instanceId, contentEl) {
    const inst = state.instances.find(i => i.instance_id === instanceId);
    if (!inst) return;
    const expandedSteps = _captureTraceState(contentEl);

    try {
        const data = await fetchJson('/api/agent/' + encodeURIComponent(instanceId) + '/status');
        if (data.error && !data.status) {
            contentEl.innerHTML = '<span class="error-message">' + esc(data.error) + '</span>';
            return;
        }

        const taskText = _extractTaskText(data.task);

        let html = '<div class="kv detail-kv">';
        html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
        html += '<span class="key">Agent Type:</span><span class="val">' + esc(data.agent_type || '') + '</span>';
        html += '<span class="key">Status:</span><span class="val">' + statusBadge(data.status) + '</span>';
        html += '<span class="key">Task:</span><span class="val word-break">' + esc(taskText) + '</span>';
        if (data.total_iterations !== undefined) {
            html += '<span class="key">Iterations:</span><span class="val">' + data.total_iterations + '</span>';
        }
        html += '</div>';

        if (data.status === 'running') {
            const iterCount = data.iteration_count || 0;
            const maxIter = data.max_iterations || 10;
            const pct = Math.min(Math.round((iterCount / maxIter) * 100), 95);
            const stateLabel = data.current_state || 'initializing';
            const stateClass = stateLabel === 'think' ? 'state-think' : stateLabel === 'act' ? 'state-act' : stateLabel === 'conclude' ? 'state-conclude' : 'state-default';
            html += '<div class="agent-progress"><div class="agent-progress-header">';
            html += '<span>Iteration <b>' + iterCount + '</b></span>';
            html += '<span class="' + stateClass + '">' + esc(stateLabel.toUpperCase()) + '</span>';
            html += '</div><div class="progress-bar"><div class="progress-fill" style="width:' + pct + '%;"></div></div>';
            if (data.last_tool_call) html += '<div class="agent-tool-info">Last tool: ' + esc(data.last_tool_call) + '</div>';
            html += '</div>';
        }

        // Render live conversation log
        if (data.conversation_log?.length > 0) {
            html += _renderAgentConversation(data.conversation_log);
        }

        if (data.answer) {
            html += '<div class="panel-title">Answer</div>';
            html += '<div class="event-log event-log-compact"><div class="entry"><span class="msg pre-wrap">' + esc(data.answer) + '</span></div></div>';
        }
        if (data.error) {
            html += '<div class="panel-title">Error</div><div class="error-message">' + esc(data.error) + '</div>';
        }

        if (data.status !== 'running') {
            await _renderAgentTrace(instanceId, html, contentEl, data, expandedSteps);
            return;
        }

        if (data.tools_used?.length) html += _renderToolCalls(data.tools_used);
        if (data.success !== undefined && data.status !== 'running') html += renderResultBanner(data.success);

        contentEl.innerHTML = html;
        _restoreTraceState(contentEl, expandedSteps);
    } catch {
        contentEl.innerHTML = '<span class="error-message">Failed to load agent detail</span>';
    }
}

function _renderToolCalls(toolsUsed) {
    let html = '<div class="panel-title">Tool Calls (' + toolsUsed.length + ')</div>';
    for (const tc of toolsUsed) {
        html += '<div class="trace-step step-act">';
        html += '<div class="step-header"><span class="step-label">' + esc(tc.tool_name) + '</span></div>';
        html += '<div class="step-body d-block">' + esc(JSON.stringify(tc.parameters || {}, null, 1)) + '</div>';
        html += '</div>';
    }
    return html;
}

export function toggleAllTraceSteps(expand) {
    document.querySelectorAll('.trace-step .step-body').forEach(el => {
        el.style.display = expand ? 'block' : 'none';
    });
}

async function _renderAgentTrace(instanceId, html, contentEl, statusData, expandedSteps) {
    try {
        const result = await fetchJson('/api/agent/' + encodeURIComponent(instanceId) + '/result');
        if (result.trace_steps?.length) {
            html += '<div class="panel-title panel-title-flex">Execution Trace (' + result.trace_steps.length + ' steps)';
            html += '<div class="flex-row-gap-4">';
            html += '<button class="btn btn-sm" data-action="expand-all-trace">Expand</button>';
            html += '<button class="btn btn-sm" data-action="collapse-all-trace">Collapse</button>';
            html += '</div></div>';
            let iteration = 0;
            for (const step of result.trace_steps) {
                const st = step.state || '';
                if (st === 'think') iteration++;
                const stateColorClass = st === 'think' ? 'state-think' : st === 'act' ? 'state-act' : st === 'conclude' ? 'state-conclude' : 'state-default';
                const stepIcon = st === 'think' ? '&#9679;' : st === 'act' ? '&#9654;' : st === 'conclude' ? '&#10003;' : '&#8226;';
                html += '<div class="trace-step step-' + st + '" data-action="toggle-trace-step">';
                html += '<div class="step-header"><span class="' + stateColorClass + '">' + stepIcon + ' ' + esc(st.toUpperCase());
                if (st === 'think') html += ' #' + iteration;
                if (step.tool_name) html += ' &mdash; ' + esc(step.tool_name);
                html += '</span></div>';
                html += '<div class="step-body">';
                if (step.reasoning) html += '<div><b>Reasoning:</b> ' + esc(step.reasoning) + '</div>';
                if (step.tool_input) html += '<div><b>Input:</b> ' + esc(step.tool_input) + '</div>';
                if (step.tool_result) html += '<div><b>Result:</b> ' + esc(step.tool_result) + '</div>';
                html += '</div></div>';
            }
        } else if (statusData.tools_used?.length) {
            html += _renderToolCalls(statusData.tools_used);
        }
    } catch { /* trace fetch failed, skip */ }

    if (statusData.success !== undefined) html += renderResultBanner(statusData.success);
    contentEl.innerHTML = html;
    _restoreTraceState(contentEl, expandedSteps || []);
}

export function updateRunningAgents(updates) {
    if (state.selectedDetailType === 'agent' && state.selectedDetailId && updates[state.selectedDetailId]) {
        const contentEl = $('ctrl-drawer-content');
        if (contentEl) renderAgentDetail(state.selectedDetailId, contentEl);
    }
}

// --- Actions ---

export async function startConversationOn(instanceId) {
    try {
        const data = await postJson('/api/fsm/' + encodeURIComponent(instanceId) + '/start', { initial_context: {} });
        refreshInstances();
        if (data.conversation_id) showConversationInDrawer(instanceId, data.conversation_id);
    } catch (e) {
        console.error('startConversationOn:', e);
        showToast('Failed to start conversation: ' + e.message);
    }
}

export async function destroyInstance(instanceId) {
    if (!confirm('Destroy this instance? This cannot be undone.')) return;
    try {
        await fetchJson('/api/instances/' + encodeURIComponent(instanceId), { method: 'DELETE' });
        if (state.selectedDetailId === instanceId) closeDrawer();
        _lastCtrlHash = '';
        refreshInstances();
        if (state.currentPage === 'control') refreshControlCenter();
    } catch (e) {
        console.error('destroyInstance:', e);
        showToast('Failed to destroy instance: ' + e.message);
    }
}

export async function cancelAgent(instanceId) {
    if (!confirm('Cancel this agent? It will stop execution.')) return;
    try {
        await postJson('/api/agent/' + encodeURIComponent(instanceId) + '/cancel', {});
        if (state.selectedDetailId === instanceId) {
            const contentEl = $('ctrl-drawer-content');
            if (contentEl) renderAgentDetail(instanceId, contentEl);
        }
        refreshControlCenter();
    } catch (e) {
        console.error('cancelAgent:', e);
        showToast('Failed to cancel agent: ' + e.message);
    }
}
