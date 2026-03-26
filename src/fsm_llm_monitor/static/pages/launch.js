// FSM-LLM Monitor — Launch Modal

import { state } from '../services/state.js';
import { fetchJson, postJson } from '../services/api.js';
import { $, esc, numVal, intVal, showError, showStatus } from '../utils/dom.js';

// Forward references (set by app.js)
let _showPage, _refreshInstances, _showConversationInDrawer;

export function setDeps(deps) {
    _showPage = deps.showPage;
    _refreshInstances = deps.refreshInstances;
    _showConversationInDrawer = deps.showConversationInDrawer;
}

export function showLaunchModal() {
    $('launch-modal').style.display = 'flex';
    loadLaunchPresets();
    checkCapabilities();
    onAgentTypeChange();
    _clearLaunchStatuses();
}

export function closeLaunchModal() {
    $('launch-modal').style.display = 'none';
}

function _clearLaunchStatuses() {
    for (const id of ['launch-fsm-status', 'launch-wf-status', 'launch-agent-status']) {
        const el = $(id);
        if (el) el.innerHTML = '';
    }
}

async function checkCapabilities() {
    try {
        state.capabilities = await fetchJson('/api/capabilities');
        const wfUnavail = $('launch-wf-unavailable');
        const agentUnavail = $('launch-agent-unavailable');
        if (wfUnavail) wfUnavail.style.display = state.capabilities.workflows ? 'none' : 'block';
        if (agentUnavail) agentUnavail.style.display = state.capabilities.agents ? 'none' : 'block';
    } catch (e) {
        console.error('checkCapabilities:', e);
    }
}

export function toggleLaunchFSMSource() {
    const source = $('launch-fsm-source').value;
    $('launch-fsm-preset-section').style.display = source === 'preset' ? 'block' : 'none';
    $('launch-fsm-json-section').style.display = source === 'json' ? 'block' : 'none';
}

async function loadLaunchPresets() {
    if (state.presets) { renderLaunchPresets(state.presets); return; }
    try {
        state.presets = await fetchJson('/api/presets');
        renderLaunchPresets(state.presets);
    } catch (e) {
        console.error('loadLaunchPresets:', e);
    }
}

export function renderLaunchPresets(presets) {
    const items = presets.fsm || [];
    const container = $('launch-preset-list');
    if (!container) return;

    const categories = ['all'];
    for (const item of items) {
        const cat = item.category || 'other';
        if (!categories.includes(cat)) categories.push(cat);
    }

    let filterHtml = '<div class="preset-filters">';
    for (const c of categories) {
        filterHtml += '<button class="btn preset-filter-btn' + (c === 'all' ? ' btn-primary' : '') + '" data-cat="' + esc(c) + '" data-action="filter-presets">' + esc(c) + '</button>';
    }
    filterHtml += '</div>';
    container.innerHTML = filterHtml;

    const listDiv = document.createElement('div');
    listDiv.id = 'preset-items';
    listDiv.className = 'preset-items';
    container.appendChild(listDiv);

    for (const p of items) {
        const card = document.createElement('div');
        card.className = 'preset-card';
        card.setAttribute('data-category', p.category || 'other');
        card.setAttribute('data-action', 'select-preset');
        card.setAttribute('data-preset-id', p.id);
        card.setAttribute('data-preset-name', p.name);
        card.innerHTML = '<div class="preset-name">' + esc(p.name) + '</div><div class="preset-category">' + esc(p.category || '') + '</div><div class="preset-desc">' + esc(p.description || '') + '</div>';
        listDiv.appendChild(card);
    }
}

export function filterPresets(cat) {
    document.querySelectorAll('#preset-items .preset-card').forEach(card => {
        card.style.display = (cat === 'all' || card.getAttribute('data-category') === cat) ? '' : 'none';
    });
    document.querySelectorAll('.preset-filter-btn').forEach(btn => {
        btn.className = btn.getAttribute('data-cat') === cat ? 'btn preset-filter-btn btn-primary' : 'btn preset-filter-btn';
    });
}

export function selectPreset(card) {
    const id = card.getAttribute('data-preset-id');
    const name = card.getAttribute('data-preset-name');
    $('launch-fsm-preset-id').value = id;
    $('launch-preset-list')?.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    if (!$('launch-fsm-label').value) {
        $('launch-fsm-label').value = name.replace(/\s*\(.*\)/, '');
    }
}

export async function doLaunchFSM(btn) {
    const _resetBtn = () => { if (btn) { btn.disabled = false; btn.textContent = 'Launch FSM'; } };
    if (btn) { btn.disabled = true; btn.textContent = 'Launching...'; }

    const source = $('launch-fsm-source').value;
    const modelVal = $('launch-fsm-model').value.trim();
    const body = { temperature: numVal('launch-fsm-temp', 0.5), label: $('launch-fsm-label').value };
    if (modelVal) body.model = modelVal;

    if (source === 'preset') {
        body.preset_id = $('launch-fsm-preset-id').value;
        if (!body.preset_id) { showError('launch-fsm-status', 'Select a preset'); _resetBtn(); return; }
    } else {
        try { body.fsm_json = JSON.parse($('launch-fsm-json').value); }
        catch { showError('launch-fsm-status', 'Invalid JSON'); _resetBtn(); return; }
    }

    try {
        const data = await postJson('/api/fsm/launch', body);
        showStatus('launch-fsm-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        _refreshInstances?.();

        const startData = await postJson('/api/fsm/' + encodeURIComponent(data.instance_id) + '/start', { initial_context: {} });
        if (startData) {
            setTimeout(() => {
                closeLaunchModal();
                _showPage?.('control');
                if (startData.conversation_id) _showConversationInDrawer?.(data.instance_id, startData.conversation_id);
            }, 500);
        }
    } catch (e) {
        showError('launch-fsm-status', 'Launch failed: ' + e.message);
    }
    _resetBtn();
}

export async function doLaunchWorkflow(btn) {
    const _resetBtn = () => { if (btn) { btn.disabled = false; btn.textContent = 'Launch Workflow'; } };
    if (btn) { btn.disabled = true; btn.textContent = 'Launching...'; }

    if (!state.capabilities.workflows) {
        showError('launch-wf-status', 'Workflow extension not installed');
        _resetBtn(); return;
    }

    const body = { label: $('launch-wf-label').value };
    const ctxText = $('launch-wf-context').value.trim();
    if (ctxText) {
        try { body.initial_context = JSON.parse(ctxText); }
        catch { showError('launch-wf-status', 'Invalid JSON context'); _resetBtn(); return; }
    }

    try {
        const data = await postJson('/api/workflow/launch', body);
        showStatus('launch-wf-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        _refreshInstances?.();
        setTimeout(() => { closeLaunchModal(); _showPage?.('control'); }, 500);
    } catch (e) {
        showError('launch-wf-status', 'Launch failed: ' + e.message);
    }
    _resetBtn();
}

// --- Stub Tools ---

export function addStubTool() {
    const container = $('launch-agent-tools');
    const idx = state.stubToolCount++;
    const row = document.createElement('div');
    row.className = 'stub-tool-row';
    row.id = 'stub-tool-' + idx;
    row.innerHTML =
        '<input type="text" placeholder="Name" class="stub-name">' +
        '<input type="text" placeholder="Description" class="stub-desc">' +
        '<input type="text" placeholder="Stub response" class="stub-resp" value="Tool executed successfully">' +
        '<button data-action="remove-stub-tool">&times;</button>';
    container.appendChild(row);
}

function getStubTools() {
    const tools = [];
    document.querySelectorAll('#launch-agent-tools .stub-tool-row').forEach(row => {
        const name = row.querySelector('.stub-name').value.trim();
        const desc = row.querySelector('.stub-desc').value.trim();
        const resp = row.querySelector('.stub-resp').value.trim();
        if (name && desc) tools.push({ name, description: desc, stub_response: resp || 'Tool executed successfully' });
    });
    return tools;
}

export function onAgentTypeChange() {
    const agentType = $('launch-agent-type')?.value;
    const needsTools = state.TOOL_BASED_AGENTS.includes(agentType);
    const toolWrapper = $('launch-agent-tools-wrapper');
    if (toolWrapper) toolWrapper.style.display = needsTools ? '' : 'none';
}

export async function doLaunchAgent(btn) {
    const _resetBtn = () => { if (btn) { btn.disabled = false; btn.textContent = 'Launch Agent'; } };
    if (btn) { btn.disabled = true; btn.textContent = 'Launching...'; }

    if (!state.capabilities.agents) {
        showError('launch-agent-status', 'Agent extension not installed');
        _resetBtn(); return;
    }

    const agentType = $('launch-agent-type').value;
    const needsTools = state.TOOL_BASED_AGENTS.includes(agentType);
    const tools = needsTools ? getStubTools() : [];
    if (needsTools && tools.length === 0) {
        showError('launch-agent-status', 'Add at least one tool for ' + agentType);
        _resetBtn(); return;
    }

    const task = $('launch-agent-task').value.trim();
    if (!task) { showError('launch-agent-status', 'Enter a task'); _resetBtn(); return; }

    const agentModel = $('launch-agent-model').value.trim();
    const body = {
        agent_type: agentType, task, max_iterations: intVal('launch-agent-iters', 10),
        tools, label: $('launch-agent-label').value,
    };
    if (agentModel) body.model = agentModel;

    try {
        const data = await postJson('/api/agent/launch', body);
        showStatus('launch-agent-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        _refreshInstances?.();
        setTimeout(() => { closeLaunchModal(); _showPage?.('control'); }, 500);
    } catch (e) {
        showError('launch-agent-status', 'Launch failed: ' + e.message);
    }
    _resetBtn();
}
