// FSM-LLM Monitor — Builder Page (Meta-Agent)

import { fetchJson, postJson } from '../services/api.js';
import { TOOL_BASED_AGENTS } from '../services/state.js';
import { $, esc, numVal, showError, showStatus, showToast, copyToClipboard } from '../utils/dom.js';
import { renderMarkdown } from '../utils/markdown.js';
import { renderGraph } from '../utils/graph.js';
import { addTypingIndicator, removeTypingIndicator } from './conversations.js';

// Agent type mapping: builder lowercase → server class names
const AGENT_TYPE_MAP = {
    react: 'ReactAgent',
    plan_execute: 'PlanExecuteAgent',
    reflexion: 'ReflexionAgent',
    rewoo: 'REWOOAgent',
    evaluator_optimizer: 'EvaluatorOptimizerAgent',
    maker_checker: 'MakerCheckerAgent',
    prompt_chain: 'PromptChainAgent',
    self_consistency: 'SelfConsistencyAgent',
    debate: 'DebateAgent',
    orchestrator: 'OrchestratorAgent',
    adapt: 'ADaPTAgent',
};

// Forward references
let _showPage, _refreshInstances;
export function setDeps(deps) {
    _showPage = deps.showPage;
    _refreshInstances = deps.refreshInstances;
}

// --- State ---
let _sessionId = null;
let _complete = false;
let _messages = [];
let _artifact = null;
let _artifactType = null;
let _sending = false;
let _following = true;

function _isNearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 60;
}

function _autoScroll(chat) {
    if (_following) chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
    _updateJump();
}

function _updateJump() {
    const btn = $('builder-jump-btn');
    const chat = $('builder-chat');
    if (btn && chat) btn.classList.toggle('visible', !_isNearBottom(chat));
}

export function builderJumpToLatest() {
    const chat = $('builder-chat');
    if (chat) {
        chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
        _following = true;
        _updateJump();
    }
}

export function onBuilderScroll() {
    const chat = $('builder-chat');
    if (chat) {
        _following = _isNearBottom(chat);
        _updateJump();
    }
}

// --- Session Management ---

export function startBuilderSession() {
    const model = $('builder-model').value.trim();
    const temp = numVal('builder-temp', 0.7);

    _sessionId = null;
    _complete = false;
    _messages = [];
    _artifact = null;
    _artifactType = null;
    _sending = false;

    $('builder-chat').innerHTML = '';
    $('builder-result-panel').style.display = 'none';
    $('builder-input-area').style.display = 'flex';
    showStatus('builder-status', 'Starting session...', 'info');

    postJson('/api/builder/start', { model: model || undefined, temperature: temp })
        .then(data => {
            _sessionId = data.session_id;
            _appendBubble('assistant', data.response);
            _renderInternalState(data.internal_state);
            showStatus('builder-status', 'Session active', 'success');
            $('builder-message-input').focus();
            if (data.is_complete) _onComplete(data);
        })
        .catch(e => showError('builder-status', `Failed to start: ${e.message}`));
}

export function sendBuilderMessage() {
    if (_sending || _complete || !_sessionId) return;

    const input = $('builder-message-input');
    const sendBtn = input?.nextElementSibling;
    const msg = input.value.trim();
    if (!msg) return;

    _sending = true;
    input.value = '';
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    _appendBubble('user', msg);

    const chat = $('builder-chat');
    addTypingIndicator(chat);

    // Hint that a long build may be in progress after 5s
    const buildTimer = setTimeout(() => {
        showStatus('builder-status', 'Building artifact... (this may take a moment)', 'info');
    }, 5000);

    postJson('/api/builder/send', { session_id: _sessionId, message: msg })
        .then(data => {
            clearTimeout(buildTimer);
            _sending = false;
            removeTypingIndicator();
            _appendBubble('assistant', data.response);
            _renderInternalState(data.internal_state);
            showStatus('builder-status', 'Session active', 'success');

            if (data.is_complete) {
                _onComplete(data);
            } else {
                input.disabled = false;
                if (sendBtn) sendBtn.disabled = false;
                input.focus();
            }
        })
        .catch(e => {
            clearTimeout(buildTimer);
            _sending = false;
            removeTypingIndicator();
            _appendBubble('error', e.message || 'Request failed');
            input.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            input.focus();
        });
}

function _onComplete(data) {
    _complete = true;
    $('builder-input-area').style.display = 'none';
    $('builder-result-panel').style.display = 'block';

    if (data.error && (!data.artifact_json || data.artifact_json === '{}')) {
        $('builder-result-json').textContent = '{}';
        $('builder-result-summary').innerHTML = '';
        $('builder-result-graph').style.display = 'none';
        $('builder-result-status').innerHTML =
            `<span class="badge badge-failed">ERROR</span> ${esc(data.error || 'Result extraction failed')}`;
        showStatus('builder-status', 'Build completed with errors', 'error');
        return;
    }

    const artifact = data.artifact || {};
    const artifactType = data.artifact_type || 'unknown';
    _artifact = data.artifact_json || JSON.stringify(artifact, null, 2);
    _artifactType = artifactType;

    let statusHtml = '';
    if (data.is_valid) {
        statusHtml = `<span class="badge badge-completed">VALID</span> ${esc(artifactType)} artifact ready`;
    } else {
        statusHtml = '<span class="badge badge-failed">INVALID</span> ';
        if (data.validation_errors?.length) statusHtml += esc(data.validation_errors.join('; '));
    }
    $('builder-result-status').innerHTML = statusHtml;
    $('builder-result-summary').innerHTML = _buildResultSummary(artifact, artifactType);

    const launchBtn = $('builder-launch-btn');
    if (launchBtn) {
        const known = ['fsm', 'workflow', 'agent'].includes(artifactType);
        launchBtn.textContent = known ? `Launch ${artifactType.toUpperCase()}` : 'Launch';
        launchBtn.style.display = known ? '' : 'none';
    }

    // Hide agent launch form from any previous result
    const agentForm = $('builder-agent-launch-form');
    if (agentForm) agentForm.style.display = 'none';

    const graphContainer = $('builder-result-graph');
    if (artifactType === 'fsm' && artifact.states && Object.keys(artifact.states).length > 0) {
        graphContainer.style.display = 'block';
        _renderBuilderGraph(artifact);
    } else {
        graphContainer.style.display = 'none';
    }

    $('builder-result-json').textContent = _artifact || '{}';
    showStatus('builder-status', 'Build complete!', 'success');
}

function _buildResultSummary(artifact, artifactType) {
    let html = '<div class="builder-summary-grid">';
    html += `<div class="builder-summary-item"><span class="key">Name</span><span class="val">${esc(artifact.name || 'Unnamed')}</span></div>`;
    if (artifact.description) {
        html += `<div class="builder-summary-item"><span class="key">Description</span><span class="val">${esc(artifact.description)}</span></div>`;
    }
    html += `<div class="builder-summary-item"><span class="key">Type</span><span class="val">${esc(artifactType.toUpperCase())}</span></div>`;

    if (artifactType === 'fsm' && artifact.states) {
        const stateIds = Object.keys(artifact.states);
        html += `<div class="builder-summary-item"><span class="key">States</span><span class="val">${stateIds.length}</span></div>`;
        if (artifact.initial_state) html += `<div class="builder-summary-item"><span class="key">Initial</span><span class="val">${esc(artifact.initial_state)}</span></div>`;
        const terminalStates = stateIds.filter(sid => !artifact.states[sid].transitions?.length);
        if (terminalStates.length) html += `<div class="builder-summary-item"><span class="key">Terminal</span><span class="val">${esc(terminalStates.join(', '))}</span></div>`;
        let transCount = 0;
        for (const sid in artifact.states) transCount += (artifact.states[sid].transitions || []).length;
        html += `<div class="builder-summary-item"><span class="key">Transitions</span><span class="val">${transCount}</span></div>`;
        if (artifact.persona) html += `<div class="builder-summary-item"><span class="key">Persona</span><span class="val">${esc(artifact.persona)}</span></div>`;
    } else if (artifactType === 'workflow' && artifact.steps) {
        html += `<div class="builder-summary-item"><span class="key">Steps</span><span class="val">${Object.keys(artifact.steps).length}</span></div>`;
        if (artifact.initial_step_id) html += `<div class="builder-summary-item"><span class="key">Initial Step</span><span class="val">${esc(artifact.initial_step_id)}</span></div>`;
    } else if (artifactType === 'agent') {
        if (artifact.agent_type) html += `<div class="builder-summary-item"><span class="key">Agent Type</span><span class="val">${esc(artifact.agent_type)}</span></div>`;
        if (artifact.tools) {
            html += `<div class="builder-summary-item"><span class="key">Tools</span><span class="val">${artifact.tools.length}</span></div>`;
            html += `<div class="builder-summary-item"><span class="key">Tool List</span><span class="val">${esc(artifact.tools.map(t => t.name || '?').join(', '))}</span></div>`;
        }
    }
    html += '</div>';
    return html;
}

function _renderBuilderGraph(fsmDef) {
    postJson('/api/fsm/visualize', fsmDef)
        .then(vizData => renderGraph('builder-result-svg', vizData, { colorVar: 'var(--primary-dim)', rx: 4, nodeClass: 'fsm' }))
        .catch(e => {
            console.error('Builder graph render failed:', e);
            const g = $('builder-result-graph');
            g.innerHTML = `<div class="empty-state">Graph render failed: ${esc(e.message || String(e))}</div>`;
        });
}

function _appendBubble(role, text) {
    _messages.push({ role, content: text });
    const chat = $('builder-chat');
    _following = _isNearBottom(chat);

    let html = `<div class="chat-bubble ${role}">`;
    html += `<div class="chat-role-tag">${esc(role)}</div>`;
    html += (role === 'user' || role === 'error') ? esc(text) : `<div class="md-body">${renderMarkdown(text)}</div>`;
    html += '</div>';

    chat.insertAdjacentHTML('beforeend', html);
    if (role === 'user') _following = true;
    _autoScroll(chat);
}

// --- Internal State Panel ---

function _renderInternalState(state) {
    const container = $('builder-state-content');
    if (!container) {
        console.warn('builder-state-content element not found — hard-refresh the page');
        return;
    }

    if (!state || typeof state !== 'object') {
        container.innerHTML = '<div class="empty-state">Waiting for state data...</div>';
        return;
    }

    let html = '';

    // FSM State + Turn
    html += '<div class="builder-state-section">';
    html += '<h4>Session</h4>';
    html += '<div class="builder-state-kv">';
    html += `<span class="key">Phase</span><span class="val"><span class="builder-state-badge">${esc(state.phase || state.current_state || 'n/a')}</span></span>`;
    html += `<span class="key">Turn</span><span class="val">${state.turn_count ?? 0}</span>`;
    if (state.artifact_type) {
        html += `<span class="key">Type</span><span class="val">${esc(state.artifact_type)}</span>`;
    }
    html += '</div></div>';

    // Progress
    if (state.builder_progress) {
        const pct = state.builder_progress.percentage ?? 0;
        html += '<div class="builder-state-section">';
        html += '<h4>Progress</h4>';
        html += `<div class="builder-state-progress"><div class="builder-state-progress-bar" style="width:${pct}%"></div></div>`;
        html += '<div class="builder-state-kv">';
        html += `<span class="key">Complete</span><span class="val">${pct.toFixed(0)}%</span>`;
        const missing = state.builder_progress.missing || [];
        if (missing.length > 0) {
            html += `<span class="key">Missing</span><span class="val">${esc(missing.join(', '))}</span>`;
        }
        html += '</div></div>';
    }

    // Builder Summary
    if (state.builder_summary) {
        html += '<div class="builder-state-section">';
        html += '<h4>Builder Summary</h4>';
        html += `<pre>${esc(state.builder_summary)}</pre>`;
        html += '</div>';
    }

    // Requirements / Context
    const reqData = state.requirements || state.context;
    if (reqData && Object.keys(reqData).length > 0) {
        html += '<div class="builder-state-section builder-state-context">';
        html += '<h4>Requirements</h4>';
        html += '<div class="builder-state-kv">';
        for (const [k, v] of Object.entries(reqData)) {
            if (v != null) {
                const display = Array.isArray(v) ? v.join(', ') : String(v);
                html += `<span class="key">${esc(k.replace(/^artifact_/, ''))}</span><span class="val">${esc(display)}</span>`;
            }
        }
        html += '</div></div>';
    }

    // Artifact Preview
    if (state.artifact_preview && Object.keys(state.artifact_preview).length > 0) {
        html += '<div class="builder-state-section builder-state-artifact">';
        html += '<h4>Artifact Preview</h4>';
        html += `<pre>${esc(JSON.stringify(state.artifact_preview, null, 2))}</pre>`;
        html += '</div>';
    }

    // Raw state fallback — always show something
    if (!state.builder_progress && !state.builder_summary && !reqData) {
        html += '<div class="builder-state-section builder-state-context">';
        html += '<h4>Raw</h4>';
        html += `<pre>${esc(JSON.stringify(state, null, 2))}</pre>`;
        html += '</div>';
    }

    container.innerHTML = html;
}

export function copyBuilderResult() {
    const json = $('builder-result-json').textContent;
    copyToClipboard(json).then(() => showStatus('builder-status', 'Copied to clipboard!', 'success'));
}

export function downloadBuilderResult() {
    const json = $('builder-result-json').textContent;
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'artifact.json';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Artifact downloaded', 'success');
}

// --- Launch ---

export function launchBuilderResult() {
    if (_artifactType === 'fsm') return _launchBuilderFSM();
    if (_artifactType === 'workflow') return _launchBuilderWorkflow();
    if (_artifactType === 'agent') return _launchBuilderAgent();
    showError('builder-status', `Unknown artifact type: ${_artifactType}`);
}

function _launchBuilderFSM() {
    let parsed;
    try { parsed = JSON.parse(_artifact); } catch { showError('builder-status', 'Invalid JSON'); return; }

    const model = $('builder-model').value.trim() || undefined;
    const temp = numVal('builder-temp', 0.5);

    postJson('/api/fsm/launch', { fsm_json: parsed, model, temperature: temp, label: parsed.name || 'Built FSM' })
        .then(data => {
            showStatus('builder-status', 'Launched! Switching to Control Center...', 'success');
            _refreshInstances?.();
            return postJson(`/api/fsm/${data.instance_id}/start`, {}).then(() => {
                setTimeout(() => _showPage?.('control'), 500);
            });
        })
        .catch(e => showError('builder-status', `Launch failed: ${e.message}`));
}

function _launchBuilderWorkflow() {
    let parsed;
    try { parsed = JSON.parse(_artifact); } catch { showError('builder-status', 'Invalid JSON'); return; }

    postJson('/api/workflow/launch', {
        definition_json: parsed,
        label: parsed.name || 'Built Workflow',
    })
        .then(() => {
            showStatus('builder-status', 'Launched! Switching to Control Center...', 'success');
            _refreshInstances?.();
            setTimeout(() => _showPage?.('control'), 500);
        })
        .catch(e => showError('builder-status', `Launch failed: ${e.message}`));
}

function _launchBuilderAgent() {
    const form = $('builder-agent-launch-form');
    const launchBtn = $('builder-launch-btn');

    // First click: show the form and populate tool stubs
    if (form.style.display === 'none') {
        form.style.display = 'block';
        launchBtn.textContent = 'Confirm Launch';
        _populateAgentToolStubs();
        $('builder-agent-task')?.focus();
        return;
    }

    // Second click: validate and submit
    const task = $('builder-agent-task').value.trim();
    if (!task) {
        showError('builder-status', 'Enter a task for the agent');
        return;
    }

    let parsed;
    try { parsed = JSON.parse(_artifact); } catch { showError('builder-status', 'Invalid JSON'); return; }

    const rawType = (parsed.agent_type || 'react').toLowerCase();
    const agentType = AGENT_TYPE_MAP[rawType] || parsed.agent_type || 'ReactAgent';
    const tools = _collectAgentToolStubs(parsed);
    const needsTools = TOOL_BASED_AGENTS.includes(agentType);

    if (needsTools && tools.length === 0) {
        showError('builder-status', 'This agent type requires at least one tool');
        return;
    }

    const body = {
        agent_type: agentType,
        task,
        tools,
        model: parsed.config?.model || $('builder-model').value.trim() || undefined,
        max_iterations: parsed.config?.max_iterations || 10,
        timeout_seconds: parsed.config?.timeout_seconds || 120,
        label: parsed.name || 'Built Agent',
    };

    launchBtn.disabled = true;
    launchBtn.textContent = 'Launching...';

    postJson('/api/agent/launch', body)
        .then(() => {
            showStatus('builder-status', 'Launched! Switching to Control Center...', 'success');
            _refreshInstances?.();
            setTimeout(() => _showPage?.('control'), 500);
        })
        .catch(e => showError('builder-status', `Launch failed: ${e.message}`))
        .finally(() => {
            launchBtn.disabled = false;
            launchBtn.textContent = 'Launch AGENT';
            form.style.display = 'none';
        });
}

function _populateAgentToolStubs() {
    let parsed;
    try { parsed = JSON.parse(_artifact); } catch { return; }

    const toolsSection = $('builder-agent-tools-section');
    const toolsList = $('builder-agent-tools-list');
    const tools = parsed.tools || [];

    if (tools.length === 0) {
        toolsSection.style.display = 'none';
        return;
    }

    toolsSection.style.display = 'block';
    toolsList.innerHTML = '';

    for (const tool of tools) {
        const row = document.createElement('div');
        row.className = 'stub-tool-row';
        row.innerHTML =
            `<span class="stub-tool-name">${esc(tool.name)}</span>` +
            `<input type="text" class="stub-resp" placeholder="Stub response" value="Tool executed successfully" title="${esc(tool.description || '')}">`;
        toolsList.appendChild(row);
    }
}

function _collectAgentToolStubs(parsed) {
    const tools = parsed.tools || [];
    const rows = document.querySelectorAll('#builder-agent-tools-list .stub-tool-row');
    return tools.map((tool, i) => ({
        name: tool.name,
        description: tool.description || '',
        stub_response: rows[i]?.querySelector('.stub-resp')?.value?.trim() || 'Tool executed successfully',
    }));
}

export function resetBuilder() {
    if (_sessionId && !_complete) {
        fetchJson(`/api/builder/${_sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    _sessionId = null;
    _complete = false;
    _messages = [];
    _artifact = null;
    _artifactType = null;
    _sending = false;

    $('builder-chat').innerHTML = '';
    $('builder-result-panel').style.display = 'none';
    $('builder-input-area').style.display = 'none';
    const agentForm = $('builder-agent-launch-form');
    if (agentForm) agentForm.style.display = 'none';
    const stateContent = $('builder-state-content');
    if (stateContent) stateContent.innerHTML = '<div class="empty-state">Start a session to see internal state</div>';
    showStatus('builder-status', '');
}
