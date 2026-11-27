// FSM-LLM Monitor — Builder Page (Meta-Agent)

import { fetchJson, postJson, loadAgentTypes, toolsRequired } from '../services/api.js';
import { $, esc, intVal, jsonViewHtml, numVal, showError, showStatus, showToast, copyToClipboard, launchedNotStartedText } from '../utils/dom.js';
import { highlightJsonIn, renderJson } from '../utils/json_highlight.js';
import { renderMarkdown } from '../utils/markdown.js';
import { renderGraph } from '../utils/graph.js';
import { state } from '../services/state.js';
import { addTypingIndicator, removeTypingIndicator, showConversationInDrawer } from './conversations.js';

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
// Bumped on every start/reset: replies carrying an older token belong to a
// session the user abandoned and are dropped.
let _sessionToken = 0;
let _gateToken = 0;

const WORKFLOW_LAUNCH_NOTE = 'Workflows built here cannot be launched from the monitor: the server only launches '
    + 'its built-in workflow presets, not a JSON definition. Copy or download the definition and build it with '
    + 'the fsm_llm.workflows Python DSL.';

// The send in flight: its abort controller and elapsed-time timer.
let _abort = null;
let _elapsedTimer = null;

// An abandoned session is dropped even with a send in flight (force): the worker
// thread cannot be stopped, so the model call runs on until the provider answers.
function _deleteServerSession(sessionId) {
    if (!sessionId) return;
    fetchJson(`/api/builder/${encodeURIComponent(sessionId)}?force=true`, { method: 'DELETE' }).catch(() => {});
}

/** The send ended: stop the elapsed timer and hide Cancel. */
function _stopSendUi() {
    _abort = null;
    clearInterval(_elapsedTimer);
    _elapsedTimer = null;
    const cancel = $('builder-cancel-btn');
    if (cancel) cancel.style.display = 'none';
}

/** Stop waiting for the send in flight: abort its request too. */
function _endSend() {
    _abort?.abort();
    _stopSendUi();
}

/** Cancel the build in flight: abandon the session (the server keeps the model call running until it ends). */
export function cancelBuilderSend() {
    if (!_sending) return;
    _endSend();
    _deleteServerSession(_sessionId);
    _sessionToken++;
    _sessionId = null;
    _sending = false;
    removeTypingIndicator();
    _appendBubble('error', 'Cancelled. The model call runs on until the provider answers; start a new session.');
    $('builder-input-area').style.display = 'none';
    showStatus('builder-status', 'Cancelled', 'info');
}

function _resetInputControls() {
    const input = $('builder-message-input');
    if (input) input.disabled = false;
    const sendBtn = input?.nextElementSibling;
    if (sendBtn) sendBtn.disabled = false;
}

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
    const maxTokens = intVal('builder-max-tokens', 4096);

    // Abandon the previous session: free it on the server and ignore any of
    // its replies still in flight.
    _endSend();
    if (_sessionId && !_complete) _deleteServerSession(_sessionId);
    const token = ++_sessionToken;

    _sessionId = null;
    _complete = false;
    _messages = [];
    _artifact = null;
    _artifactType = null;
    _sending = false;

    removeTypingIndicator();
    _resetInputControls();
    $('builder-chat').innerHTML = '';
    $('builder-result-panel').style.display = 'none';
    $('builder-input-area').style.display = 'none';
    showStatus('builder-status', 'Starting session...', 'info');

    postJson('/api/builder/start', { model: model || undefined, temperature: temp, max_tokens: maxTokens })
        .then(data => {
            if (token !== _sessionToken) {
                // A newer start or a reset superseded this one.
                _deleteServerSession(data?.session_id);
                return;
            }
            _sessionId = data.session_id;
            $('builder-input-area').style.display = 'flex';
            _appendBubble('assistant', data.response);
            _renderInternalState(data.internal_state);
            showStatus('builder-status', 'Session active', 'success');
            $('builder-message-input').focus();
            if (data.is_complete) _onComplete(data);
        })
        .catch(e => {
            if (token !== _sessionToken) return;
            showError('builder-status', `Failed to start: ${e.message}`);
        });
}

export function sendBuilderMessage() {
    if (_sending || _complete || !_sessionId) return;

    const input = $('builder-message-input');
    const sendBtn = input?.nextElementSibling;
    const msg = input.value.trim();
    if (!msg) return;

    const token = _sessionToken;
    _sending = true;
    input.value = '';
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    _appendBubble('user', msg);

    const chat = $('builder-chat');
    addTypingIndicator(chat);

    // Elapsed time of the build call, and a Cancel button while it runs.
    const began = Date.now();
    _abort = new AbortController();
    const paintElapsed = () => showStatus('builder-status', `Building artifact... ${Math.round((Date.now() - began) / 1000)} s`, 'info');
    paintElapsed();
    _elapsedTimer = setInterval(paintElapsed, 1000);
    const cancel = $('builder-cancel-btn');
    if (cancel) cancel.style.display = '';

    postJson('/api/builder/send', { session_id: _sessionId, message: msg }, { signal: _abort.signal })
        .then(data => {
            if (token !== _sessionToken) return;
            _stopSendUi();
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
            if (token !== _sessionToken) return;
            _stopSendUi();
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
        renderJson($('builder-result-json'), '{}');
        $('builder-result-summary').innerHTML = '';
        $('builder-result-graph').style.display = 'none';
        _setEditing(false);
        $('builder-edit-btn').style.display = 'none';
        const errLaunchBtn = $('builder-launch-btn');
        if (errLaunchBtn) errLaunchBtn.style.display = 'none';
        const errNote = $('builder-launch-note');
        if (errNote) errNote.style.display = 'none';
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

    _setEditing(false);
    $('builder-edit-btn').style.display = '';
    _syncLaunch(artifact, artifactType);

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

    renderJson($('builder-result-json'), _artifact || '{}');
    showStatus('builder-status', 'Build complete!', 'success');
}

/** The Launch button and its note for the artifact: workflows never launch; an agent type the manager cannot run is disabled with a note. */
function _syncLaunch(artifact, artifactType) {
    const launchBtn = $('builder-launch-btn');
    const launchNote = $('builder-launch-note');
    let note = artifactType === 'workflow' ? WORKFLOW_LAUNCH_NOTE : '';
    if (launchBtn) {
        const known = ['fsm', 'workflow', 'agent'].includes(artifactType);
        const isWorkflow = artifactType === 'workflow';
        launchBtn.textContent = known ? `Launch ${artifactType.toUpperCase()}` : 'Launch';
        launchBtn.style.display = known ? '' : 'none';
        // The server rejects definition_json launches, so do not offer one.
        launchBtn.disabled = isWorkflow;
        if (isWorkflow) launchBtn.setAttribute('aria-describedby', 'builder-launch-note');
        else launchBtn.removeAttribute('aria-describedby');
        launchBtn.title = isWorkflow ? 'Workflows cannot be launched from a JSON definition' : '';
    }
    if (launchNote) {
        launchNote.textContent = note;
        launchNote.style.display = note ? 'block' : 'none';
    }
    if (artifactType === 'agent') _gateAgentLaunch(_agentId(artifact), _sessionToken);
}

/** Disable Launch with a note when /api/capabilities does not list the agent type id (the list is the server's launchable rows). */
function _gateAgentLaunch(agentId, token) {
    const gate = ++_gateToken;  // a newer call (an Apply of Edit JSON) drops the reply of an older one
    fetchJson('/api/capabilities').then(caps => {
        if (token !== _sessionToken || gate !== _gateToken || !Array.isArray(caps?.agent_types) || caps.agent_types.includes(agentId)) return;
        const btn = $('builder-launch-btn');
        const note = $('builder-launch-note');
        if (btn) { btn.disabled = true; btn.setAttribute('aria-describedby', 'builder-launch-note'); btn.title = 'The monitor cannot launch this agent type'; }
        if (note) {
            note.textContent = `${agentId} cannot be launched from the monitor (its constructor needs arguments a form cannot supply). `
                + 'Copy or download the definition and build it with the fsm_llm.agents Python API.';
            note.style.display = 'block';
        }
    }).catch(() => {});  // unknown: leave Launch enabled, the server answers 400 for a type it cannot run
}

/** The agent type id of a built agent artifact: the Builder's type is an executable id, sent as it is (default `react`). */
function _agentId(artifact) {
    return typeof artifact.agent_type === 'string' && artifact.agent_type ? artifact.agent_type : 'react';
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
    const token = _sessionToken;
    const svg = $('builder-result-svg');
    const errEl = $('builder-result-graph-error');
    if (errEl) { errEl.textContent = ''; errEl.style.display = 'none'; }
    if (svg) svg.style.display = '';
    postJson('/api/fsm/visualize', fsmDef)
        .then(vizData => {
            if (token !== _sessionToken) return;
            renderGraph('builder-result-svg', vizData, { colorVar: 'var(--primary-dim)', rx: 4, nodeClass: 'fsm' });
        })
        .catch(e => {
            if (token !== _sessionToken) return;
            console.error('Builder graph render failed:', e);
            // Keep #builder-result-svg in the DOM for the next result; show the
            // error in its sibling instead of replacing the container.
            if (svg) { svg.innerHTML = ''; svg.style.display = 'none'; }
            if (errEl) {
                errEl.textContent = `Graph render failed: ${e.message || String(e)}`;
                errEl.style.display = '';
            }
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
    html += `<span class="key">Turn</span><span class="val">${esc(state.turn_count ?? 0)}</span>`;
    if (state.artifact_type) {
        html += `<span class="key">Type</span><span class="val">${esc(state.artifact_type)}</span>`;
    }
    html += '</div></div>';

    // Progress
    if (state.builder_progress) {
        const rawPct = Number(state.builder_progress.percentage ?? 0);
        const pct = Number.isFinite(rawPct) ? Math.max(0, Math.min(100, rawPct)) : 0;
        html += '<div class="builder-state-section">';
        html += '<h4>Progress</h4>';
        html += `<div class="builder-state-progress"><div class="builder-state-progress-bar" style="width:${pct}%"></div></div>`;
        html += '<div class="builder-state-kv">';
        html += `<span class="key">Complete</span><span class="val">${pct.toFixed(0)}%</span>`;
        const missing = Array.isArray(state.builder_progress.missing) ? state.builder_progress.missing : [];
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
        html += jsonViewHtml(JSON.stringify(state.artifact_preview, null, 2), 'pre');
        html += '</div>';
    }

    // Raw state fallback — always show something
    if (!state.builder_progress && !state.builder_summary && !reqData) {
        html += '<div class="builder-state-section builder-state-context">';
        html += '<h4>Raw</h4>';
        html += jsonViewHtml(JSON.stringify(state, null, 2), 'pre');
        html += '</div>';
    }

    container.innerHTML = html;
    highlightJsonIn(container);
}

/** Show the JSON as read-only text (`false`) or as an editable textarea (`true`). */
function _setEditing(editing) {
    $('builder-result-json').style.display = editing ? 'none' : '';
    $('builder-result-edit').style.display = editing ? 'block' : 'none';
    $('builder-edit-btn').textContent = editing ? 'Apply JSON' : 'Edit JSON';
}

/**
 * Edit JSON, then Apply: the text is only parsed here, never executed. A parse
 * error is shown and the editor stays open. The applied artifact is what Launch
 * sends, and Launch (the server loader) is the validator.
 */
export function editBuilderResult() {
    const area = $('builder-result-edit');
    if (area.style.display !== 'block') {
        area.value = _artifact || '{}';
        _setEditing(true);
        const details = $('builder-result-json').closest?.('details');
        if (details) details.open = true;
        return;
    }
    let edited;
    try { edited = JSON.parse(area.value); } catch (e) { showError('builder-status', `Invalid JSON: ${e.message}`); return; }
    if (!edited || typeof edited !== 'object' || Array.isArray(edited)) {
        showError('builder-status', 'The artifact must be a JSON object');
        return;
    }
    _artifact = JSON.stringify(edited, null, 2);
    renderJson($('builder-result-json'), _artifact);
    _setEditing(false);
    $('builder-result-summary').innerHTML = _buildResultSummary(edited, _artifactType);
    _syncLaunch(edited, _artifactType);
    const agentForm = $('builder-agent-launch-form');
    if (agentForm) agentForm.style.display = 'none';
    const hasStates = _artifactType === 'fsm' && edited.states && Object.keys(edited.states).length > 0;
    $('builder-result-graph').style.display = hasStates ? 'block' : 'none';
    if (hasStates) _renderBuilderGraph(edited);
    showStatus('builder-status', 'Edited JSON applied; Launch checks it', 'success');
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
    if ($('builder-result-edit').style.display === 'block') {
        showError('builder-status', 'Apply the JSON edit first (the editor is open)');
        return;
    }
    if (_artifactType === 'fsm') return _launchBuilderFSM();
    if (_artifactType === 'workflow') return _launchBuilderWorkflow();
    if (_artifactType === 'agent') return _launchBuilderAgent();
    showError('builder-status', `Unknown artifact type: ${_artifactType}`);
}

// One launch at a time, FSM or agent: the flag is the guard, the disabled button its sign
// (an Apply of Edit JSON re-syncs the button, so the button alone is not a guard).
let _launching = false;

/** The 500 ms window after a launch, then Control. `release` runs at its end in every case; the move (and `then`)
 *  only while the session and the page are those the launch came from: a Reset, a new start or another page cancels it. */
function _afterLaunch(release, then) {
    const token = _sessionToken, page = state.currentPage;
    setTimeout(() => {
        release();
        if (token !== _sessionToken || state.currentPage !== page) return;
        _showPage?.('control');
        then?.();
    }, 500);
}

async function _launchBuilderFSM() {
    if (_launching) return;
    let parsed;
    try { parsed = JSON.parse(_artifact); } catch { showError('builder-status', 'Invalid JSON'); return; }

    const launchBtn = $('builder-launch-btn');
    const model = $('builder-model').value.trim() || undefined;
    const temp = numVal('builder-temp', 0.5);
    const release = () => { _launching = false; if (launchBtn) launchBtn.disabled = false; };
    _launching = true;
    if (launchBtn) launchBtn.disabled = true;

    let data;
    try {
        data = await postJson('/api/fsm/launch', { fsm_json: parsed, model, temperature: temp, label: parsed.name || 'Built FSM' });
    } catch (e) {
        showError('builder-status', `Launch failed: ${e.message}`);
        release();
        return;
    }
    // The instance exists from here on: a failing start is never "Launch failed" (a retry would make a second FSM).
    showStatus('builder-status', 'Launched! Switching to Control Center...', 'success');
    _refreshInstances?.();
    let startData = null;
    try {
        startData = await postJson(`/api/fsm/${encodeURIComponent(data.instance_id)}/start`, {});
    } catch (e) {
        showError('builder-status', launchedNotStartedText(data, e));
        showToast(launchedNotStartedText(data, e), 'error');  // the view moves to Control in 500 ms
    }
    _afterLaunch(release, () => {
        // Open the new conversation, as the Launch modal does.
        if (startData?.conversation_id) showConversationInDrawer(data.instance_id, startData.conversation_id);
    });
}

function _launchBuilderWorkflow() {
    // The server cannot launch a workflow from definition_json (only presets),
    // so explain instead of sending a request that always fails.
    showError('builder-status', WORKFLOW_LAUNCH_NOTE);
}

function _launchBuilderAgent() {
    if (_launching) return;
    const form = $('builder-agent-launch-form');
    const launchBtn = $('builder-launch-btn');

    // First click: show the form and populate tool stubs
    if (form.style.display === 'none') {
        form.style.display = 'block';
        launchBtn.textContent = 'Confirm Launch';
        loadAgentTypes();  // the rows tell which types need a tool
        _populateAgentToolStubs();
        _prefillAgentForm();
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

    const agentType = _agentId(parsed);
    const tools = _collectAgentToolStubs(parsed);

    if (toolsRequired(agentType) && tools.length === 0) {
        showError('builder-status', 'This agent type requires at least one tool: use Edit JSON to add one to "tools" (a name and a description), then Apply and Launch again');
        return;
    }

    const body = {
        agent_type: agentType,
        task,
        tools,
        // Never parsed.config.model: that is a template value of every built agent.
        model: $('builder-agent-model').value.trim() || undefined,
        max_iterations: intVal('builder-agent-iters', 10),
        timeout_seconds: numVal('builder-agent-timeout', 120),
        label: $('builder-agent-label').value.trim() || parsed.name || 'Built Agent',
    };

    _launching = true;
    launchBtn.disabled = true;
    launchBtn.textContent = 'Launching...';
    const release = () => { _launching = false; launchBtn.disabled = false; };

    postJson('/api/agent/launch', body)
        .then(() => {
            form.style.display = 'none';
            launchBtn.textContent = 'Launch AGENT';
            showStatus('builder-status', 'Launched! Switching to Control Center...', 'success');
            _refreshInstances?.();
            _afterLaunch(release);
        })
        .catch(e => {
            // Keep the form (task, stub responses) open so the user can retry.
            launchBtn.textContent = 'Confirm Launch';
            showError('builder-status', `Launch failed: ${e.message}`);
            release();
        });
}

/** A positive finite number from the artifact config, else the fallback (the server bounds the value). */
function _positive(v, fallback) {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? n : fallback;
}

/** Pre-fill the launch fields: name and limits from the artifact, the model from the Builder field. */
function _prefillAgentForm() {
    let parsed;
    try { parsed = JSON.parse(_artifact); } catch { return; }
    $('builder-agent-label').value = parsed.name || 'Built Agent';
    $('builder-agent-model').value = $('builder-model').value.trim();
    $('builder-agent-iters').value = _positive(parsed.config?.max_iterations, 10);
    $('builder-agent-timeout').value = _positive(parsed.config?.timeout_seconds, 120);
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
    _endSend();
    if (_sessionId && !_complete) _deleteServerSession(_sessionId);
    _sessionToken++;
    _sessionId = null;
    _complete = false;
    _messages = [];
    _artifact = null;
    _artifactType = null;
    _sending = false;

    removeTypingIndicator();
    _resetInputControls();
    $('builder-chat').innerHTML = '';
    $('builder-result-panel').style.display = 'none';
    $('builder-input-area').style.display = 'none';
    const agentForm = $('builder-agent-launch-form');
    if (agentForm) agentForm.style.display = 'none';
    _setEditing(false);
    const stateContent = $('builder-state-content');
    if (stateContent) stateContent.innerHTML = '<div class="empty-state">Start a session to see internal state</div>';
    showStatus('builder-status', '');
}
