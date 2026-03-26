// FSM-LLM Monitor — Builder Page (Meta-Agent)

import { fetchJson, postJson } from '../services/api.js';
import { $, esc, numVal, showError, showStatus, showToast } from '../utils/dom.js';
import { renderMarkdown } from '../utils/markdown.js';
import { renderGraph } from '../utils/graph.js';
import { addTypingIndicator, removeTypingIndicator } from './conversations.js';

// Forward references
let _showPage;
export function setDeps(deps) { _showPage = deps.showPage; }

// --- State ---
let _sessionId = null;
let _complete = false;
let _messages = [];
let _artifact = null;
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
    _sending = false;

    $('builder-chat').innerHTML = '';
    $('builder-result-panel').style.display = 'none';
    $('builder-input-area').style.display = 'flex';
    showStatus('builder-status', 'Starting session...', 'info');

    postJson('/api/builder/start', { model: model || undefined, temperature: temp })
        .then(data => {
            _sessionId = data.session_id;
            _appendBubble('assistant', data.response);
            showStatus('builder-status', 'Session active', 'success');
            $('builder-message-input').focus();
            if (data.is_complete) _onComplete(data);
        })
        .catch(e => showError('builder-status', 'Failed to start: ' + e.message));
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

    postJson('/api/builder/send', { session_id: _sessionId, message: msg })
        .then(data => {
            _sending = false;
            removeTypingIndicator();
            _appendBubble('assistant', data.response);
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
            '<span class="badge badge-failed">ERROR</span> ' + esc(data.error || 'Result extraction failed');
        showStatus('builder-status', 'Build completed with errors', 'error');
        return;
    }

    const artifact = data.artifact || {};
    const artifactType = data.artifact_type || 'unknown';
    _artifact = data.artifact_json || JSON.stringify(artifact, null, 2);

    let statusHtml = '';
    if (data.is_valid) {
        statusHtml = '<span class="badge badge-completed">VALID</span> ' + esc(artifactType) + ' artifact ready';
    } else {
        statusHtml = '<span class="badge badge-failed">INVALID</span> ';
        if (data.validation_errors?.length) statusHtml += esc(data.validation_errors.join('; '));
    }
    $('builder-result-status').innerHTML = statusHtml;
    $('builder-result-summary').innerHTML = _buildResultSummary(artifact, artifactType);

    const launchBtn = $('builder-launch-btn');
    if (launchBtn) {
        if (artifactType === 'fsm') {
            launchBtn.textContent = 'Launch FSM';
            launchBtn.style.display = '';
        } else {
            launchBtn.textContent = 'Launch ' + artifactType.toUpperCase();
            launchBtn.style.display = 'none';
        }
    }

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
    html += '<div class="builder-summary-item"><span class="key">Name</span><span class="val">' + esc(artifact.name || 'Unnamed') + '</span></div>';
    if (artifact.description) {
        html += '<div class="builder-summary-item"><span class="key">Description</span><span class="val">' + esc(artifact.description) + '</span></div>';
    }
    html += '<div class="builder-summary-item"><span class="key">Type</span><span class="val">' + esc(artifactType.toUpperCase()) + '</span></div>';

    if (artifactType === 'fsm' && artifact.states) {
        const stateIds = Object.keys(artifact.states);
        html += '<div class="builder-summary-item"><span class="key">States</span><span class="val">' + stateIds.length + '</span></div>';
        if (artifact.initial_state) html += '<div class="builder-summary-item"><span class="key">Initial</span><span class="val">' + esc(artifact.initial_state) + '</span></div>';
        const terminalStates = stateIds.filter(sid => !artifact.states[sid].transitions?.length);
        if (terminalStates.length) html += '<div class="builder-summary-item"><span class="key">Terminal</span><span class="val">' + esc(terminalStates.join(', ')) + '</span></div>';
        let transCount = 0;
        for (const sid in artifact.states) transCount += (artifact.states[sid].transitions || []).length;
        html += '<div class="builder-summary-item"><span class="key">Transitions</span><span class="val">' + transCount + '</span></div>';
        if (artifact.persona) html += '<div class="builder-summary-item"><span class="key">Persona</span><span class="val">' + esc(artifact.persona) + '</span></div>';
    } else if (artifactType === 'workflow' && artifact.steps) {
        html += '<div class="builder-summary-item"><span class="key">Steps</span><span class="val">' + Object.keys(artifact.steps).length + '</span></div>';
        if (artifact.initial_step_id) html += '<div class="builder-summary-item"><span class="key">Initial Step</span><span class="val">' + esc(artifact.initial_step_id) + '</span></div>';
    } else if (artifactType === 'agent') {
        if (artifact.agent_type) html += '<div class="builder-summary-item"><span class="key">Agent Type</span><span class="val">' + esc(artifact.agent_type) + '</span></div>';
        if (artifact.tools) {
            html += '<div class="builder-summary-item"><span class="key">Tools</span><span class="val">' + artifact.tools.length + '</span></div>';
            html += '<div class="builder-summary-item"><span class="key">Tool List</span><span class="val">' + esc(artifact.tools.map(t => t.name || '?').join(', ')) + '</span></div>';
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
            $('builder-result-graph').style.display = 'none';
        });
}

function _appendBubble(role, text) {
    _messages.push({ role, content: text });
    const chat = $('builder-chat');
    _following = _isNearBottom(chat);

    let html = '<div class="chat-bubble ' + role + '">';
    html += '<div class="chat-role-tag">' + esc(role) + '</div>';
    html += (role === 'user' || role === 'error') ? esc(text) : '<div class="md-body">' + renderMarkdown(text) + '</div>';
    html += '</div>';

    chat.insertAdjacentHTML('beforeend', html);
    if (role === 'user') _following = true;
    _autoScroll(chat);
}

export function copyBuilderResult() {
    const json = $('builder-result-json').textContent;
    navigator.clipboard?.writeText(json)
        .then(() => showStatus('builder-status', 'Copied to clipboard!', 'success'))
        .catch(() => {
            const ta = document.createElement('textarea');
            ta.value = json;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            showStatus('builder-status', 'Copied to clipboard!', 'success');
        });
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

export function launchBuilderResult() {
    const json = $('builder-result-json').textContent;
    let parsed;
    try { parsed = JSON.parse(json); } catch { showError('builder-status', 'Invalid JSON'); return; }

    const model = $('builder-model').value.trim() || undefined;
    const temp = numVal('builder-temp', 0.5);

    postJson('/api/fsm/launch', { fsm_json: parsed, model, temperature: temp, label: parsed.name || 'Built FSM' })
        .then(data => {
            showStatus('builder-status', 'Launched! Switching to Control Center...', 'success');
            return postJson('/api/fsm/' + data.instance_id + '/start', {}).then(() => {
                setTimeout(() => _showPage?.('control'), 500);
            });
        })
        .catch(e => showError('builder-status', 'Launch failed: ' + e.message));
}

export function resetBuilder() {
    if (_sessionId && !_complete) {
        fetchJson('/api/builder/' + _sessionId, { method: 'DELETE' }).catch(() => {});
    }
    _sessionId = null;
    _complete = false;
    _messages = [];
    _artifact = null;
    _sending = false;

    $('builder-chat').innerHTML = '';
    $('builder-result-panel').style.display = 'none';
    $('builder-input-area').style.display = 'none';
    showStatus('builder-status', '');
}
