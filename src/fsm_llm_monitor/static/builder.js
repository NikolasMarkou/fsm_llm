// FSM-LLM Monitor — Builder Page (Meta-Agent)
// Interactive artifact builder using the meta-agent conversational flow.
// Uses the same chat-bubble components as the conversation panel.

'use strict';

var _builderSessionId = null;
var _builderComplete = false;
var _builderMessages = [];  // {role: 'user'|'assistant', content: '...'}
var _builderArtifact = null;
var _builderSending = false;
var _builderFollowing = true;

function _isBuilderNearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 60;
}

function _builderAutoScroll(chat) {
    if (_builderFollowing) {
        chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
    }
    _updateBuilderJump();
}

function _updateBuilderJump() {
    var btn = document.getElementById('builder-jump-btn');
    var chat = document.getElementById('builder-chat');
    if (btn && chat) {
        if (_isBuilderNearBottom(chat)) {
            btn.classList.remove('visible');
        } else {
            btn.classList.add('visible');
        }
    }
}

function builderJumpToLatest() {
    var chat = document.getElementById('builder-chat');
    if (chat) {
        chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
        _builderFollowing = true;
        _updateBuilderJump();
    }
}

function startBuilderSession() {
    var model = document.getElementById('builder-model').value.trim();
    var temp = numVal('builder-temp', 0.7);

    _builderSessionId = null;
    _builderComplete = false;
    _builderMessages = [];
    _builderArtifact = null;
    _builderSending = false;

    document.getElementById('builder-chat').innerHTML = '';
    document.getElementById('builder-result-panel').style.display = 'none';
    document.getElementById('builder-input-area').style.display = 'flex';
    showStatus('builder-status', 'Starting session...', 'info');

    var body = { model: model || undefined, temperature: temp };

    fetchJson('/api/builder/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    }).then(function(data) {
        _builderSessionId = data.session_id;
        _appendBuilderBubble('assistant', data.response);
        showStatus('builder-status', 'Session active', 'success');
        document.getElementById('builder-message-input').focus();
        if (data.is_complete) _onBuilderComplete(data);
    }).catch(function(e) {
        showError('builder-status', 'Failed to start: ' + e.message);
    });
}

function sendBuilderMessage() {
    if (_builderSending || _builderComplete || !_builderSessionId) return;

    var input = document.getElementById('builder-message-input');
    var sendBtn = input ? input.nextElementSibling : null;
    var msg = input.value.trim();
    if (!msg) return;

    _builderSending = true;
    input.value = '';
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    _appendBuilderBubble('user', msg);

    var chat = document.getElementById('builder-chat');
    _addTypingIndicator(chat);

    fetchJson('/api/builder/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: _builderSessionId, message: msg })
    }).then(function(data) {
        _builderSending = false;
        _removeTypingIndicator();
        _appendBuilderBubble('assistant', data.response);
        showStatus('builder-status', 'Session active', 'success');

        if (data.is_complete) {
            _onBuilderComplete(data);
        } else {
            input.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            input.focus();
        }
    }).catch(function(e) {
        _builderSending = false;
        _removeTypingIndicator();
        _appendBuilderBubble('error', e.message || 'Request failed');
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.focus();
    });
}

function _onBuilderComplete(data) {
    _builderComplete = true;
    _builderArtifact = data.artifact_json || JSON.stringify(data.artifact, null, 2);

    document.getElementById('builder-input-area').style.display = 'none';
    var resultPanel = document.getElementById('builder-result-panel');
    resultPanel.style.display = 'block';

    var json = _builderArtifact || '{}';
    document.getElementById('builder-result-json').textContent = json;

    var statusHtml = '';
    if (data.is_valid) {
        statusHtml = '<span class="badge badge-completed">VALID</span> ';
        statusHtml += esc(data.artifact_type || 'unknown') + ' artifact ready';
    } else {
        statusHtml = '<span class="badge badge-failed">INVALID</span> ';
        if (data.validation_errors && data.validation_errors.length) {
            statusHtml += esc(data.validation_errors.join('; '));
        }
    }
    document.getElementById('builder-result-status').innerHTML = statusHtml;
    showStatus('builder-status', 'Build complete!', 'success');
}

function _appendBuilderBubble(role, text) {
    _builderMessages.push({ role: role, content: text });
    var chat = document.getElementById('builder-chat');
    _builderFollowing = _isBuilderNearBottom(chat);

    var html = '<div class="chat-bubble ' + role + '">';
    html += '<div class="chat-role-tag">' + esc(role) + '</div>';
    if (role === 'user') {
        html += esc(text);
    } else if (role === 'error') {
        html += esc(text);
    } else {
        html += '<div class="md-body">' + renderMarkdown(text) + '</div>';
    }
    html += '</div>';

    chat.insertAdjacentHTML('beforeend', html);
    if (role === 'user') _builderFollowing = true;
    _builderAutoScroll(chat);
}

function copyBuilderResult() {
    var json = document.getElementById('builder-result-json').textContent;
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(json).then(function() {
            showStatus('builder-status', 'Copied to clipboard!', 'success');
        });
    } else {
        var ta = document.createElement('textarea');
        ta.value = json;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showStatus('builder-status', 'Copied to clipboard!', 'success');
    }
}

function downloadBuilderResult() {
    var json = document.getElementById('builder-result-json').textContent;
    var blob = new Blob([json], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'artifact.json';
    a.click();
    URL.revokeObjectURL(url);
}

function launchBuilderResult() {
    var json = document.getElementById('builder-result-json').textContent;
    try {
        var parsed = JSON.parse(json);
    } catch (e) {
        showError('builder-status', 'Invalid JSON');
        return;
    }

    var model = document.getElementById('builder-model').value.trim() || undefined;
    var temp = numVal('builder-temp', 0.5);

    fetchJson('/api/fsm/launch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            fsm_json: parsed,
            model: model,
            temperature: temp,
            label: parsed.name || 'Built FSM'
        })
    }).then(function(data) {
        showStatus('builder-status', 'Launched! Switching to Control Center...', 'success');
        return fetchJson('/api/fsm/' + data.instance_id + '/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        }).then(function() {
            setTimeout(function() { showPage('control'); }, 500);
        });
    }).catch(function(e) {
        showError('builder-status', 'Launch failed: ' + e.message);
    });
}

function resetBuilder() {
    if (_builderSessionId && !_builderComplete) {
        fetchJson('/api/builder/' + _builderSessionId, { method: 'DELETE' }).catch(function() {});
    }
    _builderSessionId = null;
    _builderComplete = false;
    _builderMessages = [];
    _builderArtifact = null;
    _builderSending = false;

    document.getElementById('builder-chat').innerHTML = '';
    document.getElementById('builder-result-panel').style.display = 'none';
    document.getElementById('builder-input-area').style.display = 'none';
    showStatus('builder-status', '');
}
