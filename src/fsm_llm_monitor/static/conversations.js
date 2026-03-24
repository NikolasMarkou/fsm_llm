// FSM-LLM Monitor — Conversations Page

'use strict';

async function refreshConversations() {
    try {
        var filter = document.getElementById('conv-instance-filter');
        if (filter && App.instances.length > 0) {
            var currentVal = filter.value;
            var opts = '<option value="">All instances</option>';
            for (var i = 0; i < App.instances.length; i++) {
                var inst = App.instances[i];
                if (inst.instance_type === 'fsm') {
                    opts += '<option value="' + esc(inst.instance_id) + '">' + esc(inst.label || inst.instance_id) + '</option>';
                }
            }
            filter.innerHTML = opts;
            if (currentVal && filter.querySelector('option[value="' + currentVal + '"]')) {
                filter.value = currentVal;
            } else {
                filter.value = '';
            }
        }

        var statusFilter = document.getElementById('conv-status-filter');
        var statusVal = statusFilter ? statusFilter.value : 'all';
        var includeEnded = statusVal !== 'active';
        var resp = await fetch('/api/conversations?include_ended=' + includeEnded);
        var convs = await resp.json();

        // Client-side filter by status
        if (statusVal === 'active') {
            convs = convs.filter(function(c) { return !c.is_terminal; });
        } else if (statusVal === 'ended') {
            convs = convs.filter(function(c) { return c.is_terminal; });
        }

        var body = document.getElementById('conv-list-body');
        var empty = document.getElementById('conv-list-empty');
        if (convs.length === 0) {
            body.innerHTML = '';
            empty.style.display = 'block';
            return;
        }
        empty.style.display = 'none';
        var rows = '';
        for (var i = 0; i < convs.length; i++) {
            var c = convs[i];
            var badge = c.is_terminal ? 'badge-ended' : 'badge-active';
            var label = c.is_terminal ? 'ENDED' : 'ACTIVE';
            rows += '<tr data-conv-id="' + esc(c.conversation_id) + '" class="clickable-row"><td class="cell-truncate">' + esc(c.conversation_id.substring(0, 16)) + '</td><td>' + esc(c.current_state) + '</td><td>' + (c.stack_depth || 1) + '</td><td><span class="badge ' + badge + '">' + label + '</span></td></tr>';
        }
        body.innerHTML = rows;

        body.querySelectorAll('tr').forEach(function(tr) {
            tr.addEventListener('click', function() {
                var convId = tr.getAttribute('data-conv-id');
                if (convId) showConversationDetail(convId);
            });
        });
    } catch (e) {
        console.error('refreshConversations:', e);
    }
}

async function showConversationDetail(convId) {
    App.selectedConvId = convId;
    var detail = document.getElementById('conv-detail');
    var chatInput = document.getElementById('conv-chat-input');
    try {
        var resp = await fetch('/api/conversations/' + encodeURIComponent(convId));
        var data = await resp.json();
        if (data.error) {
            detail.innerHTML = '<span class="text-error">' + esc(data.error) + '</span>';
            if (chatInput) chatInput.style.display = 'none';
            return;
        }

        if (data.instance_id) {
            App.selectedConvInstanceId = data.instance_id;
        } else {
            App.selectedConvInstanceId = null;
            for (var i = 0; i < App.instances.length; i++) {
                if (App.instances[i].instance_type === 'fsm') {
                    App.selectedConvInstanceId = App.instances[i].instance_id;
                    break;
                }
            }
        }

        var html = '<div class="kv">';
        html += '<span class="key">ID:</span><span class="val mono-id">' + esc(data.conversation_id) + '</span>';
        html += '<span class="key">State:</span><span class="val text-primary-bold">' + esc(data.current_state) + '</span>';
        html += '<span class="key">Description:</span><span class="val">' + esc(data.state_description) + '</span>';
        html += '<span class="key">Terminal:</span><span class="val">' + (data.is_terminal ? '<span class="text-error">Yes</span>' : '<span class="text-success">No</span>') + '</span>';
        html += '<span class="key">Stack Depth:</span><span class="val">' + (data.stack_depth || 1) + '</span>';
        html += '</div>';

        if (data.context_data && Object.keys(data.context_data).length > 0) {
            html += '<div class="panel-title panel-title-spaced panel-title-flex"><span>Context Data</span><button class="btn btn-sm" onclick="event.stopPropagation();copyContextData()">Copy JSON</button></div>';
            html += '<div class="kv" id="conv-context-kv">';
            for (var k in data.context_data) {
                var v = data.context_data[k];
                html += '<span class="key">' + esc(k) + ':</span><span class="val">' + esc(typeof v === 'object' ? JSON.stringify(v) : String(v)) + '</span>';
            }
            html += '</div>';
            // Store context data for clipboard copy
            App._lastContextData = data.context_data;
        } else {
            App._lastContextData = null;
        }

        if (data.last_extraction) {
            html += '<div class="panel-title panel-title-spaced">Last Extraction (Pass 1)</div>';
            html += '<div class="llm-data-panel extraction">';
            html += _renderLLMData(data.last_extraction);
            html += '</div>';
        }

        if (data.last_transition) {
            html += '<div class="panel-title panel-title-spaced">Last Transition Decision</div>';
            html += '<div class="llm-data-panel transition">';
            html += _renderLLMData(data.last_transition);
            html += '</div>';
        }

        if (data.last_response) {
            html += '<div class="panel-title panel-title-spaced">Last Response Generation (Pass 2)</div>';
            html += '<div class="llm-data-panel response">';
            html += _renderLLMData(data.last_response);
            html += '</div>';
        }

        if (data.message_history && data.message_history.length > 0) {
            html += '<div class="panel-title panel-title-spaced">Message History (' + data.message_history.length + ')</div>';
            html += '<div class="chat-container" id="conv-chat-log">';
            for (var j = 0; j < data.message_history.length; j++) {
                var msg = data.message_history[j];
                var role = msg.role || 'system';
                var content = msg.content || '';
                var bubbleClass = role === 'user' ? 'user' : 'assistant';
                html += '<div class="chat-bubble ' + bubbleClass + '">';
                html += '<div class="chat-role-tag">' + esc(role) + '</div>';
                html += esc(content);
                html += '</div>';
            }
            html += '</div>';
        }

        if (data.is_terminal) {
            html += '<div class="ended-indicator">Conversation ended</div>';
        }

        detail.innerHTML = html;

        if (chatInput) {
            chatInput.style.display = (!data.is_terminal && App.selectedConvInstanceId) ? 'block' : 'none';
        }
    } catch (e) {
        detail.innerHTML = '<span class="error-message">Failed to load conversation</span>';
        if (chatInput) chatInput.style.display = 'none';
        console.error('showConversationDetail:', e);
    }
}

async function sendChatMessage() {
    if (!App.selectedConvId || !App.selectedConvInstanceId) return;
    var input = document.getElementById('conv-message-input');
    var message = input.value.trim();
    if (!message) return;
    input.value = '';
    input.disabled = true;

    var chatLog = document.getElementById('conv-chat-log');
    if (chatLog) {
        chatLog.insertAdjacentHTML('beforeend',
            '<div class="chat-bubble user"><div class="chat-role-tag">user</div>' + esc(message) + '</div>'
        );
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    try {
        var resp = await fetch('/api/fsm/' + encodeURIComponent(App.selectedConvInstanceId) + '/converse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: App.selectedConvId, message: message })
        });
        var data = await resp.json();
        if (data.error) {
            if (chatLog) {
                chatLog.insertAdjacentHTML('beforeend',
                    '<div class="chat-bubble error"><div class="chat-role-tag">error</div>' + esc(data.error) + '</div>'
                );
            }
        } else {
            await showConversationDetail(App.selectedConvId);
            var updatedLog = document.getElementById('conv-chat-log');
            if (updatedLog) updatedLog.scrollTop = updatedLog.scrollHeight;
            refreshConversations();
            if (App.currentPage === 'dashboard') refreshConversationTable();
            if (App.currentPage === 'control' && App.selectedDetailId && App.selectedDetailType === 'fsm') {
                refreshDetailPanel(App.selectedDetailId, App.selectedDetailType);
            }
        }
    } catch (e) {
        console.error('sendChatMessage:', e);
    }
    input.disabled = false;
    input.focus();
}
