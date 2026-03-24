// FSM-LLM Monitor — Conversations (renders inside Control Center drawer)

'use strict';

// refreshConversations is kept for compat but is now a no-op outside the drawer.
// Conversation list is rendered by renderFSMDetail in control.js.
async function refreshConversations() {
    // No-op — conversations are shown inline in the control center drawer
}

function showConversationInDrawer(instanceId, convId) {
    App.selectedConvInstanceId = instanceId;
    App.selectedConvId = convId;

    // Ensure we're on control page with drawer open
    if (App.currentPage !== 'control') {
        showPage('control');
    }

    // Open drawer if not already open
    var drawer = document.getElementById('ctrl-drawer');
    var backdrop = document.getElementById('ctrl-drawer-backdrop');
    if (drawer) drawer.style.display = 'block';
    if (backdrop) backdrop.style.display = 'block';

    // Show back button, hide instance content, show conv detail
    var backBtn = document.getElementById('ctrl-drawer-back');
    var drawerContent = document.getElementById('ctrl-drawer-content');
    var convWrapper = document.getElementById('conv-detail-wrapper');
    var eventsWrapper = document.getElementById('ctrl-drawer-events-wrapper');
    if (backBtn) backBtn.style.display = 'inline-block';
    if (drawerContent) drawerContent.style.display = 'none';
    if (convWrapper) convWrapper.style.display = 'block';
    if (eventsWrapper) eventsWrapper.style.display = 'none';

    var titleEl = document.getElementById('ctrl-drawer-title');
    var inst = App.instances.find(function(i) { return i.instance_id === instanceId; });
    var instLabel = inst ? (inst.label || instanceId.substring(0, 12)) : instanceId.substring(0, 12);
    if (titleEl) titleEl.textContent = instLabel + ' \u2192 Conversation';

    showConversationDetail(convId);
}

function drawerBack() {
    // Return to instance detail view
    var backBtn = document.getElementById('ctrl-drawer-back');
    var drawerContent = document.getElementById('ctrl-drawer-content');
    var convWrapper = document.getElementById('conv-detail-wrapper');
    var eventsWrapper = document.getElementById('ctrl-drawer-events-wrapper');
    if (backBtn) backBtn.style.display = 'none';
    if (drawerContent) drawerContent.style.display = 'block';
    if (convWrapper) convWrapper.style.display = 'none';
    if (eventsWrapper) eventsWrapper.style.display = 'block';

    App.selectedConvId = null;

    // Refresh the instance detail
    if (App.selectedDetailId && App.selectedDetailType) {
        refreshDetailPanel(App.selectedDetailId, App.selectedDetailType);
    }
}

async function showConversationDetail(convId) {
    App.selectedConvId = convId;
    var detail = document.getElementById('conv-detail');
    var chatInput = document.getElementById('conv-chat-input');
    if (!detail) return;
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
        } else if (!App.selectedConvInstanceId) {
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

        // Update drawer title with breadcrumb
        var titleEl = document.getElementById('ctrl-drawer-title');
        if (titleEl) {
            var inst = App.instances.find(function(i) { return i.instance_id === App.selectedConvInstanceId; });
            var instLabel = inst ? (inst.label || App.selectedConvInstanceId.substring(0, 12)) : '';
            titleEl.textContent = (instLabel ? instLabel + ' \u2192 ' : '') + data.current_state;
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
            if (App.currentPage === 'dashboard') refreshConversationTable();
        }
    } catch (e) {
        console.error('sendChatMessage:', e);
    }
    input.disabled = false;
    input.focus();
}
