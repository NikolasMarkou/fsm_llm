// FSM-LLM Monitor — Conversation Detail & Chat

import { state } from '../services/state.js';
import { fetchJson, postJson } from '../services/api.js';
import { $, esc, renderLLMData, copyToClipboard, showToast } from '../utils/dom.js';
import { renderMarkdown } from '../utils/markdown.js';

// Forward references (set by app.js)
let _showPage, _refreshActivityTable, _refreshDetailPanel;
let _detailRequestId = 0;

export function setDeps(deps) {
    _showPage = deps.showPage;
    _refreshActivityTable = deps.refreshActivityTable;
    _refreshDetailPanel = deps.refreshDetailPanel;
}

export function showConversationInDrawer(instanceId, convId) {
    state.selectedConvInstanceId = instanceId;
    state.selectedConvId = convId;

    if (state.currentPage !== 'control') _showPage?.('control');

    const drawer = $('ctrl-drawer');
    const backdrop = $('ctrl-drawer-backdrop');
    if (drawer) drawer.style.display = 'block';
    if (backdrop) backdrop.style.display = 'block';

    const backBtn = $('ctrl-drawer-back');
    const drawerContent = $('ctrl-drawer-content');
    const convWrapper = $('conv-detail-wrapper');
    const eventsWrapper = $('ctrl-drawer-events-wrapper');
    if (backBtn) backBtn.style.display = 'inline-block';
    if (drawerContent) drawerContent.style.display = 'none';
    if (convWrapper) convWrapper.style.display = 'block';
    if (eventsWrapper) eventsWrapper.style.display = 'none';

    const titleEl = $('ctrl-drawer-title');
    const inst = state.instances.find(i => i.instance_id === instanceId);
    const instLabel = inst ? (inst.label || instanceId.substring(0, 12)) : instanceId.substring(0, 12);
    if (titleEl) titleEl.textContent = instLabel + ' \u2192 Conversation';

    showConversationDetail(convId);
}

export function drawerBack() {
    const backBtn = $('ctrl-drawer-back');
    const drawerContent = $('ctrl-drawer-content');
    const convWrapper = $('conv-detail-wrapper');
    const eventsWrapper = $('ctrl-drawer-events-wrapper');
    if (backBtn) backBtn.style.display = 'none';
    if (drawerContent) drawerContent.style.display = 'block';
    if (convWrapper) convWrapper.style.display = 'none';
    if (eventsWrapper) eventsWrapper.style.display = 'block';

    state.selectedConvId = null;

    if (state.selectedDetailId && state.selectedDetailType) {
        _refreshDetailPanel?.(state.selectedDetailId, state.selectedDetailType);
    }
}

export async function showConversationDetail(convId) {
    const thisRequest = ++_detailRequestId;
    state.selectedConvId = convId;
    const detail = $('conv-detail');
    const chatInput = $('conv-chat-input');
    if (!detail) return;

    try {
        const data = await fetchJson('/api/conversations/' + encodeURIComponent(convId));
        if (thisRequest !== _detailRequestId) return;

        if (data.instance_id) {
            state.selectedConvInstanceId = data.instance_id;
        } else if (!state.selectedConvInstanceId) {
            const fsm = state.instances.find(i => i.instance_type === 'fsm');
            if (fsm) state.selectedConvInstanceId = fsm.instance_id;
        }

        let html = '<div class="kv">';
        html += '<span class="key">ID:</span><span class="val mono-id">' + esc(data.conversation_id) + '</span>';
        html += '<span class="key">State:</span><span class="val text-primary-bold">' + esc(data.current_state) + '</span>';
        html += '<span class="key">Description:</span><span class="val">' + esc(data.state_description) + '</span>';
        html += '<span class="key">Terminal:</span><span class="val">' + (data.is_terminal ? '<span class="text-error">Yes</span>' : '<span class="text-success">No</span>') + '</span>';
        html += '<span class="key">Stack Depth:</span><span class="val">' + (data.stack_depth || 1) + '</span>';
        html += '</div>';

        if (data.context_data && Object.keys(data.context_data).length > 0) {
            html += '<div class="panel-title panel-title-spaced panel-title-flex"><span>Context Data</span><button class="btn btn-sm" data-action="copy-context">Copy JSON</button></div>';
            html += '<div class="kv" id="conv-context-kv">';
            for (const k in data.context_data) {
                const v = data.context_data[k];
                html += '<span class="key">' + esc(k) + ':</span><span class="val">' + esc(typeof v === 'object' ? JSON.stringify(v) : String(v)) + '</span>';
            }
            html += '</div>';
            state._lastContextData = data.context_data;
        } else {
            state._lastContextData = null;
        }

        if (data.last_extraction) {
            html += '<div class="panel-title panel-title-spaced">Last Extraction (Pass 1)</div>';
            html += '<div class="llm-data-panel extraction">' + renderLLMData(data.last_extraction) + '</div>';
        }
        if (data.last_transition) {
            html += '<div class="panel-title panel-title-spaced">Last Transition Decision</div>';
            html += '<div class="llm-data-panel transition">' + renderLLMData(data.last_transition) + '</div>';
        }
        if (data.last_response) {
            html += '<div class="panel-title panel-title-spaced">Last Response Generation (Pass 2)</div>';
            html += '<div class="llm-data-panel response">' + renderLLMData(data.last_response) + '</div>';
        }

        if (data.message_history?.length > 0) {
            html += '<div class="panel-title panel-title-spaced">Message History (' + data.message_history.length + ')</div>';
            html += '<div class="chat-container" id="conv-chat-log">';
            for (const msg of data.message_history) {
                const role = msg.role || 'system';
                const content = msg.content || '';
                const bubbleClass = role === 'user' ? 'user' : 'assistant';
                html += '<div class="chat-bubble ' + bubbleClass + '">';
                html += '<div class="chat-role-tag">' + esc(role) + '</div>';
                html += role === 'user' ? esc(content) : '<div class="md-body">' + renderMarkdown(content) + '</div>';
                html += '</div>';
            }
            html += '</div>';
        }

        if (data.is_terminal) html += '<div class="ended-indicator">Conversation ended</div>';

        detail.innerHTML = html;

        const chatLogEl = $('conv-chat-log');
        if (chatLogEl) chatLogEl.scrollTop = chatLogEl.scrollHeight;

        if (chatInput) {
            chatInput.style.display = (!data.is_terminal && state.selectedConvInstanceId) ? 'block' : 'none';
        }

        const titleEl = $('ctrl-drawer-title');
        if (titleEl) {
            const inst = state.instances.find(i => i.instance_id === state.selectedConvInstanceId);
            const instLabel = inst ? (inst.label || state.selectedConvInstanceId.substring(0, 12)) : '';
            titleEl.textContent = (instLabel ? instLabel + ' \u2192 ' : '') + data.current_state;
        }
    } catch (e) {
        detail.innerHTML = '<span class="error-message">Failed to load conversation</span>';
        if (chatInput) chatInput.style.display = 'none';
        console.error('showConversationDetail:', e);
    }
}

export function copyContextData() {
    const data = state._lastContextData;
    if (!data) return;
    copyToClipboard(JSON.stringify(data, null, 2)).then(() => {
        const btn = $('conv-context-kv')?.previousElementSibling?.querySelector('.btn');
        if (btn) {
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.classList.add('btn-primary');
            setTimeout(() => { btn.textContent = orig; btn.classList.remove('btn-primary'); }, 1500);
        }
    });
}

function _smoothScrollToBottom(el) {
    el?.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
}

function _addTypingIndicator(chatLog) {
    if (!chatLog) return;
    chatLog.insertAdjacentHTML('beforeend',
        '<div class="chat-bubble assistant typing-indicator" id="typing-indicator">' +
        '<div class="chat-role-tag">assistant</div>' +
        '<span class="typing-dots"><span></span><span></span><span></span></span>' +
        '</div>'
    );
    _smoothScrollToBottom(chatLog);
}

function _removeTypingIndicator() {
    $('typing-indicator')?.remove();
}

// Re-export for builder.js to reuse
export { _addTypingIndicator as addTypingIndicator, _removeTypingIndicator as removeTypingIndicator };

export async function sendChatMessage() {
    if (!state.selectedConvId || !state.selectedConvInstanceId) return;
    const input = $('conv-message-input');
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    input.disabled = true;

    const chatLog = $('conv-chat-log');
    if (chatLog) {
        chatLog.insertAdjacentHTML('beforeend',
            '<div class="chat-bubble user"><div class="chat-role-tag">user</div>' + esc(message) + '</div>'
        );
        _smoothScrollToBottom(chatLog);
    }

    _addTypingIndicator(chatLog);

    try {
        const data = await postJson('/api/fsm/' + encodeURIComponent(state.selectedConvInstanceId) + '/converse', {
            conversation_id: state.selectedConvId, message
        });
        _removeTypingIndicator();

        const responseText = data.response || '';
        if (chatLog && responseText) {
            chatLog.insertAdjacentHTML('beforeend',
                '<div class="chat-bubble assistant"><div class="chat-role-tag">assistant</div><div class="md-body">' + renderMarkdown(responseText) + '</div></div>'
            );
            _smoothScrollToBottom(chatLog);
        }
        if (data.current_state || data.is_terminal) {
            const stateEl = $('conv-detail')?.querySelector('.text-primary-bold');
            if (stateEl && data.current_state) stateEl.textContent = data.current_state;
            if (data.is_terminal) {
                const chatInputEl = $('conv-chat-input');
                if (chatInputEl) chatInputEl.style.display = 'none';
            }
        }
        if (state.currentPage === 'dashboard') _refreshActivityTable?.();
    } catch (e) {
        _removeTypingIndicator();
        if (chatLog) {
            chatLog.insertAdjacentHTML('beforeend',
                '<div class="chat-bubble error"><div class="chat-role-tag">error</div>' + esc(e.message || 'Request failed') + '</div>'
            );
            _smoothScrollToBottom(chatLog);
        }
        console.error('sendChatMessage:', e);
    }
    input.disabled = false;
    input.focus();
}
