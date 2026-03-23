// FSM-LLM Monitor — Dashboard
// Vanilla JS — no frameworks needed

'use strict';

var ws = null;
var currentPage = 'dashboard';
var _presets = null;
var _wsRetryDelay = 3000;
var WS_MAX_DELAY = 30000;
var _capabilities = { fsm: true, workflows: false, agents: false };
var _instances = [];
var _selectedConvId = null;
var _selectedConvInstanceId = null;
var _selectedDetailId = null;  // currently selected instance in control center
var _selectedDetailType = null;
var _detailPollTimer = null;
var _agentUpdates = {};

// === REFRESH SCHEDULING ===

var _refreshTimers = {};
function scheduleRefresh(key, fn, delayMs) {
    if (_refreshTimers[key]) return;
    _refreshTimers[key] = setTimeout(function() {
        _refreshTimers[key] = null;
        fn();
    }, delayMs);
}

// === UTILS ===

function esc(s) {
    if (s == null) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function formatTime(ts) {
    if (!ts) return '';
    var d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts).substring(11, 19);
    return d.toLocaleTimeString('en-US', { hour12: false });
}

function updateClock() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
}

function numVal(id, fallback) {
    var v = parseFloat(document.getElementById(id).value);
    return Number.isFinite(v) ? v : fallback;
}

function intVal(id, fallback) {
    var v = parseInt(document.getElementById(id).value, 10);
    return Number.isFinite(v) ? v : fallback;
}

function showError(elementId, msg) {
    var el = document.getElementById(elementId);
    if (el) el.innerHTML = '<span class="error-message">' + esc(msg) + '</span>';
}

function renderResultBanner(success) {
    var cls = success ? 'success' : 'failure';
    var text = success ? 'Agent completed successfully' : 'Agent failed';
    return '<div class="result-banner ' + cls + '">' + text + '</div>';
}

function showStatus(elementId, msg, color) {
    var el = document.getElementById(elementId);
    if (el) el.innerHTML = '<span style="color:var(--' + color + ');">' + esc(msg) + '</span>';
}

function statusBadge(status) {
    var cls = 'badge-' + status;
    return '<span class="badge ' + cls + '">' + esc(status.toUpperCase()) + '</span>';
}

function _renderLLMData(obj) {
    if (!obj || typeof obj !== 'object') return '<span style="color:var(--text-dim);">No data</span>';
    var html = '';
    var keys = Object.keys(obj);
    for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        var v = obj[k];
        if (v === null || v === undefined) continue;
        var display;
        if (typeof v === 'object') {
            display = '<pre style="margin:2px 0;white-space:pre-wrap;font-size:11px;color:var(--text);">' + esc(JSON.stringify(v, null, 2)) + '</pre>';
        } else {
            display = '<span style="color:var(--text);">' + esc(String(v)) + '</span>';
        }
        html += '<div style="margin-bottom:4px;"><span style="color:var(--cyan);font-weight:600;font-size:10px;text-transform:uppercase;">' + esc(k) + ':</span> ' + display + '</div>';
    }
    return html || '<span style="color:var(--text-dim);">Empty</span>';
}

// === NAV ===

function showPage(page) {
    document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
    document.querySelectorAll('.sidebar-items button[data-page]').forEach(function(b) { b.classList.remove('active'); });
    var pageEl = document.getElementById('page-' + page);
    if (pageEl) pageEl.classList.add('active');
    var btn = document.querySelector('.sidebar-items button[data-page="' + page + '"]');
    if (btn) btn.classList.add('active');
    currentPage = page;

    var refreshMap = {
        'conversations': refreshConversations,
        'logs': refreshLogs,
        'settings': loadSettings,
        'control': refreshControlCenter
    };
    if (refreshMap[page]) refreshMap[page]();

    if (page === 'visualizer') {
        var activeTab = document.querySelector('.tab-content.active');
        if (activeTab && activeTab.id === 'tab-agents') {
            var sel = document.getElementById('viz-agent-type');
            if (sel && sel.value) visualizeGraph('agent', sel.value);
        } else if (activeTab && activeTab.id === 'tab-workflows') {
            var sel2 = document.getElementById('viz-wf-type');
            if (sel2 && sel2.value) visualizeGraph('workflow', sel2.value);
        }
    }
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

// === TABS ===

function switchTab(tabId, btn) {
    // Find parent tab-bar to scope tab switching
    var tabBar = btn ? btn.parentElement : null;
    var scope = tabBar ? tabBar.parentElement : document;
    scope.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
    if (tabBar) tabBar.querySelectorAll('.tab').forEach(function(b) { b.classList.remove('active'); });
    var tabEl = document.getElementById(tabId);
    if (tabEl) tabEl.classList.add('active');
    if (btn) btn.classList.add('active');
}

// === WEBSOCKET ===

function connectWS() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen = function() {
        document.getElementById('ws-status').textContent = 'CONNECTED';
        document.getElementById('ws-status').className = 'connected';
        _wsRetryDelay = 3000;
    };
    ws.onmessage = function(event) {
        try {
            var data = JSON.parse(event.data);
            if (data.type === 'metrics') updateMetrics(data.data);
            if (data.events) updateEvents(data.events);
            if (data.instances) {
                _instances = data.instances;
                renderInstanceGrid();
                if (currentPage === 'control') {
                    renderControlFSMs();
                    renderControlWorkflows();
                    renderControlAgents();
                }
            }
            if (data.agent_updates) {
                _agentUpdates = data.agent_updates;
                updateRunningAgents(data.agent_updates);
            }
            // Refresh conversation tables on conversation-relevant events only
            if (data.events && data.events.length > 0) {
                var hasConvEvent = data.events.some(function(e) {
                    var t = e.event_type;
                    return t === 'conversation_start' || t === 'conversation_end'
                        || t === 'state_transition' || t === 'post_processing';
                });
                if (hasConvEvent) {
                    if (currentPage === 'dashboard') scheduleRefresh('dash-conv', refreshConversationTable, 3000);
                    if (currentPage === 'conversations') {
                        scheduleRefresh('conv-list', refreshConversations, 2000);
                        if (_selectedConvId) {
                            var convId = _selectedConvId;
                            scheduleRefresh('conv-detail', function() { showConversationDetail(convId); }, 2000);
                        }
                    }
                    if (currentPage === 'control' && _selectedDetailId && _selectedDetailType === 'fsm') {
                        var detailId = _selectedDetailId;
                        scheduleRefresh('ctrl-detail', function() { refreshDetailPanel(detailId, 'fsm'); }, 2000);
                    }
                }
            }
        } catch (e) {
            console.error('WS message parse error:', e);
        }
    };
    ws.onclose = function() {
        document.getElementById('ws-status').textContent = 'DISCONNECTED';
        document.getElementById('ws-status').className = 'blink';
        setTimeout(connectWS, _wsRetryDelay);
        _wsRetryDelay = Math.min(_wsRetryDelay * 2, WS_MAX_DELAY);
    };
    ws.onerror = function() { ws.close(); };
}

// === DASHBOARD ===

function updateMetrics(m) {
    document.getElementById('m-conversations').textContent = m.active_conversations;
    document.getElementById('m-events').textContent = m.total_events;
    document.getElementById('m-transitions').textContent = m.total_transitions;
    document.getElementById('m-errors').textContent = m.total_errors;
}

function updateEvents(events) {
    var log = document.getElementById('event-log');
    var emptyHint = document.getElementById('event-empty');
    if (emptyHint) emptyHint.remove();

    var html = '';
    for (var i = 0; i < events.length; i++) {
        var e = events[i];
        var ts = formatTime(e.timestamp);
        var level = (e.level || 'INFO').toLowerCase();
        html += '<div class="entry ' + level + '"><span class="ts">' + ts + '</span><span class="type">' + esc(e.event_type) + '</span><span class="msg">' + esc(e.message) + '</span></div>';
    }
    log.insertAdjacentHTML('afterbegin', html);
    while (log.children.length > 200) log.removeChild(log.lastChild);
}

// === INSTANCE GRID (dashboard) ===

function _relativeTime(dateStr) {
    if (!dateStr) return '';
    var diff = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
    if (diff < 60) return diff + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return Math.floor(diff / 86400) + 'd ago';
}

function renderInstanceGrid() {
    var grid = document.getElementById('instances-grid');
    var empty = document.getElementById('instances-empty');
    if (!grid) return;
    if (_instances.length === 0) {
        grid.innerHTML = '';
        if (empty) empty.style.display = 'block';
        return;
    }
    if (empty) empty.style.display = 'none';
    var html = '';
    for (var i = 0; i < _instances.length; i++) {
        var inst = _instances[i];
        html += '<div class="instance-card" onclick="navigateToInstance(\'' + esc(inst.instance_id) + '\',\'' + esc(inst.instance_type) + '\')">';
        html += '<div class="inst-label">' + esc(inst.label || inst.instance_id) + '</div>';
        html += '<div style="display:flex;justify-content:space-between;align-items:center;">';
        html += '<div class="inst-type">' + esc(inst.instance_type) + '</div>';
        html += '<div class="inst-status">' + statusBadge(inst.status) + '</div>';
        html += '</div>';
        // Extra info by type
        var extra = '';
        if (inst.instance_type === 'fsm' && inst.conversation_count !== undefined) {
            extra = inst.conversation_count + ' conversation' + (inst.conversation_count !== 1 ? 's' : '');
        } else if (inst.instance_type === 'agent' && inst.agent_type) {
            extra = inst.agent_type;
        } else if (inst.instance_type === 'workflow' && inst.active_workflows !== undefined) {
            extra = inst.active_workflows + ' active';
        }
        if (extra || inst.created_at) {
            html += '<div style="font-size:10px;color:var(--text-dim);margin-top:4px;display:flex;justify-content:space-between;">';
            html += '<span>' + esc(extra) + '</span>';
            if (inst.created_at) html += '<span>' + _relativeTime(inst.created_at) + '</span>';
            html += '</div>';
        }
        html += '</div>';
    }
    grid.innerHTML = html;
}

async function refreshInstances() {
    try {
        var resp = await fetch('/api/instances');
        _instances = await resp.json();
        renderInstanceGrid();
    } catch (e) {
        console.error('refreshInstances:', e);
    }
}

async function refreshConversationTable() {
    try {
        var resp = await fetch('/api/conversations');
        var convs = await resp.json();
        var body = document.getElementById('conv-table-body');
        var empty = document.getElementById('conv-empty');
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
            rows += '<tr><td class="cell-truncate">' + esc(c.conversation_id.substring(0, 16)) + '</td><td>' + esc(c.current_state) + '</td><td>' + c.message_history.length + '</td><td><span class="badge ' + badge + '">' + label + '</span></td></tr>';
        }
        body.innerHTML = rows;
    } catch (e) {
        console.error('refreshConversationTable:', e);
    }
}

// === CONVERSATIONS (interactive inspector) ===

async function refreshConversations() {
    try {
        // Update instance filter dropdown
        var filter = document.getElementById('conv-instance-filter');
        if (filter && _instances.length > 0) {
            var currentVal = filter.value;
            var opts = '<option value="">All instances</option>';
            for (var i = 0; i < _instances.length; i++) {
                var inst = _instances[i];
                if (inst.instance_type === 'fsm') {
                    opts += '<option value="' + esc(inst.instance_id) + '">' + esc(inst.label || inst.instance_id) + '</option>';
                }
            }
            filter.innerHTML = opts;
            // Reset to "All instances" if previous selection no longer exists
            if (currentVal && filter.querySelector('option[value="' + currentVal + '"]')) {
                filter.value = currentVal;
            } else {
                filter.value = '';
            }
        }

        var resp = await fetch('/api/conversations');
        var convs = await resp.json();
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
            rows += '<tr data-conv-id="' + esc(c.conversation_id) + '" style="cursor:pointer;"><td class="cell-truncate">' + esc(c.conversation_id.substring(0, 16)) + '</td><td>' + esc(c.current_state) + '</td><td>' + (c.stack_depth || 1) + '</td><td><span class="badge ' + badge + '">' + label + '</span></td></tr>';
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
    _selectedConvId = convId;
    var detail = document.getElementById('conv-detail');
    var chatInput = document.getElementById('conv-chat-input');
    try {
        var resp = await fetch('/api/conversations/' + encodeURIComponent(convId));
        var data = await resp.json();
        if (data.error) {
            detail.innerHTML = '<span style="color:var(--red);">' + esc(data.error) + '</span>';
            if (chatInput) chatInput.style.display = 'none';
            return;
        }

        // Use instance_id from snapshot (set by backend), fallback to search
        if (data.instance_id) {
            _selectedConvInstanceId = data.instance_id;
        } else {
            _selectedConvInstanceId = null;
            for (var i = 0; i < _instances.length; i++) {
                if (_instances[i].instance_type === 'fsm') {
                    _selectedConvInstanceId = _instances[i].instance_id;
                    break;
                }
            }
        }

        var html = '<div class="kv">';
        html += '<span class="key">ID:</span><span class="val">' + esc(data.conversation_id) + '</span>';
        html += '<span class="key">State:</span><span class="val" style="color:var(--primary);font-weight:600;">' + esc(data.current_state) + '</span>';
        html += '<span class="key">Description:</span><span class="val">' + esc(data.state_description) + '</span>';
        html += '<span class="key">Terminal:</span><span class="val">' + (data.is_terminal ? '<span style="color:var(--red);">YES</span>' : '<span style="color:var(--success);">NO</span>') + '</span>';
        html += '<span class="key">Stack Depth:</span><span class="val">' + (data.stack_depth || 1) + '</span>';
        html += '</div>';

        // Context data
        if (data.context_data && Object.keys(data.context_data).length > 0) {
            html += '<div class="panel-title" style="margin-top:12px;">CONTEXT DATA</div>';
            html += '<div class="kv">';
            for (var k in data.context_data) {
                var v = data.context_data[k];
                html += '<span class="key">' + esc(k) + ':</span><span class="val">' + esc(typeof v === 'object' ? JSON.stringify(v) : String(v)) + '</span>';
            }
            html += '</div>';
        }

        // LLM Interaction: Last Extraction
        if (data.last_extraction) {
            html += '<div class="panel-title" style="margin-top:12px;">LAST EXTRACTION (Pass 1)</div>';
            html += '<div class="llm-panel" style="background:rgba(50,116,217,0.06);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;max-height:200px;overflow-y:auto;">';
            html += _renderLLMData(data.last_extraction);
            html += '</div>';
        }

        // LLM Interaction: Last Transition
        if (data.last_transition) {
            html += '<div class="panel-title" style="margin-top:12px;">LAST TRANSITION DECISION</div>';
            html += '<div class="llm-panel" style="background:rgba(255,152,48,0.06);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;max-height:200px;overflow-y:auto;">';
            html += _renderLLMData(data.last_transition);
            html += '</div>';
        }

        // LLM Interaction: Last Response
        if (data.last_response) {
            html += '<div class="panel-title" style="margin-top:12px;">LAST RESPONSE GENERATION (Pass 2)</div>';
            html += '<div class="llm-panel" style="background:rgba(115,191,105,0.06);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;max-height:200px;overflow-y:auto;">';
            html += _renderLLMData(data.last_response);
            html += '</div>';
        }

        // Message history
        if (data.message_history && data.message_history.length > 0) {
            html += '<div class="panel-title" style="margin-top:12px;">MESSAGE HISTORY (' + data.message_history.length + ')</div>';
            html += '<div class="event-log" id="conv-chat-log" style="max-height:300px;">';
            for (var j = 0; j < data.message_history.length; j++) {
                var msg = data.message_history[j];
                var role = msg.role || 'system';
                var content = msg.content || '';
                var roleColor = role === 'user' ? 'var(--cyan)' : 'var(--primary)';
                html += '<div class="entry"><span class="type" style="color:' + roleColor + ';width:60px;">' + esc(role.toUpperCase()) + '</span><span class="msg">' + esc(content) + '</span></div>';
            }
            html += '</div>';
        }

        // Show ended indicator for terminal conversations
        if (data.is_terminal) {
            html += '<div class="ended-indicator">Conversation ended</div>';
        }

        detail.innerHTML = html;

        // Show chat input if conversation is active and belongs to a managed FSM
        if (chatInput) {
            chatInput.style.display = (!data.is_terminal && _selectedConvInstanceId) ? 'block' : 'none';
        }
    } catch (e) {
        detail.innerHTML = '<span class="error-message">Failed to load conversation</span>';
        if (chatInput) chatInput.style.display = 'none';
        console.error('showConversationDetail:', e);
    }
}

async function sendChatMessage() {
    if (!_selectedConvId || !_selectedConvInstanceId) return;
    var input = document.getElementById('conv-message-input');
    var message = input.value.trim();
    if (!message) return;
    input.value = '';
    input.disabled = true;

    // Append user message immediately
    var chatLog = document.getElementById('conv-chat-log');
    if (chatLog) {
        chatLog.insertAdjacentHTML('beforeend',
            '<div class="entry"><span class="type" style="color:var(--cyan);width:60px;">USER</span><span class="msg">' + esc(message) + '</span></div>'
        );
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    try {
        var resp = await fetch('/api/fsm/' + encodeURIComponent(_selectedConvInstanceId) + '/converse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: _selectedConvId, message: message })
        });
        var data = await resp.json();
        if (data.error) {
            if (chatLog) {
                chatLog.insertAdjacentHTML('beforeend',
                    '<div class="entry error"><span class="type" style="width:60px;">ERROR</span><span class="msg">' + esc(data.error) + '</span></div>'
                );
            }
        } else {
            // Refresh full detail to update LLM panels (extraction, transition, response)
            await showConversationDetail(_selectedConvId);
            // Auto-scroll chat log to bottom
            var updatedLog = document.getElementById('conv-chat-log');
            if (updatedLog) updatedLog.scrollTop = updatedLog.scrollHeight;
            // Immediately sync all visible conversation panels
            refreshConversations();
            if (currentPage === 'dashboard') refreshConversationTable();
            if (currentPage === 'control' && _selectedDetailId && _selectedDetailType === 'fsm') {
                refreshDetailPanel(_selectedDetailId, _selectedDetailType);
            }
        }
    } catch (e) {
        console.error('sendChatMessage:', e);
    }
    input.disabled = false;
    input.focus();
}

// === LAUNCH MODAL ===

function showLaunchModal() {
    document.getElementById('launch-modal').style.display = 'flex';
    loadLaunchPresets();
    checkCapabilities();
}

function closeLaunchModal() {
    document.getElementById('launch-modal').style.display = 'none';
}

async function checkCapabilities() {
    try {
        var resp = await fetch('/api/capabilities');
        _capabilities = await resp.json();
        var wfUnavail = document.getElementById('launch-wf-unavailable');
        var agentUnavail = document.getElementById('launch-agent-unavailable');
        if (wfUnavail) wfUnavail.style.display = _capabilities.workflows ? 'none' : 'block';
        if (agentUnavail) agentUnavail.style.display = _capabilities.agents ? 'none' : 'block';
    } catch (e) {
        console.error('checkCapabilities:', e);
    }
}

function toggleLaunchFSMSource() {
    var source = document.getElementById('launch-fsm-source').value;
    document.getElementById('launch-fsm-preset-section').style.display = source === 'preset' ? 'block' : 'none';
    document.getElementById('launch-fsm-json-section').style.display = source === 'json' ? 'block' : 'none';
}

async function loadLaunchPresets() {
    if (_presets) {
        renderLaunchPresets(_presets);
        return;
    }
    try {
        var resp = await fetch('/api/presets');
        _presets = await resp.json();
        renderLaunchPresets(_presets);
    } catch (e) {
        console.error('loadLaunchPresets:', e);
    }
}

function renderLaunchPresets(presets) {
    var items = presets.fsm || [];
    var container = document.getElementById('launch-preset-list');
    if (!container) return;

    // Category filter
    var categories = ['all'];
    for (var ci = 0; ci < items.length; ci++) {
        var cat = items[ci].category || 'other';
        if (categories.indexOf(cat) === -1) categories.push(cat);
    }
    var filterHtml = '<div class="preset-filters" style="margin-bottom:8px;display:flex;gap:4px;flex-wrap:wrap;">';
    for (var fi = 0; fi < categories.length; fi++) {
        var c = categories[fi];
        filterHtml += '<button class="btn preset-filter-btn' + (c === 'all' ? ' btn-primary' : '') + '" data-cat="' + esc(c) + '" style="font-size:10px;padding:2px 8px;" onclick="filterPresets(\'' + esc(c) + '\')">' + esc(c) + '</button>';
    }
    filterHtml += '</div>';
    container.innerHTML = filterHtml;

    var listDiv = document.createElement('div');
    listDiv.id = 'preset-items';
    listDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;';
    container.appendChild(listDiv);

    for (var i = 0; i < items.length; i++) {
        var p = items[i];
        var card = document.createElement('div');
        card.className = 'preset-card';
        card.setAttribute('data-category', p.category || 'other');
        card.innerHTML = '<div class="preset-name">' + esc(p.name) + '</div><div class="preset-category">' + esc(p.category || '') + '</div><div class="preset-desc">' + esc(p.description || '') + '</div>';
        (function(id, name) {
            card.addEventListener('click', function() {
                document.getElementById('launch-fsm-preset-id').value = id;
                // Highlight selected
                container.querySelectorAll('.preset-card').forEach(function(c) { c.style.borderColor = ''; });
                card.style.borderColor = 'var(--primary)';
                if (!document.getElementById('launch-fsm-label').value) {
                    document.getElementById('launch-fsm-label').value = name.replace(/\s*\(.*\)/, '');
                }
            });
        })(p.id, p.name);
        listDiv.appendChild(card);
    }
}

function filterPresets(cat) {
    var cards = document.querySelectorAll('#preset-items .preset-card');
    for (var i = 0; i < cards.length; i++) {
        cards[i].style.display = (cat === 'all' || cards[i].getAttribute('data-category') === cat) ? '' : 'none';
    }
    var btns = document.querySelectorAll('.preset-filter-btn');
    for (var j = 0; j < btns.length; j++) {
        btns[j].className = btns[j].getAttribute('data-cat') === cat ? 'btn preset-filter-btn btn-primary' : 'btn preset-filter-btn';
    }
}

async function doLaunchFSM() {
    var source = document.getElementById('launch-fsm-source').value;
    var body = {
        model: document.getElementById('launch-fsm-model').value || 'ollama_chat/qwen3.5:4b',
        temperature: numVal('launch-fsm-temp', 0.5),
        label: document.getElementById('launch-fsm-label').value
    };

    if (source === 'preset') {
        body.preset_id = document.getElementById('launch-fsm-preset-id').value;
        if (!body.preset_id) {
            showError('launch-fsm-status', 'Select a preset');
            return;
        }
    } else {
        try {
            body.fsm_json = JSON.parse(document.getElementById('launch-fsm-json').value);
        } catch (e) {
            showError('launch-fsm-status', 'Invalid JSON');
            return;
        }
    }

    try {
        var resp = await fetch('/api/fsm/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        var data = await resp.json();
        if (data.error) {
            showError('launch-fsm-status', data.error);
            return;
        }
        showStatus('launch-fsm-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        refreshInstances();

        // Auto-start a conversation
        var startResp = await fetch('/api/fsm/' + encodeURIComponent(data.instance_id) + '/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_context: {} })
        });
        var startData = await startResp.json();
        if (!startData.error) {
            _selectedConvInstanceId = data.instance_id;
            _selectedConvId = startData.conversation_id;
            setTimeout(function() {
                closeLaunchModal();
                showPage('conversations');
                refreshConversations();
                if (_selectedConvId) showConversationDetail(_selectedConvId);
            }, 500);
        }
    } catch (e) {
        showError('launch-fsm-status', 'Launch failed: ' + e.message);
    }
}

async function doLaunchWorkflow() {
    if (!_capabilities.workflows) {
        showError('launch-wf-status', 'Workflow extension not installed');
        return;
    }
    var body = {
        label: document.getElementById('launch-wf-label').value
    };
    var ctxText = document.getElementById('launch-wf-context').value.trim();
    if (ctxText) {
        try {
            body.initial_context = JSON.parse(ctxText);
        } catch (e) {
            showError('launch-wf-status', 'Invalid JSON context');
            return;
        }
    }
    try {
        var resp = await fetch('/api/workflow/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        var data = await resp.json();
        if (data.error) {
            showError('launch-wf-status', data.error);
            return;
        }
        showStatus('launch-wf-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        refreshInstances();
    } catch (e) {
        showError('launch-wf-status', 'Launch failed');
    }
}

// === STUB TOOL BUILDER ===

var _stubToolCount = 0;

function addStubTool() {
    var container = document.getElementById('launch-agent-tools');
    var idx = _stubToolCount++;
    var row = document.createElement('div');
    row.className = 'stub-tool-row';
    row.id = 'stub-tool-' + idx;
    row.innerHTML =
        '<input type="text" placeholder="Name" class="stub-name">' +
        '<input type="text" placeholder="Description" class="stub-desc">' +
        '<input type="text" placeholder="Stub response" class="stub-resp" value="Tool executed successfully">' +
        '<button onclick="this.parentElement.remove()">&times;</button>';
    container.appendChild(row);
}

function getStubTools() {
    var tools = [];
    document.querySelectorAll('#launch-agent-tools .stub-tool-row').forEach(function(row) {
        var name = row.querySelector('.stub-name').value.trim();
        var desc = row.querySelector('.stub-desc').value.trim();
        var resp = row.querySelector('.stub-resp').value.trim();
        if (name && desc) {
            tools.push({ name: name, description: desc, stub_response: resp || 'Tool executed successfully' });
        }
    });
    return tools;
}

var TOOL_BASED_AGENTS = ['ReactAgent', 'ReflexionAgent', 'PlanExecuteAgent', 'REWOOAgent', 'ADaPTAgent'];

function onAgentTypeChange() {
    var agentType = document.getElementById('launch-agent-type').value;
    var needsTools = TOOL_BASED_AGENTS.indexOf(agentType) !== -1;
    var toolSection = document.getElementById('launch-agent-tools');
    var addToolBtn = toolSection ? toolSection.nextElementSibling : null;
    var toolTitle = toolSection ? toolSection.previousElementSibling : null;
    if (toolSection) toolSection.style.display = needsTools ? '' : 'none';
    if (addToolBtn && addToolBtn.tagName === 'BUTTON') addToolBtn.style.display = needsTools ? '' : 'none';
    if (toolTitle && toolTitle.classList.contains('panel-title')) toolTitle.style.display = needsTools ? '' : 'none';
}

async function doLaunchAgent() {
    if (!_capabilities.agents) {
        showError('launch-agent-status', 'Agent extension not installed');
        return;
    }
    var agentType = document.getElementById('launch-agent-type').value;
    var needsTools = TOOL_BASED_AGENTS.indexOf(agentType) !== -1;
    var tools = needsTools ? getStubTools() : [];
    if (needsTools && tools.length === 0) {
        showError('launch-agent-status', 'Add at least one tool for ' + agentType);
        return;
    }
    var task = document.getElementById('launch-agent-task').value.trim();
    if (!task) {
        showError('launch-agent-status', 'Enter a task');
        return;
    }
    var body = {
        agent_type: agentType,
        task: task,
        model: document.getElementById('launch-agent-model').value || 'ollama_chat/qwen3.5:4b',
        max_iterations: intVal('launch-agent-iters', 10),
        tools: tools,
        label: document.getElementById('launch-agent-label').value
    };
    try {
        var resp = await fetch('/api/agent/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        var data = await resp.json();
        if (data.error) {
            showError('launch-agent-status', data.error);
            return;
        }
        showStatus('launch-agent-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        refreshInstances();
        setTimeout(function() {
            closeLaunchModal();
            showPage('control');
        }, 500);
    } catch (e) {
        showError('launch-agent-status', 'Launch failed');
    }
}

// === CONTROL CENTER ===

async function refreshControlCenter() {
    await refreshInstances();
    renderControlFSMs();
    renderControlWorkflows();
    renderControlAgents();
    // Refresh detail panel if one is open
    if (_selectedDetailId) {
        refreshDetailPanel(_selectedDetailId, _selectedDetailType);
    }
}

function closeDetail(panelId) {
    document.getElementById(panelId).style.display = 'none';
    _selectedDetailId = null;
    _selectedDetailType = null;
    if (_detailPollTimer) { clearInterval(_detailPollTimer); _detailPollTimer = null; }
    // Deselect rows
    document.querySelectorAll('tr.clickable-row.selected').forEach(function(r) { r.classList.remove('selected'); });
}

function navigateToInstance(instanceId, instanceType) {
    showPage('control');
    var tabMap = { fsm: 'ctrl-tab-fsm', workflow: 'ctrl-tab-workflows', agent: 'ctrl-tab-agents' };
    var tabId = tabMap[instanceType];
    if (tabId) {
        var tabBtns = document.querySelectorAll('#page-control .tab-bar .tab');
        tabBtns.forEach(function(btn) {
            if (btn.getAttribute('data-tab') === tabId) {
                switchTab(tabId, btn);
            }
        });
    }
    setTimeout(function() { selectInstance(instanceId, instanceType); }, 100);
}

function selectInstance(instanceId, type) {
    // Deselect previous
    document.querySelectorAll('tr.clickable-row.selected').forEach(function(r) { r.classList.remove('selected'); });
    // Select new row
    var row = document.querySelector('tr[data-instance-id="' + instanceId + '"]');
    if (row) row.classList.add('selected');

    _selectedDetailId = instanceId;
    _selectedDetailType = type;

    // Show the right detail panel, hide others
    var panels = { fsm: 'ctrl-fsm-detail', workflow: 'ctrl-wf-detail', agent: 'ctrl-agent-detail' };
    for (var t in panels) {
        var p = document.getElementById(panels[t]);
        if (p) p.style.display = (t === type) ? 'block' : 'none';
    }

    refreshDetailPanel(instanceId, type);

    // Start polling for live updates
    if (_detailPollTimer) clearInterval(_detailPollTimer);
    _detailPollTimer = setInterval(function() {
        if (_selectedDetailId && currentPage === 'control') {
            refreshDetailPanel(_selectedDetailId, _selectedDetailType);
        }
    }, 2000);
}

async function refreshDetailPanel(instanceId, type) {
    if (type === 'fsm') await renderFSMDetail(instanceId);
    else if (type === 'workflow') await renderWorkflowDetail(instanceId);
    else if (type === 'agent') await renderAgentDetail(instanceId);
    // Always refresh events
    await refreshDetailEvents(instanceId, type);
}

async function refreshDetailEvents(instanceId, type) {
    var eventsIds = { fsm: 'ctrl-fsm-events', workflow: 'ctrl-wf-events', agent: 'ctrl-agent-events' };
    var logEl = document.getElementById(eventsIds[type]);
    if (!logEl) return;
    try {
        var resp = await fetch('/api/instances/' + encodeURIComponent(instanceId) + '/events?limit=50');
        var events = await resp.json();
        if (events.length === 0) {
            logEl.innerHTML = '<div class="empty-hint" style="padding:8px;">No events yet...</div>';
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

// --- FSM Detail ---

async function renderFSMDetail(instanceId) {
    var titleEl = document.getElementById('ctrl-fsm-detail-title');
    var contentEl = document.getElementById('ctrl-fsm-detail-content');
    var inst = _instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    titleEl.textContent = inst.label || instanceId;

    try {
        var resp = await fetch('/api/fsm/' + encodeURIComponent(instanceId) + '/conversations');
        var convs = await resp.json();

        var html = '<div class="kv" style="margin-bottom:8px;">';
        html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
        html += '<span class="key">Source:</span><span class="val">' + esc(inst.source || 'custom') + '</span>';
        html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
        html += '</div>';

        html += '<div class="panel-title">CONVERSATIONS (' + convs.length + ')</div>';
        if (convs.length === 0 || (convs.length === 1 && convs[0].error)) {
            html += '<div class="empty-hint" style="padding:8px;">No active conversations.</div>';
            if (inst.status === 'running') {
                html += '<button class="btn btn-primary btn-sm" style="margin-top:4px;" onclick="startConversationOn(\'' + esc(instanceId) + '\')">START CONVERSATION</button>';
            }
        } else {
            for (var i = 0; i < convs.length; i++) {
                var c = convs[i];
                if (c.error) continue;
                html += '<div class="conv-card" onclick="goToConversation(\'' + esc(instanceId) + '\',\'' + esc(c.conversation_id) + '\')">';
                html += '<div class="conv-info">';
                html += '<span class="mono-id">' + esc(c.conversation_id.substring(0, 12)) + '</span>';
                html += '<span class="conv-state">' + esc(c.current_state) + '</span>';
                html += '<span style="color:var(--text-dim);">' + (c.message_history ? c.message_history.length : 0) + ' msgs</span>';
                html += '</div>';
                html += '<div>' + (c.is_terminal ? statusBadge('ended') : statusBadge('active')) + '</div>';
                html += '</div>';
            }
            if (inst.status === 'running') {
                html += '<button class="btn btn-sm" style="margin-top:4px;" onclick="startConversationOn(\'' + esc(instanceId) + '\')">+ NEW CONVERSATION</button>';
            }
        }
        contentEl.innerHTML = html;
    } catch (e) {
        contentEl.innerHTML = '<span class="error-message">Failed to load FSM detail</span>';
    }
}

function goToConversation(instanceId, convId) {
    _selectedConvInstanceId = instanceId;
    showPage('conversations');
    setTimeout(function() { showConversationDetail(convId); }, 300);
}

// --- Workflow Detail ---

async function renderWorkflowDetail(instanceId) {
    var titleEl = document.getElementById('ctrl-wf-detail-title');
    var contentEl = document.getElementById('ctrl-wf-detail-content');
    var inst = _instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    titleEl.textContent = inst.label || instanceId;

    var html = '<div class="kv" style="margin-bottom:8px;">';
    html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
    html += '<span class="key">Status:</span><span class="val">' + statusBadge(inst.status) + '</span>';
    html += '<span class="key">Active Workflows:</span><span class="val">' + (inst.active_workflows || 0) + '</span>';
    html += '</div>';

    try {
        var resp = await fetch('/api/workflow/' + encodeURIComponent(instanceId) + '/instances');
        var wfInstances = await resp.json();

        html += '<div class="panel-title">WORKFLOW INSTANCES (' + wfInstances.length + ')</div>';
        if (wfInstances.length === 0) {
            html += '<div class="empty-hint" style="padding:8px;">No workflow instances.</div>';
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
                    html += '<span style="color:var(--text-dim);">' + formatTime(wf.created_at) + '</span>';
                }
                html += '</div>';
                html += '<div>' + statusBadge(wfStatus) + '</div>';
                html += '</div>';
            }
        }
    } catch (e) {
        html += '<div class="panel-title">WORKFLOW INSTANCES</div>';
        html += '<div class="empty-hint" style="padding:8px;">Could not load workflow instances.</div>';
    }

    contentEl.innerHTML = html;
}

// --- Agent Detail ---

async function renderAgentDetail(instanceId) {
    var titleEl = document.getElementById('ctrl-agent-detail-title');
    var contentEl = document.getElementById('ctrl-agent-detail-content');
    var inst = _instances.find(function(i) { return i.instance_id === instanceId; });
    if (!inst) return;

    titleEl.textContent = inst.label || instanceId;

    try {
        var resp = await fetch('/api/agent/' + encodeURIComponent(instanceId) + '/status');
        var data = await resp.json();
        if (data.error && !data.status) {
            contentEl.innerHTML = '<span class="error-message">' + esc(data.error) + '</span>';
            return;
        }

        var html = '<div class="kv" style="margin-bottom:8px;">';
        html += '<span class="key">Instance ID:</span><span class="val mono-id">' + esc(instanceId) + '</span>';
        html += '<span class="key">Agent Type:</span><span class="val">' + esc(data.agent_type || '') + '</span>';
        html += '<span class="key">Status:</span><span class="val">' + statusBadge(data.status) + '</span>';
        html += '<span class="key">Task:</span><span class="val" style="word-break:break-word;">' + esc(data.task || '') + '</span>';
        if (data.total_iterations !== undefined) {
            html += '<span class="key">Iterations:</span><span class="val">' + data.total_iterations + '</span>';
        }
        html += '</div>';

        // Show real-time progress if running
        if (data.status === 'running') {
            var iterCount = data.iteration_count || 0;
            var maxIter = 10;
            var pct = Math.min(Math.round((iterCount / maxIter) * 100), 95);
            var stateLabel = data.current_state || 'initializing';
            var stateColor = stateLabel === 'think' ? 'var(--primary)' : stateLabel === 'act' ? 'var(--yellow)' : stateLabel === 'conclude' ? 'var(--success)' : 'var(--text-dim)';

            html += '<div style="margin-bottom:8px;">';
            html += '<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px;">';
            html += '<span>Iteration <b>' + iterCount + '</b></span>';
            html += '<span style="color:' + stateColor + ';font-weight:600;">' + esc(stateLabel.toUpperCase()) + '</span>';
            html += '</div>';
            html += '<div class="progress-bar"><div class="progress-fill" style="width:' + pct + '%;transition:width 0.5s;"></div></div>';
            if (data.last_tool_call) {
                html += '<div style="font-size:11px;color:var(--yellow);margin-top:4px;">Last tool: ' + esc(data.last_tool_call) + '</div>';
            }
            html += '</div>';
        }

        // Show answer if completed
        if (data.answer) {
            html += '<div class="panel-title">ANSWER</div>';
            html += '<div class="event-log" style="max-height:150px;margin-bottom:8px;">';
            html += '<div class="entry"><span class="msg" style="white-space:pre-wrap;color:var(--text);">' + esc(data.answer) + '</span></div>';
            html += '</div>';
        }

        // Show error if failed
        if (data.error) {
            html += '<div class="panel-title">ERROR</div>';
            html += '<div class="error-message">' + esc(data.error) + '</div>';
        }

        // Show trace steps if available (completed agents with full result)
        if (data.status !== 'running') {
            await _renderAgentTrace(instanceId, html, contentEl, data);
            return;
        }

        // Show tool calls trace as fallback
        if (data.tools_used && data.tools_used.length > 0) {
            html += _renderToolCalls(data.tools_used);
        }

        // Success indicator
        if (data.success !== undefined && data.status !== 'running') {
            html += renderResultBanner(data.success);
        }

        contentEl.innerHTML = html;
    } catch (e) {
        contentEl.innerHTML = '<span class="error-message">Failed to load agent detail</span>';
    }
}

function _renderToolCalls(toolsUsed) {
    var html = '<div class="panel-title">TOOL CALLS (' + toolsUsed.length + ')</div>';
    for (var i = 0; i < toolsUsed.length; i++) {
        var tc = toolsUsed[i];
        html += '<div class="trace-step step-act">';
        html += '<div class="step-header"><span class="step-label">' + esc(tc.tool_name) + '</span></div>';
        html += '<div class="step-body" style="display:block;">' + esc(JSON.stringify(tc.parameters || {}, null, 1)) + '</div>';
        html += '</div>';
    }
    return html;
}

function toggleAllTraceSteps(expand) {
    document.querySelectorAll('.trace-step .step-body').forEach(function(el) {
        el.style.display = expand ? 'block' : 'none';
    });
}

async function _renderAgentTrace(instanceId, html, contentEl, statusData) {
    // Fetch full result with trace steps
    try {
        var resp = await fetch('/api/agent/' + encodeURIComponent(instanceId) + '/result');
        var result = await resp.json();
        if (result.trace_steps && result.trace_steps.length > 0) {
            html += '<div class="panel-title panel-title-flex">EXECUTION TRACE (' + result.trace_steps.length + ' steps)';
            html += '<div style="display:flex;gap:4px;">';
            html += '<button class="btn btn-sm" onclick="event.stopPropagation();toggleAllTraceSteps(true)">EXPAND ALL</button>';
            html += '<button class="btn btn-sm" onclick="event.stopPropagation();toggleAllTraceSteps(false)">COLLAPSE ALL</button>';
            html += '</div></div>';
            var iteration = 0;
            for (var i = 0; i < result.trace_steps.length; i++) {
                var step = result.trace_steps[i];
                var state = step.state || '';
                if (state === 'think') iteration++;

                var stepClass = 'step-' + state;
                var stepColor = state === 'think' ? 'var(--primary)' : state === 'act' ? 'var(--yellow)' : state === 'conclude' ? 'var(--success)' : 'var(--text-dim)';
                var stepIcon = state === 'think' ? '&#9679;' : state === 'act' ? '&#9654;' : state === 'conclude' ? '&#10003;' : '&#8226;';

                html += '<div class="trace-step ' + stepClass + '" onclick="this.querySelector(\'.step-body\').style.display=this.querySelector(\'.step-body\').style.display===\'none\'?\'block\':\'none\'">';
                html += '<div class="step-header">';
                html += '<span style="color:' + stepColor + ';font-weight:600;">' + stepIcon + ' ' + esc(state.toUpperCase());
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

    // Success indicator
    if (statusData.success !== undefined) {
        html += renderResultBanner(statusData.success);
    }

    contentEl.innerHTML = html;
}

function updateRunningAgents(updates) {
    // Update agent detail panel if currently viewing a running agent
    if (_selectedDetailType === 'agent' && _selectedDetailId && updates[_selectedDetailId]) {
        renderAgentDetail(_selectedDetailId);
    }
}

// --- Control Center Table Renderers ---

function renderControlFSMs() {
    var body = document.getElementById('ctrl-fsm-body');
    var empty = document.getElementById('ctrl-fsm-empty');
    var fsms = _instances.filter(function(i) { return i.instance_type === 'fsm'; });
    if (fsms.length === 0) {
        body.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';
    var rows = '';
    for (var i = 0; i < fsms.length; i++) {
        var f = fsms[i];
        var sel = (_selectedDetailId === f.instance_id) ? ' selected' : '';
        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(f.instance_id) + '" onclick="selectInstance(\'' + esc(f.instance_id) + '\',\'fsm\')">';
        rows += '<td>' + esc(f.label || f.instance_id) + '</td>';
        rows += '<td style="font-size:11px;max-width:180px;overflow:hidden;text-overflow:ellipsis;">' + esc(f.source || 'custom') + '</td>';
        rows += '<td>' + (f.conversation_count || 0) + '</td>';
        rows += '<td>' + statusBadge(f.status) + '</td>';
        rows += '<td>';
        if (f.status === 'running') {
            rows += '<button class="btn" style="font-size:11px;padding:2px 8px;" onclick="event.stopPropagation();startConversationOn(\'' + esc(f.instance_id) + '\')">+ CONV</button> ';
        }
        rows += '<button class="btn" style="font-size:11px;padding:2px 8px;color:var(--red);" onclick="event.stopPropagation();destroyInstance(\'' + esc(f.instance_id) + '\')">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;
}

function renderControlWorkflows() {
    var body = document.getElementById('ctrl-wf-body');
    var empty = document.getElementById('ctrl-wf-empty');
    var wfs = _instances.filter(function(i) { return i.instance_type === 'workflow'; });
    if (wfs.length === 0) {
        body.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';
    var rows = '';
    for (var i = 0; i < wfs.length; i++) {
        var w = wfs[i];
        var sel = (_selectedDetailId === w.instance_id) ? ' selected' : '';
        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(w.instance_id) + '" onclick="selectInstance(\'' + esc(w.instance_id) + '\',\'workflow\')">';
        rows += '<td>' + esc(w.label || w.instance_id) + '</td>';
        rows += '<td>' + statusBadge(w.status) + '</td>';
        rows += '<td>' + (w.active_workflows || 0) + '</td>';
        rows += '<td>';
        rows += '<button class="btn" style="font-size:11px;padding:2px 8px;color:var(--red);" onclick="event.stopPropagation();destroyInstance(\'' + esc(w.instance_id) + '\')">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;
}

function renderControlAgents() {
    var body = document.getElementById('ctrl-agent-body');
    var empty = document.getElementById('ctrl-agent-empty');
    var agents = _instances.filter(function(i) { return i.instance_type === 'agent'; });
    if (agents.length === 0) {
        body.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';
    var rows = '';
    for (var i = 0; i < agents.length; i++) {
        var a = agents[i];
        var sel = (_selectedDetailId === a.instance_id) ? ' selected' : '';
        rows += '<tr class="clickable-row' + sel + '" data-instance-id="' + esc(a.instance_id) + '" onclick="selectInstance(\'' + esc(a.instance_id) + '\',\'agent\')">';
        rows += '<td>' + esc(a.label || a.instance_id) + '</td>';
        rows += '<td>' + esc(a.agent_type || '') + '</td>';
        rows += '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(a.task || '') + '</td>';
        rows += '<td>' + statusBadge(a.status) + '</td>';
        rows += '<td>';
        if (a.status === 'running') {
            rows += '<button class="btn" style="font-size:11px;padding:2px 8px;color:var(--yellow);" onclick="event.stopPropagation();cancelAgent(\'' + esc(a.instance_id) + '\')">CANCEL</button> ';
        }
        rows += '<button class="btn" style="font-size:11px;padding:2px 8px;color:var(--red);" onclick="event.stopPropagation();destroyInstance(\'' + esc(a.instance_id) + '\')">&times;</button>';
        rows += '</td></tr>';
    }
    body.innerHTML = rows;
}

async function startConversationOn(instanceId) {
    try {
        var resp = await fetch('/api/fsm/' + encodeURIComponent(instanceId) + '/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_context: {} })
        });
        var data = await resp.json();
        if (data.error) {
            console.error('startConversation:', data.error);
            return;
        }
        _selectedConvInstanceId = instanceId;
        _selectedConvId = data.conversation_id;
        refreshInstances();
        // Navigate to conversation detail
        showPage('conversations');
        setTimeout(function() {
            refreshConversations();
            if (data.conversation_id) showConversationDetail(data.conversation_id);
        }, 300);
    } catch (e) {
        console.error('startConversationOn:', e);
    }
}

async function destroyInstance(instanceId) {
    try {
        await fetch('/api/instances/' + encodeURIComponent(instanceId), { method: 'DELETE' });
        if (_selectedDetailId === instanceId) {
            closeDetail('ctrl-fsm-detail');
            closeDetail('ctrl-wf-detail');
            closeDetail('ctrl-agent-detail');
        }
        refreshInstances();
        if (currentPage === 'control') refreshControlCenter();
    } catch (e) {
        console.error('destroyInstance:', e);
    }
}

async function cancelAgent(instanceId) {
    try {
        await fetch('/api/agent/' + encodeURIComponent(instanceId) + '/cancel', { method: 'POST' });
        if (_selectedDetailId === instanceId) renderAgentDetail(instanceId);
        refreshControlCenter();
    } catch (e) {
        console.error('cancelAgent:', e);
    }
}

// === SHARED GRAPH RENDERER ===

function layoutNodes(nodes, edges) {
    if (!nodes || nodes.length === 0) return;
    var nodeMap = {};
    for (var i = 0; i < nodes.length; i++) nodeMap[nodes[i].id] = nodes[i];

    var adj = {};
    for (var i = 0; i < nodes.length; i++) adj[nodes[i].id] = [];
    for (var i = 0; i < edges.length; i++) {
        if (adj[edges[i].from] && edges[i].from !== edges[i].to) {
            adj[edges[i].from].push(edges[i].to);
        }
    }

    var start = nodes[0].id;
    for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].is_initial) { start = nodes[i].id; break; }
    }

    var layers = {};
    var visited = {};
    var queue = [start];
    visited[start] = true;
    layers[start] = 0;
    while (queue.length > 0) {
        var cur = queue.shift();
        var neighbors = adj[cur] || [];
        for (var i = 0; i < neighbors.length; i++) {
            var n = neighbors[i];
            if (!visited[n]) {
                visited[n] = true;
                layers[n] = (layers[cur] || 0) + 1;
                queue.push(n);
            }
        }
    }
    var maxLayer = 0;
    for (var k in layers) { if (layers[k] > maxLayer) maxLayer = layers[k]; }
    for (var i = 0; i < nodes.length; i++) {
        if (layers[nodes[i].id] === undefined) {
            layers[nodes[i].id] = maxLayer + 1;
        }
    }

    var layerGroups = {};
    for (var i = 0; i < nodes.length; i++) {
        var l = layers[nodes[i].id];
        if (!layerGroups[l]) layerGroups[l] = [];
        layerGroups[l].push(nodes[i]);
    }

    var W = 180, H = 60, XPAD = 120, YPAD = 100;
    var layerKeys = Object.keys(layerGroups).map(Number).sort(function(a, b) { return a - b; });

    var MAX_COLS = 5;
    var totalLayers = layerKeys.length;
    var wrapRow = totalLayers > MAX_COLS ? MAX_COLS : totalLayers;

    for (var li = 0; li < layerKeys.length; li++) {
        var group = layerGroups[layerKeys[li]];
        var col = li % wrapRow;
        var row = Math.floor(li / wrapRow);
        var effectiveCol = (row % 2 === 0) ? col : (wrapRow - 1 - col);
        var rowHeight = 0;
        for (var ri = row * wrapRow; ri < Math.min((row + 1) * wrapRow, layerKeys.length); ri++) {
            var g = layerGroups[layerKeys[ri]];
            if (g.length > rowHeight) rowHeight = g.length;
        }
        var x = 140 + effectiveCol * (W + XPAD);
        var rowYBase = 60 + row * (rowHeight * (H + YPAD) + 60);
        for (var ni = 0; ni < group.length; ni++) {
            group[ni].x = x;
            group[ni].y = rowYBase + ni * (H + YPAD);
        }
    }
}

function rectEdgePoint(cx, cy, tx, ty, W, H) {
    var dx = tx - cx, dy = ty - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    var hw = W / 2, hh = H / 2;
    var sx = dx !== 0 ? hw / Math.abs(dx) : Infinity;
    var sy = dy !== 0 ? hh / Math.abs(dy) : Infinity;
    var s = Math.min(sx, sy);
    return { x: cx + dx * s, y: cy + dy * s };
}

function renderGraph(svgId, data, opts) {
    opts = opts || {};
    var svg = document.getElementById(svgId);
    if (!svg) return;
    var nodes = data.nodes;
    var edges = data.edges;
    var colorVar = opts.colorVar || 'var(--primary-dim)';
    var arrowColor = opts.arrowColor || colorVar;
    var rx = opts.rx || 4;
    var markerId = 'arrow-' + svgId;
    var W = 180, H = 60;

    layoutNodes(nodes, edges);

    var nodeMap = {};
    for (var i = 0; i < nodes.length; i++) nodeMap[nodes[i].id] = nodes[i];

    var maxX = 0, maxY = 0;
    for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].x + W / 2 > maxX) maxX = nodes[i].x + W / 2;
        if (nodes[i].y + H / 2 > maxY) maxY = nodes[i].y + H / 2;
    }
    var svgW = maxX + 80;
    var svgH = maxY + 80;
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
    svg.setAttribute('viewBox', '0 0 ' + svgW + ' ' + svgH);

    var html = '<defs><marker id="' + markerId + '" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="' + arrowColor + '"/></marker></defs>';

    var edgePairs = {};
    for (var i = 0; i < edges.length; i++) {
        var key = [edges[i].from, edges[i].to].sort().join('|');
        if (!edgePairs[key]) edgePairs[key] = 0;
        edgePairs[key]++;
    }

    var drawnPairs = {};
    for (var i = 0; i < edges.length; i++) {
        var e = edges[i];
        var from = nodeMap[e.from];
        var to = nodeMap[e.to];
        if (!from || !to) continue;
        if (e.from === e.to) {
            var cx = from.x, cy = from.y - H / 2;
            html += '<path class="edge-line" d="M ' + (cx - 20) + ' ' + cy + ' C ' + (cx - 30) + ' ' + (cy - 50) + ', ' + (cx + 30) + ' ' + (cy - 50) + ', ' + (cx + 20) + ' ' + cy + '" fill="none" marker-end="url(#' + markerId + ')"/>';
            if (e.label) html += '<text class="edge-label" x="' + cx + '" y="' + (cy - 38) + '">' + esc(e.label) + '</text>';
            continue;
        }

        var pairKey = [e.from, e.to].sort().join('|');
        var isBidi = edgePairs[pairKey] > 1;
        var offset = 0;
        if (isBidi) {
            if (!drawnPairs[pairKey]) drawnPairs[pairKey] = 0;
            offset = drawnPairs[pairKey] === 0 ? 12 : -12;
            drawnPairs[pairKey]++;
        }

        var p1 = rectEdgePoint(from.x, from.y, to.x, to.y, W, H);
        var p2 = rectEdgePoint(to.x, to.y, from.x, from.y, W + 16, H + 16);

        if (offset !== 0) {
            var dx = to.x - from.x, dy = to.y - from.y;
            var len = Math.sqrt(dx * dx + dy * dy) || 1;
            var nx = -dy / len * offset, ny = dx / len * offset;
            var mx = (p1.x + p2.x) / 2 + nx, my = (p1.y + p2.y) / 2 + ny;
            html += '<path class="edge-line" d="M ' + p1.x + ' ' + p1.y + ' Q ' + mx + ' ' + my + ' ' + p2.x + ' ' + p2.y + '" fill="none" marker-end="url(#' + markerId + ')"/>';
            if (e.label) html += '<text class="edge-label" x="' + mx + '" y="' + (my - 14) + '">' + esc(e.label) + '</text>';
        } else {
            html += '<line class="edge-line" x1="' + p1.x + '" y1="' + p1.y + '" x2="' + p2.x + '" y2="' + p2.y + '" marker-end="url(#' + markerId + ')"/>';
            if (e.label) {
                html += '<text class="edge-label" x="' + ((p1.x + p2.x) / 2) + '" y="' + ((p1.y + p2.y) / 2 - 14) + '">' + esc(e.label) + '</text>';
            }
        }
    }

    var nodeClass = opts.nodeClass || 'fsm';
    for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var cls = n.is_initial ? 'initial' : n.is_terminal ? 'terminal' : '';
        html += '<rect class="node-rect node-' + nodeClass + ' ' + cls + '" x="' + (n.x - W / 2) + '" y="' + (n.y - H / 2) + '" width="' + W + '" height="' + H + '" rx="' + rx + '"/>';
        html += '<text class="node-label" x="' + n.x + '" y="' + (n.y - 6) + '">' + esc(n.label || n.id) + '</text>';
        var subtitle = n.step_type || (n.description ? n.description.substring(0, 24) : '');
        if (subtitle) {
            html += '<text class="node-subtitle" x="' + n.x + '" y="' + (n.y + 12) + '">' + esc(subtitle) + '</text>';
        }
    }

    svg.innerHTML = html;
}

// === UNIFIED VISUALIZER ===

async function visualizeGraph(type, typeValue) {
    var endpoints = {
        fsm: { post: '/api/fsm/visualize', presetGet: '/api/fsm/visualize/preset/' },
        agent: { get: '/api/agent/visualize?agent_type=' },
        workflow: { get: '/api/workflow/visualize?workflow_id=' }
    };
    var svgIds = { fsm: 'viz-svg', agent: 'viz-agent-svg', workflow: 'viz-wf-svg' };
    var infoIds = { fsm: 'viz-info', agent: 'viz-agent-info', workflow: 'viz-wf-info' };
    var transIds = { fsm: 'viz-trans-body', agent: 'viz-agent-trans-body', workflow: 'viz-wf-trans-body' };
    var styles = {
        fsm: { colorVar: 'var(--primary-dim)', rx: 4, nodeClass: 'fsm' },
        agent: { colorVar: 'var(--yellow)', arrowColor: 'var(--yellow)', rx: 16, nodeClass: 'agent' },
        workflow: { colorVar: 'var(--cyan)', arrowColor: 'var(--cyan)', rx: 12, nodeClass: 'wf' }
    };

    try {
        var resp;
        if (type === 'fsm') {
            if (typeValue && typeof typeValue === 'string' && typeValue.includes('/')) {
                resp = await fetch(endpoints.fsm.presetGet + encodeURIComponent(typeValue));
            } else if (typeValue && typeof typeValue === 'object') {
                resp = await fetch(endpoints.fsm.post, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(typeValue),
                });
            } else {
                return;
            }
        } else {
            if (!typeValue) return;
            resp = await fetch(endpoints[type].get + encodeURIComponent(typeValue));
        }

        var data = await resp.json();
        if (data.error) {
            console.error('visualizeGraph ' + type + ':', data.error);
            return;
        }

        renderGraph(svgIds[type], data, styles[type]);

        var info = data.info || data.fsm || {};
        var infoEl = document.getElementById(infoIds[type]);
        if (infoEl) {
            var infoHtml = '';
            var fields = { Name: info.name, Description: info.description, States: info.state_count || info.step_count, Version: info.version, Initial: info.initial_state };
            for (var k in fields) {
                if (fields[k] !== undefined && fields[k] !== '') {
                    infoHtml += '<span class="key">' + k + ':</span><span class="val">' + esc(fields[k]) + '</span>';
                }
            }
            infoEl.innerHTML = infoHtml;
        }

        var tbody = document.getElementById(transIds[type]);
        if (tbody && data.edges) {
            var rows = '';
            for (var i = 0; i < data.edges.length; i++) {
                var e = data.edges[i];
                rows += '<tr><td>' + esc(e.from) + '</td><td>' + esc(e.to) + '</td>';
                if (type === 'fsm') rows += '<td>' + (e.priority || '') + '</td>';
                rows += '<td>' + esc(e.label) + '</td></tr>';
            }
            tbody.innerHTML = rows;
        }
    } catch (e) {
        console.error('visualizeGraph ' + type + ':', e);
    }
}

// === FSM: VISUALIZER ===

async function visualizeFSM(presetId) {
    var statusEl = document.getElementById('viz-fsm-status');
    if (presetId) {
        await visualizeGraph('fsm', presetId);
        if (statusEl) showStatus('viz-fsm-status', 'OK', 'success');
        return;
    }
    var jsonText = document.getElementById('viz-fsm-json').value.trim();
    if (!jsonText) return;
    try {
        var fsmDef = JSON.parse(jsonText);
        await visualizeGraph('fsm', fsmDef);
        if (statusEl) showStatus('viz-fsm-status', 'OK', 'success');
    } catch (e) {
        showError('viz-fsm-status', 'Invalid JSON');
    }
}

// === PRESETS (inline in visualizer) ===

function showPresetPicker() {
    var picker = document.getElementById('preset-picker');
    if (picker.style.display === 'none') {
        picker.style.display = 'block';
        loadFSMPresets();
    } else {
        picker.style.display = 'none';
    }
}

async function loadFSMPresets() {
    if (_presets) {
        renderPresets(_presets);
        return;
    }
    try {
        var resp = await fetch('/api/presets');
        _presets = await resp.json();
        renderPresets(_presets);
    } catch (e) {
        console.error('loadFSMPresets:', e);
        var empty = document.getElementById('preset-empty');
        if (empty) empty.textContent = 'Failed to load presets';
    }
}

function renderPresets(presets) {
    var items = presets.fsm || [];
    var container = document.getElementById('preset-list');
    var empty = document.getElementById('preset-empty');
    if (items.length === 0) {
        if (empty) empty.textContent = 'No presets found';
        return;
    }
    if (empty) empty.style.display = 'none';
    container.innerHTML = '';
    for (var i = 0; i < items.length; i++) {
        var p = items[i];
        var card = document.createElement('div');
        card.className = 'preset-card';
        card.innerHTML = '<div class="preset-name">' + esc(p.name) + '</div><div class="preset-category">' + esc(p.category || '') + '</div><div class="preset-desc">' + esc(p.description || '') + '</div>';
        (function(id) {
            card.addEventListener('click', function() { useFSMPreset(id); });
        })(p.id);
        container.appendChild(card);
    }
}

async function useFSMPreset(presetId) {
    try {
        var resp = await fetch('/api/preset/fsm/' + encodeURIComponent(presetId));
        var data = await resp.json();
        if (data.error) return;
        document.getElementById('viz-fsm-json').value = JSON.stringify(data, null, 2);
        document.getElementById('preset-picker').style.display = 'none';
        visualizeFSM();
    } catch (e) {
        console.error('useFSMPreset:', e);
    }
}

// === LOGS ===

async function refreshLogs() {
    var level = document.getElementById('log-level').value;
    var filter = document.getElementById('log-filter').value.trim().toLowerCase();
    try {
        var resp = await fetch('/api/logs?limit=500&level=' + encodeURIComponent(level));
        var logs = await resp.json();
        if (filter) {
            logs = logs.filter(function(r) { return r.message.toLowerCase().indexOf(filter) !== -1; });
        }
        var stream = document.getElementById('log-stream');
        var logEmpty = document.getElementById('log-empty');
        if (logEmpty) logEmpty.remove();
        var colors = { DEBUG: 'var(--text-dim)', INFO: 'var(--primary)', WARNING: 'var(--yellow)', ERROR: 'var(--red)', CRITICAL: 'var(--red)' };
        logs.reverse();
        var html = '';
        if (logs.length === 0) {
            html = '<div class="empty-hint" style="padding:12px;">No log entries matching filter</div>';
        }
        for (var i = 0; i < logs.length; i++) {
            var r = logs[i];
            var ts = formatTime(r.timestamp);
            var c = colors[r.level] || 'var(--primary)';
            var conv = r.conversation_id ? ' [' + r.conversation_id + ']' : '';
            html += '<div class="entry"><span class="ts" style="color:' + c + '">' + ts + '</span><span class="type" style="color:' + c + ';width:70px;">' + r.level + '</span><span class="msg" style="color:var(--text-dim)">' + esc(r.module) + ':' + r.line + conv + ' ' + esc(r.message) + '</span></div>';
        }
        stream.innerHTML = html;
        document.getElementById('log-stats').textContent = 'Total: ' + logs.length + ' | Level: >= ' + level;
    } catch (e) {
        console.error('refreshLogs:', e);
    }
}

// === SETTINGS ===

async function loadSettings() {
    try {
        var resp = await fetch('/api/config');
        var cfg = await resp.json();
        document.getElementById('set-refresh').value = cfg.refresh_interval;
        document.getElementById('set-max-events').value = cfg.max_events;
        document.getElementById('set-max-logs').value = cfg.max_log_lines;
        document.getElementById('set-level').value = cfg.log_level;
    } catch (e) {
        console.error('loadSettings config:', e);
    }
    try {
        var resp = await fetch('/api/info');
        var info = await resp.json();
        var el = document.getElementById('sys-info');
        var html = '';
        for (var k in info) {
            html += '<span class="key">' + esc(k.replace(/_/g, ' ')) + ':</span><span class="val">' + esc(info[k]) + '</span>';
        }
        el.innerHTML = html;
        document.getElementById('version-info').textContent = 'v' + info.monitor_version;
    } catch (e) {
        console.error('loadSettings info:', e);
    }
}

async function saveSettings() {
    try {
        await fetch('/api/config', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                refresh_interval: numVal('set-refresh', 1.0),
                max_events: intVal('set-max-events', 1000),
                max_log_lines: intVal('set-max-logs', 5000),
                log_level: document.getElementById('set-level').value,
                show_internal_keys: false,
                auto_scroll_logs: true,
            }),
        });
        showStatus('conn-status', 'Settings saved', 'success');
    } catch (e) {
        showError('conn-status', 'Save failed');
        console.error('saveSettings:', e);
    }
}

function resetSettings() {
    document.getElementById('set-refresh').value = '1.0';
    document.getElementById('set-max-events').value = '1000';
    document.getElementById('set-max-logs').value = '5000';
    document.getElementById('set-level').value = 'INFO';
}

// === KEYBOARD SHORTCUTS ===

document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
        case '1': showPage('dashboard'); break;
        case '2': showPage('visualizer'); break;
        case '3': showPage('conversations'); break;
        case '4': showPage('control'); break;
        case '5': showPage('logs'); break;
        case '6': showPage('settings'); break;
        case 'Escape': closeLaunchModal(); break;
    }
});

// === INIT ===

connectWS();
loadSettings();
refreshInstances();
setInterval(updateClock, 1000);
updateClock();
// Periodically refresh instances for agent status updates
setInterval(function() {
    if (currentPage === 'control' || currentPage === 'dashboard') {
        refreshInstances();
        if (currentPage === 'control') refreshControlCenter();
    }
}, 5000);
