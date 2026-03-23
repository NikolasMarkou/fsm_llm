// FSM-LLM Monitor — Retro Terminal Dashboard
// Vanilla JS — no frameworks needed

'use strict';

var ws = null;
var currentPage = 'dashboard';
var _presets = null;
var _wsRetryDelay = 3000;
var WS_MAX_DELAY = 30000;

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
    if (el) el.innerHTML = '<span style="color:var(--red);">' + esc(msg) + '</span>';
}

function showStatus(elementId, msg, color) {
    var el = document.getElementById(elementId);
    if (el) el.innerHTML = '<span style="color:var(--' + color + ');">' + esc(msg) + '</span>';
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
        'settings': loadSettings
    };
    if (refreshMap[page]) refreshMap[page]();

    if (page === 'visualizer') {
        // Auto-render selected agent/workflow if tab is active
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
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.tab').forEach(function(b) { b.classList.remove('active'); });
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
    // Remove empty-state hint if present
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
            rows += '<tr><td>' + esc(c.conversation_id.substring(0, 16)) + '</td><td>' + esc(c.current_state) + '</td><td>' + c.message_history.length + '</td><td><span class="badge ' + badge + '">' + label + '</span></td></tr>';
        }
        body.innerHTML = rows;
    } catch (e) {
        console.error('refreshConversationTable:', e);
    }
}

// Refresh conversations on dashboard view, throttled
var _convRefreshTimer = null;
function scheduleConversationRefresh() {
    if (_convRefreshTimer) return;
    _convRefreshTimer = setTimeout(function() {
        _convRefreshTimer = null;
        if (currentPage === 'dashboard') refreshConversationTable();
    }, 5000);
}

// === CONVERSATIONS (read-only inspector) ===

async function refreshConversations() {
    try {
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
            rows += '<tr data-conv-id="' + esc(c.conversation_id) + '" style="cursor:pointer;"><td>' + esc(c.conversation_id.substring(0, 16)) + '</td><td>' + esc(c.current_state) + '</td><td>' + (c.stack_depth || 1) + '</td><td><span class="badge ' + badge + '">' + label + '</span></td></tr>';
        }
        body.innerHTML = rows;

        // Click handler for rows
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
    var detail = document.getElementById('conv-detail');
    try {
        var resp = await fetch('/api/conversations/' + encodeURIComponent(convId));
        var data = await resp.json();
        if (data.error) {
            detail.innerHTML = '<span style="color:var(--red);">' + esc(data.error) + '</span>';
            return;
        }

        var html = '<div class="kv">';
        html += '<span class="key">ID:</span><span class="val">' + esc(data.conversation_id) + '</span>';
        html += '<span class="key">State:</span><span class="val">' + esc(data.current_state) + '</span>';
        html += '<span class="key">Description:</span><span class="val">' + esc(data.state_description) + '</span>';
        html += '<span class="key">Terminal:</span><span class="val">' + (data.is_terminal ? 'YES' : 'NO') + '</span>';
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

        // Message history
        if (data.message_history && data.message_history.length > 0) {
            html += '<div class="panel-title" style="margin-top:12px;">MESSAGE HISTORY (' + data.message_history.length + ')</div>';
            html += '<div class="event-log" style="max-height:300px;">';
            for (var i = 0; i < data.message_history.length; i++) {
                var msg = data.message_history[i];
                var role = msg.role || 'system';
                var roleColor = role === 'user' ? 'var(--cyan)' : 'var(--green)';
                html += '<div class="entry"><span class="type" style="color:' + roleColor + ';width:60px;">' + esc(role.toUpperCase()) + '</span><span class="msg">' + esc(msg.content || msg.message || '') + '</span></div>';
            }
            html += '</div>';
        }

        detail.innerHTML = html;
    } catch (e) {
        detail.innerHTML = '<span style="color:var(--red);">Failed to load conversation</span>';
        console.error('showConversationDetail:', e);
    }
}

// === SHARED GRAPH RENDERER ===

function layoutNodes(nodes, edges) {
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

    // For long linear chains (>5 layers), wrap into rows to avoid infinite horizontal scroll
    var MAX_COLS = 5;
    var totalLayers = layerKeys.length;
    var wrapRow = totalLayers > MAX_COLS ? MAX_COLS : totalLayers;

    for (var li = 0; li < layerKeys.length; li++) {
        var group = layerGroups[layerKeys[li]];
        var col = li % wrapRow;
        var row = Math.floor(li / wrapRow);
        // Alternate row direction (boustrophedon) so edges don't cross the whole width
        var effectiveCol = (row % 2 === 0) ? col : (wrapRow - 1 - col);
        var rowHeight = 0;
        // Compute how many nodes are in any layer of this row to offset vertically
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
    var colorVar = opts.colorVar || 'var(--green-dim)';
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
            if (e.label) html += '<text class="edge-label" x="' + mx + '" y="' + (my - 6) + '">' + esc(e.label) + '</text>';
        } else {
            html += '<line class="edge-line" x1="' + p1.x + '" y1="' + p1.y + '" x2="' + p2.x + '" y2="' + p2.y + '" marker-end="url(#' + markerId + ')"/>';
            if (e.label) {
                html += '<text class="edge-label" x="' + ((p1.x + p2.x) / 2) + '" y="' + ((p1.y + p2.y) / 2 - 6) + '">' + esc(e.label) + '</text>';
            }
        }
    }

    var nodeClass = opts.nodeClass || 'fsm';
    for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var cls = n.is_initial ? 'initial' : n.is_terminal ? 'terminal' : '';
        html += '<rect class="node-rect node-' + nodeClass + ' ' + cls + '" x="' + (n.x - W / 2) + '" y="' + (n.y - H / 2) + '" width="' + W + '" height="' + H + '" rx="' + rx + '"/>';
        html += '<text class="node-label" x="' + n.x + '" y="' + (n.y - 6) + '">' + esc(n.label || n.id) + '</text>';
        // Show description or step_type as subtitle
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
        fsm: { colorVar: 'var(--green-dim)', rx: 4, nodeClass: 'fsm' },
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
        if (statusEl) showStatus('viz-fsm-status', 'OK', 'green');
        return;
    }
    var jsonText = document.getElementById('viz-fsm-json').value.trim();
    if (!jsonText) return;
    try {
        var fsmDef = JSON.parse(jsonText);
        await visualizeGraph('fsm', fsmDef);
        if (statusEl) showStatus('viz-fsm-status', 'OK', 'green');
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
        var colors = { DEBUG: 'var(--green-dim)', INFO: 'var(--green)', WARNING: 'var(--yellow)', ERROR: 'var(--red)', CRITICAL: 'var(--red)' };
        logs.reverse();
        var html = '';
        if (logs.length === 0) {
            html = '<div class="empty-hint" style="padding:12px;">No log entries matching filter</div>';
        }
        for (var i = 0; i < logs.length; i++) {
            var r = logs[i];
            var ts = formatTime(r.timestamp);
            var c = colors[r.level] || 'var(--green)';
            var conv = r.conversation_id ? ' [' + r.conversation_id + ']' : '';
            html += '<div class="entry"><span class="ts" style="color:' + c + '">' + ts + '</span><span class="type" style="color:' + c + ';width:70px;">' + r.level + '</span><span class="msg" style="color:var(--green-dim)">' + esc(r.module) + ':' + r.line + conv + ' ' + esc(r.message) + '</span></div>';
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
        showStatus('conn-status', 'Settings saved', 'green');
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
        case '4': showPage('logs'); break;
        case '5': showPage('settings'); break;
    }
});

// === INIT ===

connectWS();
loadSettings();
setInterval(updateClock, 1000);
updateClock();
setInterval(scheduleConversationRefresh, 10000);
