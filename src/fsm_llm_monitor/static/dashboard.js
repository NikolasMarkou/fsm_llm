// FSM-LLM Monitor — Dashboard Page

'use strict';

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
    if (App.instances.length === 0) {
        grid.innerHTML = '';
        if (empty) empty.style.display = 'block';
        return;
    }
    if (empty) empty.style.display = 'none';
    var html = '';
    for (var i = 0; i < App.instances.length; i++) {
        var inst = App.instances[i];
        html += '<div class="instance-card" onclick="navigateToInstance(\'' + esc(inst.instance_id) + '\',\'' + esc(inst.instance_type) + '\')">';
        html += '<div class="inst-label">' + esc(inst.label || inst.instance_id) + '</div>';
        html += '<div class="flex-between">';
        html += '<div class="inst-type">' + esc(inst.instance_type) + '</div>';
        html += '<div class="inst-status">' + statusBadge(inst.status) + '</div>';
        html += '</div>';
        var extra = '';
        if (inst.instance_type === 'fsm' && inst.conversation_count !== undefined) {
            extra = inst.conversation_count + ' conversation' + (inst.conversation_count !== 1 ? 's' : '');
        } else if (inst.instance_type === 'agent' && inst.agent_type) {
            extra = inst.agent_type;
        } else if (inst.instance_type === 'workflow' && inst.active_workflows !== undefined) {
            extra = inst.active_workflows + ' active';
        }
        if (extra || inst.created_at) {
            html += '<div class="text-small-dim">';
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
        App.instances = await resp.json();
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
