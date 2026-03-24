// FSM-LLM Monitor — Dashboard Page

'use strict';

function updateMetrics(m) {
    var activeEl = document.getElementById('m-active-convs');
    var convsEl = document.getElementById('m-conversations');
    var eventsEl = document.getElementById('m-events');
    var transEl = document.getElementById('m-transitions');
    var errorsEl = document.getElementById('m-errors');
    var errorsCard = document.getElementById('m-errors-card');

    if (activeEl) activeEl.textContent = m.active_conversations;
    if (convsEl) convsEl.textContent = m.active_conversations;
    if (eventsEl) eventsEl.textContent = m.total_events;
    if (transEl) transEl.textContent = m.total_transitions;
    if (errorsEl) errorsEl.textContent = m.total_errors;

    // Error card visual feedback
    if (errorsCard) {
        if (m.total_errors > 0) {
            errorsCard.classList.add('has-errors');
        } else {
            errorsCard.classList.remove('has-errors');
        }
    }
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
        var cat = e.event_type.indexOf('transition') !== -1 ? ' transition' : e.event_type.indexOf('conversation') !== -1 ? ' conversation' : '';
        var cid = e.conversation_id ? '<span class="conv-id">' + esc(e.conversation_id.substring(0, 8)) + '</span>' : '';
        html += '<div class="entry ' + level + cat + '"><span class="ts">' + ts + '</span>' + cid + '<span class="type">' + esc(e.event_type) + '</span><span class="msg">' + esc(e.message) + '</span></div>';
    }
    log.insertAdjacentHTML('afterbegin', html);
    while (log.children.length > 200) log.removeChild(log.lastChild);
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
        var typeClass = 'type-' + (inst.instance_type || 'fsm');
        html += '<div class="instance-card ' + typeClass + '" onclick="navigateToInstance(\'' + esc(inst.instance_id) + '\',\'' + esc(inst.instance_type) + '\')">';
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
            if (inst.created_at) html += '<span>' + relativeTime(inst.created_at) + '</span>';
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
