// FSM-LLM Monitor — Dashboard Page

'use strict';

// --- Instance Grid Pagination State ---

var _instPage = 0;
var _instPerPage = 12;
var _instSearch = '';
var _lastInstancesHash = '';

function updateMetrics(m) {
    var activeEl = document.getElementById('m-active-convs');
    var convsEl = document.getElementById('m-conversations');
    var eventsEl = document.getElementById('m-events');
    var transEl = document.getElementById('m-transitions');
    var errorsEl = document.getElementById('m-errors');
    var errorsCard = document.getElementById('m-errors-card');

    if (activeEl) activeEl.textContent = formatNumber(m.active_conversations);
    if (convsEl) convsEl.textContent = formatNumber((m.events_per_type && m.events_per_type.conversation_start) || 0);
    if (eventsEl) eventsEl.textContent = formatNumber(m.total_events);
    if (transEl) transEl.textContent = formatNumber(m.total_transitions);
    if (errorsEl) errorsEl.textContent = formatNumber(m.total_errors);

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
    if (!log) return;
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
    // Trim to 50 entries (reduced from 200 — keeps panel scannable)
    while (log.children.length > 50) log.removeChild(log.lastChild);
}

// --- Instance Grid (paginated) ---

function _getFilteredInstances() {
    if (!_instSearch) return App.instances;
    var q = _instSearch.toLowerCase();
    return App.instances.filter(function(inst) {
        return (inst.label || inst.instance_id).toLowerCase().indexOf(q) !== -1
            || inst.instance_type.toLowerCase().indexOf(q) !== -1;
    });
}

function renderInstanceGrid() {
    var grid = document.getElementById('instances-grid');
    var empty = document.getElementById('instances-empty');
    var paginationEl = document.getElementById('instances-pagination');
    var titleEl = document.getElementById('instances-title-count');
    if (!grid) return;

    // Quick hash to skip re-render if nothing changed
    var hash = App.instances.length + ':' + App.instances.map(function(i) { return i.instance_id + i.status; }).join(',');
    if (hash === _lastInstancesHash && !_instSearch) {
        return;
    }
    _lastInstancesHash = hash;

    var filtered = _getFilteredInstances();

    // Update count badge
    if (titleEl) titleEl.textContent = filtered.length;

    if (filtered.length === 0) {
        grid.innerHTML = '';
        if (empty) empty.style.display = 'block';
        if (paginationEl) paginationEl.style.display = 'none';
        return;
    }
    if (empty) empty.style.display = 'none';

    // Paginate
    var totalPages = Math.ceil(filtered.length / _instPerPage);
    if (_instPage >= totalPages) _instPage = totalPages - 1;
    if (_instPage < 0) _instPage = 0;
    var start = _instPage * _instPerPage;
    var pageItems = filtered.slice(start, start + _instPerPage);

    var html = '';
    for (var i = 0; i < pageItems.length; i++) {
        var inst = pageItems[i];
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

    // Pagination controls
    if (paginationEl) {
        if (totalPages <= 1) {
            paginationEl.style.display = 'none';
        } else {
            paginationEl.style.display = 'flex';
            paginationEl.innerHTML = '<button class="btn btn-sm" onclick="instPagePrev()"' + (_instPage === 0 ? ' disabled' : '') + '>&laquo; Prev</button>'
                + '<span class="pagination-info">Page ' + (_instPage + 1) + ' of ' + totalPages + '</span>'
                + '<button class="btn btn-sm" onclick="instPageNext()"' + (_instPage >= totalPages - 1 ? ' disabled' : '') + '>Next &raquo;</button>';
        }
    }
}

function instPagePrev() {
    if (_instPage > 0) { _instPage--; _lastInstancesHash = ''; renderInstanceGrid(); }
}

function instPageNext() {
    var filtered = _getFilteredInstances();
    var totalPages = Math.ceil(filtered.length / _instPerPage);
    if (_instPage < totalPages - 1) { _instPage++; _lastInstancesHash = ''; renderInstanceGrid(); }
}

function onInstSearchInput() {
    _instSearch = (document.getElementById('inst-search') || {}).value || '';
    _instPage = 0;
    _lastInstancesHash = '';
    renderInstanceGrid();
}

async function refreshInstances() {
    try {
        App.instances = await fetchJson('/api/instances');
        renderInstanceGrid();
    } catch (e) {
        console.error('refreshInstances:', e);
    }
}

// --- Conversation Table (bounded) ---

var _convShowEnded = false;

function toggleConvEnded() {
    _convShowEnded = !_convShowEnded;
    var btn = document.getElementById('conv-toggle-ended');
    if (btn) btn.textContent = _convShowEnded ? 'Hide ended' : 'Show ended';
    refreshConversationTable();
}

async function refreshConversationTable() {
    try {
        var convs = await fetchJson('/api/conversations');
        var body = document.getElementById('conv-table-body');
        var empty = document.getElementById('conv-empty');
        var countEl = document.getElementById('conv-title-count');

        // Filter ended if toggle is off
        if (!_convShowEnded) {
            convs = convs.filter(function(c) { return !c.is_terminal; });
        }

        if (countEl) countEl.textContent = convs.length;

        if (convs.length === 0) {
            body.innerHTML = '';
            empty.style.display = 'block';
            return;
        }
        empty.style.display = 'none';

        // Limit to 20 rows
        var display = convs.slice(0, 20);
        var rows = '';
        for (var i = 0; i < display.length; i++) {
            var c = display[i];
            var badge = c.is_terminal ? 'badge-ended' : 'badge-active';
            var label = c.is_terminal ? 'ENDED' : 'ACTIVE';
            var instId = c.instance_id || '';
            var onclick = instId ? ' onclick="navigateToInstance(\'' + esc(instId) + '\',\'fsm\');setTimeout(function(){showConversationInDrawer(\'' + esc(instId) + '\',\'' + esc(c.conversation_id) + '\')},200)"' : '';
            rows += '<tr class="clickable-row"' + onclick + '><td class="cell-truncate">' + esc(c.conversation_id.substring(0, 16)) + '</td><td>' + esc(c.current_state) + '</td><td>' + c.message_history.length + '</td><td><span class="badge ' + badge + '">' + label + '</span></td></tr>';
        }
        if (convs.length > 20) {
            rows += '<tr><td colspan="4" class="text-dim" style="text-align:center;">...and ' + (convs.length - 20) + ' more</td></tr>';
        }
        body.innerHTML = rows;
    } catch (e) {
        console.error('refreshConversationTable:', e);
    }
}
