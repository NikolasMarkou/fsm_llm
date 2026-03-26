// FSM-LLM Monitor — Dashboard Page

import { state, scheduleRefresh } from '../services/state.js';
import { fetchJson } from '../services/api.js';
import { $, esc, statusBadge, hashInstances } from '../utils/dom.js';
import { formatTime, relativeTime, formatNumber } from '../utils/format.js';

// --- Instance Grid State ---
let _instPage = 0;
const _instPerPage = 12;
let _instSearch = '';
let _lastInstancesHash = '';

// --- Conversation Table State ---
let _convShowEnded = false;
let _convPage = 0;
const _convPerPage = 20;
let _convSearch = '';
let _convData = [];

// Exposed for WS + external use
export let navigateToInstance;

export function setNavigateToInstance(fn) {
    navigateToInstance = fn;
}

// --- Metrics ---

export function updateMetrics(m) {
    const set = (id, v) => { const el = $(id); if (el) el.textContent = formatNumber(v); };
    set('m-active-convs', m.active_conversations);
    set('m-conversations', m.events_per_type?.conversation_start || 0);
    set('m-events', m.total_events);
    set('m-transitions', m.total_transitions);
    set('m-errors', m.total_errors);

    const errorsCard = $('m-errors-card');
    if (errorsCard) errorsCard.classList.toggle('has-errors', m.total_errors > 0);
}

// --- Events ---

export function updateEvents(events) {
    const log = $('event-log');
    if (!log) return;
    const emptyHint = $('event-empty');
    if (emptyHint) emptyHint.remove();

    let html = '';
    for (const e of events) {
        const ts = formatTime(e.timestamp);
        const level = (e.level || 'INFO').toLowerCase();
        const cat = e.event_type.includes('transition') ? ' transition' : e.event_type.includes('conversation') ? ' conversation' : '';
        const cid = e.conversation_id ? '<span class="conv-id">' + esc(e.conversation_id.substring(0, 8)) + '</span>' : '';
        html += '<div class="entry ' + level + cat + '"><span class="ts">' + ts + '</span>' + cid + '<span class="type">' + esc(e.event_type) + '</span><span class="msg">' + esc(e.message) + '</span></div>';
    }
    log.insertAdjacentHTML('afterbegin', html);
    while (log.children.length > 50) log.removeChild(log.lastChild);
}

// --- Instance Grid ---

function _getFilteredInstances() {
    if (!_instSearch) return state.instances;
    const q = _instSearch.toLowerCase();
    return state.instances.filter(inst =>
        (inst.label || inst.instance_id).toLowerCase().includes(q) ||
        inst.instance_type.toLowerCase().includes(q)
    );
}

export function renderInstanceGrid() {
    const grid = $('instances-grid');
    const empty = $('instances-empty');
    const paginationEl = $('instances-pagination');
    const titleEl = $('instances-title-count');
    if (!grid) return;

    const hash = hashInstances(state.instances);
    if (hash === _lastInstancesHash && !_instSearch) return;
    _lastInstancesHash = hash;

    const filtered = _getFilteredInstances();
    if (titleEl) titleEl.textContent = filtered.length;

    if (filtered.length === 0) {
        grid.innerHTML = '';
        if (empty) empty.style.display = 'block';
        if (paginationEl) paginationEl.style.display = 'none';
        return;
    }
    if (empty) empty.style.display = 'none';

    const totalPages = Math.ceil(filtered.length / _instPerPage);
    _instPage = Math.max(0, Math.min(_instPage, totalPages - 1));
    const start = _instPage * _instPerPage;
    const pageItems = filtered.slice(start, start + _instPerPage);

    let html = '';
    for (const inst of pageItems) {
        const typeClass = 'type-' + (inst.instance_type || 'fsm');
        html += '<div class="instance-card ' + typeClass + '" data-action="navigate-instance" data-instance-id="' + esc(inst.instance_id) + '" data-instance-type="' + esc(inst.instance_type) + '">';
        html += '<div class="inst-label">' + esc(inst.label || inst.instance_id) + '</div>';
        html += '<div class="flex-between">';
        html += '<div class="inst-type">' + esc(inst.instance_type) + '</div>';
        html += '<div class="inst-status">' + statusBadge(inst.status) + '</div>';
        html += '</div>';
        let extra = '';
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

    if (paginationEl) {
        if (totalPages <= 1) {
            paginationEl.style.display = 'none';
        } else {
            paginationEl.style.display = 'flex';
            paginationEl.innerHTML =
                '<button class="btn btn-sm" data-action="inst-page-prev"' + (_instPage === 0 ? ' disabled' : '') + '>&laquo; Prev</button>' +
                '<span class="pagination-info">Page ' + (_instPage + 1) + ' of ' + totalPages + '</span>' +
                '<button class="btn btn-sm" data-action="inst-page-next"' + (_instPage >= totalPages - 1 ? ' disabled' : '') + '>Next &raquo;</button>';
        }
    }
}

export function instPagePrev() {
    if (_instPage > 0) { _instPage--; _lastInstancesHash = ''; renderInstanceGrid(); }
}

export function instPageNext() {
    const totalPages = Math.ceil(_getFilteredInstances().length / _instPerPage);
    if (_instPage < totalPages - 1) { _instPage++; _lastInstancesHash = ''; renderInstanceGrid(); }
}

export function onInstSearchInput() {
    _instSearch = $('inst-search')?.value || '';
    _instPage = 0;
    _lastInstancesHash = '';
    scheduleRefresh('inst-search', renderInstanceGrid, 250);
}

export async function refreshInstances() {
    const grid = $('instances-grid');
    if (grid && state.instances.length === 0 && !grid.innerHTML) {
        grid.innerHTML = '<div class="loading-spinner">Loading instances...</div>';
    }
    try {
        state.instances = await fetchJson('/api/instances');
        renderInstanceGrid();
    } catch (e) {
        console.error('refreshInstances:', e);
    }
}

// --- Conversation Table ---

export function toggleConvEnded() {
    _convShowEnded = !_convShowEnded;
    const btn = $('conv-toggle-ended');
    if (btn) btn.textContent = _convShowEnded ? 'Hide ended' : 'Show ended';
    _convPage = 0;
    refreshConversationTable();
}

export function onConvSearchInput() {
    _convSearch = $('conv-search')?.value || '';
    _convPage = 0;
    scheduleRefresh('conv-search', _renderConvTable, 250);
}

export function convPagePrev() {
    if (_convPage > 0) { _convPage--; _renderConvTable(); }
}

export function convPageNext() {
    const totalPages = Math.ceil(_getFilteredConvs().length / _convPerPage);
    if (_convPage < totalPages - 1) { _convPage++; _renderConvTable(); }
}

function _getFilteredConvs() {
    let convs = _convData;
    if (!_convShowEnded) convs = convs.filter(c => !c.is_terminal);
    if (_convSearch) {
        const q = _convSearch.toLowerCase();
        convs = convs.filter(c =>
            c.conversation_id.toLowerCase().includes(q) ||
            (c.current_state || '').toLowerCase().includes(q)
        );
    }
    return convs;
}

function _renderConvTable() {
    const body = $('conv-table-body');
    const empty = $('conv-empty');
    const countEl = $('conv-title-count');
    const pagEl = $('conv-pagination');
    if (!body) return;

    const filtered = _getFilteredConvs();
    if (countEl) countEl.textContent = filtered.length;

    if (filtered.length === 0) {
        body.innerHTML = '';
        if (empty) empty.style.display = 'block';
        if (pagEl) pagEl.style.display = 'none';
        return;
    }
    if (empty) empty.style.display = 'none';

    const totalPages = Math.ceil(filtered.length / _convPerPage);
    _convPage = Math.max(0, Math.min(_convPage, totalPages - 1));
    const start = _convPage * _convPerPage;
    const display = filtered.slice(start, start + _convPerPage);

    let rows = '';
    for (const c of display) {
        const badge = c.is_terminal ? 'badge-ended' : 'badge-active';
        const label = c.is_terminal ? 'ENDED' : 'ACTIVE';
        const instId = c.instance_id || '';
        const action = instId ? ' data-action="conv-row-click" data-instance-id="' + esc(instId) + '" data-conv-id="' + esc(c.conversation_id) + '"' : '';
        rows += '<tr class="clickable-row"' + action + '><td class="cell-truncate" title="' + esc(c.conversation_id) + '">' + esc(c.conversation_id.substring(0, 16)) + '</td><td>' + esc(c.current_state) + '</td><td>' + c.message_history.length + '</td><td><span class="badge ' + badge + '">' + label + '</span></td></tr>';
    }
    body.innerHTML = rows;

    if (pagEl) {
        if (totalPages <= 1) {
            pagEl.style.display = 'none';
        } else {
            pagEl.style.display = 'flex';
            pagEl.innerHTML =
                '<button class="btn btn-sm" data-action="conv-page-prev"' + (_convPage === 0 ? ' disabled' : '') + '>&laquo; Prev</button>' +
                '<span class="pagination-info">Page ' + (_convPage + 1) + ' of ' + totalPages + '</span>' +
                '<button class="btn btn-sm" data-action="conv-page-next"' + (_convPage >= totalPages - 1 ? ' disabled' : '') + '>Next &raquo;</button>';
        }
    }
}

export async function refreshConversationTable() {
    const body = $('conv-table-body');
    if (body && _convData.length === 0 && !body.innerHTML) {
        body.innerHTML = '<tr><td colspan="4"><div class="loading-spinner">Loading conversations...</div></td></tr>';
    }
    try {
        _convData = await fetchJson('/api/conversations');
        _renderConvTable();
    } catch (e) {
        console.error('refreshConversationTable:', e);
    }
}
