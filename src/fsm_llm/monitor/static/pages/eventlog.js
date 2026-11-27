// FSM-LLM Monitor: Event log browser (read only; the server cuts every event to a field whitelist)

import { fetchJson } from '../services/api.js';
import { $, esc, jsonViewHtml } from '../utils/dom.js';
import { highlightJsonIn } from '../utils/json_highlight.js';

const failure = e => `<span class="error-message">${esc(e.message)}</span>`;

export async function loadEventLogs() {
    const list = $('eventlog-list');
    if (!list) return;
    try {
        const data = await fetchJson('/api/eventlog');
        const buttons = data.conversations.map(id =>
            `<button class="btn btn-sm" data-action="open-eventlog" data-conv-id="${esc(id)}">${esc(id)}</button>`);
        list.innerHTML = !data.available
            ? '<span class="text-dim">No event log: start the monitor with --event-log DIR, or attach an API that has one.</span>'
            : buttons.join(' ') || '<span class="text-dim">No logs yet</span>';
    } catch (e) {
        list.innerHTML = failure(e);
    }
}

// DOM bound of one open log: past this many rows the view stops paging and says where the log continues.
const EVENTLOG_MAX_ROWS = 2000;
// The log on screen ({id, head, rows, next, pages, busy, error}) and a token bumped by every open:
// a reply that is not the latest belongs to a log or page that is no longer wanted and is dropped, never painted.
let _log = null;
let _token = 0;

const _row = ({ seq, ts, type, frame, ...rest }) =>
    `<tr><td>${esc(seq)}</td><td>${esc(ts)}</td><td>${esc(frame)}</td><td>${esc(type)}</td><td>${Object.keys(rest).length ? jsonViewHtml(JSON.stringify(rest)) : ''}</td></tr>`;

function _take(log, d) {
    const u = d.usage;
    const head = `<div class="text-small-dim">${esc(d.conversation_id)}: ${esc(u.calls)} LLM calls, ${esc(u.total_tokens)} tokens, ${esc(u.errors)} errors (from the log)${d.next_seq != null ? '; the first events only' : ''}${d.skipped ? `; damaged after seq ${esc(d.corrupt_after_seq)}, the rest is not shown` : ''}</div>`;
    const rows = d.events.map(_row);   // all of it can throw before the log is touched: a bad page changes nothing
    Object.assign(log, { head, next: d.next_seq, error: null });
    log.rows.push(...rows);
    log.pages++;
}

function _paint(log) {
    const view = $('eventlog-view');
    if (!view) return;
    const tail = log.next != null
        ? (log.rows.length >= EVENTLOG_MAX_ROWS
            ? `<div class="text-small-dim">Showing the first ${esc(log.rows.length)} events; the log continues after seq ${esc(log.next)}</div>`
            : '<button class="btn btn-sm" data-action="eventlog-more">Load more</button>')
        : (log.pages > 1 ? '<div class="text-small-dim">No more events to show</div>' : '');
    view.innerHTML = `${log.head}<table class="viz-trans-table"><thead><tr><th>seq</th><th>time</th><th>frame</th><th>type</th><th>fields</th></tr></thead><tbody>${log.rows.join('')}</tbody></table>${log.error ? failure(log.error) : ''}${tail}`;
    highlightJsonIn(view);
}

export async function openEventLog(convId) {
    const view = $('eventlog-view');
    if (!view || !convId) return;
    _log = null;    // a Load more of the log being replaced must not supersede this open
    const token = ++_token;
    try {
        const d = await fetchJson(`/api/eventlog/${encodeURIComponent(convId)}`);
        if (token !== _token) return;
        _log = { id: convId, rows: [], pages: 0, busy: false };
        _take(_log, d);
        _paint(_log);
    } catch (e) {
        if (token === _token) view.innerHTML = failure(e);
    }
}

/** Append the page after the last seq shown (same log, no first-page refetch); an error keeps the pages already shown. */
export async function loadMoreEventLog() {
    const log = _log;
    if (!log || log.busy || log.next == null || log.rows.length >= EVENTLOG_MAX_ROWS) return;
    log.busy = true;
    const token = _token;   // only an open moves it: the record is null while one runs, so no Load more overlaps an open
    try {
        const d = await fetchJson(`/api/eventlog/${encodeURIComponent(log.id)}?since_seq=${encodeURIComponent(log.next)}`);
        if (token !== _token) return;
        _take(log, d);
    } catch (e) {
        if (token !== _token) return;
        log.error = e;
    } finally {
        log.busy = false;
    }
    _paint(log);
}
