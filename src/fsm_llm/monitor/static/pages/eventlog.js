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

export async function openEventLog(convId) {
    const view = $('eventlog-view');
    if (!view || !convId) return;
    try {
        const d = await fetchJson(`/api/eventlog/${encodeURIComponent(convId)}`);
        const rows = d.events.map(({ seq, ts, type, frame, ...rest }) =>
            `<tr><td>${esc(seq)}</td><td>${esc(ts)}</td><td>${esc(frame)}</td><td>${esc(type)}</td><td>${Object.keys(rest).length ? jsonViewHtml(JSON.stringify(rest)) : ''}</td></tr>`);
        const u = d.usage;
        view.innerHTML = `<div class="text-small-dim">${esc(d.conversation_id)}: ${esc(u.calls)} LLM calls, ${esc(u.total_tokens)} tokens, ${esc(u.errors)} errors (from the log)${d.next_seq != null ? '; the first events only' : ''}</div>`
            + `<table class="viz-trans-table"><thead><tr><th>seq</th><th>time</th><th>frame</th><th>type</th><th>fields</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
        highlightJsonIn(view);
    } catch (e) {
        view.innerHTML = failure(e);
    }
}
