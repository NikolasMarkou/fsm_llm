import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as conv from '../static/pages/conversations.js';
import * as control from '../static/pages/control.js';

// Conversation A is sent a message; its reply (terminal, END_A) arrives while the drawer shows A or B.
async function scenario(switchAway) {
    let reply;
    serve((url, opts) => {
        if (opts?.method === 'POST') return new Promise(res => { reply = () => res({ response: 'REPLY-FOR-A', current_state: 'END_A', is_terminal: true }); });
        return { conversation_id: url.endsWith('/B') ? 'B' : 'A', instance_id: 'I', current_state: 'live', is_terminal: false, message_history: [], context_data: {} };
    });
    const badge = { textContent: 'stateB' };
    el('conv-detail').querySelector = () => badge;
    el('conv-chat-input').style.display = '';
    el('conv-chat-log').html.length = 0;
    conv.showConversationInDrawer('I', 'A'); await sleep(20);
    el('conv-message-input').value = 'hello';
    const sent = conv.sendChatMessage(); await sleep(20);
    if (switchAway) { conv.drawerBack(); conv.showConversationInDrawer('I', 'B'); await sleep(20); }
    reply(); await sent;
    return { badge: badge.textContent, input_display: el('conv-chat-input').style.display, bubbles: el('conv-chat-log').html.filter(h => h.includes('REPLY-FOR-A')).length };
}
conv.setDeps({});
state.instances = [{ instance_id: 'I', instance_type: 'fsm', label: 'L' }];
state.currentPage = 'control';
const r = { stay: await scenario(false), switched: await scenario(true) };

// Plan 9283c1fd step 6 (O3, B4): a conversation of the attached API (no instance) opens read-only and
// never binds to a managed FSM; a failed send in a conversation the user left is a toast and the typed
// text comes back. Nothing below decides by a clock: timers are inert and the replies are gates.
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {};
globalThis.setTimeout = () => 1;
globalThis.confirm = () => true;
console.error = () => {};
const flush = async () => { for (let i = 0; i < 12; i++) await new Promise(r => setImmediate(r)); };
const toasts = () => (document.body.children || []).map(c => c.textContent);
const MANAGED = 'managedFSM';
const posts = [], gets = [], snaps = {};
const snap = (id, owner) => ({ conversation_id: id, instance_id: owner, current_state: 'live', is_terminal: false, message_history: [], context_data: {} });
const answer = (url, opts) => {
    if (opts?.method === 'POST') { posts.push(url); return { response: 'REPLY', current_state: 'live', is_terminal: false }; }
    const m = url.match(/^\/api\/conversations\/(.+)$/);
    if (m) return snaps[m[1]];
    gets.push(url);
    return [];
};
conv.setDeps({ showPage() {}, openDrawer: control.openDrawer, refreshDetailPanel: control.refreshDetailPanel });
function fresh() {
    serve(answer);
    state.instances = [{ instance_id: MANAGED, instance_type: 'fsm', label: 'M' }];
    state.currentPage = 'control';
    state.selectedDetailId = 'STALE'; state.selectedDetailType = 'fsm';
    state.selectedConvId = null; state.selectedConvInstanceId = null; state.selectedConvReadOnly = false;
    el('ctrl-drawer-content').innerHTML = 'STALE-PANEL';
    el('ctrl-drawer-events').innerHTML = 'STALE-EVENTS';
    el('conv-detail').innerHTML = '';
    el('conv-chat-input').style.display = '';
    el('conv-chat-log').html.length = 0;
    el('conv-message-input').value = '';
    document.body.children = [];
    posts.length = 0; gets.length = 0;
}
async function open(instanceId, convId, owner) {
    fresh();
    snaps[convId] = snap(convId, owner);
    conv.showConversationInDrawer(instanceId, convId); await flush();
    const html = el('conv-detail').innerHTML;
    return {
        chat_input: el('conv-chat-input').style.display,
        end_button: html.includes('end-conversation'),
        end_target: (html.match(/data-instance-id="([^"]*)"/) || [])[1] ?? null,
        conv_inst: state.selectedConvInstanceId,
        read_only: state.selectedConvReadOnly,
        detail_id: state.selectedDetailId,
    };
}
const tryActions = async (convId, endAs) => {
    el('conv-message-input').value = 'hi';
    await conv.sendChatMessage();
    await conv.endConversation(endAs, convId);
    return [...posts];
};

// an attached-API conversation (instance id '') beside a managed FSM in the list
r.attached = await open('', 'att1', '');
r.attached.posts = await tryActions('att1', MANAGED);
conv.drawerBack(); await flush();
r.attached.after_back = { read_only: state.selectedConvReadOnly, detail_id: state.selectedDetailId, panel: el('ctrl-drawer-content').innerHTML, events: el('ctrl-drawer-events').innerHTML, gets: [...gets] };
// the flag does not outlive the entry that set it: a plain open of an instance clears it
await open('', 'att1', '');
control.openDrawer(MANAGED, 'fsm', { fetchDetail: false });
r.attached.after_open_drawer = state.selectedConvReadOnly;
// a managed conversation keeps its chat input and End, bound to the row's instance
r.managed = await open(MANAGED, 'c1', MANAGED);
r.managed.posts = await tryActions('c1', MANAGED);
// the flag is up from the entry, before the snapshot is back; and the two senders refuse on the flag alone
{
    fresh();
    let release;
    const held = new Promise(res => { release = res; });
    serve(async (url, opts) => { if (url.startsWith('/api/conversations/')) await held; return answer(url, opts); });
    snaps.att2 = snap('att2', '');
    conv.showConversationInDrawer('', 'att2'); await flush();
    r.during_load = state.selectedConvReadOnly;
    release(); await flush();
    await open(MANAGED, 'c1', MANAGED);
    state.selectedConvReadOnly = true;
    r.read_only_guards = await tryActions('c1', MANAGED);
}
// no explicit id, the owner comes from the snapshot
r.snapshot_owner = await open('', 'c2', 'OWN');
r.snapshot_owner.posts = await tryActions('c2', 'OWN');

// a failed send: a toast always, the bubble and the typed text only where the user still is
async function sending(fails, away, typed) {
    fresh();
    snaps.A = snap('A', MANAGED); snaps.B = snap('B', MANAGED);
    let settle;
    serve((url, opts) => {
        if (opts?.method === 'POST') return new Promise(res => { settle = () => res(fails ? { fail: 'LLM down' } : { response: 'REPLY', current_state: 'live', is_terminal: false }); });
        return answer(url, opts);
    });
    conv.showConversationInDrawer(MANAGED, 'A'); await flush();
    el('conv-message-input').value = 'hello';
    const sent = conv.sendChatMessage(); await flush();
    if (away) { conv.drawerBack(); conv.showConversationInDrawer(MANAGED, 'B'); await flush(); }
    if (typed !== undefined) el('conv-message-input').value = typed;
    settle(); await sent;
    return { error_bubbles: el('conv-chat-log').html.filter(h => h.includes('LLM down')).length, reply_bubbles: el('conv-chat-log').html.filter(h => h.includes('REPLY')).length, toasts: toasts(), input: el('conv-message-input').value, disabled: el('conv-message-input').disabled };
}
r.send = {
    stay: await sending(true, false),
    stay_typed: await sending(true, false, 'draft'),
    away: await sending(true, true),
    away_typed: await sending(true, true, 'draft B'),
    ok: await sending(false, false),
};
out(r);
process.exit(0);
