import { createHash } from 'node:crypto';
import { El, el, serve, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as launch from '../static/pages/launch.js';
import * as builder from '../static/pages/builder.js';
import * as conv from '../static/pages/conversations.js';
import * as eventlog from '../static/pages/eventlog.js';
import * as control from '../static/pages/control.js';
import * as visualizer from '../static/pages/visualizer.js';
import * as logs from '../static/pages/logs.js';
import { renderGraph } from '../static/utils/graph.js';
import { renderMarkdown } from '../static/utils/markdown.js';

// A view or a timer must not act on a request, a modal or a state that is no longer the
// current one. No sleep decides anything here: the timers are a fake scheduler, the replies are gates.
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push({ fn, ms, live: true }); return timers.length; };
globalThis.clearTimeout = id => { if (timers[id - 1]) timers[id - 1].live = false; };
/** Run the pending 500 ms launch windows (the toast timers of 4000 ms are left alone); how many ran. */
const fire = () => {
    const due = timers.filter(t => t.live && t.ms === 500);
    for (const t of due) { t.live = false; t.fn(); }
    return due.length;
};
const flush = async () => { for (let i = 0; i < 12; i++) await new Promise(r => setImmediate(r)); };
const gate = () => { let open; const p = new Promise(r => { open = r; }); return { p, open }; };
console.error = () => {};                        // checkCapabilities logs the failure it also toasts
const toasts = () => (document.body.children || []).map(c => c.textContent);

const defaultCaps = { ...state.capabilities };   // read before anything sets it
const calls = [];
const route = { fail: {}, gates: {}, custom: null };  // route[url] -> fail text, or a gate the reply waits on; custom(url, opts) answers first
serve(async (url, opts) => {
    calls.push(`${opts?.method || 'GET'} ${url}`);
    if (route.custom) { const r = await route.custom(url, opts); if (r !== undefined) return r; }
    if (route.gates[url]) await route.gates[url].p;
    if (route.fail[url] !== undefined) return { fail: route.fail[url] };
    if (url === '/api/agent/types') return { agent_types: [{ id: 'debate', name: 'debate', launchable: true, needs_tools: false, takes_tools: false, tool_names: [], drawable: true }] };
    if (url === '/api/presets') return { fsm: [] };
    if (url === '/api/workflow/presets') return { workflows: [] };
    if (url === '/api/capabilities') return { fsm: true, workflows: true, agents: true };
    if (url === '/api/fsm/launch') return { instance_id: 'i1', label: 'L1' };
    if (url === '/api/fsm/i1/start') return { conversation_id: 'c1' };
    if (url === '/api/workflow/launch' || url === '/api/agent/launch') return { instance_id: 'x1', label: 'X1' };
    if (url.startsWith('/api/fsm/visualize')) return { nodes: [], edges: [] };
    return { session_id: 's', response: 'done', internal_state: {}, is_complete: true, artifact: FSM, artifact_json: JSON.stringify(FSM), artifact_type: 'fsm', is_valid: true };
});
const FSM = { name: 'F', initial_state: 'a', states: { a: { id: 'a', description: 'd', purpose: 'p' } } };
const count = url => calls.filter(c => c === `POST ${url}`).length;
const reset = () => { calls.length = 0; route.fail = {}; route.gates = {}; route.custom = null; timers.length = 0; };

const pages = [];
const drawer = [];
launch.setDeps({ showPage: p => pages.push(p), refreshInstances() {}, showConversationInDrawer: (...a) => drawer.push(a) });
builder.setDeps({ showPage: p => pages.push(p), refreshInstances() {} });
const result = {};

// --- the capabilities fetch fails (the server decides; the failure is shown once) ---
route.fail['/api/capabilities'] = 'boom';
launch.showLaunchModal();
await flush();
const tmpl = el('launch-tool-template');
result.capabilities = {
    defaults: defaultCaps,
    kept: { ...state.capabilities },
    toasts: toasts().filter(t => t.includes('boom')).length,
};
result.tool_templates = {
    count: (tmpl.children || []).length,
    first: (tmpl.children || [])[0] ? [tmpl.children[0].value, tmpl.children[0].textContent] : null,
    search: (tmpl.children || [])[1] ? [tmpl.children[1].value, tmpl.children[1].textContent] : null,
    innerHTML: tmpl.innerHTML,
};
Object.assign(el('launch-wf-preset'), { value: 'linear' });
Object.assign(el('launch-fsm-source'), { value: 'preset' });
Object.assign(el('launch-fsm-preset-id'), { value: 'demo' });
Object.assign(el('launch-agent-type'), { value: 'debate' });
el('launch-agent-task').value = 'do it';
reset();
{
    const btn = { disabled: false, textContent: 'Launch Workflow' };
    await launch.doLaunchWorkflow(btn);
    result.capabilities.workflow_posts = count('/api/workflow/launch');
    result.capabilities.workflow_status = el('launch-wf-status').innerHTML;
    fire();
}

// --- the launch created the instance, then the first start failed ---
const status = id => el(id).innerHTML;
reset(); pages.length = 0; drawer.length = 0;
route.fail['/api/fsm/i1/start'] = 'LLM operation timed out';
const startFailedToasts = () => toasts().filter(t => t.includes('starting the first conversation failed')).length;
{
    const btn = { disabled: false, textContent: 'Launch FSM' };
    const toastsBefore = startFailedToasts();
    await launch.doLaunchFSM(btn);
    const after = { status: status('launch-fsm-status'), disabled: btn.disabled, pages: [...pages] };
    const ran = fire();
    result.fsm_partial = { after, ran, pages: [...pages], drawer: drawer.length, disabled: btn.disabled, launches: count('/api/fsm/launch'), modal: el('launch-modal').style.display, toast: startFailedToasts() - toastsBefore };
}
reset(); pages.length = 0; drawer.length = 0;
route.fail['/api/fsm/launch'] = 'bad preset';
{
    const btn = { disabled: false, textContent: 'Launch FSM' };
    await launch.doLaunchFSM(btn);
    result.fsm_first_failed = { status: status('launch-fsm-status'), disabled: btn.disabled, timers: timers.filter(t => t.live).length, pages: [...pages], posts: calls.filter(c => c.startsWith('POST')).length };
}

// --- the modal closed inside the window (or while the start was in flight): no navigation ---
const launchers = {
    fsm: [launch.doLaunchFSM, 'Launch FSM'],
    workflow: [launch.doLaunchWorkflow, 'Launch Workflow'],
    agent: [launch.doLaunchAgent, 'Launch Agent'],
};
for (const [name, [run, label]] of Object.entries(launchers)) {
    for (const mode of ['stays_open', 'closed', 'closed_and_reopened']) {
        reset(); pages.length = 0; drawer.length = 0;
        el('launch-modal').style.display = 'flex';
        const btn = { disabled: false, textContent: label };
        await run(btn);
        if (mode !== 'stays_open') launch.closeLaunchModal();
        if (mode === 'closed_and_reopened') { launch.showLaunchModal(); await flush(); }
        const before = btn.disabled;
        fire();
        result[`${name}_${mode}`] = { disabled_in_window: before, pages: [...pages], drawer: drawer.length, disabled: btn.disabled, modal: el('launch-modal').style.display };
    }
}
reset(); pages.length = 0; drawer.length = 0;
el('launch-modal').style.display = 'flex';
route.gates['/api/fsm/i1/start'] = gate();
{
    const btn = { disabled: false, textContent: 'Launch FSM' };
    const running = launch.doLaunchFSM(btn);
    await flush();
    launch.closeLaunchModal();                  // closed while the start request is in flight
    route.gates['/api/fsm/i1/start'].open();
    await running;
    fire();
    result.fsm_closed_during_start = { pages: [...pages], drawer: drawer.length, disabled: btn.disabled };
}

// --- a newer launch owns the button: the 500 ms window of an earlier one never frees it ---
reset(); pages.length = 0; drawer.length = 0;
el('launch-modal').style.display = 'flex';
{
    const btn = { disabled: false, textContent: 'Launch Workflow' };
    await launch.doLaunchWorkflow(btn);                    // A lands, its window is pending
    launch.closeLaunchModal();
    launch.showLaunchModal(); await flush();
    route.gates['/api/workflow/launch'] = gate();
    const second = launch.doLaunchWorkflow(btn);           // B is in flight on the same button
    await flush();
    const before = btn.disabled;
    fire();                                                // A's window ends
    const afterA = { disabled: btn.disabled, text: btn.textContent };
    route.gates['/api/workflow/launch'].open(); await second;
    fire();                                                // B's own window frees the button
    result.launch_newer_owns_button = { before, afterA, afterB: { disabled: btn.disabled, text: btn.textContent }, posts: count('/api/workflow/launch') };
}

// --- the builder FSM launch ---
async function builderReady() {
    builder.startBuilderSession();
    await flush();
}
const bbtn = el('builder-launch-btn');
reset(); pages.length = 0;
await builderReady();
reset();
route.fail['/api/fsm/i1/start'] = 'LLM operation timed out';
const builderToastsBefore = startFailedToasts();
builder.launchBuilderResult();
await flush();
result.builder_partial = { status: status('builder-status'), disabled: bbtn.disabled, pages_before: [...pages], toast: startFailedToasts() - builderToastsBefore };
fire();
result.builder_partial.pages = [...pages];
result.builder_partial.disabled_after = bbtn.disabled;
result.builder_partial.launches = count('/api/fsm/launch');

pages.length = 0;
await builderReady();
reset();
route.gates['/api/fsm/launch'] = gate();
builder.launchBuilderResult();
builder.launchBuilderResult();                   // a second click while the first is in flight
await flush();
const inFlight = { disabled: bbtn.disabled, launches: count('/api/fsm/launch') };
route.gates['/api/fsm/launch'].open();
await flush();
fire();
result.builder_double = { inFlight, launches: count('/api/fsm/launch'), disabled_after: bbtn.disabled };

await builderReady();
reset();
route.fail['/api/fsm/launch'] = 'bad definition';
route.gates['/api/fsm/launch'] = gate();
builder.launchBuilderResult();
await flush();
const failingFlight = { disabled: bbtn.disabled };
route.gates['/api/fsm/launch'].open();
await flush();
result.builder_first_failed = { ...failingFlight, status: status('builder-status'), disabled_after: bbtn.disabled, timers: timers.filter(t => t.live && t.ms === 500).length };
delete route.gates['/api/fsm/launch'];
builder.launchBuilderResult();                   // a retry after the failure is a new launch
await flush();
result.builder_first_failed.retry_launches = count('/api/fsm/launch');

// --- a failing conversation detail load that was superseded does not touch the view ---
conv.setDeps({});
state.instances = [{ instance_id: 'I', instance_type: 'fsm', label: 'L' }];
state.currentPage = 'control';
const convBody = (id, st) => ({ conversation_id: id, instance_id: 'I', current_state: st, is_terminal: false, message_history: [], context_data: {} });
const shown = () => {
    const html = el('conv-detail').innerHTML;
    return { live: html.includes('liveB'), prior: html === 'PRIOR', error: (html.match(/<span class="error-message">(.*?)<\/span>/) || [])[1] ?? null, input: el('conv-chat-input').style.display };
};
{
    reset();
    const gA = gate();
    route.custom = async url => {
        if (url.endsWith('/A')) { await gA.p; return { fail: 'ended A' }; }
        return convBody('B', 'liveB');
    };
    const a = conv.showConversationDetail('A');
    await conv.showConversationDetail('B');
    const beforeA = shown();
    gA.open();
    await a;
    result.detail_superseded = { before: beforeA, after: shown(), selected: state.selectedConvId };
}
{
    reset();
    route.custom = url => (url.endsWith('/A') ? { fail: 'ended <b>A</b>' } : undefined);
    await conv.showConversationDetail('A');
    result.detail_current_fails = shown();
}
{
    reset();
    const gA = gate();
    route.custom = async () => { await gA.p; return { fail: 'late' }; };
    el('conv-detail').innerHTML = 'PRIOR';
    el('conv-chat-input').style.display = 'block';
    const a = conv.showConversationDetail('A');
    state.selectedConvId = null;               // the drawer went back: no conversation is shown
    gA.open();
    await a;
    result.detail_after_back = shown();
}
// --- the input is focused only while the same conversation is still shown and the input is visible ---
async function focusScenario(mode) {
    reset();
    const gate2 = gate();
    route.custom = async (url, opts) => {
        if (opts?.method === 'POST') { await gate2.p; return { response: 'R', current_state: mode === 'terminal' ? 'END' : 'live', is_terminal: mode === 'terminal' }; }
        return convBody(url.endsWith('/B') ? 'B' : 'A', 'live');
    };
    const input = el('conv-message-input');
    let focused = 0;
    input.focus = () => { focused++; };
    el('conv-chat-input').style.display = '';
    conv.showConversationInDrawer('I', 'A');
    await flush();
    input.value = 'hello';
    const sent = conv.sendChatMessage();
    await flush();
    if (mode === 'switched') { conv.drawerBack(); conv.showConversationInDrawer('I', 'B'); await flush(); }
    gate2.open();
    await sent;
    return { focused, disabled: input.disabled };
}
result.focus = { stays: await focusScenario('stays'), switched: await focusScenario('switched'), terminal: await focusScenario('terminal') };

// --- the event log: Load more by since_seq, a request token, an error keeps the earlier pages ---
const seqs = html => [...html.matchAll(/<tr><td>(\d+)<\/td>/g)].map(m => Number(m[1]));
const logView = () => el('eventlog-view').innerHTML;
const usage = { calls: 1, total_tokens: 2, errors: 0 };
/** A log of `total` events served 200 at a time after `since_seq`, as the route does (next_seq only on a full page). */
const pageOf = (id, url, total, mark) => {
    const since = Number(new URL(url, 'http://x').searchParams.get('since_seq') || 0);
    const last = Math.min(since + 200, total);
    const events = [];
    for (let s = since + 1; s <= last; s++) events.push({ seq: s, ts: `t${s}`, type: 'turn/start', frame: 0, ...(s === 201 ? { state: '<script>x</script>' } : {}), ...(mark ? { log: id } : {}) });
    return { conversation_id: id, events, usage, skipped: 0, corrupt_after_seq: null, next_seq: events.length === 200 ? last : null };
};
const more = eventlog.loadMoreEventLog ?? (async () => {});   // absent before the fix: the rows then fail on values
const asks = () => calls.filter(c => c.startsWith('GET /api/eventlog'));
{
    reset();
    route.custom = url => (url.startsWith('/api/eventlog/abc') ? pageOf('abc', url, 450) : undefined);
    await eventlog.openEventLog('abc');
    const first = logView();
    await more();
    const second = logView();
    await more();
    const third = logView();
    const askedBefore = asks().length;
    await more();           // nothing is left: no request
    result.eventlog_more = {
        first: { seqs: seqs(first).length, more: first.includes('data-action="eventlog-more"'), notice: first.includes('the first events only'), end: first.includes('No more events to show') },
        second: { seqs: seqs(second).length, more: second.includes('data-action="eventlog-more"'), notice: second.includes('the first events only'), headers: second.split('<thead>').length - 1, escaped: second.includes('&lt;script&gt;'), raw: second.includes('<script>') },
        third: { rows: seqs(third).length, unique: new Set(seqs(third)).size, last: seqs(third).at(-1), more: third.includes('data-action="eventlog-more"'), notice: third.includes('the first events only'), end: third.includes('No more events to show'), headers: third.split('<thead>').length - 1 },
        asked: asks(),
        asked_after_end: asks().length - askedBefore,
    };
}
{
    reset();
    const g = gate();
    route.custom = async url => { if (url.includes('since_seq=200')) await g.p; return url.startsWith('/api/eventlog/abc') ? pageOf('abc', url, 450) : undefined; };
    await eventlog.openEventLog('abc');
    const one = more();
    const two = more();     // a second click while the first is in flight
    g.open();
    await Promise.all([one, two]);
    result.eventlog_double_click = { asked: asks(), rows: seqs(logView()).length };
}
{
    reset();
    let broken = true;
    route.custom = url => {
        if (url.startsWith('/api/eventlog/abc')) return broken && url.includes('since_seq') ? { fail: 'boom <i>x</i>' } : pageOf('abc', url, 450);
        return undefined;
    };
    await eventlog.openEventLog('abc');
    await more();
    const failed = logView();
    broken = false;
    await more();
    const retried = logView();
    result.eventlog_error = { rows: seqs(failed).length, error: failed.includes('boom &lt;i&gt;x&lt;/i&gt;'), raw: failed.includes('<i>x'), more: failed.includes('data-action="eventlog-more"'), rows_after: seqs(retried).length, error_after: retried.includes('boom') };
}
{
    reset();
    const gA = gate();
    route.custom = async url => {
        if (url.startsWith('/api/eventlog/A')) { await gA.p; return pageOf('A', url, 450, true); }
        if (url.startsWith('/api/eventlog/B')) return pageOf('B', url, 450, true);
        return undefined;
    };
    // log A (slow) opened, then log B (fast)
    const openA = eventlog.openEventLog('A');
    await eventlog.openEventLog('B');
    const afterB = logView();
    gA.open();
    await openA;
    result.eventlog_open_race = { b_shown: afterB.includes('B:'), final_b: logView().includes('B:') && !logView().includes('A:'), final_rows: seqs(logView()).length };
    // A Load more of log A lands after log B was opened: dropped, and B continues from its own next_seq
    reset();
    const gA2 = gate();
    route.custom = async url => {
        if (url.startsWith('/api/eventlog/A')) { if (url.includes('since_seq')) await gA2.p; return pageOf('A', url, 450, true); }
        if (url.startsWith('/api/eventlog/B')) return pageOf('B', url, 450, true);
        return undefined;
    };
    await eventlog.openEventLog('A');
    const moreA = more();
    await eventlog.openEventLog('B');
    gA2.open();
    await moreA;
    const mixed = logView();
    await more();
    const next = logView();
    result.eventlog_more_race = { rows: seqs(mixed).length, b_only: mixed.includes('B:') && !mixed.includes('A:'), next_rows: seqs(next).length, next_b_only: !next.includes('A:'), asked: asks().filter(a => a.includes('since_seq')) };
}
{
    // A Load more clicked while another log is still opening must not supersede that open
    reset();
    const gB = gate();
    route.custom = async url => {
        if (url.startsWith('/api/eventlog/B')) await gB.p;
        return url.startsWith('/api/eventlog/') ? pageOf(url.split('/')[3].split('?')[0], url, 450, true) : undefined;
    };
    await eventlog.openEventLog('A');
    const openB = eventlog.openEventLog('B');
    await more();                                // the stale button of log A, clicked while B loads
    gB.open();
    await openB;
    result.eventlog_more_during_open = { b_only: logView().includes('B:') && !logView().includes('A:'), asked: asks() };
}
{
    // A stale open that FAILS must not paint over the newer log either; hostile fields of every column are escaped
    reset();
    const gA = gate();
    route.custom = async url => {
        if (url.startsWith('/api/eventlog/A')) { await gA.p; return { fail: 'late failure of A' }; }
        if (url.startsWith('/api/eventlog/B')) return pageOf('B', url, 10, true);
        return undefined;
    };
    const openA = eventlog.openEventLog('A');
    await eventlog.openEventLog('B');
    gA.open();
    await openA;
    result.eventlog_stale_failure = { b_only: logView().includes('B:') && !logView().includes('late failure') };
    reset();
    route.custom = () => ({ conversation_id: 'h<u>x', usage, next_seq: null, events: [{ seq: 1, ts: '<i>ts</i>', frame: '<s>f</s>', type: '<b>t</b>', note: '<em>n</em>' }] });
    await eventlog.openEventLog('h');
    const hostile = logView();
    result.eventlog_hostile = { raw: ['<u>', '<i>', '<s>', '<b>', '<em>'].filter(t => hostile.includes(t)), escaped: ['h&lt;u&gt;x', '&lt;i&gt;ts', '&lt;s&gt;f', '&lt;b&gt;t', '&lt;em&gt;n'].filter(t => hostile.includes(t)).length };
}
{
    reset();
    route.custom = url => (url.startsWith('/api/eventlog/big') ? pageOf('big', url, 2600) : undefined);
    await eventlog.openEventLog('big');
    for (let i = 0; i < 12; i++) await more();
    const capped = logView();
    result.eventlog_cap = { rows: seqs(capped).length, more: capped.includes('eventlog-more'), line: (capped.match(/Showing the first[^<]*/) || [null])[0], asked: asks().length };
}
{
    reset();
    route.custom = url => (url.startsWith('/api/eventlog/zzz') ? { fail: 'no such log' } : undefined);
    await eventlog.openEventLog('zzz');
    const failedView = logView();
    const before = asks().length;
    await more();
    result.eventlog_failed_open = { error: failedView.includes('no such log'), more: failedView.includes('eventlog-more'), asked_more: asks().length - before };
}

// --- the workflow step data in the control drawer, and the <details> open state across the 2 s repaint ---
{
    // The stub has no DOM: the drawer content gets the two behaviours the paint relies on. Assigning innerHTML re-creates
    // every <details> CLOSED (as a real DOM does) and querySelectorAll('details') returns the current ones.
    const box = el('ctrl-drawer-content');
    let markup = '';
    let details = [];
    Object.defineProperty(box, 'innerHTML', { get: () => markup, set: v => { markup = v; details = [...v.matchAll(/<details/g)].map(() => ({ open: false })); } });
    box.querySelectorAll = sel => (sel === 'details' ? details : []);
    state.instances = [{ instance_id: 'W', instance_type: 'workflow', label: 'W', status: 'running' }];
    state.selectedDetailId = 'W'; state.selectedDetailType = 'workflow';
    state.selectedConvId = null;
    let wf = {};
    route.custom = url => (url === '/api/workflow/W/instances' ? [wf] : url.includes('/events') ? [] : undefined);
    const paint = async w => { wf = w; await control.refreshDetailPanel('W', 'workflow'); await flush(); return markup; };
    const decode = t => t.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
    const viewTexts = h => [...h.matchAll(/<span class="json-view">([\s\S]*?)<\/span>/g)].map(m => decode(m[1]));
    const stepWith = data => ({ workflow_instance_id: 'w1', status: 'running', history: [{ step_id: 's', data }] });
    const parses = t => { try { JSON.parse(t); return true; } catch { return false; } };

    // a small payload (822 characters) is shown whole: it parses and equals the pretty text
    const small = Object.fromEntries(Array.from({ length: 40 }, (_, i) => [`key${i}`, `value${i}`]));
    const smallText = JSON.stringify(small, null, 1);
    const hs = await paint(stepWith(small));
    const shown = viewTexts(hs);
    result.step_data_small = { length: smallText.length, whole: shown.length === 1 && shown[0] === smallText, parses: parses(shown[0] ?? ''), note: /Truncated|more characters/.test(hs) };

    // exactly the cap is whole, one more character is the labelled cut
    const padTo = n => { const base = JSON.stringify({ k: '' }, null, 1).length; return { k: 'x'.repeat(n - base) }; };
    const edge = JSON.stringify(padTo(20000), null, 1).length;
    const hEdge = await paint(stepWith(padTo(20000)));
    const hOver = await paint(stepWith(padTo(20001)));
    result.step_data_edge = { edge, at_cap_whole: viewTexts(hEdge).length === 1 && !/Truncated/.test(hEdge), over_cut: /Truncated/.test(hOver) && /1 more characters not shown/.test(hOver) };

    // a 50000 character payload with hostile text: labelled raw cut, outside any json-view, escaped, the count right
    const huge = { a: '<i>x</i>', blob: 'y'.repeat(50000) };
    const hugeText = JSON.stringify(huge, null, 1);
    const hh = await paint(stepWith(huge));
    const rawM = hh.match(/<pre class="step-data-raw">([\s\S]*?)<\/pre>/);
    const rawText = rawM ? decode(rawM[1]) : null;
    result.step_data_huge = {
        label: hh.includes('Truncated raw text, not valid JSON'),
        count: hh.includes(`${hugeText.length - 20000} more characters not shown`),
        shown: rawText === hugeText.slice(0, 20000),
        in_json_view: viewTexts(hh).some(t => t.includes('blob')),
        raw_tags: hh.includes('<i>x'),
        escaped: hh.includes('&lt;i&gt;x'),
    };

    // the open state of the Context Data <details> survives a repaint that changes the markup; a closed one stays closed
    const withCtx = (stamp) => ({ workflow_instance_id: 'w1', status: 'running', updated_at: stamp, context: { a: 1 } });
    await paint(withCtx('2026-01-01T00:00:00Z'));
    const closedBefore = details.map(d => d.open);
    await paint(withCtx('2026-01-01T00:00:02Z'));
    const closedAfter = details.map(d => d.open);
    details[0].open = true;                                  // the user opens it
    const h2 = await paint(withCtx('2026-01-01T00:00:04Z'));
    const openedAfter = details.map(d => d.open);
    const h3 = await paint(withCtx('2026-01-01T00:00:04Z'));  // identical markup: no repaint, no change
    result.details_state = { count: details.length, closed_before: closedBefore, closed_after: closedAfter, opened_after: openedAfter, repainted: h2 !== '', unchanged: details.map(d => d.open), same: h2 === h3 };
}

// --- the builder input area waits for the start reply; an edit with no states hides the old graph; the first status paint ---
{
    const input = () => el('builder-input-area').style.display;
    const status = () => el('builder-status').innerHTML;
    const notDone = { session_id: 's', response: 'hi', internal_state: {}, is_complete: false };
    reset();
    const g = gate();
    route.custom = url => (url === '/api/builder/start' ? g.p.then(() => notDone) : undefined);
    builder.startBuilderSession();
    await flush();
    const pending = input();
    g.open(); await flush();
    result.builder_input_start = { pending, after: input() };
    // a failed start leaves it hidden and shows the error
    reset();
    route.fail['/api/builder/start'] = 'no model';
    builder.startBuilderSession(); await flush();
    result.builder_input_failed = { input: input(), error: status().includes('Failed to start: no model') };
    // a start that is superseded before its reply: the dropped reply must not show the input
    reset();
    const gA = gate(), gB = gate();
    let starts = 0;
    route.custom = url => { if (url !== '/api/builder/start') return undefined; return (++starts === 1 ? gA : gB).p.then(() => notDone); };
    builder.startBuilderSession();
    builder.startBuilderSession();
    await flush();
    gA.open(); await flush();
    const afterStale = input();
    gB.open(); await flush();
    result.builder_input_superseded = { after_stale: afterStale, after_current: input() };
    // an already finished build hides it again (the reply completes the session)
    reset();
    route.custom = undefined;
    builder.startBuilderSession(); await flush();
    result.builder_input_complete = input();

    // the first status paint of a send is at once, not after the first 1000 ms tick
    reset();
    route.custom = url => (url === '/api/builder/start' ? notDone : undefined);
    builder.startBuilderSession(); await flush();
    const gS = gate();
    route.custom = url => (url === '/api/builder/send' ? gS.p.then(() => notDone) : undefined);
    el('builder-message-input').value = 'build it';
    builder.sendBuilderMessage();
    const first = status();
    gS.open(); await flush();
    result.builder_status_first = { first, after: status() };

    // an edit of the built FSM: with states the graph stays, with no states it is hidden, a non-object is refused
    reset();
    route.custom = undefined;
    builder.startBuilderSession(); await flush();
    const graph = el('builder-result-graph');
    const apply = async text => { builder.editBuilderResult(); el('builder-result-edit').value = text; builder.editBuilderResult(); await flush(); };
    const before = graph.style.display;
    await apply(JSON.stringify({ ...FSM, name: 'E' }));
    const withStates = graph.style.display;
    await apply(JSON.stringify({ name: 'NoStates', initial_state: 'a' }));
    const noStates = graph.style.display;
    await apply(JSON.stringify({ ...FSM, name: 'Again' }));      // states again: the hidden graph must come back
    const again = graph.style.display;
    await apply(JSON.stringify({ name: 'EmptyStates', initial_state: 'a', states: {} }));
    result.builder_edit_graph = { before, with_states: withStates, no_states: noStates, again, empty_states: graph.style.display };
    // a workflow artifact has no graph: an edit that happens to hold `states` does not show one
    reset();
    route.custom = url => (url === '/api/builder/start' ? { session_id: 's', response: 'r', internal_state: {}, is_complete: true, artifact: { name: 'W' }, artifact_json: '{"name":"W"}', artifact_type: 'workflow', is_valid: true } : undefined);
    builder.startBuilderSession(); await flush();
    await apply(JSON.stringify(FSM));
    result.builder_edit_workflow_graph = graph.style.display;
}

// --- graph rows as a running sum, the empty graph, CRLF markdown ---
{
    const W = 180, H = 60;
    const overlapsOf = ns => {
        const o = [];
        for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
            if (Math.abs(ns[i].x - ns[j].x) < W && Math.abs(ns[i].y - ns[j].y) < H) o.push([ns[i].id, ns[j].id]);
        }
        return o;
    };
    const draw = (id, graph) => { el(id).parentElement = null; renderGraph(id, graph); return el(id); };
    const hash = t => createHash('sha1').update(t).digest('hex');
    // a start node, `wide` children in layer 1, then a chain: layer l holds the node n<l> (and `extra` more at `tallLayer`)
    const chain = (wide, last, extra = 0, tallLayer = 0) => {
        const nodes = [{ id: 'a', is_initial: true }], edges = [], layer = { a: 0 };
        for (let i = 1; i <= wide; i++) { nodes.push({ id: 'b' + i }); edges.push({ from: 'a', to: 'b' + i }); layer['b' + i] = 1; }
        let prev = 'b1';
        for (let l = 2; l <= last; l++) {
            nodes.push({ id: 'n' + l }); edges.push({ from: prev, to: 'n' + l }); layer['n' + l] = l; prev = 'n' + l;
            if (l === tallLayer) for (let k = 0; k < extra; k++) { nodes.push({ id: `x${l}_${k}` }); edges.push({ from: 'n' + (l - 1), to: `x${l}_${k}` }); layer[`x${l}_${k}`] = l; }
        }
        return { nodes, edges, layer };
    };
    const rowTops = g => {
        const tops = [];
        for (const n of g.nodes) { const r = Math.floor(g.layer[n.id] / 5); tops[r] = Math.min(tops[r] ?? Infinity, n.y); }
        return tops;
    };
    // row 0 holds a layer of 5 nodes: row 1 starts below THAT height (5 * 160 + 60), row 2 below row 1 (1 * 160 + 60)
    const g1 = chain(5, 11);
    draw('g-rows', g1);
    const t1 = rowTops(g1);
    result.graph_rows = { overlaps: overlapsOf(g1.nodes), row1_minus_row0: t1[1] - t1[0], row2_minus_row1: t1[2] - t1[1] };
    // row 1 is the taller one (3 nodes in layer 6): row 1 still starts below row 0's own height (1 layer deep: 160 + 60)
    const g2 = chain(1, 11, 2, 6);
    draw('g-rows2', g2);
    const t2 = rowTops(g2);
    result.graph_rows_tall_second = { overlaps: overlapsOf(g2.nodes), row1_minus_row0: t2[1] - t2[0], row2_minus_row1: t2[2] - t2[1] };

    // a graph of one row, and one of two rows of equal height, keep every byte (the hashes were computed once for the layout of a one-row graph)
    const one = () => ({
        nodes: [{ id: 's', is_initial: true, label: 'S<b>' }, { id: 'x', description: 'a long description of the state' }, { id: 'y' }, { id: 'z', is_terminal: true }],
        edges: [{ from: 's', to: 'x', label: 'go' }, { from: 's', to: 'y' }, { from: 'x', to: 'x', label: 'loop' }, { from: 'x', to: 'z' }, { from: 'z', to: 'x' }, { from: 'y', to: 'z' }],
    });
    const two = chain(1, 6);   // 7 layers in two rows, every layer one node tall
    const sizedSvg = (id, g) => { const svg = el(id); svg.parentElement = { clientWidth: 800, clientHeight: 400 }; renderGraph(id, g); return svg; };
    const sOne = sizedSvg('g-one', one()), sTwo = sizedSvg('g-two', two);
    result.graph_unmoved = { one: [hash(sOne.innerHTML), sOne.attrs.viewBox], two: [hash(sTwo.innerHTML), sTwo.attrs.viewBox] };

    // no nodes: the svg is cleared and the viewBox is never written from infinite bounds
    const e1 = el('g-empty'); e1.innerHTML = '<g/>'; e1.parentElement = null;
    renderGraph('g-empty', { nodes: [], edges: [] });
    const e2 = el('g-empty2'); e2.parentElement = null;
    let threw = null;
    try { renderGraph('g-empty2', {}); } catch (e) { threw = String(e.message); }
    result.graph_empty = { html: e1.innerHTML, viewBox: e1.attrs.viewBox ?? null, bad: /Infinity|NaN/.test(JSON.stringify([e1.attrs, e2.attrs, e1.style, e2.style])), threw, no_nodes_key: e2.attrs.viewBox ?? null };
}
{
    const lf = 'Intro\n```js\nlet a = 1;\n```\nafter\n\n- a\n- b\n\n1. x\n2. y\n\none\n\ntwo';
    const crlf = lf.replace(/\n/g, '\r\n'), cr = lf.replace(/\n/g, '\r');
    const want = renderMarkdown(lf), got = renderMarkdown(crlf);
    result.markdown_crlf = {
        fence: got.includes('<pre class="md-code-block"><code>let a = 1;</code></pre>'),
        same_as_lf: got === want, cr_same_as_lf: renderMarkdown(cr) === want, stray_cr: got.includes('\r'),
        lists: [got.includes('<ul><li>a</li><li>b</li></ul>'), got.includes('<ol><li>x</li><li>y</li></ol>')],
        paragraphs: got.includes('one</p><p>two'),
    };
    // the escape still comes first: hostile text inside and around a CRLF fence stays text
    const bad = renderMarkdown('```\r\n<script>alert(1)</script>\r\n```\r\n<img src=x onerror=alert(1)>\r\n**<b>**');
    result.markdown_crlf_hostile = {
        raw: /<script|<img|<b>/.test(bad),
        fence: bad.includes('<pre class="md-code-block"><code>&lt;script&gt;alert(1)&lt;/script&gt;</code></pre>'),
        img: bad.includes('&lt;img src=x onerror=alert(1)&gt;'),
    };
}

// --- the empty dropdown option clears the view, and a late reply cannot paint over the clear ---
{
    const view = (svgId, infoId, rowsId) => ({ svg: el(svgId).innerHTML, info: el(infoId).innerHTML, rows: el(rowsId).innerHTML });
    const wfGates = {};
    const graphFor = id => ({ nodes: [{ id: 'a', is_initial: true }, { id: 'b' }], edges: [{ from: 'a', to: 'b', label: 'go' }], info: { name: 'WF ' + id } });
    reset();
    route.custom = async url => {
        const m = /workflow_id=(.+)$/.exec(url);
        if (m) { await wfGates[m[1]]?.p; return graphFor(m[1]); }
        if (url.startsWith('/api/agent/visualize')) return { ...graphFor('agent'), info: { name: 'AG<' } };
        return undefined;
    };
    const wf = () => view('viz-wf-svg', 'viz-wf-info', 'viz-wf-trans-body');
    const drawn = await visualizer.visualizeGraph('workflow', 'order_processing');
    const afterDraw = wf();
    const cleared = await visualizer.visualizeGraph('workflow', '');
    const afterClear = wf();
    result.viz_empty_workflow = {
        drawn, cleared,
        drew: afterDraw.svg.includes('<rect') && afterDraw.info.includes('WF order_processing') && afterDraw.rows.includes('<tr>'),
        svg: afterClear.svg, rows: afterClear.rows,
        info: afterClear.info,
    };
    const ag = () => view('viz-agent-svg', 'viz-agent-info', 'viz-agent-trans-body');
    await visualizer.visualizeGraph('agent', 'react');
    const agDrawn = ag().info.includes('AG&lt;');
    await visualizer.visualizeGraph('agent', '');
    result.viz_empty_agent = { drew: agDrawn, svg: ag().svg, rows: ag().rows, info: ag().info };

    // a reply that lands after the empty option was chosen does not bring the old graph back
    wfGates.slow = gate();
    const late = visualizer.visualizeGraph('workflow', 'slow');
    await visualizer.visualizeGraph('workflow', '');
    wfGates.slow.open();
    const lateRet = await late;
    result.viz_late_after_clear = { svg: wf().svg, rows: wf().rows, info: wf().info.includes('WF slow'), returned: lateRet };
    // A (slow) then B (fast): the final view is B
    wfGates.A = gate();
    const slowA = visualizer.visualizeGraph('workflow', 'A');
    await visualizer.visualizeGraph('workflow', 'B');
    wfGates.A.open(); await slowA;
    result.viz_late_after_newer = { info_is_b: wf().info.includes('WF B'), info_is_a: wf().info.includes('WF A') };
    // a superseded request that FAILS shows no toast (the view is another one now); the current one still does
    const graphRoute = route.custom;
    wfGates.bad = gate();
    route.custom = async url => {
        if (url.endsWith('workflow_id=bad')) { await wfGates.bad.p; return { fail: 'late failure' }; }
        if (url.endsWith('workflow_id=now')) return { fail: 'current failure' };
        return graphRoute(url);
    };
    const failToasts = text => toasts().filter(t => t.includes(text)).length;
    const slowBad = visualizer.visualizeGraph('workflow', 'bad');
    await visualizer.visualizeGraph('workflow', '');
    wfGates.bad.open();
    const lateFailed = await slowBad;
    const current = await visualizer.visualizeGraph('workflow', 'now');
    result.viz_late_failure = { returned: lateFailed, late_toasts: failToasts('late failure'), current_returned: current, current_toasts: failToasts('current failure') };
}
// --- Clear survives a pill toggle, a search and a page show; a record that arrives later shows ---
{
    const pill = lvl => { const p = new El('pill-' + lvl); p.attrs['data-level'] = lvl; p.classList.add('active'); return p; };
    const pills = [pill('INFO'), pill('ERROR')];
    const realAll = document.querySelectorAll;
    document.querySelectorAll = sel => (sel === '#log-pills .log-pill.active' ? pills.filter(p => p.classList.contains('active')) : sel === '#log-pills .log-pill' ? pills : []);
    const stream = el('log-stream');
    let markup = '';
    Object.defineProperty(stream, 'innerHTML', { get: () => markup, set: v => { markup = v; stream.html = []; } });
    stream.children = [];
    const visible = () => (markup + stream.html.join('')).split('class="entry').length - 1;
    const rec = (i, level = 'INFO') => ({ timestamp: `2026-11-27T10:00:0${i}Z`, level, module: 'm', line: i, message: 'rec' + i });
    let server = [rec(1), rec(2), rec(3, 'ERROR')];
    reset();
    // the server answers like the route: the records at or above `level`, the newest `limit` of them (oldest first here)
    const RANK = { TRACE: 5, DEBUG: 10, INFO: 20, ERROR: 40 };
    route.custom = url => {
        if (!url.startsWith('/api/logs')) return undefined;
        const q = new URL('http://x' + url).searchParams;
        return server.filter(r => RANK[r.level] >= RANK[q.get('level')]).slice(-Number(q.get('limit')));
    };
    state.currentPage = 'logs';
    const seen = {};
    await logs.refreshLogs();
    seen.refreshed = visible();
    logs.clearLogs();
    seen.cleared = visible();
    await logs.refreshLogs();
    seen.refresh_after_clear = visible();
    logs.toggleLogPill(pills[0]); await flush();               // INFO off
    seen.pill_off = visible();
    logs.toggleLogPill(pills[0]); await flush();               // INFO on again: the cleared records stay hidden
    seen.pill_on = visible();
    el('log-filter').value = 'rec';
    await logs.refreshLogs();
    seen.search = visible();
    el('log-filter').value = '';
    logs.onShowLogs(); await flush();
    seen.page_show = visible();
    await logs.syncLogs();
    seen.sync = visible();
    // a record that arrives after the Clear shows, by refresh and by append, and survives later rebuilds
    server = [...server, rec(4)];
    await logs.refreshLogs();
    seen.new_by_refresh = visible();
    logs.appendLogs([rec(5)]);
    seen.new_by_append = visible();
    server = [...server, rec(5)];
    logs.toggleLogPill(pills[0]); await flush(); logs.toggleLogPill(pills[0]); await flush();
    seen.new_after_toggles = visible();
    seen.pill_text = pills[0].textContent;
    // a second Clear hides these too, and a later record shows again
    logs.clearLogs();
    await logs.refreshLogs();
    seen.second_clear = visible();
    server = [...server, rec(6)];
    await logs.refreshLogs();
    seen.after_second_clear = visible();
    result.logs_clear = seen;
    // Clear holds across a pill that LOWERS the minimum level: the records below the level shown at Clear were never fetched
    const lr = (m, n, level) => ({ timestamp: `2026-11-27T10:0${m}:0${n}Z`, level, module: 'm', line: 10 * m + n, message: `lr${m}${n}` });
    const debugPill = pill('DEBUG');
    debugPill.classList.remove('active');
    pills.push(debugPill);                                      // DEBUG off, INFO and ERROR on
    const lv = {};
    server = [lr(1, 1, 'INFO'), lr(1, 2, 'ERROR'), lr(1, 3, 'DEBUG')];  // the newest record is a DEBUG one
    await logs.refreshLogs();
    lv.shown_before = visible();                                // the INFO and the ERROR
    const cursorCalls = () => calls.filter(c => c === 'GET /api/logs?limit=1&level=TRACE').length;
    const cursorBefore = cursorCalls();
    logs.clearLogs(); await flush();                            // the one request that learns the newest record of any level
    lv.cursor_request = cursorCalls() - cursorBefore;
    await logs.refreshLogs();
    lv.after_clear = visible();
    logs.toggleLogPill(debugPill); await flush();               // Debug on: the old DEBUG record stays hidden
    lv.debug_on = visible();
    server = [...server, lr(1, 4, 'DEBUG')];
    await logs.refreshLogs();
    lv.new_debug = visible();                                   // a record after the Clear shows at any level
    server = [...server, lr(1, 5, 'INFO')];
    await logs.refreshLogs();
    lv.new_info = visible();
    logs.toggleLogPill(debugPill); await flush();               // Debug off again
    // INFO off at Clear time (the newest record is an INFO one), then on again
    server = [lr(2, 1, 'ERROR'), lr(2, 2, 'INFO')];
    logs.toggleLogPill(pills[0]); await flush();                // INFO off (ERROR left)
    lv.info_off_shown = visible();
    logs.clearLogs(); await flush();
    logs.toggleLogPill(pills[0]); await flush();                // INFO on: the old INFO record stays hidden
    lv.info_on = visible();
    server = [...server, lr(2, 3, 'INFO')];
    await logs.refreshLogs();
    lv.info_new = visible();
    // the reply of an earlier Clear arrives after a later one: the cursor never moves back
    const real = route.custom;
    let release;
    const held = new Promise(r => { release = r; });
    let asked = 0;
    route.custom = async url => {
        if (!url.includes('limit=1&')) return real(url);
        const answer = real(url);                                // the snapshot of the moment of the request
        if (asked++ === 0) await held;
        return answer;
    };
    logs.clearLogs();                                           // its reply is held
    server = [...server, lr(3, 1, 'INFO'), lr(3, 2, 'INFO')];
    logs.clearLogs(); await flush();                            // answers at once with the newer record
    release(); await flush();
    await logs.refreshLogs();
    lv.late_reply = visible();
    // the cursor request fails: the keys of the records seen still hold
    route.custom = url => (url.includes('limit=1&') ? { fail: 'down' } : real(url));
    server = [...server, lr(4, 1, 'INFO')];
    await logs.refreshLogs();                                   // seen now
    logs.clearLogs(); await flush();
    await logs.refreshLogs();
    lv.cursor_failed = visible();
    route.custom = real;
    result.logs_clear_levels = lv;
    // Clear waits for its cursor (B5a): a pill toggled while the cursor reply is held rebuilds nothing until it lands
    const hr = (n, level) => ({ timestamp: `2026-11-27T11:00:0${n}Z`, level, module: 'hm', line: n, message: `hold${n}` });
    server = [hr(1, 'INFO'), hr(2, 'DEBUG')];                   // DEBUG is off: that record was never fetched
    await logs.refreshLogs();
    const hold = { shown_before: visible() };
    let landed;
    const cursorHeld = new Promise(r => { landed = r; });
    route.custom = async url => {
        if (!url.includes('limit=1&')) return real(url);
        const answer = real(url);
        await cursorHeld;
        return answer;
    };
    logs.clearLogs();
    logs.toggleLogPill(debugPill); await flush();               // Debug on, the cursor reply still held
    hold.during = visible();
    landed(); await flush();
    hold.after_landing = visible();
    logs.toggleLogPill(debugPill); await flush();               // Debug off again
    route.custom = real;
    result.logs_clear_pending = hold;
    // A cursor request that never answers does not freeze later refreshes (review F4): the wait is a bounded timer
    // (the fake scheduler here, never a real sleep); once it fires the refresh proceeds and fetches.
    const hungSeen = [];
    route.custom = async url => { hungSeen.push(url); return url.includes('limit=1&') ? new Promise(() => {}) : real(url); };
    const timersBefore = timers.length;
    logs.clearLogs();
    let settled = false;
    logs.refreshLogs().then(() => { settled = true; });
    await flush();
    const wait = timers.slice(timersBefore).filter(t => t.live && t.ms >= 1000 && t.ms <= 10000);
    const bounded = { timers: wait.length, ms_in_range: wait.length === 1, settled_before_timer: settled, requests_before_timer: hungSeen.filter(u => !u.includes('limit=1&')).length };
    for (const t of wait) { t.live = false; t.fn(); }
    await flush();                                             // never await the refresh itself: on a tree with no bound it hangs
    bounded.settled_after_timer = settled;
    bounded.requests_after_timer = hungSeen.filter(u => !u.includes('limit=1&')).length;
    route.custom = real;
    logs.clearLogs(); await flush();                            // a Clear that lands replaces the hung cursor request
    result.logs_clear_hung_cursor = bounded;
    // The search matches what the entry shows: module:line, [conversation id] and message, on the refresh and the append path
    const sr = (n, extra = {}) => ({ timestamp: `2026-11-27T12:00:0${n}Z`, level: 'INFO', module: 'mod.x', line: n, message: `srec${n}`, ...extra });
    server = [sr(1, { conversation_id: 'conv-abc-123' }), sr(2, { module: 'other.y' }), sr(3, { message: 'has a(b inside' })];
    const search = {};
    const find = async text => { el('log-filter').value = text; await logs.refreshLogs(); return visible(); };
    search.module = await find('mod.x');
    search.module_highlight = markup.includes('<span class="search-highlight">mod.x</span>:1');
    search.conversation = await find('conv-abc');
    search.conversation_upper = await find('CONV-ABC');
    search.conversation_highlight = markup.includes('<span class="search-highlight">conv-abc</span>-123]');
    search.message = await find('srec2');
    search.line_and_message = await find('other.y:2 srec2');
    search.no_match = await find('nothing-here');
    search.regex_chars = await find('a(b');
    search.regex_highlight = markup.includes('<span class="search-highlight">a(b</span>');
    search.regex_toasts = toasts().filter(t => t.includes('Failed to load logs')).length;
    el('log-filter').value = 'conv-xyz';
    await logs.refreshLogs();
    search.append_before = visible();
    logs.appendLogs([sr(4, { conversation_id: 'conv-xyz-9' }), sr(5)]);
    search.append_by_conversation = visible();
    logs.appendLogs([sr(6, { module: 'conv-xyz.mod' })]);
    search.append_by_module = visible();
    el('log-filter').value = '';
    result.logs_search = search;
    document.querySelectorAll = realAll;
}
// --- opening a conversation straight in the drawer makes no request for the hidden detail panel ---
{
    const urls = [];
    const convOk = { conversation_id: 'c', instance_id: 'I', current_state: 's', is_terminal: false, message_history: [], context_data: {} };
    const run = async (before, act) => {
        reset();
        urls.length = 0;
        route.custom = url => { urls.push(url); return url.startsWith('/api/conversations/') ? convOk : []; };
        state.instances = [{ instance_id: 'I', instance_type: 'fsm', label: 'L', status: 'running' }];
        state.currentPage = 'control';
        state.selectedConvId = null;
        Object.assign(state, before);
        await act();
        await flush();
        const census = [...urls];
        const chosen = { id: state.selectedDetailId, type: state.selectedDetailType, conv: state.selectedConvId, polling: !!state.detailPollTimer };
        clearInterval(state.detailPollTimer); state.detailPollTimer = null;
        return { census, chosen };
    };
    conv.setDeps({ showPage() {}, openDrawer: control.openDrawer, refreshDetailPanel: control.refreshDetailPanel });
    const direct = await run({ selectedDetailId: null, selectedDetailType: null }, () => conv.showConversationInDrawer('I', 'c'));
    result.drawer_direct = direct;
    // Back shows the instance: it asks for the detail then
    const back = await run({ selectedDetailId: null, selectedDetailType: null }, async () => { conv.showConversationInDrawer('I', 'c'); await flush(); urls.length = 0; conv.drawerBack(); });
    result.drawer_back = { census: back.census, chosen: back.chosen };
    // the plain open (a row click) still fetches, and so does an instance already selected for another conversation
    result.drawer_plain_open = (await run({ selectedDetailId: null, selectedDetailType: null }, () => control.openDrawer('I', 'fsm'))).census;
    result.drawer_already_selected = (await run({ selectedDetailId: 'I', selectedDetailType: 'fsm' }, () => conv.showConversationInDrawer('I', 'c'))).census;
}
out(result);
process.exit(0);
