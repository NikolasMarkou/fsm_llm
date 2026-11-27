import { el, serve, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as launch from '../static/pages/launch.js';
import * as builder from '../static/pages/builder.js';
import * as conv from '../static/pages/conversations.js';
import * as eventlog from '../static/pages/eventlog.js';
import * as control from '../static/pages/control.js';

// Plan 1f0925d2: a view or a timer must not act on a request, a modal or a state that is no longer the
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

// --- F1: the capabilities fetch fails (the server decides; the failure is shown once) ---
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

// --- F1: the launch created the instance, then the first start failed ---
const status = id => el(id).innerHTML;
reset(); pages.length = 0; drawer.length = 0;
route.fail['/api/fsm/i1/start'] = 'LLM operation timed out';
{
    const btn = { disabled: false, textContent: 'Launch FSM' };
    await launch.doLaunchFSM(btn);
    const after = { status: status('launch-fsm-status'), disabled: btn.disabled, pages: [...pages] };
    const ran = fire();
    result.fsm_partial = { after, ran, pages: [...pages], drawer: drawer.length, disabled: btn.disabled, launches: count('/api/fsm/launch'), modal: el('launch-modal').style.display };
}
reset(); pages.length = 0; drawer.length = 0;
route.fail['/api/fsm/launch'] = 'bad preset';
{
    const btn = { disabled: false, textContent: 'Launch FSM' };
    await launch.doLaunchFSM(btn);
    result.fsm_first_failed = { status: status('launch-fsm-status'), disabled: btn.disabled, timers: timers.filter(t => t.live).length, pages: [...pages], posts: calls.filter(c => c.startsWith('POST')).length };
}

// --- F1: the modal closed inside the window (or while the start was in flight): no navigation ---
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

// --- F1: the builder FSM launch ---
async function builderReady() {
    builder.startBuilderSession();
    await flush();
}
const bbtn = el('builder-launch-btn');
reset(); pages.length = 0;
await builderReady();
reset();
route.fail['/api/fsm/i1/start'] = 'LLM operation timed out';
builder.launchBuilderResult();
await flush();
result.builder_partial = { status: status('builder-status'), disabled: bbtn.disabled, pages_before: [...pages] };
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

// --- F2: a failing conversation detail load that was superseded does not touch the view ---
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
// --- F2: the input is focused only while the same conversation is still shown and the input is visible ---
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

// --- F2: the event log: Load more by since_seq, a request token, an error keeps the earlier pages ---
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

// --- F3 (D-009): the workflow step data in the control drawer, and the <details> open state across the 2 s repaint ---
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

// --- F3 (D-009): the builder input area waits for the start reply; an edit with no states hides the old graph; the first status paint ---
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
out(result);
process.exit(0);
