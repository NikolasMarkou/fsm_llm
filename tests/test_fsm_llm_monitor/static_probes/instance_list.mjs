import { createHash } from 'node:crypto';
import { el, serve, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as dash from '../static/pages/dashboard.js';
import * as control from '../static/pages/control.js';
import * as ws from '../static/services/ws.js';

// The instance list (plan 9283c1fd step 5): the "Loading instances..." spinner never outlives its
// request, a superseded reply paints and toasts nothing, the counts of an instance refresh the grid,
// the table and the WebSocket gate, and a finished agent's drawer has no Execution Trace block.
// No sleep decides anything: the replies are gates and the timers are inert.
globalThis.setTimeout = () => 1;
globalThis.clearTimeout = () => {};
console.error = () => {};
const flush = async () => { for (let i = 0; i < 12; i++) await new Promise(r => setImmediate(r)); };
const gate = () => { let open; const p = new Promise(r => { open = r; }); return { p, open }; };
const toasts = () => (document.body.children || []).map(c => c.textContent);
const clearToasts = () => { document.body.children = []; };
const grid = () => el('instances-grid');
const SPINNER = 'loading-spinner';

const inst = (id, extra = {}) => ({ instance_id: id, instance_type: 'fsm', label: id, status: 'running', ...extra });
const route = { body: [], gates: [] };   // route.gates: replies waiting, in request order
serve(async url => {
    if (url !== '/api/instances') return [];
    const mine = route.gates.shift();
    if (mine) await mine.p;
    return typeof route.body === 'function' ? route.body() : route.body;
});
const r = {};

// 1. a first fetch that fails leaves no spinner behind and says so once
route.body = () => ({ fail: 'down' });
await dash.refreshInstances();
r.failed_first = { grid: grid().innerHTML, spinner: grid().innerHTML.includes(SPINNER), toasts: toasts() };
clearToasts();

// 2. an empty monitor polled twice: the second poll must not leave the spinner (the early return of the hash)
route.body = [];
await dash.refreshInstances();
r.empty_first = grid().innerHTML;
await dash.refreshInstances();
r.empty_second = grid().innerHTML;
r.empty_hint = el('instances-empty').style.display;

// 3. an older, slower reply that lands after a newer one changes nothing
state.instances = [];
grid().innerHTML = '';
let older = gate();
route.gates = [older];                    // the first request waits, the second does not
route.body = () => [inst('new')];
const p1 = dash.refreshInstances();
await dash.refreshInstances();
route.body = [inst('old')];               // what the older reply will carry when it is let go
older.open(); await p1;
r.late_ok = { ids: state.instances.map(i => i.instance_id), grid_old: grid().innerHTML.includes('old'), grid_new: grid().innerHTML.includes('new'), spinner: grid().innerHTML.includes(SPINNER) };

// 3b. an older reply that FAILS after a newer one succeeded: no toast, the newer list stands
clearToasts();
older = gate();
route.gates = [older];
route.body = () => [inst('new2')];
const p2 = dash.refreshInstances();
await dash.refreshInstances();
route.body = () => ({ fail: 'late failure' });
older.open(); await p2;
r.late_fail = { ids: state.instances.map(i => i.instance_id), toasts: toasts(), spinner: grid().innerHTML.includes(SPINNER) };

// 3c. the latest request is the one that clears the spinner: the older one finishing last does not leave it
state.instances = [];
grid().innerHTML = '';
clearToasts();
older = gate();
route.gates = [older];
route.body = [];
const p3 = dash.refreshInstances();       // writes the spinner, waits
await dash.refreshInstances();            // the latest: empty list
older.open(); await p3;
r.latest_clears = { grid: grid().innerHTML, toasts: toasts() };

// 3d. a superseded reply that lands while the latest request is still out leaves the spinner to it
state.instances = [];
grid().innerHTML = '';
clearToasts();
const g1 = gate(), g2 = gate();
route.gates = [g1, g2];
route.body = [];
const q1 = dash.refreshInstances();
const q2 = dash.refreshInstances();
g1.open(); await q1; await flush();
const whileLatestOut = grid().innerHTML.includes(SPINNER);
g2.open(); await q2;
r.superseded_keeps_spinner = { whileLatestOut, afterLatest: grid().innerHTML };

// 4. a change of conversation_count or active_workflows alone refreshes the grid and the table
const shown = () => ({
    grid: grid().innerHTML.match(/\d+ (?:conversations?|active)/g),
    table: el('ctrl-unified-body').innerHTML.match(/\d+ (?:convs?|active)/g),
});
state.instances = [];
grid().innerHTML = '';
route.gates = [];
route.body = () => [inst('f1', { conversation_count: 0 }), inst('w1', { instance_type: 'workflow', active_workflows: 0 })];
await dash.refreshInstances();
control.renderUnifiedTable();
r.counts_before = shown();
route.body = () => [inst('f1', { conversation_count: 3 }), inst('w1', { instance_type: 'workflow', active_workflows: 0 })];
await dash.refreshInstances();
control.renderUnifiedTable();
r.conversation_count_change = shown();
route.body = () => [inst('f1', { conversation_count: 3 }), inst('w1', { instance_type: 'workflow', active_workflows: 2 })];
await dash.refreshInstances();
control.renderUnifiedTable();
r.active_workflows_change = shown();

// 5. the WebSocket gate lets a count-only change through
const socks = [];
globalThis.WebSocket = class { static OPEN = 1; static CONNECTING = 0; constructor() { this.readyState = 0; socks.push(this); } send() {} close() {} };
let paints = 0;
ws.registerHandlers({ renderInstanceGrid: () => { paints++; } });
ws.connectWS();
const push = instances => socks[0].onmessage({ data: JSON.stringify({ instances }) });
push([inst('f1', { conversation_count: 0 })]);
const afterFirst = paints;
push([inst('f1', { conversation_count: 0 })]);
const afterSame = paints;
push([inst('f1', { conversation_count: 3 })]);
const afterCount = paints;
push([inst('w1', { instance_type: 'workflow', active_workflows: 0 })]);
push([inst('w1', { instance_type: 'workflow', active_workflows: 2 })]);
r.ws_gate = { paints: [afterFirst, afterSame, afterCount, paints], count: state.instances[0].active_workflows };

// 6. the agent drawer of a finished run: the tool-call fallback, and no Execution Trace block
const agentRun = { result: { status: 'completed', success: true, answer: 'done' } };
let resultFetches = 0;
serve(async url => {
    if (url.endsWith('/status')) return { status: 'completed', agent_type: 'react', task: 't', success: true, total_iterations: 1, tools_used: [{ tool_name: 'lookup', parameters: { q: '<b>x</b>' } }], conversation_log: [] };
    if (url.endsWith('/result')) { resultFetches++; return agentRun.result; }
    if (url.includes('/events')) return [];
    return [];
});
state.instances = [{ instance_id: 'G', instance_type: 'agent', label: 'G' }];
state.selectedDetailId = 'G'; state.selectedDetailType = 'agent';
const drawerHtml = async () => {
    el('ctrl-drawer-content').innerHTML = '';
    control.closeDrawer(); state.selectedDetailId = 'G'; state.selectedDetailType = 'agent';
    await control.refreshDetailPanel('G', 'agent'); await flush();
    return el('ctrl-drawer-content').innerHTML;
};
const plain = await drawerHtml();
agentRun.result = { status: 'completed', success: true, answer: 'done', trace_steps: [{ state: 'think', reasoning: 'r', tool_name: 'lookup' }] };
const withSteps = await drawerHtml();
r.agent_drawer = {
    tool_calls: plain.includes('Tool Calls (1)'),
    escaped: plain.includes('&lt;b&gt;x&lt;/b&gt;') && !plain.includes('<b>x</b>'),
    html_sha: createHash('sha256').update(plain).digest('hex').slice(0, 16),
    same_with_steps_in_result: withSteps === plain,
    result_fetches: resultFetches,
    trace_block: /Execution Trace|toggle-trace-step|expand-all-trace/.test(withSteps),
};
out(r);
