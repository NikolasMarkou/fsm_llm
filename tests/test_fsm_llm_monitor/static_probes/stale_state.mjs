import { el, serve, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as launch from '../static/pages/launch.js';
import * as builder from '../static/pages/builder.js';

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
const route = { fail: {}, gates: {} };           // route[url] -> fail text, or a gate the reply waits on
serve(async (url, opts) => {
    calls.push(`${opts?.method || 'GET'} ${url}`);
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
const reset = () => { calls.length = 0; route.fail = {}; route.gates = {}; timers.length = 0; };

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

out(result);
process.exit(0);
