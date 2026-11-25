import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as launch from '../static/pages/launch.js';
import * as visualizer from '../static/pages/visualizer.js';

// An empty payload and a failed fetch never crash and never leave a hardcoded option behind.
let mode = 'empty';
let typeCalls = 0;
serve(async (url, opts) => {
    if (url === '/api/agent/types') {
        typeCalls++;
        if (mode === 'fail') return { fail: 'down' };
        if (mode === 'junk') return { agent_types: 'no', other: 1 };
        if (mode === 'badrows') return { agent_types: [null, 5, { name: 'x' }, { id: 7 }, 'react'] };
        if (mode === 'rows') return { agent_types: [{ id: 'react', name: 'react', launchable: true, needs_tools: false, takes_tools: false, tool_names: [], drawable: true }] };
        return { agent_types: [] };
    }
    if (url === '/api/presets') return { fsm: [] };
    if (url === '/api/workflow/presets') return { workflows: [] };
    if (url === '/api/capabilities') return { fsm: true, workflows: true, agents: true };
    if (opts?.method === 'POST') return { instance_id: 'x' };
    return { nodes: [], edges: [] };
});
state.capabilities = { fsm: true, workflows: true, agents: true };
launch.setDeps({ showPage() {}, refreshInstances() {} });
const sel = el('launch-agent-type'), vsel = el('viz-agent-type');
const shape = s => (s.children || []).map(c => (c.children ? { group: c.label, options: c.children.map(o => [o.value, o.textContent]) } : [c.value, c.textContent, c.disabled]));
const snap = () => ({ launch: shape(sel), viz: shape(vsel), launchValue: sel.value, vizValue: vsel.value, toast: (document.body.children || []).map(c => c.textContent) });

const result = {};
for (const m of ['empty', 'junk', 'badrows', 'fail']) {
    mode = m; state.agentTypes = null;
    launch.showLaunchModal(); await visualizer.showAgentGraph(); await sleep(30);
    result[m] = snap();
}
// a failed load caches nothing: the next opening asks again and fills the select
const failedCalls = typeCalls;
mode = 'rows';
launch.showLaunchModal(); await visualizer.showAgentGraph(); await sleep(30);
result.recovered = { ...snap(), asked: typeCalls - failedCalls };
// no type to launch: an error line and no request
mode = 'empty'; state.agentTypes = null;
launch.showLaunchModal(); await visualizer.showAgentGraph(); await sleep(30);
let posted = 0;
const first = globalThis.fetch;
globalThis.fetch = async (u, o) => { if (o?.method === 'POST') posted++; return first(u, o); };
el('launch-agent-task').value = 't';
await launch.doLaunchAgent({ disabled: false, textContent: 'Launch Agent' });
result.nothingToLaunch = { posted, status: el('launch-agent-status').innerHTML, ...snap() };
out(result);
process.exit(0);
