import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as launch from '../static/pages/launch.js';
import * as visualizer from '../static/pages/visualizer.js';

// The two agent selects are filled from GET /api/agent/types and nothing else (plan 659c07c3 step 6).
const HOSTILE = '<img src=x onerror=alert(1)>';
const row = (id, o = {}) => ({ id, name: id, launchable: true, needs_tools: false, takes_tools: false, tool_names: [], drawable: true, ...o });
const rows = [
    row('debate'),
    row('auto_memory', { needs_tools: true, takes_tools: true, tool_names: ['search'] }),
    row('react', { needs_tools: true, takes_tools: true }),
    row('adapt', { takes_tools: true }),
    row(HOSTILE),
    row('swarm', { launchable: false, drawable: false }),
    row('evaluator_optimizer', { launchable: false }),
];
const urls = [];
const posts = [];
serve(async (url, opts) => {
    urls.push(url);
    if (url === '/api/agent/types') return { agent_types: rows };
    if (url === '/api/presets') return { fsm: [] };
    if (url === '/api/workflow/presets') return { workflows: [] };
    if (url === '/api/capabilities') return { fsm: true, workflows: true, agents: true };
    if (opts?.method === 'POST') { posts.push(JSON.parse(opts.body)); return { instance_id: 'x', label: 'L' }; }
    return { fsm: { name: 'N', state_count: 1 }, nodes: [], edges: [] };
});
state.capabilities = { fsm: true, workflows: true, agents: true };
launch.setDeps({ showPage() {}, refreshInstances() {} });

const shape = sel => (sel.children || []).map(c => (c.children
    ? { group: c.label, options: c.children.map(o => [o.value, o.textContent]) }
    : [c.value, c.textContent]));
const wrapper = el('launch-agent-tools-wrapper');
launch.showLaunchModal();
await sleep(30);
const sel = el('launch-agent-type');
const launchSelect = { shape: shape(sel), value: sel.value, innerHTML: sel.innerHTML, wrapper: wrapper.style.display };
sel.value = 'debate'; launch.onAgentTypeChange();
const noTools = wrapper.style.display;
sel.value = 'adapt'; launch.onAgentTypeChange();
const optionalTools = wrapper.style.display;
launch.showLaunchModal(); await sleep(30);          // opened again: no second fetch, the choice is kept
const reopened = { value: sel.value, fetches: urls.filter(u => u === '/api/agent/types').length };

el('launch-agent-task').value = 'task';
sel.value = 'react';
launch.onAgentTypeChange();
await launch.doLaunchAgent({ disabled: false, textContent: 'Launch Agent' });
const needsOne = { posts: posts.length, status: el('launch-agent-status').innerHTML };
sel.value = 'auto_memory';                           // a file that names its tools: the server stubs them
await launch.doLaunchAgent({ disabled: false, textContent: 'Launch Agent' });
const namedTools = { posts: posts.length, body: posts[0] };

const vsel = el('viz-agent-type');
await visualizer.showAgentGraph();
await sleep(30);
const viz = { shape: shape(vsel), value: vsel.value, innerHTML: vsel.innerHTML, drawn: urls.filter(u => u.startsWith('/api/agent/visualize')) };
out({ launchSelect, noTools, optionalTools, reopened, needsOne, namedTools, viz });
process.exit(0);
