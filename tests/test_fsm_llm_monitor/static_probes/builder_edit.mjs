import { el, serve, sleep, out } from './stub.mjs';
import * as builder from '../static/pages/builder.js';
// 13.6: Edit JSON then Apply, then Launch sends the edited artifact; a parse error is shown and nothing is sent.
const fsm = { name: 'Built', initial_state: 'a', states: { a: { id: 'a', description: 'd', purpose: 'p' } } };
const posts = [];
serve((url, opts) => {
    if (url === '/api/fsm/launch') { posts.push(JSON.parse(opts.body)); return { instance_id: 'i1' }; }
    if (url === '/api/fsm/i1/start') return {};
    if (url.startsWith('/api/fsm/visualize')) return { nodes: [], edges: [] };
    return { session_id: 's', response: 'hi', internal_state: {}, is_complete: true, artifact: fsm, artifact_json: JSON.stringify(fsm), artifact_type: 'fsm', is_valid: true };
});
builder.setDeps({ showPage() {}, refreshInstances() {} });
builder.startBuilderSession(); await sleep(30);
const area = el('builder-edit-btn') && el('builder-result-edit'), btn = el('builder-edit-btn'), pre = el('builder-result-json'), status = () => el('builder-status').innerHTML;
const before = pre.textContent;
builder.editBuilderResult();
const opened = { label: btn.textContent, editor: area.style.display, pre: pre.style.display, text: JSON.parse(area.value).name };
area.value = '{ not json';
builder.editBuilderResult();
const bad = { status: status(), stillEditing: area.style.display, preUnchanged: pre.textContent === before };
builder.launchBuilderResult(); await sleep(20);                       // the editor is open: no launch
const openLaunch = { posts: posts.length, status: status() };
area.value = '[1, 2]';
builder.editBuilderResult();
const array = { status: status(), stillEditing: area.style.display };
el('builder-launch-btn').textContent = 'stale';
area.value = JSON.stringify({ ...fsm, name: 'Edited' });
builder.editBuilderResult();
const applied = { label: btn.textContent, editor: area.style.display, pre: JSON.parse(pre.textContent).name, launchLabel: el('builder-launch-btn').textContent, summary: el('builder-result-summary').innerHTML.includes('Edited') };
builder.launchBuilderResult(); await sleep(30);
out({ opened, bad, openLaunch, array, applied, launched: posts.map(p => p.fsm_json.name), label: posts[0]?.label });
