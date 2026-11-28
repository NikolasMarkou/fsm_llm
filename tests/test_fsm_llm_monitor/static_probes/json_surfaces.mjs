// JSON colouring on every surface (plan 51fde95b D-040): each page marks its JSON `json-view` (text escaped), plain text stays plain,
// and the colouring pass leaves exactly the page's text. The stub cannot parse HTML, so `wire` finds the marked elements in innerHTML.
import { El, el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as eventlog from '../static/pages/eventlog.js';
import * as conv from '../static/pages/conversations.js';
import * as builder from '../static/pages/builder.js';
import * as control from '../static/pages/control.js';

const decode = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
/** Make `box.querySelectorAll` return elements for the `.json-view` markup of its innerHTML; `seen` keeps the last ones. */
function wire(box) {
    const seen = [];
    box.querySelectorAll = sel => {
        if (sel !== '.json-view') return [];
        seen.length = 0;
        for (const m of box.innerHTML.matchAll(/<(pre|span) class="json-view">([\s\S]*?)<\/\1>/g)) Object.assign(seen[seen.push(new El('j')) - 1], { textContent: decode(m[2]) });
        return seen;
    };
    return seen;
}
// [text, coloured] of each marked element, after the page ran its colouring pass
const view = seen => seen.map(e => [e.textContent, (e.children ?? []).length > 0 && e.children.some(c => c.className === 'json-key')]);
const rawTags = html => ['<script', '<i>x', '<b>d'].some(t => html.includes(t));

// Event log: the fields cell
serve(() => ({ conversation_id: 'c', next_seq: null, usage: { calls: 0, total_tokens: 0, errors: 0 }, events: [{ seq: 1, ts: 't', type: 'x', frame: 0, state: '<script>s</script>', n: 1 }, { seq: 2, ts: 't', type: 'y', frame: 0 }] }));
const evSeen = wire(el('eventlog-view'));
await eventlog.openEventLog('c');
const eventlogRow = { views: view(evSeen), raw: rawTags(el('eventlog-view').innerHTML) };

// Conversation drawer: an object context value and the llm data panels are JSON, text and numbers are not
conv.setDeps({});
state.instances = [{ instance_id: 'I', instance_type: 'fsm', label: 'L' }];
state.currentPage = 'control';
serve(() => ({ conversation_id: 'A', instance_id: 'I', current_state: 's', is_terminal: false, message_history: [], context_data: { name: 'plain', n: 5, obj: { a: [1, '<i>x</i>'] } }, last_extraction: { fields: { q: 1 }, note: 'text' } }));
const convSeen = wire(el('conv-detail'));
conv.showConversationInDrawer('I', 'A'); await sleep(20);
const conversationRow = { views: view(convSeen), raw: rawTags(el('conv-detail').innerHTML) };

// Builder: the result text and the internal state panel
const fsm = { name: 'Built', initial_state: 'a', states: { a: { id: 'a', description: '<b>d</b>', purpose: 'p' } } };
serve(url => {
    if (url.startsWith('/api/fsm/visualize')) return { nodes: [], edges: [] };
    return { session_id: 's', response: 'hi', internal_state: { artifact_preview: { x: 1 }, builder_summary: 'plain summary' }, is_complete: true, artifact: fsm, artifact_json: JSON.stringify(fsm, null, 2), artifact_type: 'fsm', is_valid: true };
});
builder.setDeps({ showPage() {}, refreshInstances() {} });
const stateSeen = wire(el('builder-state-content'));
builder.startBuilderSession(); await sleep(30);
const resultPre = el('builder-result-json');
const builderRow = {
    result: [resultPre.textContent === JSON.stringify(fsm, null, 2), (resultPre.children ?? []).some(c => c.className === 'json-key')],
    state: view(stateSeen), stateRaw: rawTags(el('builder-state-content').innerHTML),
};

// Builder again: the raw state fallback, the edited result after Apply, and a failed build's `{}`
serve(url => url.startsWith('/api/fsm/visualize') ? { nodes: [], edges: [] } : ({ session_id: 's', response: 'hi', internal_state: { note: { k: [1] } }, is_complete: true, artifact: fsm, artifact_json: JSON.stringify(fsm), artifact_type: 'fsm', is_valid: true }));
builder.startBuilderSession(); await sleep(30);
const rawState = view(stateSeen);
builder.editBuilderResult();
const edited = { ...fsm, name: 'Edited' };
el('builder-result-edit').value = JSON.stringify(edited);
builder.editBuilderResult();
const nodeText = pre => (pre.children ?? []).map(c => c.textContent).join('');
const applied = [resultPre.textContent === JSON.stringify(edited, null, 2), nodeText(resultPre) === JSON.stringify(edited, null, 2)];
serve(() => ({ session_id: 's', response: 'hi', internal_state: {}, is_complete: true, error: 'no', artifact_json: '{}' }));
builder.startBuilderSession(); await sleep(30);
const failedBuild = [resultPre.textContent, (resultPre.children ?? []).map(c => c.className)];

// Builder: a key the server never sends (`requirements`) is shown by the Raw block, as JSON (plan c9d908d4 step 2, C3)
serve(url => url.startsWith('/api/fsm/visualize') ? { nodes: [], edges: [] } : ({ session_id: 's', response: 'hi', internal_state: { phase: 'x', requirements: { n: { a: 1 } } }, is_complete: true, artifact: fsm, artifact_json: JSON.stringify(fsm), artifact_type: 'fsm', is_valid: true }));
builder.startBuilderSession(); await sleep(30);
const requirementsState = view(stateSeen);
const requirementsHtml = el('builder-state-content').innerHTML;

// Control drawer: workflow context and step data, agent tool-call parameters and context
const workflow = { workflow_instance_id: 'w1', status: 'running', current_step: 's', context: { cfg: { a: 1 }, s: 'x' }, history: [{ step_id: 's', data: { k: 1 } }] };
serve(url => {
    if (url.endsWith('/instances')) return [workflow];
    if (url.includes('/events')) return [];
    if (url.endsWith('/status')) return { status: 'running', agent_type: 'T', tools_used: [{ tool_name: 't', parameters: { q: '<script>x</script>' } }], conversation_log: [{ type: 'context', state: 'act', data: { tool_name: 'm', tool_input: { a: 1 }, note: 'plain' } }] };
    return { instance_id: 'W', status: 'running', label: 'W' };
});
state.instances = [{ instance_id: 'W', instance_type: 'workflow', label: 'W' }, { instance_id: 'G', instance_type: 'agent', label: 'G' }];
const drawer = el('ctrl-drawer-content');
const ctrlSeen = wire(drawer);
const control_ = {};
for (const [id, type] of [['W', 'workflow'], ['G', 'agent']]) {
    state.selectedDetailId = id; state.selectedDetailType = type;
    await control.refreshDetailPanel(id, type); await sleep(20);
    control_[type] = { views: view(ctrlSeen), raw: rawTags(drawer.innerHTML) };
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const texts = v => v.views.map(([t]) => t), coloured = v => v.views.every(([, c]) => c);
const w = control_.workflow, a = control_.agent;
out({
    eventlog: same(texts(eventlogRow), ['{"state":"<script>s</script>","n":1}']) && coloured(eventlogRow) && !eventlogRow.raw,
    conversation: same(texts(conversationRow), ['{"a":[1,"<i>x</i>"]}', '{\n  "q": 1\n}']) && coloured(conversationRow) && !conversationRow.raw,
    builderResult: builderRow.result.every(Boolean),
    builderRawState: same(texts({ views: rawState }), ['{\n  "note": {\n    "k": [\n      1\n    ]\n  }\n}']) && coloured({ views: rawState }),
    builderUnknownKeyIsRaw: same(texts({ views: requirementsState }), ['{\n  "phase": "x",\n  "requirements": {\n    "n": {\n      "a": 1\n    }\n  }\n}']) && coloured({ views: requirementsState }) && !requirementsHtml.includes('[object Object]') && requirementsHtml.includes('<h4>Raw</h4>') && !requirementsHtml.includes('<h4>Requirements</h4>'),
    builderApplied: applied.every(Boolean),
    builderFailed: same(failedBuild, ['{}', ['json-punct']]),
    builderState: same(texts({ views: builderRow.state }), ['{\n  "x": 1\n}']) && coloured({ views: builderRow.state }) && !builderRow.stateRaw,
    workflowDrawer: same(texts(w), ['{\n "k": 1\n}', '{"a":1}']) && coloured(w) && !w.raw,
    agentDrawer: same(texts(a), ['{\n  "a": 1\n}', '{\n  "a": 1\n}', '{\n "q": "<script>x</script>"\n}']) && coloured(a) && !a.raw,
});
