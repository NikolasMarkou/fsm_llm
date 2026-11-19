import { serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as builder from '../static/pages/builder.js';
// 13.3: launching a built FSM opens its conversation in the drawer, as the Launch modal does.
const fsm = { name: 'F', initial_state: 'a', states: { a: { id: 'a', description: 'd', purpose: 'p' } } };
let startBody = { conversation_id: 'conv-77' };
serve(url => {
    if (url === '/api/fsm/launch') return { instance_id: 'inst-1' };
    if (url === '/api/fsm/inst-1/start') return startBody;
    if (url.startsWith('/api/fsm/visualize')) return { nodes: [], edges: [] };
    return { session_id: 's', response: 'done', internal_state: {}, is_complete: true, artifact: fsm, artifact_json: JSON.stringify(fsm), artifact_type: 'fsm', is_valid: true };
});
const pages = [];
builder.setDeps({ showPage: p => pages.push(p), refreshInstances() {} });
async function run(start) {
    startBody = start;
    state.selectedConvId = null; state.selectedConvInstanceId = null; pages.length = 0;
    builder.startBuilderSession(); await sleep(30);
    builder.launchBuilderResult(); await sleep(650);
    return { conv: state.selectedConvId, instance: state.selectedConvInstanceId, pages: [...pages] };
}
const withConversation = await run({ conversation_id: 'conv-77' });
const withoutConversation = await run({});
out({ withConversation, withoutConversation });
