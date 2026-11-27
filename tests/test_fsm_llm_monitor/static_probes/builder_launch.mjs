import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as builder from '../static/pages/builder.js';

const artifact = { name: 'A', agent_type: 'react', tools: [{ name: 't1', description: 'd' }], config: {} };
const fsm = { name: 'F', initial_state: 'a', states: { a: { id: 'a', description: 'd', purpose: 'p' } } };
let cur = artifact, curType = 'agent';
let refuse = true;
const posts = [], fsmPosts = [];
serve(url => {
    if (url === '/api/agent/types') return { agent_types: [] };
    if (url === '/api/agent/launch') { posts.push(url); return refuse ? { fail: 'model not configured' } : {}; }
    if (url === '/api/fsm/launch') { fsmPosts.push(url); return { instance_id: 'i1' }; }
    if (url === '/api/fsm/i1/start') return { conversation_id: 'c1' };
    if (url.startsWith('/api/fsm/visualize')) return { nodes: [], edges: [] };
    return { session_id: 's', response: 'done', internal_state: {}, is_complete: true, artifact: cur, artifact_json: JSON.stringify(cur), artifact_type: curType, is_valid: true };
});
const pages = [];
builder.setDeps({ showPage: p => pages.push(p), refreshInstances() {} });
el('builder-agent-launch-form').style.display = 'none';
const form = el('builder-agent-launch-form'), btn = el('builder-launch-btn');
builder.startBuilderSession(); await sleep(30);
builder.launchBuilderResult();                      // first click opens the form
el('builder-agent-task').value = 'my task';
const opened = [form.style.display, btn.textContent];
const snap = () => ({ form: form.style.display, label: btn.textContent, disabled: btn.disabled, posts: posts.length });
builder.launchBuilderResult(); await sleep(30);     // second click: refused
const failed = snap();
refuse = false;
builder.launchBuilderResult(); await sleep(30);     // retry from the same open form: accepted
const retried = snap();
// Plan 9283c1fd step 8 (G3a): the button holds until the redirect; the redirect follows only the session and page it came from.
const g3a = {};
await sleep(560);                                    // the 500 ms window of the accepted launch is over
g3a.window_over = { disabled: btn.disabled, pages: [...pages] }; pages.length = 0;
builder.launchBuilderResult();                       // click 1 opens the form
el('builder-agent-task').value = 'my task';
const before = posts.length;
builder.launchBuilderResult(); await sleep(30);      // click 2 submits
builder.launchBuilderResult(); builder.launchBuilderResult(); builder.launchBuilderResult(); await sleep(30);  // clicks 3 to 5 inside the window
g3a.four_clicks = { posts: posts.length - before, disabled: btn.disabled, form: form.style.display };
await sleep(560);
g3a.after_window = { disabled: btn.disabled, pages: [...pages] }; pages.length = 0;
refuse = true;                                       // a refusal frees the button and the guard at once
builder.launchBuilderResult(); el('builder-agent-task').value = 'my task';
builder.launchBuilderResult(); await sleep(30);
g3a.refused = snap();
refuse = false;
const afterRefusal = posts.length;
builder.launchBuilderResult(); await sleep(30);
g3a.retry_posts = posts.length - afterRefusal;
await sleep(560); pages.length = 0;
async function launchThen(type, art, act) {          // launch from a fresh session, then `act` inside the window
    cur = art; curType = type; pages.length = 0; state.selectedConvId = null;
    builder.startBuilderSession(); await sleep(30);
    builder.launchBuilderResult();
    if (type === 'agent') { el('builder-agent-task').value = 'my task'; builder.launchBuilderResult(); }
    await sleep(30); act(); await sleep(560);
    return { pages: [...pages], conv: state.selectedConvId, disabled: btn.disabled };
}
state.currentPage = 'builder';
g3a.agent_stays = await launchThen('agent', artifact, () => {});
g3a.agent_reset = await launchThen('agent', artifact, () => builder.resetBuilder());
g3a.agent_navigated = await launchThen('agent', artifact, () => { state.currentPage = 'dashboard'; });
state.currentPage = 'builder';
g3a.fsm_stays = await launchThen('fsm', fsm, () => {});
g3a.fsm_reset = await launchThen('fsm', fsm, () => builder.resetBuilder());
g3a.fsm_navigated = await launchThen('fsm', fsm, () => { state.currentPage = 'dashboard'; });
out({ opened, failed, retried, g3a });
