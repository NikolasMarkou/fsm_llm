import { el, serve, sleep, out } from './stub.mjs';
import * as builder from '../static/pages/builder.js';
// 13.2: Launch is disabled, with a note, for an agent type /api/capabilities does not list.
let agentType = 'react', capsDelay = 0, capsCalls = 0, slowFirst = false, caps = { fsm: true, agent_types: ['react', 'debate'] }, posts = 0;
serve(async url => {
    if (url === '/api/capabilities') { const n = ++capsCalls; await sleep(slowFirst && n === 1 ? 60 : capsDelay); return caps.fail ? { fail: 'down' } : caps; }
    if (url === '/api/agent/launch') { posts++; return {}; }
    const a = { name: 'A', agent_type: agentType, tools: [{ name: 't', description: 'd' }], config: {} };
    return { session_id: 's', response: 'done', internal_state: {}, is_complete: true, artifact: a, artifact_json: JSON.stringify(a), artifact_type: 'agent', is_valid: true };
});
builder.setDeps({ showPage() {}, refreshInstances() {} });
const btn = el('builder-launch-btn'), note = el('builder-launch-note');
async function run(type) {
    agentType = type;
    builder.startBuilderSession(); await sleep(40);
    return { disabled: btn.disabled, note: note.textContent, noteShown: note.style.display, label: btn.textContent, posts };
}
const listed = await run('react');
const unlisted = await run('evaluator_optimizer');
const unlistedPromptChain = await run('prompt_chain');
const backToListed = await run('debate');                    // a later result re-enables it
capsDelay = 30;                                              // a reply that arrives after the session was reset is dropped
agentType = 'evaluator_optimizer';
builder.startBuilderSession(); await sleep(15); builder.resetBuilder(); await sleep(50);
const staleAfterReset = { disabled: btn.disabled, note: note.textContent };
capsDelay = 0;
caps = { fsm: true };                                        // an older server with no list
const noList = await run('maker_checker');
caps = { fail: true };
const capsDown = await run('maker_checker');
caps = { fsm: true, agent_types: ['react', 'debate'] };
const inherited = await run('constructor');                  // a key of Object.prototype is no agent type (D-058)
const protoKey = await run('__proto__');
capsCalls = 0; slowFirst = true;                             // the reply for the pre-edit artifact arrives after the Apply
agentType = 'evaluator_optimizer';
builder.startBuilderSession(); await sleep(20);
const area = el('builder-result-edit');
builder.editBuilderResult();
area.value = JSON.stringify({ name: 'A', agent_type: 'react', tools: [{ name: 't', description: 'd' }], config: {} });
builder.editBuilderResult(); await sleep(120);
slowFirst = false;
const staleAfterEdit = { disabled: btn.disabled, note: note.textContent };
out({ inherited, protoKey, staleAfterEdit, staleAfterReset, listed, unlisted, unlistedPromptChain, backToListed, noList, capsDown });
