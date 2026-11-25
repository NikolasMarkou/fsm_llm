import { el, serve, sleep, out } from './stub.mjs';
import * as builder from '../static/pages/builder.js';
// 13.1: the model of a launched built agent is the Builder field, then the form field, never artifact.config.model.
let config = { model: 'gpt-4o-mini', max_iterations: 7, timeout_seconds: 300.0 };
let launchBody = null;
let tools = [{ name: 't1', description: 'd' }];
const artifact = () => ({ name: 'Weather bot', agent_type: 'react', tools, config });
const rows = [{ id: 'react', name: 'react', launchable: true, needs_tools: true, takes_tools: true, tool_names: [], drawable: true }];
serve((url, opts) => {
    if (url === '/api/agent/types') return { agent_types: rows };
    if (url === '/api/agent/launch') { launchBody = JSON.parse(opts.body); return {}; }
    const a = artifact();
    return { session_id: 's', response: 'done', internal_state: {}, is_complete: true, artifact: a, artifact_json: JSON.stringify(a), artifact_type: 'agent', is_valid: true };
});
builder.setDeps({ showPage() {}, refreshInstances() {} });
const field = id => el(id).value;
const fields = () => ({ label: field('builder-agent-label'), model: field('builder-agent-model'), iters: field('builder-agent-iters'), timeout: field('builder-agent-timeout') });
async function run(builderModel, edit) {
    launchBody = null;
    el('builder-model').value = builderModel;
    el('builder-agent-launch-form').style.display = 'none';
    builder.startBuilderSession(); await sleep(30);
    builder.launchBuilderResult();                      // first click opens and pre-fills the form
    const prefilled = fields();
    el('builder-agent-task').value = 'task';
    edit?.();
    builder.launchBuilderResult(); await sleep(30);     // second click sends
    return { prefilled, sent: launchBody };
}
const fromBuilderField = await run('ollama_chat/my-local-model');
const typedInForm = await run('ollama_chat/my-local-model', () => {
    el('builder-agent-model').value = 'ollama_chat/other'; el('builder-agent-label').value = 'Mine';
    el('builder-agent-iters').value = '3'; el('builder-agent-timeout').value = '45';
});
const emptyField = await run('');
config = { model: 'gpt-4o-mini', max_iterations: 'many', timeout_seconds: -5 };
const badConfig = await run('');
tools = [];                                          // react needs a tool and the artifact has none: nothing is sent
const noTool = await run('');
out({ fromBuilderField, typedInForm, emptyField, badConfig, noTool });
