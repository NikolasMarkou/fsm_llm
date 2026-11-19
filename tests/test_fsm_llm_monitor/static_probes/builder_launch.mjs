import { el, serve, sleep, out } from './stub.mjs';
import * as builder from '../static/pages/builder.js';

const artifact = { name: 'A', agent_type: 'react', tools: [{ name: 't1', description: 'd' }], config: {} };
let refuse = true;
const posts = [];
serve(url => {
    if (url === '/api/agent/launch') { posts.push(url); return refuse ? { fail: 'model not configured' } : {}; }
    return { session_id: 's', response: 'done', internal_state: {}, is_complete: true, artifact, artifact_json: JSON.stringify(artifact), artifact_type: 'agent', is_valid: true };
});
builder.setDeps({ showPage() {}, refreshInstances() {} });
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
out({ opened, failed, retried: snap() });
