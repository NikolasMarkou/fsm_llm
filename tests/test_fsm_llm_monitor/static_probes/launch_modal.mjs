import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as launch from '../static/pages/launch.js';

state.capabilities = { fsm: true, workflows: true, agents: true };
launch.setDeps({ showPage() {}, refreshInstances() {} });
Object.assign(el('launch-agent-type'), { value: 'DebateAgent' });
el('launch-agent-task').value = 'do it';
el('launch-fsm-source').value = 'preset';
el('launch-fsm-preset-id').value = 'demo';
el('launch-wf-preset').value = 'linear';
const posts = [];
let failing = false;
serve(async (url, opts) => {
    if (opts?.method !== 'POST') return {};
    posts.push(url);
    await sleep(10);
    return failing ? { fail: 'refused' } : { instance_id: 'x', label: 'L', conversation_id: 'c' };
});
const launchers = { agent: [launch.doLaunchAgent, 'Launch Agent'], fsm: [launch.doLaunchFSM, 'Launch FSM'], workflow: [launch.doLaunchWorkflow, 'Launch Workflow'] };
const result = {};
for (const [name, [run, label]] of Object.entries(launchers)) {
    for (failing of [false, true]) {
        posts.length = 0;
        const btn = { disabled: false, textContent: label };
        await run(btn);
        const returned = btn.disabled;
        let window = null;
        if (!failing) {
            await sleep(150);                   // inside the 500 ms window before the modal closes
            window = [btn.disabled, posts.length];
            await sleep(450);
        }
        result[`${name}_${failing ? 'failed' : 'ok'}`] = { returned, window, end: [btn.disabled, btn.textContent] };
    }
}
out(result);
