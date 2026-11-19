import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as control from '../static/pages/control.js';

const runs = [['done-1', 'completed'], ['live-1', 'running'], ['wait-1', 'waiting'], ['dead-1', 'failed']];
serve(url => {
    if (url.endsWith('/instances')) return runs.map(([id, status]) => ({ workflow_instance_id: id, status, current_step_id: 's', history: [] }));
    return { instance_id: 'W', instance_type: 'workflow', status: 'running', label: 'W', active_workflows: 2 };
});
state.instances = [{ instance_id: 'W', instance_type: 'workflow', label: 'W' }];
state.selectedDetailId = 'W'; state.selectedDetailType = 'workflow';
await control.refreshDetailPanel('W', 'workflow'); await sleep(20);
const select = el('ctrl-drawer-content').innerHTML.match(/<select id="wf-event-target"[^>]*>(.*?)<\/select>/s);
out({ options: [...select[1].matchAll(/<option value="([^"]*)"( selected)?>/g)].map(m => [m[1], !!m[2]]) });
