import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as conv from '../static/pages/conversations.js';

const base = { conversation_id: 'A', instance_id: 'I', current_state: 'live', is_terminal: false, message_history: [], context_data: {} };
async function show(extra) {
    serve(() => ({ ...base, ...extra }));
    conv.showConversationInDrawer('I', 'A'); await sleep(20);
    return el('conv-detail').innerHTML;
}
conv.setDeps({});
state.instances = [{ instance_id: 'I', instance_type: 'fsm', label: 'L' }];
state.currentPage = 'control';
const none = await show({});
const some = await show({
    cut_off: { reply: true, fields: ['<b>f</b>'] },
    classification_results: { intent: { name: '<i>buy</i>', confidence: 0.9 } },
    transition_classification: { chosen: 'go' },
});
out({
    none: ['Cut off', 'Classification Results', 'Transition Classifier'].map(t => none.includes(t)),
    some: ['Cut off at max_tokens: reply, &lt;b&gt;f&lt;/b&gt;', 'Classification Results', 'Transition Classifier', '&lt;i&gt;buy&lt;/i&gt;'].map(t => some.includes(t)),
    raw_tags: some.includes('<b>f') || some.includes('<i>buy'),
});
