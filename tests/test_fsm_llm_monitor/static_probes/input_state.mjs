import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as conv from '../static/pages/conversations.js';

// A is sent a message and its reply is pending; the drawer then shows B.
let reply, posts = 0;
serve((url, opts) => {
    if (opts?.method === 'POST') { posts++; return new Promise(res => { reply = () => res({ response: 'R', current_state: 'x', is_terminal: false }); }); }
    return { conversation_id: url.endsWith('/B') ? 'B' : 'A', instance_id: 'I', current_state: 'live', is_terminal: false, message_history: [], context_data: {} };
});
conv.setDeps({});
state.instances = [{ instance_id: 'I', instance_type: 'fsm', label: 'L' }];
state.currentPage = 'control';
const input = el('conv-message-input');
conv.showConversationInDrawer('I', 'A'); await sleep(20);
input.value = 'hello';
const sent = conv.sendChatMessage(); await sleep(20);
const sendingA = input.disabled;
conv.drawerBack(); conv.showConversationInDrawer('I', 'B'); await sleep(20);
const shownB = input.disabled;
input.value = 'typed in B';
await conv.sendChatMessage();  // refused while A is answered: a toast, the text kept, nothing posted
await sleep(80);
const refused = { value: input.value, posts, toast: el('toast-live').textContent };
reply(); await sent;
out({ sendingA, shownB, refused, after: input.disabled });
process.exit(0);
