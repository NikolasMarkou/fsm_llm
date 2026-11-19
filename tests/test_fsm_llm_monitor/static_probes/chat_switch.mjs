import { el, serve, sleep, out } from './stub.mjs';
import { state } from '../static/services/state.js';
import * as conv from '../static/pages/conversations.js';

// Conversation A is sent a message; its reply (terminal, END_A) arrives while the drawer shows A or B.
async function scenario(switchAway) {
    let reply;
    serve((url, opts) => {
        if (opts?.method === 'POST') return new Promise(res => { reply = () => res({ response: 'REPLY-FOR-A', current_state: 'END_A', is_terminal: true }); });
        return { conversation_id: url.endsWith('/B') ? 'B' : 'A', instance_id: 'I', current_state: 'live', is_terminal: false, message_history: [], context_data: {} };
    });
    const badge = { textContent: 'stateB' };
    el('conv-detail').querySelector = () => badge;
    el('conv-chat-input').style.display = '';
    el('conv-chat-log').html.length = 0;
    conv.showConversationInDrawer('I', 'A'); await sleep(20);
    el('conv-message-input').value = 'hello';
    const sent = conv.sendChatMessage(); await sleep(20);
    if (switchAway) { conv.drawerBack(); conv.showConversationInDrawer('I', 'B'); await sleep(20); }
    reply(); await sent;
    return { badge: badge.textContent, input_display: el('conv-chat-input').style.display, bubbles: el('conv-chat-log').html.filter(h => h.includes('REPLY-FOR-A')).length };
}
conv.setDeps({});
state.instances = [{ instance_id: 'I', instance_type: 'fsm', label: 'L' }];
state.currentPage = 'control';
out({ stay: await scenario(false), switched: await scenario(true) });
