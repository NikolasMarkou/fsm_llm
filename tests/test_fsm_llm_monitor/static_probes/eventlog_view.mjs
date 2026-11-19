import { el, serve, out } from './stub.mjs';
import * as eventlog from '../static/pages/eventlog.js';

const usage = { calls: 2, total_tokens: 30, errors: 1 };
let answer;
serve(url => (typeof answer === 'function' ? answer(url) : answer));

answer = { available: false, conversations: [] };
await eventlog.loadEventLogs();
const unavailable = el('eventlog-list').innerHTML;

answer = { available: true, conversations: ['abc', '<b>x</b>'] };
await eventlog.loadEventLogs();
const list = el('eventlog-list').innerHTML;

answer = { conversation_id: 'abc', next_seq: 2, usage, events: [
    { seq: 1, ts: 't1', type: 'turn/start', frame: 0, state: '<script>s</script>' },
    { seq: 2, ts: 't2', type: 'turn/end', frame: 0 },
] };
await eventlog.openEventLog('abc');
const view = el('eventlog-view').innerHTML;

answer = { fail: 'no such log' };
await eventlog.openEventLog('zzz');
const failed = el('eventlog-view').innerHTML;

let asked;
answer = url => { asked = url; return { conversation_id: 'a b', next_seq: null, usage, events: [] }; };
await eventlog.openEventLog('a b/c');
out({
    unavailable: unavailable.includes('--event-log DIR'),
    list: [list.includes('data-conv-id="abc"'), list.includes('&lt;b&gt;x&lt;/b&gt;'), list.includes('<b>x')],
    view: [view.includes('2 LLM calls, 30 tokens, 1 errors'), view.includes('the first events only'), view.includes('&lt;script&gt;'), view.includes('<script>'), view.split('<tr>').length - 1],
    failure: failed.includes('no such log') && failed.includes('error-message'),
    path: asked,
    complete: !el('eventlog-view').innerHTML.includes('first events only'),
});
