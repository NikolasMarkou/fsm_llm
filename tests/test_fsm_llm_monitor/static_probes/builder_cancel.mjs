import { el, serve, sleep, out } from './stub.mjs';
import * as builder from '../static/pages/builder.js';
// 13.5: Cancel aborts the send, drops the session with ?force=true and ends the elapsed timer.
const calls = [];
let mode = 'hang', signal = null, n = 0;
serve((url, opts) => {
    if (opts?.method === 'DELETE') { calls.push(url); return { deleted: true }; }
    if (url === '/api/builder/send') {
        signal = opts.signal;
        if (mode === 'ok') return { response: 'ok', internal_state: {}, is_complete: false };
        return new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))));
    }
    return { session_id: `s${++n}`, response: 'hi', internal_state: {}, is_complete: false };
});
builder.setDeps({ showPage() {}, refreshInstances() {} });
const status = () => el('builder-status').innerHTML, cancelBtn = el('builder-cancel-btn');
async function begin(msg = 'build it') {
    builder.startBuilderSession(); await sleep(20);
    el('builder-message-input').value = msg;
    builder.sendBuilderMessage(); await sleep(20);
}
await begin();
const whileSending = { cancelShown: cancelBtn.style.display };
await sleep(1100);
whileSending.status = status();
builder.cancelBuilderSend(); await sleep(20);
const cancelled = { deletes: [...calls], aborted: signal.aborted, cancelBtn: cancelBtn.style.display, input: el('builder-input-area').style.display,
    bubble: el('builder-chat').html.some(h => h.includes('Cancelled')), noAbortBubble: !el('builder-chat').html.some(h => h.includes('aborted')), status: status() };
await begin();                                   // a new session, then Reset while sending
builder.resetBuilder(); await sleep(20);
const reset = { deletes: calls.slice(1), aborted: signal.aborted, cancelBtn: cancelBtn.style.display };
await begin();                                   // Start Building again while a send is in flight
const abortedFirst = signal;
builder.startBuilderSession(); await sleep(20);
const restart = { deletes: calls.slice(2), aborted: abortedFirst.aborted };
mode = 'ok';
await begin('again');                            // a finished send stops the timer and hides Cancel
const before = status(); await sleep(1100);
const finished = { cancelBtn: cancelBtn.style.display, timerStopped: status() === before };
builder.cancelBuilderSend();                     // nothing in flight: a no-op
out({ whileSending, cancelled, reset, restart, finished, deletesAfterNoop: calls.length });
setTimeout(() => process.exit(0), 100);  // a leaked interval must fail an assertion, not hang the probe
