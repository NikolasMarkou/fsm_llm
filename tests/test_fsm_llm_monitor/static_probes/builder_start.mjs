import { el, serve, sleep, out } from './stub.mjs';
import * as builder from '../static/pages/builder.js';
// 13.4: the start request carries the Max Tokens field (4096 when it is empty or not a number).
const bodies = [];
serve((url, opts) => {
    if (url === '/api/builder/start') bodies.push(JSON.parse(opts.body));
    return { session_id: 's', response: 'hi', internal_state: {}, is_complete: false };
});
builder.setDeps({ showPage() {}, refreshInstances() {} });
for (const v of ['8000', '', 'abc']) {
    el('builder-max-tokens').value = v;
    builder.startBuilderSession(); await sleep(20);
}
out({ tokens: bodies.map(b => b.max_tokens) });
