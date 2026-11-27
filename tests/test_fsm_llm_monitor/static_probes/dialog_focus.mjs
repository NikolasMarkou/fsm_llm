import { el, sleep, out } from './stub.mjs';
// Plan 9283c1fd step 8 (G7): a 4401 close that arrives after the user dismissed the key prompt does not open it again;
// the dismissal is honoured until a key is saved (Settings). Step 9 adds the focus-trap rows below.
console.error = () => {};
const socks = [];
globalThis.WebSocket = class { static OPEN = 1; static CONNECTING = 0; constructor(u) { this.url = u; this.readyState = 0; socks.push(this); } send() {} close() {} };
const mem = new Map();
globalThis.sessionStorage = { getItem: k => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: k => mem.delete(k) };
const auth = await import('../static/services/auth.js');
const ws = await import('../static/services/ws.js');

const modal = el('apikey-modal'), input = el('apikey-input'), status = el('ws-status');
modal.style.display = 'none';
let opens = 0;
input.focus = () => { opens++; };                    // openDialog focuses the input once per real open
const newSock = () => { ws.connectWS(); return socks[socks.length - 1]; };
const reset = () => { auth.setApiKey('k'); auth.clearApiKey(); };   // a saved key clears the dismissal
const open = () => auth.isApiKeyModalOpen();
const r = {};

// 4401 first, then the boot prompt: one modal, whatever the order.
let sock = newSock();
sock.onclose({ code: 4401 });
r.early = { open_after_4401: open() };
auth.requestApiKey({ force: true });
r.early.opens = opens; r.early.status = status.textContent;
auth.cancelApiKeyModal();
r.early.open_after_cancel = open();

// The boot prompt, Cancel, then the 4401 close arrives: the dismissal holds.
reset(); opens = 0;
sock = newSock();
auth.requestApiKey({ force: true });
auth.cancelApiKeyModal();
sock.onclose({ code: 4401 });
r.late = { open: open(), opens, status: status.textContent };
if (open()) auth.cancelApiKeyModal();

// Saving a key (Settings) after the dismissal reconnects and the dismissal is over.
const before = socks.length;
auth.setApiKey('k');
r.save = { reconnects: socks.length - before, open: open() };

// No dismissal: a 4401 prompts as before.
auth.clearApiKey(); opens = 0;
sock = newSock();
sock.onclose({ code: 4401 });
r.undismissed = { open: open(), opens, status: status.textContent };
auth.cancelApiKeyModal();
out(r);
