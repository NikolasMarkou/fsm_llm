import { el, sleep, out } from './stub.mjs';
// Plan 9283c1fd step 8 (G7): a 4401 close that arrives after the user dismissed the key prompt does not open it again;
// the dismissal is honoured until a key is saved (Settings). Step 9 adds the focus-trap rows below (O4, D-007):
// Tab and Shift+Tab stay inside the open dialog or drawer. Proven over this stub DOM only: no browser ran here.
console.error = () => {};
const listeners = new Set();                          // the keydown listeners registered on the document
let adds = 0, removes = 0;
document.addEventListener = (type, f) => { if (type === 'keydown' && !listeners.has(f)) { listeners.add(f); adds++; } };
document.removeEventListener = (type, f) => { if (listeners.delete(f)) removes++; };
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

// --- Step 9: the focus trap ---
document.contains = () => true;
const widget = (id, { visible = true, disabled = false } = {}) => {
    const w = el(id);
    Object.assign(w, { disabled, getClientRects: () => (visible ? [1] : []), focus: () => { document.activeElement = w; } });
    return w;
};
const dialog = (id, kids, inside = []) => {            // inside: focusable nodes the selector does not list
    const d = widget(id); d.style.display = 'none'; d.querySelectorAll = () => kids;
    d.contains = n => n === d || kids.includes(n) || inside.includes(n);
    return d;
};
const focused = () => document.activeElement?.id ?? null;
const press = (shift = false) => {                     // one Tab keydown to every registered listener
    const e = { key: 'Tab', shiftKey: shift, prevented: false, preventDefault() { this.prevented = true; } };
    [...listeners].forEach(f => f(e));
    return { prevented: e.prevented, focused: focused() };
};
const dom = await import('../static/utils/dom.js');
const [a, b, c] = ['ta', 'tb', 'tc'].map(id => widget(id));
const hidden = widget('t-hidden', { visible: false }), off = widget('t-off', { disabled: true }), outside = widget('t-out');
const d1 = dialog('launch-modal', [a, b, c, hidden, off]);
const t = {};
outside.focus();
dom.openDialog(d1, 'flex');
t.opened = { focused: focused(), listeners: listeners.size };
c.focus(); t.tab_on_last_wraps_to_first = press();
a.focus(); t.shift_tab_on_first_wraps_to_last = press(true);
b.focus(); t.tab_in_middle_not_prevented = { tab: press(), shift: press(true), listeners: listeners.size };
outside.focus(); t.tab_with_focus_outside_enters_dialog = { tab: press(), shift: (outside.focus(), press(true)) };
// A dialog opened twice is one entry and one listener.
dom.openDialog(d1, 'flex');
t.reopen_one_listener = listeners.size;
dom.closeDialog(d1);
t.existing_focus_return_unchanged = { focused: focused(), listeners: listeners.size, display: d1.style.display };

// An empty dialog keeps focus on the dialog element.
const d2 = dialog('ctrl-drawer', []);
outside.focus(); dom.openDialog(d2, 'block');
t.empty_dialog_keeps_focus_on_container = { tab: press(), shift: press(true), tabindex: d2.getAttribute('tabindex') };
dom.closeDialog(d2);

// The drawer is opened as 'block' like a dialog and traps the same way.
const [x, y] = ['dx', 'dy'].map(id => widget(id));
const d3 = dialog('ctrl-drawer', [x, y]);
dom.openDialog(d3, 'block');
y.focus(); const wrap = press(); x.focus(); const back = press(true);
t.drawer_traps_like_a_dialog = { wrap, back };
dom.closeDialog(d3);

// Two dialogs stacked: the top traps; after the top closes the lower one does; closed out of order, the stack survives.
const [k1, k2] = ['k1', 'k2'].map(id => widget(id));
const d4 = dialog('apikey-modal', [k1, k2]);
dom.openDialog(d1, 'flex'); dom.openDialog(d4, 'flex', k1);
k2.focus(); const top = press(); c.focus(); const lowerFocusEntersTop = press();
dom.closeDialog(d4); c.focus(); const lower = press();
t.stacked_dialogs_top_traps_then_lower_after_close = { top, lowerFocusEntersTop, lower, listeners: listeners.size };
dom.openDialog(d4, 'flex', k1); dom.closeDialog(d1); k2.focus();
t.stack_survives_out_of_order_close = { top: press(), listeners: listeners.size };
dom.closeDialog(d4);

// A dialog hidden by hand (style.display written directly) is still on the stack: reopening it makes no second entry,
// and closeDialog pops it even though it is already hidden.
dom.openDialog(d1, 'flex'); d1.style.display = 'none';
dom.openDialog(d1, 'flex'); dom.closeDialog(d1);
const reopened = listeners.size;
dom.openDialog(d1, 'flex'); d1.style.display = 'none'; dom.closeDialog(d1);
t.hidden_by_hand_is_popped = { reopened, closed: listeners.size };
t.close_removes_listener = { listeners: listeners.size, adds, removes, after_close: press() };

// Plan 9283c1fd review F1: "outside" is decided by the dialog's own subtree, never by membership in the focusable list.
// A <summary> (the Context Data toggle of the drawer) is in the selector; the stub honours the selector by tag.
const tagged = (id, tag) => Object.assign(widget(id), { tag });
const selects = (sel, kids) => sel.split(',').map(x => x.trim()).includes('summary') ? kids : kids.filter(k => k.tag !== 'summary');
const [wClose, wAdv, wSum, wEv, wSend] = [['wclose', 'button'], ['wadv', 'button'], ['wsum', 'summary'], ['wev', 'input'], ['wsend', 'button']].map(a => tagged(...a));
const wf = dialog('ctrl-drawer', []);
wf.querySelectorAll = sel => selects(sel, [wClose, wAdv, wSum, wEv, wSend]);
wf.contains = n => [wf, wClose, wAdv, wSum, wEv, wSend].includes(n);
outside.focus(); dom.openDialog(wf, 'block');
wSum.focus(); const sumMid = press();
wSend.focus(); const sendWraps = press();
t.tab_from_summary_reaches_next_control = { sumMid, sendWraps };
dom.closeDialog(wf);
// The summary is the LAST control: it is in the list, so Tab on it wraps and Shift+Tab on the first reaches it.
const wf2 = dialog('ctrl-drawer', []);
wf2.querySelectorAll = sel => selects(sel, [wClose, wAdv, wSum]);
wf2.contains = n => [wf2, wClose, wAdv, wSum].includes(n);
outside.focus(); dom.openDialog(wf2, 'block');
wSum.focus(); const lastWraps = press(); wClose.focus(); const firstBack = press(true);
t.summary_last_is_in_the_trap = { lastWraps, firstBack };
dom.closeDialog(wf2);
// A focusable inside that the selector does not list (a scroller): Tab and Shift+Tab are left to the browser.
const scroller = tagged('scroller', 'div');
const d5 = dialog('launch-modal', [a, b, c], [scroller]);
outside.focus(); dom.openDialog(d5, 'flex');
scroller.focus(); const unlistedTab = press(); scroller.focus(); const unlistedShift = press(true);
t.tab_from_unlisted_focusable_inside_is_not_redirected = { unlistedTab, unlistedShift, focused_after: focused() };
outside.focus(); t.tab_from_outside_still_enters = press();
dom.closeDialog(d5);
r.trap = t;
out(r);
