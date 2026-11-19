// A minimal DOM stand-in for the monitor ES modules (probes run by test_static_js.py, plan D-009).
// Elements appear on the first getElementById; fetch is answered by a handler the probe supplies.
export const els = new Map();
export class El {
    constructor(id) {
        Object.assign(this, { id, style: {}, dataset: {}, value: '', disabled: false, textContent: '', innerHTML: '', html: [], attrs: {} });
        const s = new Set();
        this.classList = { add: c => s.add(c), remove: c => s.delete(c), contains: c => s.has(c), toggle: c => (s.has(c) ? s.delete(c) : s.add(c)) };
    }
    insertAdjacentHTML(_pos, h) { this.html.push(h); }
    setAttribute(k, v) { this.attrs[k] = v; }
    getAttribute(k) { return this.attrs[k] ?? null; }
    remove() { els.delete(this.id); }
    querySelector() { return null; }
    querySelectorAll() { return []; }
    closest() { return null; }
    focus() {}
    scrollTo() {}
    appendChild() {}
    after() {}
}
export const el = id => els.get(id) ?? (els.set(id, new El(id)), els.get(id));
Object.assign(globalThis, {
    document: { getElementById: el, querySelector: () => null, querySelectorAll: () => [], createElement: () => new El('created'), body: new El('body') },
    requestAnimationFrame: f => f(),
    sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    Headers: class { constructor() { this.m = new Map(); } set(k, v) { this.m.set(k, v); } delete(k) { this.m.delete(k); } },
    location: { hash: '', protocol: 'http:', host: 'x' },
    history: { replaceState() {} },
});
globalThis.window = globalThis;
export const sleep = ms => new Promise(r => setTimeout(r, ms));
/** Answer every fetch with handler(url, opts); `{fail: 'text'}` answers a 400 with that detail. */
export function serve(handler) {
    globalThis.fetch = async (url, opts) => {
        const body = await handler(url, opts);
        if (body && body.fail !== undefined) return { ok: false, status: 400, statusText: 'Bad', json: async () => ({ detail: body.fail }) };
        return { ok: true, json: async () => body };
    };
}
export const out = obj => console.log(JSON.stringify(obj));
