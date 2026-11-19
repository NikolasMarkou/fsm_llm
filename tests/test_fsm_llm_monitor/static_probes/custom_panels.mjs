import { El, els, serve, sleep, out } from './stub.mjs';
import { loadDashboardConfig } from '../static/pages/dashboard.js';

// A section element that registers the ids of its innerHTML, as the real DOM does.
class Section extends El {
    constructor() {
        super('created');
        let html = '';
        Object.defineProperty(this, 'innerHTML', {
            get: () => html,
            set: h => { html = h; this.kids = [...h.matchAll(/id="([^"]+)"/g)].map(m => m[1]); this.kids.forEach(k => els.set(k, new El(k))); },
        });
    }
    remove() { els.delete(this.id); (this.kids || []).forEach(k => els.delete(k)); }
}
document.getElementById = id => els.get(id) ?? null; // no auto-created elements: absence matters here
document.createElement = () => new Section();
document.querySelector = sel => (sel === '.metrics-row' ? { after: sec => els.set(sec.id, sec) } : null);

let cfg;
serve(() => ({ active: true, config: cfg }));
const labels = html => [...(html || '').matchAll(/metric-label">([^<]*)/g)].map(m => m[1]);
async function stage(config) {
    cfg = config; loadDashboardConfig(); await sleep(20);
    const sec = els.get('custom-panels-section');
    return {
        section: !!sec, title: sec?.innerHTML.match(/<h3>([^<]*)/)?.[1] ?? null, desc: sec?.innerHTML.includes('desc ONE') ?? null,
        panels: labels(els.get('custom-panels')?.innerHTML), alerts: els.has('custom-alerts') ? labels(els.get('custom-alerts').innerHTML) : null,
    };
}
const panel = (id, title) => ({ panel_id: id, title, metric: 'total_events' });
out({
    one: await stage({ name: 'Config ONE', description: 'desc ONE', panels: [panel('p1', 'P1')], alerts: [{ alert_id: 'a1', description: 'ALERT ONE', metric: 'total_errors', condition: '>', threshold: 1 }] }),
    two: await stage({ name: 'Config TWO', description: '', panels: [panel('p2', 'P2')], alerts: [] }),
    three: await stage({ name: 'Config THREE', description: '', panels: [], alerts: [] }),
});
