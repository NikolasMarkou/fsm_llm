import { el, out } from './stub.mjs';
import { updateMetrics } from '../static/pages/dashboard.js';

const base = { active_conversations: 1, total_events: 2, total_transitions: 3, total_errors: 0 };
const shown = () => ({
    display: el('m-usage-row').style.display,
    tiles: ['m-llm-calls', 'm-llm-tokens', 'm-llm-errors'].map(id => el(id).textContent),
});
el('m-usage-row').style.display = 'none';
updateMetrics({ ...base, usage: { calls: 7, total_tokens: 12, errors: 2 } });
const withUsage = shown();
updateMetrics({ ...base, usage: null });
const without = shown();
out({ withUsage, without });
