// FSM-LLM Monitor — Visualizer Page

import { state } from '../services/state.js';
import { fetchJson } from '../services/api.js';
import { $, esc, showToast } from '../utils/dom.js';
import { renderGraph } from '../utils/graph.js';

let _vizStatusTimer = null;

export async function visualizeGraph(type, typeValue) {
    const endpoints = {
        fsm: { post: '/api/fsm/visualize', presetGet: '/api/fsm/visualize/preset/' },
        agent: { get: '/api/agent/visualize?agent_type=' },
        workflow: { get: '/api/workflow/visualize?workflow_id=' },
    };
    const svgIds = { fsm: 'viz-svg', agent: 'viz-agent-svg', workflow: 'viz-wf-svg' };
    const infoIds = { fsm: 'viz-info', agent: 'viz-agent-info', workflow: 'viz-wf-info' };
    const transIds = { fsm: 'viz-trans-body', agent: 'viz-agent-trans-body', workflow: 'viz-wf-trans-body' };
    const styles = {
        fsm: { colorVar: 'var(--primary-dim)', rx: 4, nodeClass: 'fsm' },
        agent: { colorVar: 'var(--yellow)', arrowColor: 'var(--yellow)', rx: 16, nodeClass: 'agent' },
        workflow: { colorVar: 'var(--cyan)', arrowColor: 'var(--cyan)', rx: 12, nodeClass: 'wf' },
    };

    try {
        let data;
        if (type === 'fsm') {
            if (typeValue && typeof typeValue === 'string' && typeValue.includes('/')) {
                data = await fetchJson(endpoints.fsm.presetGet + encodeURIComponent(typeValue));
            } else if (typeValue && typeof typeValue === 'object') {
                data = await fetchJson(endpoints.fsm.post, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(typeValue),
                });
            } else {
                return;
            }
        } else {
            if (!typeValue) return;
            data = await fetchJson(endpoints[type].get + encodeURIComponent(typeValue));
        }

        renderGraph(svgIds[type], data, styles[type]);

        // Info panel
        const info = data.info || data.fsm || {};
        const infoEl = $(infoIds[type]);
        if (infoEl) {
            const fields = { Name: info.name, Description: info.description, States: info.state_count || info.step_count, Version: info.version, Initial: info.initial_state };
            let html = '';
            for (const k in fields) {
                if (fields[k] !== undefined && fields[k] !== '') {
                    if (type === 'fsm') {
                        html += '<span class="viz-info-item"><span class="key">' + k + '</span><span class="val">' + esc(fields[k]) + '</span></span>';
                    } else {
                        html += '<span class="key">' + k + ':</span><span class="val">' + esc(fields[k]) + '</span>';
                    }
                }
            }
            infoEl.innerHTML = html;
        }

        // Transitions table
        const tbody = $(transIds[type]);
        if (tbody && data.edges) {
            let rows = '';
            for (const e of data.edges) {
                rows += '<tr><td>' + esc(e.from) + '</td><td>' + esc(e.to) + '</td>';
                if (type === 'fsm') rows += '<td>' + (e.priority || '') + '</td>';
                rows += '<td title="' + esc(e.label) + '">' + esc(e.label) + '</td></tr>';
            }
            tbody.innerHTML = rows;
        }

        if (type === 'fsm') {
            const hint = $('viz-empty-hint');
            if (hint) hint.style.display = 'none';
            const details = $('viz-details');
            if (details) details.style.display = '';
        }
    } catch (e) {
        console.error('visualizeGraph ' + type + ':', e);
        showToast('Visualization failed', 'error');
    }
}

export async function visualizeFSM() {
    const jsonText = $('viz-fsm-json')?.value.trim();
    if (!jsonText) return;
    try {
        const fsmDef = JSON.parse(jsonText);
        await visualizeGraph('fsm', fsmDef);
        _showVizStatus('OK', 'success');
    } catch {
        _showVizStatus('Invalid JSON', 'error');
    }
}

function _showVizStatus(text, type) {
    const el = $('viz-fsm-status');
    if (!el) return;
    if (_vizStatusTimer) { clearTimeout(_vizStatusTimer); _vizStatusTimer = null; }
    el.textContent = text;
    el.className = 'viz-status-overlay visible ' + type;
    if (type === 'success') {
        _vizStatusTimer = setTimeout(() => el.classList.remove('visible'), 2000);
    }
}

export function switchVizDetail(tab, btn) {
    const infoPanel = $('viz-detail-info');
    const transPanel = $('viz-detail-transitions');
    if (!infoPanel || !transPanel) return;
    infoPanel.style.display = tab === 'info' ? '' : 'none';
    transPanel.style.display = tab === 'transitions' ? '' : 'none';
    const tabs = btn.parentElement.querySelectorAll('.viz-dtab');
    tabs.forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
}

// --- Presets ---

export async function loadFSMPresets() {
    if (state.presets) {
        _populatePresetDropdown(state.presets);
        return;
    }
    try {
        state.presets = await fetchJson('/api/presets');
        _populatePresetDropdown(state.presets);
    } catch (e) {
        console.error('loadFSMPresets:', e);
        showToast('Failed to load presets', 'error');
    }
}

function _populatePresetDropdown(presets) {
    const select = $('viz-preset-select');
    if (!select || select.options.length > 1) return;
    const items = presets.fsm || [];
    if (items.length === 0) return;

    const groups = {};
    for (const item of items) {
        const cat = item.category || 'other';
        (groups[cat] ??= []).push(item);
    }

    for (const cat of Object.keys(groups).sort()) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = cat.charAt(0).toUpperCase() + cat.slice(1);
        for (const item of groups[cat]) {
            const opt = document.createElement('option');
            opt.value = item.id;
            opt.textContent = item.name;
            if (item.description) opt.title = item.description;
            optgroup.appendChild(opt);
        }
        select.appendChild(optgroup);
    }
}

export async function useFSMPreset(presetId) {
    if (!presetId) return;
    try {
        const data = await fetchJson('/api/preset/fsm/' + encodeURIComponent(presetId));
        $('viz-fsm-json').value = JSON.stringify(data, null, 2);
        visualizeFSM();
    } catch (e) {
        console.error('useFSMPreset:', e);
        showToast('Failed to load preset', 'error');
    }
}

// --- Split-Pane Resize ---

export function initVizDivider() {
    const divider = $('viz-divider');
    if (!divider) return;

    const editorPane = divider.previousElementSibling;
    const container = divider.parentElement;

    divider.addEventListener('mousedown', (e) => {
        e.preventDefault();
        divider.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        const onMouseMove = (ev) => {
            const rect = container.getBoundingClientRect();
            const x = ev.clientX - rect.left;
            const w = Math.max(250, Math.min(rect.width * 0.6, x));
            editorPane.style.width = w + 'px';
        };

        const onMouseUp = () => {
            divider.classList.remove('active');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    });

    loadFSMPresets();
}
