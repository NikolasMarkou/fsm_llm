// FSM-LLM Monitor — Visualizer Page

'use strict';

async function visualizeGraph(type, typeValue) {
    var endpoints = {
        fsm: { post: '/api/fsm/visualize', presetGet: '/api/fsm/visualize/preset/' },
        agent: { get: '/api/agent/visualize?agent_type=' },
        workflow: { get: '/api/workflow/visualize?workflow_id=' }
    };
    var svgIds = { fsm: 'viz-svg', agent: 'viz-agent-svg', workflow: 'viz-wf-svg' };
    var infoIds = { fsm: 'viz-info', agent: 'viz-agent-info', workflow: 'viz-wf-info' };
    var transIds = { fsm: 'viz-trans-body', agent: 'viz-agent-trans-body', workflow: 'viz-wf-trans-body' };
    var styles = {
        fsm: { colorVar: 'var(--primary-dim)', rx: 4, nodeClass: 'fsm' },
        agent: { colorVar: 'var(--yellow)', arrowColor: 'var(--yellow)', rx: 16, nodeClass: 'agent' },
        workflow: { colorVar: 'var(--cyan)', arrowColor: 'var(--cyan)', rx: 12, nodeClass: 'wf' }
    };

    try {
        var resp;
        if (type === 'fsm') {
            if (typeValue && typeof typeValue === 'string' && typeValue.includes('/')) {
                resp = await fetch(endpoints.fsm.presetGet + encodeURIComponent(typeValue));
            } else if (typeValue && typeof typeValue === 'object') {
                resp = await fetch(endpoints.fsm.post, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(typeValue),
                });
            } else {
                return;
            }
        } else {
            if (!typeValue) return;
            resp = await fetch(endpoints[type].get + encodeURIComponent(typeValue));
        }

        var data = await resp.json();
        if (data.error) {
            console.error('visualizeGraph ' + type + ':', data.error);
            return;
        }

        renderGraph(svgIds[type], data, styles[type]);

        var info = data.info || data.fsm || {};
        var infoEl = document.getElementById(infoIds[type]);
        if (infoEl) {
            var infoHtml = '';
            var fields = { Name: info.name, Description: info.description, States: info.state_count || info.step_count, Version: info.version, Initial: info.initial_state };
            for (var k in fields) {
                if (fields[k] !== undefined && fields[k] !== '') {
                    infoHtml += '<span class="key">' + k + ':</span><span class="val">' + esc(fields[k]) + '</span>';
                }
            }
            infoEl.innerHTML = infoHtml;
        }

        var tbody = document.getElementById(transIds[type]);
        if (tbody && data.edges) {
            var rows = '';
            for (var i = 0; i < data.edges.length; i++) {
                var e = data.edges[i];
                rows += '<tr><td>' + esc(e.from) + '</td><td>' + esc(e.to) + '</td>';
                if (type === 'fsm') rows += '<td>' + (e.priority || '') + '</td>';
                rows += '<td>' + esc(e.label) + '</td></tr>';
            }
            tbody.innerHTML = rows;
        }
    } catch (e) {
        console.error('visualizeGraph ' + type + ':', e);
    }
}

async function visualizeFSM(presetId) {
    var statusEl = document.getElementById('viz-fsm-status');
    if (presetId) {
        await visualizeGraph('fsm', presetId);
        if (statusEl) showStatus('viz-fsm-status', 'OK', 'success');
        return;
    }
    var jsonText = document.getElementById('viz-fsm-json').value.trim();
    if (!jsonText) return;
    try {
        var fsmDef = JSON.parse(jsonText);
        await visualizeGraph('fsm', fsmDef);
        if (statusEl) showStatus('viz-fsm-status', 'OK', 'success');
    } catch (e) {
        showError('viz-fsm-status', 'Invalid JSON');
    }
}

// === PRESETS (inline in visualizer) ===

function showPresetPicker() {
    var picker = document.getElementById('preset-picker');
    if (picker.style.display === 'none') {
        picker.style.display = 'block';
        loadFSMPresets();
    } else {
        picker.style.display = 'none';
    }
}

async function loadFSMPresets() {
    if (App.presets) {
        renderPresets(App.presets);
        return;
    }
    try {
        var resp = await fetch('/api/presets');
        App.presets = await resp.json();
        renderPresets(App.presets);
    } catch (e) {
        console.error('loadFSMPresets:', e);
        var empty = document.getElementById('preset-empty');
        if (empty) empty.textContent = 'Failed to load presets';
    }
}

function renderPresets(presets) {
    var items = presets.fsm || [];
    var container = document.getElementById('preset-list');
    var empty = document.getElementById('preset-empty');
    if (items.length === 0) {
        if (empty) empty.textContent = 'No presets found';
        return;
    }
    if (empty) empty.style.display = 'none';
    container.innerHTML = '';
    for (var i = 0; i < items.length; i++) {
        var p = items[i];
        var card = document.createElement('div');
        card.className = 'preset-card';
        card.innerHTML = '<div class="preset-name">' + esc(p.name) + '</div><div class="preset-category">' + esc(p.category || '') + '</div><div class="preset-desc">' + esc(p.description || '') + '</div>';
        (function(id) {
            card.addEventListener('click', function() { useFSMPreset(id); });
        })(p.id);
        container.appendChild(card);
    }
}

async function useFSMPreset(presetId) {
    try {
        var resp = await fetch('/api/preset/fsm/' + encodeURIComponent(presetId));
        var data = await resp.json();
        if (data.error) return;
        document.getElementById('viz-fsm-json').value = JSON.stringify(data, null, 2);
        document.getElementById('preset-picker').style.display = 'none';
        visualizeFSM();
    } catch (e) {
        console.error('useFSMPreset:', e);
    }
}
