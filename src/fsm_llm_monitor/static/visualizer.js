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
        var data;
        if (type === 'fsm') {
            if (typeValue && typeof typeValue === 'string' && typeValue.includes('/')) {
                data = await fetchJson(endpoints.fsm.presetGet + encodeURIComponent(typeValue));
            } else if (typeValue && typeof typeValue === 'object') {
                data = await fetchJson(endpoints.fsm.post, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
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

async function visualizeFSM() {
    var jsonText = document.getElementById('viz-fsm-json').value.trim();
    if (!jsonText) return;
    try {
        var fsmDef = JSON.parse(jsonText);
        await visualizeGraph('fsm', fsmDef);
        showStatus('viz-fsm-status', 'OK', 'success');
    } catch (e) {
        showError('viz-fsm-status', 'Invalid JSON');
    }
}

// === PRESETS (dropdown in editor header) ===

async function loadFSMPresets() {
    if (App.presets) {
        _populatePresetDropdown(App.presets);
        return;
    }
    try {
        App.presets = await fetchJson('/api/presets');
        _populatePresetDropdown(App.presets);
    } catch (e) {
        console.error('loadFSMPresets:', e);
    }
}

function _populatePresetDropdown(presets) {
    var select = document.getElementById('viz-preset-select');
    if (!select || select.options.length > 1) return;
    var items = presets.fsm || [];
    if (items.length === 0) return;

    var groups = {};
    for (var i = 0; i < items.length; i++) {
        var cat = items[i].category || 'other';
        if (!groups[cat]) groups[cat] = [];
        groups[cat].push(items[i]);
    }

    var cats = Object.keys(groups).sort();
    for (var c = 0; c < cats.length; c++) {
        var optgroup = document.createElement('optgroup');
        optgroup.label = cats[c].charAt(0).toUpperCase() + cats[c].slice(1);
        var catItems = groups[cats[c]];
        for (var j = 0; j < catItems.length; j++) {
            var opt = document.createElement('option');
            opt.value = catItems[j].id;
            opt.textContent = catItems[j].name;
            if (catItems[j].description) opt.title = catItems[j].description;
            optgroup.appendChild(opt);
        }
        select.appendChild(optgroup);
    }
}

async function useFSMPreset(presetId) {
    if (!presetId) return;
    try {
        var data = await fetchJson('/api/preset/fsm/' + encodeURIComponent(presetId));
        document.getElementById('viz-fsm-json').value = JSON.stringify(data, null, 2);
        visualizeFSM();
    } catch (e) {
        console.error('useFSMPreset:', e);
    }
}

// === SPLIT-PANE RESIZE HANDLE ===

function initVizDivider() {
    var divider = document.getElementById('viz-divider');
    if (!divider) return;

    var editorPane = divider.previousElementSibling;
    var container = divider.parentElement;

    divider.addEventListener('mousedown', function(e) {
        e.preventDefault();
        divider.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        function onMouseMove(ev) {
            var rect = container.getBoundingClientRect();
            var x = ev.clientX - rect.left;
            var minW = 250;
            var maxW = rect.width * 0.6;
            var w = Math.max(minW, Math.min(maxW, x));
            editorPane.style.width = w + 'px';
        }

        function onMouseUp() {
            divider.classList.remove('active');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    });

    // Load presets on init
    loadFSMPresets();
}
