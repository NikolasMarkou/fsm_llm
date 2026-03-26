// FSM-LLM Monitor — Visualizer Page

'use strict';

var _vizStatusTimer = null;

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

        // Populate info
        var info = data.info || data.fsm || {};
        var infoEl = document.getElementById(infoIds[type]);
        if (infoEl) {
            var fields = { Name: info.name, Description: info.description, States: info.state_count || info.step_count, Version: info.version, Initial: info.initial_state };
            if (type === 'fsm') {
                // Badge flow layout for FSM
                var html = '';
                for (var k in fields) {
                    if (fields[k] !== undefined && fields[k] !== '') {
                        html += '<span class="viz-info-item"><span class="key">' + k + '</span><span class="val">' + esc(fields[k]) + '</span></span>';
                    }
                }
                infoEl.innerHTML = html;
            } else {
                // Standard kv layout for agents/workflows
                var html = '';
                for (var k in fields) {
                    if (fields[k] !== undefined && fields[k] !== '') {
                        html += '<span class="key">' + k + ':</span><span class="val">' + esc(fields[k]) + '</span>';
                    }
                }
                infoEl.innerHTML = html;
            }
        }

        // Populate transitions table
        var tbody = document.getElementById(transIds[type]);
        if (tbody && data.edges) {
            var rows = '';
            for (var i = 0; i < data.edges.length; i++) {
                var e = data.edges[i];
                rows += '<tr><td>' + esc(e.from) + '</td><td>' + esc(e.to) + '</td>';
                if (type === 'fsm') rows += '<td>' + (e.priority || '') + '</td>';
                rows += '<td title="' + esc(e.label) + '">' + esc(e.label) + '</td></tr>';
            }
            tbody.innerHTML = rows;
        }

        // FSM-specific: show details panel, hide empty hint
        if (type === 'fsm') {
            var hint = document.getElementById('viz-empty-hint');
            if (hint) hint.style.display = 'none';
            var details = document.getElementById('viz-details');
            if (details) details.style.display = '';
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
        _showVizStatus('OK', 'success');
    } catch (e) {
        _showVizStatus('Invalid JSON', 'error');
    }
}

function _showVizStatus(text, type) {
    var el = document.getElementById('viz-fsm-status');
    if (!el) return;
    if (_vizStatusTimer) { clearTimeout(_vizStatusTimer); _vizStatusTimer = null; }
    el.textContent = text;
    el.className = 'viz-status-overlay visible ' + type;
    if (type === 'success') {
        _vizStatusTimer = setTimeout(function() {
            el.classList.remove('visible');
        }, 2000);
    }
}

// === DETAILS PANEL TAB SWITCHING ===

function switchVizDetail(tab, btn) {
    var infoPanel = document.getElementById('viz-detail-info');
    var transPanel = document.getElementById('viz-detail-transitions');
    if (!infoPanel || !transPanel) return;
    infoPanel.style.display = tab === 'info' ? '' : 'none';
    transPanel.style.display = tab === 'transitions' ? '' : 'none';
    // Update active tab button
    var tabs = btn.parentElement.querySelectorAll('.viz-dtab');
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
    btn.classList.add('active');
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
