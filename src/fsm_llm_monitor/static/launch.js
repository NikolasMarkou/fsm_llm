// FSM-LLM Monitor — Launch Modal

'use strict';

function showLaunchModal() {
    document.getElementById('launch-modal').style.display = 'flex';
    loadLaunchPresets();
    checkCapabilities();
}

function closeLaunchModal() {
    document.getElementById('launch-modal').style.display = 'none';
}

async function checkCapabilities() {
    try {
        var resp = await fetch('/api/capabilities');
        App.capabilities = await resp.json();
        var wfUnavail = document.getElementById('launch-wf-unavailable');
        var agentUnavail = document.getElementById('launch-agent-unavailable');
        if (wfUnavail) wfUnavail.style.display = App.capabilities.workflows ? 'none' : 'block';
        if (agentUnavail) agentUnavail.style.display = App.capabilities.agents ? 'none' : 'block';
    } catch (e) {
        console.error('checkCapabilities:', e);
    }
}

function toggleLaunchFSMSource() {
    var source = document.getElementById('launch-fsm-source').value;
    document.getElementById('launch-fsm-preset-section').style.display = source === 'preset' ? 'block' : 'none';
    document.getElementById('launch-fsm-json-section').style.display = source === 'json' ? 'block' : 'none';
}

async function loadLaunchPresets() {
    if (App.presets) {
        renderLaunchPresets(App.presets);
        return;
    }
    try {
        var resp = await fetch('/api/presets');
        App.presets = await resp.json();
        renderLaunchPresets(App.presets);
    } catch (e) {
        console.error('loadLaunchPresets:', e);
    }
}

function renderLaunchPresets(presets) {
    var items = presets.fsm || [];
    var container = document.getElementById('launch-preset-list');
    if (!container) return;

    var categories = ['all'];
    for (var ci = 0; ci < items.length; ci++) {
        var cat = items[ci].category || 'other';
        if (categories.indexOf(cat) === -1) categories.push(cat);
    }
    var filterHtml = '<div class="preset-filters">';
    for (var fi = 0; fi < categories.length; fi++) {
        var c = categories[fi];
        filterHtml += '<button class="btn preset-filter-btn' + (c === 'all' ? ' btn-primary' : '') + '" data-cat="' + esc(c) + '" onclick="filterPresets(\'' + esc(c) + '\')">' + esc(c) + '</button>';
    }
    filterHtml += '</div>';
    container.innerHTML = filterHtml;

    var listDiv = document.createElement('div');
    listDiv.id = 'preset-items';
    listDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;';
    container.appendChild(listDiv);

    for (var i = 0; i < items.length; i++) {
        var p = items[i];
        var card = document.createElement('div');
        card.className = 'preset-card';
        card.setAttribute('data-category', p.category || 'other');
        card.innerHTML = '<div class="preset-name">' + esc(p.name) + '</div><div class="preset-category">' + esc(p.category || '') + '</div><div class="preset-desc">' + esc(p.description || '') + '</div>';
        (function(id, name) {
            card.addEventListener('click', function() {
                document.getElementById('launch-fsm-preset-id').value = id;
                container.querySelectorAll('.preset-card').forEach(function(c) { c.classList.remove('selected'); });
                card.classList.add('selected');
                if (!document.getElementById('launch-fsm-label').value) {
                    document.getElementById('launch-fsm-label').value = name.replace(/\s*\(.*\)/, '');
                }
            });
        })(p.id, p.name);
        listDiv.appendChild(card);
    }
}

function filterPresets(cat) {
    var cards = document.querySelectorAll('#preset-items .preset-card');
    for (var i = 0; i < cards.length; i++) {
        cards[i].style.display = (cat === 'all' || cards[i].getAttribute('data-category') === cat) ? '' : 'none';
    }
    var btns = document.querySelectorAll('.preset-filter-btn');
    for (var j = 0; j < btns.length; j++) {
        btns[j].className = btns[j].getAttribute('data-cat') === cat ? 'btn preset-filter-btn btn-primary' : 'btn preset-filter-btn';
    }
}

async function doLaunchFSM() {
    var source = document.getElementById('launch-fsm-source').value;
    var body = {
        model: document.getElementById('launch-fsm-model').value || 'ollama_chat/qwen3.5:4b',
        temperature: numVal('launch-fsm-temp', 0.5),
        label: document.getElementById('launch-fsm-label').value
    };

    if (source === 'preset') {
        body.preset_id = document.getElementById('launch-fsm-preset-id').value;
        if (!body.preset_id) {
            showError('launch-fsm-status', 'Select a preset');
            return;
        }
    } else {
        try {
            body.fsm_json = JSON.parse(document.getElementById('launch-fsm-json').value);
        } catch (e) {
            showError('launch-fsm-status', 'Invalid JSON');
            return;
        }
    }

    try {
        var resp = await fetch('/api/fsm/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        var data = await resp.json();
        if (data.error) {
            showError('launch-fsm-status', data.error);
            return;
        }
        showStatus('launch-fsm-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        refreshInstances();

        var startResp = await fetch('/api/fsm/' + encodeURIComponent(data.instance_id) + '/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_context: {} })
        });
        var startData = await startResp.json();
        if (!startData.error) {
            setTimeout(function() {
                closeLaunchModal();
                showPage('control');
                if (startData.conversation_id) {
                    showConversationInDrawer(data.instance_id, startData.conversation_id);
                }
            }, 500);
        }
    } catch (e) {
        showError('launch-fsm-status', 'Launch failed: ' + e.message);
    }
}

async function doLaunchWorkflow() {
    if (!App.capabilities.workflows) {
        showError('launch-wf-status', 'Workflow extension not installed');
        return;
    }
    var body = {
        label: document.getElementById('launch-wf-label').value
    };
    var ctxText = document.getElementById('launch-wf-context').value.trim();
    if (ctxText) {
        try {
            body.initial_context = JSON.parse(ctxText);
        } catch (e) {
            showError('launch-wf-status', 'Invalid JSON context');
            return;
        }
    }
    try {
        var resp = await fetch('/api/workflow/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        var data = await resp.json();
        if (data.error) {
            showError('launch-wf-status', data.error);
            return;
        }
        showStatus('launch-wf-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        refreshInstances();
    } catch (e) {
        showError('launch-wf-status', 'Launch failed');
    }
}

// === STUB TOOL BUILDER ===

function addStubTool() {
    var container = document.getElementById('launch-agent-tools');
    var idx = App.stubToolCount++;
    var row = document.createElement('div');
    row.className = 'stub-tool-row';
    row.id = 'stub-tool-' + idx;
    row.innerHTML =
        '<input type="text" placeholder="Name" class="stub-name">' +
        '<input type="text" placeholder="Description" class="stub-desc">' +
        '<input type="text" placeholder="Stub response" class="stub-resp" value="Tool executed successfully">' +
        '<button onclick="this.parentElement.remove()">&times;</button>';
    container.appendChild(row);
}

function getStubTools() {
    var tools = [];
    document.querySelectorAll('#launch-agent-tools .stub-tool-row').forEach(function(row) {
        var name = row.querySelector('.stub-name').value.trim();
        var desc = row.querySelector('.stub-desc').value.trim();
        var resp = row.querySelector('.stub-resp').value.trim();
        if (name && desc) {
            tools.push({ name: name, description: desc, stub_response: resp || 'Tool executed successfully' });
        }
    });
    return tools;
}

function onAgentTypeChange() {
    var agentType = document.getElementById('launch-agent-type').value;
    var needsTools = App.TOOL_BASED_AGENTS.indexOf(agentType) !== -1;
    var toolSection = document.getElementById('launch-agent-tools');
    var addToolBtn = toolSection ? toolSection.nextElementSibling : null;
    var toolTitle = toolSection ? toolSection.previousElementSibling : null;
    if (toolSection) toolSection.style.display = needsTools ? '' : 'none';
    if (addToolBtn && addToolBtn.tagName === 'BUTTON') addToolBtn.style.display = needsTools ? '' : 'none';
    if (toolTitle && toolTitle.classList.contains('panel-title')) toolTitle.style.display = needsTools ? '' : 'none';
}

async function doLaunchAgent() {
    if (!App.capabilities.agents) {
        showError('launch-agent-status', 'Agent extension not installed');
        return;
    }
    var agentType = document.getElementById('launch-agent-type').value;
    var needsTools = App.TOOL_BASED_AGENTS.indexOf(agentType) !== -1;
    var tools = needsTools ? getStubTools() : [];
    if (needsTools && tools.length === 0) {
        showError('launch-agent-status', 'Add at least one tool for ' + agentType);
        return;
    }
    var task = document.getElementById('launch-agent-task').value.trim();
    if (!task) {
        showError('launch-agent-status', 'Enter a task');
        return;
    }
    var body = {
        agent_type: agentType,
        task: task,
        model: document.getElementById('launch-agent-model').value || 'ollama_chat/qwen3.5:4b',
        max_iterations: intVal('launch-agent-iters', 10),
        tools: tools,
        label: document.getElementById('launch-agent-label').value
    };
    try {
        var resp = await fetch('/api/agent/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        var data = await resp.json();
        if (data.error) {
            showError('launch-agent-status', data.error);
            return;
        }
        showStatus('launch-agent-status', 'Launched: ' + (data.label || data.instance_id), 'success');
        refreshInstances();
        setTimeout(function() {
            closeLaunchModal();
            showPage('control');
        }, 500);
    } catch (e) {
        showError('launch-agent-status', 'Launch failed');
    }
}
