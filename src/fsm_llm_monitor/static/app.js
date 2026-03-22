// FSM-LLM Monitor — Retro Terminal Dashboard
// Vanilla JS — no frameworks needed

let ws = null;
let currentPage = 'dashboard';
let activeChatSession = '';

// === NAV ===

function showPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-items button').forEach(b => b.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    const btn = document.querySelector(`.sidebar-items button[data-page="${page}"]`);
    if (btn) btn.classList.add('active');
    currentPage = page;

    // Refresh page-specific data
    if (page === 'fsm-launch') refreshFSMSessions();
    if (page === 'fsm-chat') refreshChatSessions();
    if (page === 'fsm-presets') loadFSMPresets();
    if (page === 'agent-jobs') refreshAgentJobs();
    if (page === 'agent-presets') loadAgentPresets();
    if (page === 'wf-presets') loadWorkflowPresets();
    if (page === 'logs') refreshLogs();
    if (page === 'settings') loadSettings();
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

function toggleSection(id) {
    const el = document.getElementById(id);
    el.classList.toggle('collapsed');
}

// === WEBSOCKET ===

function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen = () => {
        document.getElementById('ws-status').textContent = 'CONNECTED';
        document.getElementById('ws-status').classList.remove('blink');
        document.getElementById('ws-status').classList.add('connected');
    };
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'metrics') updateMetrics(data.data);
        if (data.events) updateEvents(data.events);
    };
    ws.onclose = () => {
        document.getElementById('ws-status').textContent = 'DISCONNECTED';
        document.getElementById('ws-status').classList.add('blink');
        document.getElementById('ws-status').classList.remove('connected');
        setTimeout(connectWS, 3000);
    };
    ws.onerror = () => ws.close();
}

// === DASHBOARD ===

function updateMetrics(m) {
    document.getElementById('m-conversations').textContent = m.active_conversations;
    document.getElementById('m-events').textContent = m.total_events;
    document.getElementById('m-transitions').textContent = m.total_transitions;
    document.getElementById('m-errors').textContent = m.total_errors;
    refreshConversationTable();
}

function updateEvents(events) {
    const log = document.getElementById('event-log');
    events.forEach(e => {
        const ts = formatTime(e.timestamp);
        const level = (e.level || 'INFO').toLowerCase();
        const div = document.createElement('div');
        div.className = 'entry ' + level;
        div.innerHTML = `<span class="ts">${ts}</span><span class="type">${esc(e.event_type)}</span><span class="msg">${esc(e.message)}</span>`;
        log.insertBefore(div, log.firstChild);
    });
    while (log.children.length > 200) log.removeChild(log.lastChild);
}

async function refreshConversationTable() {
    try {
        const resp = await fetch('/api/conversations');
        const convs = await resp.json();
        const body = document.getElementById('conv-table-body');
        const empty = document.getElementById('conv-empty');
        if (convs.length === 0) { body.innerHTML = ''; empty.style.display = 'block'; return; }
        empty.style.display = 'none';
        body.innerHTML = convs.map(c =>
            `<tr><td>${esc(c.conversation_id.substring(0, 16))}</td>` +
            `<td>${esc(c.current_state)}</td><td>${c.message_history.length}</td>` +
            `<td><span class="badge ${c.is_terminal ? 'badge-ended' : 'badge-active'}">${c.is_terminal ? 'ENDED' : 'ACTIVE'}</span></td></tr>`
        ).join('');
    } catch (e) {}
}

// === FSM: LAUNCH ===

async function launchFSM() {
    const path = document.getElementById('launch-fsm-path').value.trim();
    if (!path) return;
    const model = document.getElementById('launch-fsm-model').value.trim() || 'gpt-4o-mini';
    const temp = parseFloat(document.getElementById('launch-fsm-temp').value) || 0.5;
    const tokens = parseInt(document.getElementById('launch-fsm-tokens').value) || 1000;
    const status = document.getElementById('launch-fsm-status');
    status.innerHTML = '<span class="blink" style="color:var(--yellow);">LAUNCHING...</span>';

    try {
        const resp = await fetch('/api/launch/fsm', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fsm_path: path, model, temperature: temp, max_tokens: tokens }),
        });
        const data = await resp.json();
        if (data.error) { status.innerHTML = `<span style="color:var(--red);">ERROR: ${esc(data.error)}</span>`; return; }
        status.innerHTML = `<span style="color:var(--green);">STARTED: ${esc(data.conversation_id.substring(0,12))}</span>`;
        activeChatSession = data.conversation_id;
        refreshFSMSessions();
        showPage('fsm-chat');
        refreshChatSessions();
        switchChatSession(data.conversation_id);
        appendChatMessage('SYSTEM', data.initial_response);
    } catch (e) {
        status.innerHTML = `<span style="color:var(--red);">FAILED: ${esc(e.message)}</span>`;
    }
}

async function refreshFSMSessions() {
    try {
        const resp = await fetch('/api/fsm/sessions');
        const sessions = await resp.json();
        const body = document.getElementById('launch-fsm-sessions');
        const empty = document.getElementById('launch-fsm-empty');
        if (sessions.length === 0) { body.innerHTML = ''; empty.style.display = 'block'; return; }
        empty.style.display = 'none';
        body.innerHTML = sessions.map(s =>
            `<tr><td>${esc(s.conversation_id.substring(0,12))}</td><td>${esc(s.state)}</td>` +
            `<td><span class="badge ${s.ended ? 'badge-ended' : 'badge-active'}">${s.ended ? 'ENDED' : 'ACTIVE'}</span></td>` +
            `<td><button class="btn" style="padding:2px 8px;font-size:11px;" onclick="showPage('fsm-chat');switchChatSession('${s.conversation_id}')">CHAT</button></td></tr>`
        ).join('');
    } catch (e) {}
}

// === FSM: CHAT ===

async function refreshChatSessions() {
    try {
        const resp = await fetch('/api/fsm/sessions');
        const sessions = await resp.json();
        const select = document.getElementById('chat-session');
        const current = select.value;
        select.innerHTML = '<option value="">-- Select Session --</option>';
        sessions.filter(s => !s.ended).forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.conversation_id;
            opt.textContent = s.conversation_id.substring(0, 16) + ' [' + s.state + ']';
            select.appendChild(opt);
        });
        if (current) select.value = current;
    } catch (e) {}
}

function switchChatSession(convId) {
    activeChatSession = convId;
    document.getElementById('chat-session').value = convId;
    document.getElementById('chat-messages').innerHTML = '';
    document.getElementById('chat-state').textContent = convId ? 'Session: ' + convId.substring(0, 12) : 'No active session';
    document.getElementById('chat-state').style.color = convId ? 'var(--green)' : 'var(--green-dim)';
}

async function sendChat() {
    if (!activeChatSession) return;
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    appendChatMessage('YOU', msg);

    try {
        const resp = await fetch(`/api/fsm/${encodeURIComponent(activeChatSession)}/converse`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg }),
        });
        const data = await resp.json();
        if (data.error) { appendChatMessage('ERROR', data.error); return; }
        appendChatMessage('FSM', data.response);
        document.getElementById('chat-state').textContent = `State: ${data.state}` + (data.ended ? ' [ENDED]' : '');
        if (data.ended) document.getElementById('chat-state').style.color = 'var(--red)';
    } catch (e) { appendChatMessage('ERROR', e.message); }
}

async function endChat() {
    if (!activeChatSession) return;
    try {
        await fetch(`/api/fsm/${encodeURIComponent(activeChatSession)}/end`, { method: 'POST' });
        appendChatMessage('SYSTEM', 'Conversation ended.');
        document.getElementById('chat-state').textContent = 'ENDED';
        document.getElementById('chat-state').style.color = 'var(--red)';
        activeChatSession = '';
        refreshChatSessions();
        refreshFSMSessions();
    } catch (e) {}
}

function appendChatMessage(role, text) {
    const log = document.getElementById('chat-messages');
    const colors = { YOU: 'var(--cyan)', FSM: 'var(--green)', SYSTEM: 'var(--yellow)', ERROR: 'var(--red)' };
    const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
    log.innerHTML += `<div class="entry"><span class="ts">${ts}</span><span class="type" style="color:${colors[role] || 'var(--green)'};width:60px;">${role}</span><span class="msg">${esc(text)}</span></div>`;
    log.scrollTop = log.scrollHeight;
}

// === FSM: VISUALIZER ===

async function visualizeFSM(pathOverride) {
    const path = pathOverride || document.getElementById('viz-fsm-path').value.trim();
    if (!path) return;
    if (!pathOverride) document.getElementById('viz-fsm-path').value = path;

    try {
        const resp = await fetch('/api/fsm/visualize?path=' + encodeURIComponent(path));
        const data = await resp.json();
        if (data.error) return;
        renderFSMGraph(data);

        // Info panel
        const f = data.fsm;
        document.getElementById('viz-info').innerHTML =
            `<span class="key">Name:</span><span class="val">${esc(f.name)}</span>` +
            `<span class="key">Description:</span><span class="val">${esc(f.description)}</span>` +
            `<span class="key">Version:</span><span class="val">${esc(f.version)}</span>` +
            `<span class="key">Initial:</span><span class="val">${esc(f.initial_state)}</span>` +
            `<span class="key">States:</span><span class="val">${f.state_count}</span>`;

        // Transitions
        const tbody = document.getElementById('viz-trans-body');
        tbody.innerHTML = '';
        data.edges.forEach(e => {
            tbody.innerHTML += `<tr><td>${esc(e.from)}</td><td>${esc(e.to)}</td><td>${e.priority}</td><td>${esc(e.label)}</td></tr>`;
        });
    } catch (e) {}
}

function renderFSMGraph(data) {
    const svg = document.getElementById('viz-svg');
    const nodes = data.nodes;
    const edges = data.edges;

    // Calculate layout
    const W = 160, H = 50, PAD = 40;
    const cols = Math.min(nodes.length, 4);
    const svgW = cols * (W + PAD) + PAD;
    const rows = Math.ceil(nodes.length / cols);
    const svgH = rows * (H + 80) + 60;
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);

    // Assign positions
    nodes.forEach((n, i) => {
        n.x = PAD + (i % cols) * (W + PAD) + W / 2;
        n.y = 40 + Math.floor(i / cols) * (H + 80) + H / 2;
    });

    const nodeMap = {};
    nodes.forEach(n => nodeMap[n.id] = n);

    let html = `<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="var(--green-dim)"/></marker></defs>`;

    // Draw edges
    edges.forEach(e => {
        const from = nodeMap[e.from];
        const to = nodeMap[e.to];
        if (!from || !to) return;
        const dx = to.x - from.x, dy = to.y - from.y;
        const len = Math.sqrt(dx*dx + dy*dy) || 1;
        const x1 = from.x + (dx/len) * (W/2);
        const y1 = from.y + (dy/len) * (H/2);
        const x2 = to.x - (dx/len) * (W/2 + 8);
        const y2 = to.y - (dy/len) * (H/2 + 8);
        html += `<line class="edge-line" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
        if (e.label) {
            html += `<text class="edge-label" x="${(x1+x2)/2}" y="${(y1+y2)/2 - 6}">${esc(e.label)}</text>`;
        }
    });

    // Draw nodes
    nodes.forEach(n => {
        const cls = n.is_initial ? 'initial' : n.is_terminal ? 'terminal' : '';
        html += `<rect class="node-rect ${cls}" x="${n.x - W/2}" y="${n.y - H/2}" width="${W}" height="${H}" rx="4"/>`;
        html += `<text class="node-label" x="${n.x}" y="${n.y}">${esc(n.id)}</text>`;
    });

    svg.innerHTML = html;
}

// === FSM: PRESETS ===

let _presets = null;

async function loadPresets() {
    if (_presets) return _presets;
    try {
        const resp = await fetch('/api/presets');
        _presets = await resp.json();
        return _presets;
    } catch (e) { return { fsm: [], agent: [], workflow: [] }; }
}

async function loadFSMPresets() {
    const presets = await loadPresets();
    const container = document.getElementById('fsm-presets-list');
    const empty = document.getElementById('fsm-presets-empty');
    if (presets.fsm.length === 0) { empty.textContent = 'No FSM presets found in examples/'; return; }
    empty.style.display = 'none';
    container.innerHTML = presets.fsm.map(p =>
        `<div class="preset-card" onclick="useFSMPreset('${esc(p.path)}')">` +
        `<div class="preset-name">${esc(p.name)}</div>` +
        `<div class="preset-category">${esc(p.category)}</div>` +
        `<div class="preset-desc">${esc(p.description)}</div>` +
        `<div class="preset-path">${esc(p.path)}</div></div>`
    ).join('');
}

function useFSMPreset(path) {
    document.getElementById('launch-fsm-path').value = path;
    showPage('fsm-launch');
}

async function loadAgentPresets() {
    const presets = await loadPresets();
    const container = document.getElementById('agent-presets-list');
    const empty = document.getElementById('agent-presets-empty');
    if (presets.agent.length === 0) { empty.textContent = 'No agent presets found'; return; }
    empty.style.display = 'none';
    container.innerHTML = presets.agent.map(p =>
        `<div class="preset-card" onclick="useAgentPreset('${esc(p.id)}')">` +
        `<div class="preset-name">${esc(p.name)}</div>` +
        `<div class="preset-category">AGENT</div>` +
        `<div class="preset-desc">${esc(p.description)}</div>` +
        `<div class="preset-path">${esc(p.path)}</div></div>`
    ).join('');
}

function useAgentPreset(id) {
    document.getElementById('launch-agent-task').value = `Run the ${id.replace(/_/g, ' ')} example`;
    showPage('agent-launch');
}

async function loadWorkflowPresets() {
    const presets = await loadPresets();
    const container = document.getElementById('wf-presets-list');
    const empty = document.getElementById('wf-presets-empty');
    if (presets.workflow.length === 0) { empty.textContent = 'No workflow presets found'; return; }
    empty.style.display = 'none';
    container.innerHTML = presets.workflow.map(p =>
        `<div class="preset-card" onclick="useWorkflowPreset('${esc(p.id)}')">` +
        `<div class="preset-name">${esc(p.name)}</div>` +
        `<div class="preset-category">WORKFLOW</div>` +
        `<div class="preset-path">${esc(p.path)}</div></div>`
    ).join('');
}

function useWorkflowPreset(id) {
    document.getElementById('launch-wf-id').value = id;
    showPage('wf-launch');
}

// === AGENTS: LAUNCH ===

async function launchAgent() {
    const task = document.getElementById('launch-agent-task').value.trim();
    if (!task) return;
    const model = document.getElementById('launch-agent-model').value.trim() || 'gpt-4o-mini';
    const maxIter = parseInt(document.getElementById('launch-agent-iter').value) || 10;
    const temp = parseFloat(document.getElementById('launch-agent-temp').value) || 0.5;
    const timeout = parseFloat(document.getElementById('launch-agent-timeout').value) || 300;
    const status = document.getElementById('launch-agent-status');
    status.innerHTML = '<span class="blink" style="color:var(--yellow);">LAUNCHING...</span>';

    try {
        const resp = await fetch('/api/launch/agent', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task, model, max_iterations: maxIter, temperature: temp, timeout_seconds: timeout }),
        });
        const data = await resp.json();
        if (data.error) { status.innerHTML = `<span style="color:var(--red);">ERROR: ${esc(data.error)}</span>`; return; }
        status.innerHTML = `<span style="color:var(--green);">JOB: ${esc(data.job_id)} — RUNNING</span>`;
        pollAgentJob(data.job_id);
    } catch (e) {
        status.innerHTML = `<span style="color:var(--red);">FAILED: ${esc(e.message)}</span>`;
    }
}

async function pollAgentJob(jobId) {
    const poll = async () => {
        try {
            const resp = await fetch(`/api/agent/${jobId}`);
            const data = await resp.json();
            if (data.status === 'running') setTimeout(poll, 2000);
            else { refreshAgentJobs(); document.getElementById('launch-agent-status').innerHTML = `<span style="color:${data.status === 'completed' ? 'var(--green)' : 'var(--red)'};">${data.status.toUpperCase()}: ${esc((data.answer || data.error || '').substring(0, 60))}</span>`; }
        } catch (e) {}
    };
    setTimeout(poll, 2000);
}

async function refreshAgentJobs() {
    try {
        const resp = await fetch('/api/agent/jobs');
        const jobs = await resp.json();
        const body = document.getElementById('agent-jobs-body');
        const empty = document.getElementById('agent-jobs-empty');
        if (jobs.length === 0) { body.innerHTML = ''; empty.style.display = 'block'; return; }
        empty.style.display = 'none';
        body.innerHTML = jobs.map(j => {
            const sc = j.status === 'completed' ? 'var(--green)' : j.status === 'failed' ? 'var(--red)' : 'var(--yellow)';
            return `<tr onclick="showAgentDetail('${j.job_id}')" style="cursor:pointer;">` +
                `<td>${esc(j.job_id)}</td><td>${esc(j.type || 'react')}</td>` +
                `<td>${esc((j.task||'').substring(0, 30))}</td>` +
                `<td style="color:${sc}">${j.status.toUpperCase()}</td>` +
                `<td>${j.iterations_used || '-'}</td><td>${(j.tools_used || []).join(', ') || '-'}</td>` +
                `<td>${esc((j.answer || j.error || '-').substring(0, 40))}</td></tr>`;
        }).join('');
    } catch (e) {}
}

async function showAgentDetail(jobId) {
    try {
        const resp = await fetch(`/api/agent/${jobId}`);
        const j = await resp.json();
        const el = document.getElementById('agent-job-detail');
        el.innerHTML =
            `<span class="key">Job ID:</span><span class="val">${esc(j.job_id || jobId)}</span>` +
            `<span class="key">Status:</span><span class="val">${esc(j.status)}</span>` +
            `<span class="key">Task:</span><span class="val">${esc(j.task)}</span>` +
            `<span class="key">Answer:</span><span class="val">${esc(j.answer || '-')}</span>` +
            `<span class="key">Iterations:</span><span class="val">${j.iterations_used || '-'}</span>` +
            `<span class="key">Tools Used:</span><span class="val">${(j.tools_used || []).join(', ') || '-'}</span>` +
            (j.error ? `<span class="key">Error:</span><span class="val" style="color:var(--red)">${esc(j.error)}</span>` : '');
    } catch (e) {}
}

// === WORKFLOWS: LAUNCH ===

async function launchWorkflow() {
    const wfId = document.getElementById('launch-wf-id').value.trim() || 'demo';
    const ctxRaw = document.getElementById('launch-wf-ctx').value.trim();
    let ctx = null;
    if (ctxRaw) { try { ctx = JSON.parse(ctxRaw); } catch (e) {} }
    const status = document.getElementById('launch-wf-status');
    status.innerHTML = '<span class="blink" style="color:var(--yellow);">LAUNCHING...</span>';

    try {
        const resp = await fetch('/api/launch/workflow', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workflow_id: wfId, initial_context: ctx }),
        });
        const data = await resp.json();
        if (data.error) { status.innerHTML = `<span style="color:var(--red);">ERROR: ${esc(data.error)}</span>`; return; }
        status.innerHTML = `<span style="color:var(--green);">INSTANCE: ${esc(data.instance_id)} — STARTED</span>`;
    } catch (e) {
        status.innerHTML = `<span style="color:var(--red);">FAILED: ${esc(e.message)}</span>`;
    }
}

// === LOGS ===

async function refreshLogs() {
    const level = document.getElementById('log-level').value;
    const filter = document.getElementById('log-filter').value.trim().toLowerCase();
    try {
        const resp = await fetch(`/api/logs?limit=500&level=${level}`);
        let logs = await resp.json();
        if (filter) logs = logs.filter(r => r.message.toLowerCase().includes(filter));
        const stream = document.getElementById('log-stream');
        stream.innerHTML = '';
        const colors = { DEBUG: 'var(--green-dim)', INFO: 'var(--green)', WARNING: 'var(--yellow)', ERROR: 'var(--red)', CRITICAL: 'var(--red)' };
        logs.reverse().forEach(r => {
            const ts = formatTime(r.timestamp);
            const c = colors[r.level] || 'var(--green)';
            const conv = r.conversation_id ? ` [${r.conversation_id}]` : '';
            stream.innerHTML += `<div class="entry"><span class="ts" style="color:${c}">${ts}</span><span class="type" style="color:${c};width:70px;">${r.level}</span><span class="msg" style="color:var(--green-dim)">${esc(r.module)}:${r.line}${conv} ${esc(r.message)}</span></div>`;
        });
        document.getElementById('log-stats').textContent = `Total: ${logs.length} | Level: >= ${level}`;
    } catch (e) {}
}

function clearLogs() { document.getElementById('log-stream').innerHTML = ''; }

// === SETTINGS ===

async function loadSettings() {
    try {
        const resp = await fetch('/api/config');
        const cfg = await resp.json();
        document.getElementById('set-refresh').value = cfg.refresh_interval;
        document.getElementById('set-max-events').value = cfg.max_events;
        document.getElementById('set-max-logs').value = cfg.max_log_lines;
        document.getElementById('set-level').value = cfg.log_level;
    } catch (e) {}
    try {
        const resp = await fetch('/api/info');
        const info = await resp.json();
        const el = document.getElementById('sys-info');
        el.innerHTML = '';
        Object.entries(info).forEach(([k, v]) => { el.innerHTML += `<span class="key">${esc(k.replace(/_/g, ' '))}:</span><span class="val">${esc(v)}</span>`; });
        document.getElementById('version-info').textContent = `v${info.monitor_version}`;
    } catch (e) {}
}

async function saveSettings() {
    try {
        await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                refresh_interval: parseFloat(document.getElementById('set-refresh').value),
                max_events: parseInt(document.getElementById('set-max-events').value),
                max_log_lines: parseInt(document.getElementById('set-max-logs').value),
                log_level: document.getElementById('set-level').value,
                show_internal_keys: false, auto_scroll_logs: true,
            }),
        });
        document.getElementById('conn-status').innerHTML = '<span style="color:var(--green);">Settings saved</span>';
    } catch (e) { document.getElementById('conn-status').innerHTML = '<span style="color:var(--red);">Save failed</span>'; }
}

function resetSettings() {
    document.getElementById('set-refresh').value = '1.0';
    document.getElementById('set-max-events').value = '1000';
    document.getElementById('set-max-logs').value = '5000';
    document.getElementById('set-level').value = 'INFO';
}

// === UTILS ===

function esc(s) {
    if (s === null || s === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(s);
    return div.innerHTML;
}

function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    if (isNaN(d)) return String(ts).substring(11, 19);
    return d.toLocaleTimeString('en-US', { hour12: false });
}

function updateClock() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
}

// === KEYBOARD SHORTCUTS ===

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
        case '1': showPage('dashboard'); break;
        case '2': showPage('fsm-launch'); break;
        case '3': showPage('agent-launch'); break;
        case '4': showPage('wf-launch'); break;
    }
});

// === INIT ===

connectWS();
loadSettings();
setInterval(updateClock, 1000);
updateClock();
