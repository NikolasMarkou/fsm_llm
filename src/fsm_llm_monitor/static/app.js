// FSM-LLM Monitor — Retro Terminal Dashboard
// Vanilla JS — no frameworks needed

let ws = null;
let currentPage = 'dashboard';

// === NAV ===

function showPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-items button').forEach(b => b.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    const btn = document.querySelector(`.sidebar-items button[data-page="${page}"]`);
    if (btn) btn.classList.add('active');
    currentPage = page;

    // Refresh page-specific data
    if (page === 'conversations') refreshConversationList();
    if (page === 'logs') refreshLogs();
    if (page === 'settings') loadSettings();
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
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
        if (data.type === 'metrics') {
            updateMetrics(data.data);
        }
        if (data.events) {
            updateEvents(data.events);
        }
    };

    ws.onclose = () => {
        document.getElementById('ws-status').textContent = 'DISCONNECTED';
        document.getElementById('ws-status').classList.add('blink');
        document.getElementById('ws-status').classList.remove('connected');
        setTimeout(connectWS, 3000);
    };

    ws.onerror = () => {
        ws.close();
    };
}

// === DASHBOARD ===

function updateMetrics(m) {
    document.getElementById('m-conversations').textContent = m.active_conversations;
    document.getElementById('m-events').textContent = m.total_events;
    document.getElementById('m-transitions').textContent = m.total_transitions;
    document.getElementById('m-errors').textContent = m.total_errors;

    // State distribution
    const dist = document.getElementById('state-dist');
    if (m.states_visited && Object.keys(m.states_visited).length > 0) {
        const sorted = Object.entries(m.states_visited).sort((a, b) => b[1] - a[1]);
        dist.innerHTML = sorted.map(([s, c]) =>
            `<div style="display:flex;justify-content:space-between;padding:2px 0;">` +
            `<span>${esc(s)}</span><span style="color:var(--green-bright)">${c}</span></div>`
        ).join('');
    }

    // Refresh conversation table
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
    // Cap at 200 entries
    while (log.children.length > 200) {
        log.removeChild(log.lastChild);
    }
}

async function refreshConversationTable() {
    try {
        const resp = await fetch('/api/conversations');
        const convs = await resp.json();
        const body = document.getElementById('conv-table-body');
        const empty = document.getElementById('conv-empty');

        if (convs.length === 0) {
            body.innerHTML = '';
            empty.style.display = 'block';
            return;
        }
        empty.style.display = 'none';
        body.innerHTML = convs.map(c =>
            `<tr><td>${esc(c.conversation_id.substring(0, 16))}</td>` +
            `<td>${esc(c.current_state)}</td>` +
            `<td>${c.message_history.length}</td>` +
            `<td>${c.stack_depth}</td>` +
            `<td><span class="badge ${c.is_terminal ? 'badge-ended' : 'badge-active'}">${c.is_terminal ? 'ENDED' : 'ACTIVE'}</span></td></tr>`
        ).join('');
    } catch (e) {}
}

// === FSM VIEWER ===

async function loadFSM() {
    const path = document.getElementById('fsm-path').value.trim();
    if (!path) return;

    try {
        const resp = await fetch('/api/fsm/load?path=' + encodeURIComponent(path));
        const fsm = await resp.json();
        if (fsm.error) {
            document.getElementById('fsm-details').innerHTML = `<span class="key" style="color:var(--red);">Error:</span><span class="val" style="color:var(--red);">${esc(fsm.error)}</span>`;
            return;
        }
        renderFSM(fsm);
    } catch (e) {
        document.getElementById('fsm-details').innerHTML = `<span class="key" style="color:var(--red);">Error:</span><span class="val" style="color:var(--red);">Failed to load</span>`;
    }
}

function renderFSM(fsm) {
    // Details
    document.getElementById('fsm-details').innerHTML =
        `<span class="key">Name:</span><span class="val">${esc(fsm.name)}</span>` +
        `<span class="key">Description:</span><span class="val">${esc(fsm.description)}</span>` +
        `<span class="key">Version:</span><span class="val">${esc(fsm.version)}</span>` +
        `<span class="key">Initial State:</span><span class="val">${esc(fsm.initial_state)}</span>` +
        `<span class="key">Persona:</span><span class="val">${esc(fsm.persona || 'None')}</span>` +
        `<span class="key">States:</span><span class="val">${fsm.state_count}</span>`;

    // Tree
    const tree = document.getElementById('fsm-tree');
    tree.innerHTML = `<div style="color:var(--green-bright);font-weight:bold;margin-bottom:8px;">${esc(fsm.name)} (v${esc(fsm.version)})</div>`;
    fsm.states.forEach(s => {
        let marker = '';
        if (s.is_initial) marker = ' <span class="initial">[>>]</span>';
        else if (s.is_terminal) marker = ' <span class="terminal">[XX]</span>';

        let node = `<div class="tree-node">${esc(s.state_id)}${marker}</div>`;
        s.transitions.forEach(t => {
            node += `<div class="tree-node" style="padding-left:40px;"><span class="arrow">-> </span>${esc(t.target_state)} <span style="color:var(--green-dim);">(p=${t.priority})</span></div>`;
        });
        tree.innerHTML += node;
    });

    // Transitions table
    const tbody = document.getElementById('fsm-trans-body');
    tbody.innerHTML = '';
    fsm.states.forEach(s => {
        s.transitions.forEach(t => {
            tbody.innerHTML += `<tr><td>${esc(s.state_id)}</td><td>${esc(t.target_state)}</td>` +
                `<td>${t.priority}</td><td>${esc(t.description.substring(0, 50))}</td>` +
                `<td>${t.condition_count}</td><td>${t.has_logic ? 'Yes' : 'No'}</td></tr>`;
        });
    });
}

// === CONVERSATIONS ===

async function refreshConversationList() {
    try {
        const resp = await fetch('/api/conversations');
        const convs = await resp.json();
        const select = document.getElementById('conv-select');
        const current = select.value;
        select.innerHTML = '<option value="">-- Select --</option>';
        convs.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.conversation_id;
            opt.textContent = c.conversation_id.substring(0, 24) + ' [' + c.current_state + ']';
            select.appendChild(opt);
        });
        if (current) select.value = current;
    } catch (e) {}
}

async function loadConversation(id) {
    if (!id) return;
    try {
        const resp = await fetch('/api/conversations/' + encodeURIComponent(id));
        const c = await resp.json();
        if (c.error) return;

        // State
        const status = c.is_terminal ? '<span class="badge badge-ended">ENDED</span>' : '<span class="badge badge-active">ACTIVE</span>';
        document.getElementById('conv-state').innerHTML =
            `<span class="key">State:</span><span class="val">${esc(c.current_state)} ${status}</span>` +
            `<span class="key">Description:</span><span class="val">${esc(c.state_description)}</span>` +
            `<span class="key">Stack Depth:</span><span class="val">${c.stack_depth}</span>` +
            `<span class="key">ID:</span><span class="val">${esc(c.conversation_id)}</span>`;

        // Context
        const ctx = document.getElementById('conv-ctx-body');
        ctx.innerHTML = '';
        Object.entries(c.context_data).sort().forEach(([k, v]) => {
            const vs = String(v).substring(0, 60);
            ctx.innerHTML += `<tr><td>${esc(k)}</td><td>${esc(vs)}</td></tr>`;
        });

        // Messages
        const msgs = document.getElementById('conv-messages');
        msgs.innerHTML = '';
        c.message_history.forEach(ex => {
            Object.entries(ex).forEach(([role, msg]) => {
                const color = role === 'user' ? 'var(--cyan)' : 'var(--green)';
                msgs.innerHTML += `<div class="entry"><span class="type" style="color:${color}">${role.toUpperCase()}</span><span class="msg">${esc(msg)}</span></div>`;
            });
            msgs.innerHTML += '<div style="border-bottom:1px solid var(--border);margin:4px 0;"></div>';
        });

        // LLM
        const llm = document.getElementById('conv-llm');
        let parts = [];
        if (c.last_extraction) parts.push(`<span class="key">Confidence:</span><span class="val">${c.last_extraction.confidence || 'N/A'}</span>`);
        if (c.last_transition) parts.push(`<span class="key">Transition:</span><span class="val">${esc(c.last_transition.selected_transition || 'N/A')}</span>`);
        if (c.last_response) parts.push(`<span class="key">Response Type:</span><span class="val">${esc(c.last_response.message_type || 'N/A')}</span>`);
        llm.innerHTML = parts.length > 0 ? parts.join('') : '<span class="key">Status:</span><span class="val">No data</span>';
    } catch (e) {}
}

// === LOGS ===

async function refreshLogs() {
    const level = document.getElementById('log-level').value;
    const filter = document.getElementById('log-filter').value.trim().toLowerCase();

    try {
        const resp = await fetch(`/api/logs?limit=500&level=${level}`);
        let logs = await resp.json();

        if (filter) {
            logs = logs.filter(r => r.message.toLowerCase().includes(filter));
        }

        const stream = document.getElementById('log-stream');
        stream.innerHTML = '';

        const colors = { DEBUG: 'var(--green-dim)', INFO: 'var(--green)', WARNING: 'var(--yellow)', ERROR: 'var(--red)', CRITICAL: 'var(--red)' };

        // Reverse to show oldest first
        logs.reverse().forEach(r => {
            const ts = formatTime(r.timestamp);
            const c = colors[r.level] || 'var(--green)';
            const conv = r.conversation_id ? ` [${r.conversation_id}]` : '';
            stream.innerHTML += `<div class="entry"><span class="ts" style="color:${c}">${ts}</span>` +
                `<span class="type" style="color:${c};width:70px;">${r.level}</span>` +
                `<span class="msg" style="color:var(--green-dim)">${esc(r.module)}:${r.line}${conv} ${esc(r.message)}</span></div>`;
        });

        document.getElementById('log-stats').textContent = `Total: ${logs.length} | Level: >= ${level}`;
    } catch (e) {}
}

function clearLogs() {
    document.getElementById('log-stream').innerHTML = '';
}

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
        Object.entries(info).forEach(([k, v]) => {
            el.innerHTML += `<span class="key">${esc(k.replace(/_/g, ' '))}:</span><span class="val">${esc(v)}</span>`;
        });
        document.getElementById('version-info').textContent = `v${info.monitor_version}`;
    } catch (e) {}
}

async function saveSettings() {
    try {
        const cfg = {
            refresh_interval: parseFloat(document.getElementById('set-refresh').value),
            max_events: parseInt(document.getElementById('set-max-events').value),
            max_log_lines: parseInt(document.getElementById('set-max-logs').value),
            log_level: document.getElementById('set-level').value,
            show_internal_keys: false,
            auto_scroll_logs: true,
        };
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cfg),
        });
        document.getElementById('conn-status').innerHTML = '<span style="color:var(--green);">Settings saved</span>';
    } catch (e) {
        document.getElementById('conn-status').innerHTML = '<span style="color:var(--red);">Save failed</span>';
    }
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

// === CLOCK ===

function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', { hour12: false });
}

// === KEYBOARD SHORTCUTS ===

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
        case 'd': showPage('dashboard'); break;
        case 'f': showPage('fsm'); break;
        case 'c': showPage('conversations'); break;
        case 'a': showPage('agents'); break;
        case 'w': showPage('workflows'); break;
        case 'l': showPage('logs'); break;
        case 's': showPage('settings'); break;
    }
});

// === INIT ===

connectWS();
loadSettings();
setInterval(updateClock, 1000);
updateClock();
