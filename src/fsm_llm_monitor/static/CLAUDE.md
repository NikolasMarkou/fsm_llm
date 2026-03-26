# fsm_llm_monitor/static — Frontend (PURE Design, ES Modules)

## What This Is

Vanilla JS frontend for the FSM-LLM monitoring dashboard, following PURE design principles (framework-less, web-standards-first). Uses ES modules with `import`/`export`, Proxy-based reactive state, and event delegation. No framework, no build step, no npm. Single `<script type="module">` entry point.

## Architecture

```
static/
├── app.js              # Main entry: boot, navigation, event delegation, keyboard shortcuts
├── style.css           # Grafana-inspired dark theme with CSS custom properties
├── flows.json          # Agent/workflow pattern flow definitions for visualizer
├── services/
│   ├── state.js        # Proxy-based reactive state (replaces global App namespace)
│   ├── api.js          # HTTP client (fetchJson, postJson)
│   └── ws.js           # WebSocket connection, reconnect, message dispatch
├── utils/
│   ├── dom.js          # DOM helpers: esc(), $(), showToast, statusBadge, etc.
│   ├── format.js       # formatTime, relativeTime, formatNumber
│   ├── markdown.js     # Lightweight Markdown → HTML renderer (XSS-safe)
│   └── graph.js        # BFS graph layout + SVG rendering
└── pages/
    ├── dashboard.js    # Metric cards, instance grid, conversation table
    ├── control.js      # Unified instance table, drawer, FSM/workflow/agent detail
    ├── conversations.js # Conversation detail + chat interface
    ├── visualizer.js   # FSM/agent/workflow graph viewer with split pane
    ├── logs.js         # Level-filtered log stream with live/pause
    ├── builder.js      # Meta-agent conversational artifact builder
    ├── settings.js     # Runtime config + system info
    └── launch.js       # Launch modal for FSM/workflow/agent instances
```

## Key Patterns

### ES Modules with Explicit Imports
All files use ES module `import`/`export`. Dependencies are explicit. No global namespace pollution. The HTML loads a single `<script type="module" src="/static/app.js">`.

### Proxy-Based Reactive State
`services/state.js` exports a `Proxy`-wrapped state object. All mutations emit `statechange` events via an internal `EventTarget`. Modules import `state` and read/write properties directly.

### Event Delegation
All user interactions use `data-action` attributes in HTML instead of inline `onclick` handlers. A single `click` listener in `app.js` dispatches to the appropriate handler via an `ACTIONS` map. Similarly, `input` and `change` events are delegated by element ID.

### Dependency Injection via `setDeps()`
Page modules that need cross-module references (e.g., `showPage`, `refreshInstances`) receive them via a `setDeps()` call during boot in `app.js`. This avoids circular imports.

### WebSocket Handler Registration
`services/ws.js` provides `registerHandlers()` to register page-level functions as WS message handlers. `app.js` wires everything during boot.

## Cross-Module Dependencies

```
services/state.js ← (all modules)
services/api.js ← (all page modules)
utils/dom.js ← (all modules)
utils/format.js ← dashboard.js, control.js, logs.js
utils/markdown.js ← conversations.js, builder.js
utils/graph.js ← visualizer.js, builder.js
services/ws.js ← app.js (registerHandlers + connectWS)
pages/conversations.js ← control.js, builder.js (typing indicator)
pages/dashboard.js ← control.js (refreshInstances), app.js
```

## Testing

```bash
pytest tests/test_fsm_llm_monitor/test_app.py  # Verifies all modules serve HTTP 200
```

## Adding a New Module

1. Create `static/pages/newpage.js` or `static/utils/newutil.js` as an ES module
2. Import from `services/state.js` for shared state, `services/api.js` for HTTP
3. Export functions that `app.js` needs to call
4. Import in `app.js` and wire up event delegation + WS handlers
5. Add to the test list in `test_app.py` (`test_static_js_modules` and `test_static_files_exist`)
