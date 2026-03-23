# FSM-LLM Monitor — Frontend

Vanilla JavaScript single-page application for the FSM-LLM monitoring dashboard. No frameworks, no build tools, no npm — just plain JS served as static files by FastAPI.

## Architecture

All modules share the global scope via regular `<script>` tags loaded in dependency order from `templates/index.html`. Shared mutable state lives in a single `App` namespace object defined in `state.js`.

```
Browser ← WebSocket ← FastAPI (server.py)
  │
  ├── state.js         App namespace (shared globals)
  ├── utils.js         DOM helpers, escaping, formatting
  ├── nav.js           Page routing, sidebar, tabs
  ├── websocket.js     WebSocket connect/reconnect/dispatch
  ├── dashboard.js     Metrics cards, event log, instance grid
  ├── conversations.js Conversation inspector + live chat
  ├── launch.js        Launch modal (FSM / workflow / agent)
  ├── control.js       Control center (detail panels, tables, actions)
  ├── graph.js         SVG graph layout (BFS layering) + rendering
  ├── visualizer.js    Visualizer page + preset picker
  ├── logs.js          Log viewer with level/text filtering
  ├── settings.js      Config panel + system info
  └── init.js          Keyboard shortcuts + boot sequence
```

## Module Loading Order

Scripts are loaded synchronously in `index.html` in this order:

1. `state.js` — must be first (defines `App`)
2. `utils.js` — used by everything
3. `nav.js` — references page refresh functions by name (resolved at call time)
4. `dashboard.js`, `conversations.js`, `launch.js`, `control.js` — page modules (independent)
5. `graph.js` — pure layout/rendering functions
6. `visualizer.js` — depends on `graph.js`
7. `logs.js`, `settings.js` — small page modules
8. `websocket.js` — dispatches to functions from all modules (all defined by now)
9. `init.js` — must be last (runs boot sequence)

## Shared State (`App` namespace)

All cross-module state is accessed via `App.*`:

| Property | Type | Used By |
|----------|------|---------|
| `App.ws` | WebSocket | websocket.js |
| `App.currentPage` | string | nav.js, websocket.js, control.js, conversations.js, init.js |
| `App.instances` | array | dashboard.js, conversations.js, control.js, websocket.js |
| `App.presets` | object\|null | launch.js, visualizer.js |
| `App.capabilities` | object | launch.js |
| `App.selectedConvId` | string\|null | conversations.js, websocket.js |
| `App.selectedConvInstanceId` | string\|null | conversations.js, launch.js, control.js |
| `App.selectedDetailId` | string\|null | control.js, websocket.js |
| `App.selectedDetailType` | string\|null | control.js, websocket.js |
| `App.detailPollTimer` | number\|null | control.js |
| `App.agentUpdates` | object | websocket.js |
| `App.refreshTimers` | object | utils.js |
| `App.wsRetryDelay` | number | websocket.js |
| `App.WS_MAX_DELAY` | number | websocket.js |
| `App.stubToolCount` | number | launch.js |
| `App.TOOL_BASED_AGENTS` | array | launch.js |

## Conventions

- **No modules/imports**: Regular `<script>` tags, global scope. All functions are on `window`.
- **HTML event handlers**: `onclick`, `onchange`, `onkeydown` attributes in `index.html` call global functions directly.
- **HTML escaping**: Always use `esc()` for user-supplied strings rendered as HTML.
- **Async/await**: Used for all `fetch()` calls. Error handling via try/catch with `console.error`.
- **DOM IDs**: Functions reference elements by ID. IDs are defined in `templates/index.html`.
- **Refresh scheduling**: `scheduleRefresh(key, fn, delayMs)` debounces rapid updates from WebSocket events.

## Pages

| Page | Key | Module | Description |
|------|-----|--------|-------------|
| Dashboard | `1` | dashboard.js | Metrics cards, event log, instance grid |
| Visualizer | `2` | visualizer.js + graph.js | FSM/agent/workflow graph visualization |
| Conversations | `3` | conversations.js | Interactive conversation inspector with live chat |
| Control Center | `4` | control.js | Instance management, detail panels, actions |
| Logs | `5` | logs.js | Filterable log viewer |
| Settings | `6` | settings.js | Config + system info |

## Other Static Files

| File | Purpose |
|------|---------|
| `style.css` | Grafana-inspired dark theme (CSS custom properties) |
| `flows.json` | Agent/workflow pattern definitions for the visualizer |
| `app.js` | Legacy barrel file (empty, kept for backward compatibility) |
