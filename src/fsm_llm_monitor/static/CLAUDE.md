# fsm_llm_monitor/static — Frontend JavaScript Modules

## What This Is

Modular vanilla JS frontend for the FSM-LLM monitoring dashboard. 13 script files loaded via `<script>` tags in `templates/index.html`. No framework, no build step, no npm.

## File Map

| File | Lines | Functions | Purpose |
|------|-------|-----------|---------|
| `state.js` | 23 | — | `var App = {...}` — single namespace for all shared mutable state |
| `utils.js` | 77 | `esc`, `formatTime`, `updateClock`, `numVal`, `intVal`, `showError`, `showStatus`, `statusBadge`, `renderResultBanner`, `_renderLLMData`, `scheduleRefresh` | DOM helpers, HTML escaping, formatting, debounced refresh |
| `nav.js` | 46 | `showPage`, `toggleSidebar`, `switchTab` | Page navigation + tab switching |
| `websocket.js` | 63 | `connectWS` | WebSocket connection, reconnect with exponential backoff, message dispatch |
| `dashboard.js` | 108 | `updateMetrics`, `updateEvents`, `_relativeTime`, `renderInstanceGrid`, `refreshInstances`, `refreshConversationTable` | Dashboard page rendering |
| `conversations.js` | 192 | `refreshConversations`, `showConversationDetail`, `sendChatMessage` | Conversation inspector with live chat |
| `launch.js` | 280 | `showLaunchModal`, `closeLaunchModal`, `checkCapabilities`, `toggleLaunchFSMSource`, `loadLaunchPresets`, `renderLaunchPresets`, `filterPresets`, `doLaunchFSM`, `doLaunchWorkflow`, `addStubTool`, `getStubTools`, `onAgentTypeChange`, `doLaunchAgent` | Launch modal for FSM/workflow/agent instances |
| `control.js` | 484 | `refreshControlCenter`, `closeDetail`, `navigateToInstance`, `selectInstance`, `refreshDetailPanel`, `refreshDetailEvents`, `renderFSMDetail`, `goToConversation`, `renderWorkflowDetail`, `renderAgentDetail`, `_renderToolCalls`, `toggleAllTraceSteps`, `_renderAgentTrace`, `updateRunningAgents`, `renderControlFSMs`, `renderControlWorkflows`, `renderControlAgents`, `startConversationOn`, `destroyInstance`, `cancelAgent` | Control center — instance tables, detail panels, actions |
| `graph.js` | 181 | `layoutNodes`, `rectEdgePoint`, `renderGraph` | BFS-based graph layout + SVG rendering (used by visualizer) |
| `visualizer.js` | 154 | `visualizeGraph`, `visualizeFSM`, `showPresetPicker`, `loadFSMPresets`, `renderPresets`, `useFSMPreset` | Visualizer page — FSM/agent/workflow graphs + preset picker |
| `logs.js` | 34 | `refreshLogs` | Log viewer with level and text filtering |
| `settings.js` | 56 | `loadSettings`, `saveSettings`, `resetSettings` | Settings page + system info display |
| `init.js` | 30 | — | Keyboard shortcuts (1-6 for pages, Esc) + boot sequence |
| `app.js` | 6 | — | Empty barrel file (backward compat, not loaded by HTML) |
| `style.css` | — | — | Grafana-inspired dark theme with CSS custom properties |
| `flows.json` | — | — | Agent/workflow pattern flow definitions for visualizer |

## Key Patterns

### Shared State via `App` Namespace
All shared mutable state lives in `state.js` as `var App = {...}`. Modules read/write `App.instances`, `App.currentPage`, `App.selectedConvId`, etc. Never use bare globals for cross-module state.

### Global Functions for HTML Handlers
All functions are global (plain `function` declarations in non-module `<script>` tags). This is intentional — `templates/index.html` uses ~35 `onclick`/`onchange` attributes that call these functions directly. Do NOT wrap modules in IIFEs or switch to ES modules without updating all HTML event handlers.

### Script Load Order Matters
`index.html` loads scripts in dependency order. `state.js` first, `init.js` last. Adding a new module requires inserting the `<script>` tag at the right position. Key constraints:
- `state.js` before everything (defines `App`)
- `utils.js` before any module that calls `esc()`, `formatTime()`, etc.
- `graph.js` before `visualizer.js`
- `websocket.js` before `init.js` (boot calls `connectWS()`)
- `init.js` must be last (executes boot sequence immediately)

### WebSocket Message Dispatch
`connectWS()` in `websocket.js` receives JSON messages and dispatches to functions across multiple modules (`updateMetrics`, `renderInstanceGrid`, `renderControlFSMs`, etc.). These functions must all be defined (loaded) before the first WS message arrives — guaranteed because all scripts load synchronously before `init.js` calls `connectWS()`.

## Cross-Module Dependencies

```
state.js ← (everything)
utils.js ← nav.js, dashboard.js, conversations.js, launch.js, control.js, visualizer.js, logs.js, settings.js, graph.js
nav.js ← control.js (navigateToInstance calls showPage), launch.js, init.js
dashboard.js ← websocket.js (updateMetrics, updateEvents, renderInstanceGrid, refreshConversationTable)
conversations.js ← websocket.js (refreshConversations, showConversationDetail), control.js (goToConversation)
control.js ← websocket.js (renderControlFSMs/Workflows/Agents, updateRunningAgents, refreshDetailPanel)
graph.js ← visualizer.js (renderGraph)
```

## Gotchas

- **`control.js` is 484 lines** — largest module. Contains FSM, workflow, and agent detail renderers + table renderers + actions. Splitting further would create tight cross-file coupling. Subsections are marked with `// ---` comments.
- **`'use strict'` in every file** — but since these are regular scripts (not modules), `'use strict'` applies per-file, not globally. Function declarations are still hoisted to global scope.
- **`_prefixed` functions** (e.g., `_relativeTime`, `_renderLLMData`, `_renderToolCalls`, `_renderAgentTrace`) are "private by convention" — still global, but not called from HTML templates.
- **`app.js` still exists** — empty comment file kept so existing tests for `/static/app.js` continue to pass. It is NOT loaded by `index.html`.
- **No CSS-in-JS** — all styles are in `style.css`. Exception: `renderLaunchPresets()` sets one inline `style.cssText` on the preset items container.

## Testing

```bash
pytest tests/test_fsm_llm_monitor/test_app.py  # Verifies all 13 modules + app.js serve HTTP 200
```

No JS unit tests — only static file serving is verified. Runtime behavior is validated by structural analysis (function cross-references, handler resolution).

## Adding a New Module

1. Create `static/newmodule.js` with `'use strict';` header
2. Use `App.*` for shared state, plain `function` declarations for globals
3. Add `<script src="/static/newmodule.js"></script>` to `index.html` in the correct load order position
4. Add the module to the test list in `test_app.py` (`test_static_js_modules` and `test_static_files_exist`)
5. Update this CLAUDE.md file map
