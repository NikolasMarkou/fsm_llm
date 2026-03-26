# fsm_llm_monitor/static — Frontend JavaScript Modules

## What This Is

Modular vanilla JS frontend for the FSM-LLM monitoring dashboard. 16 JS files in this directory (15 loaded via `<script>` tags in `templates/index.html`; `app.js` exists for backward compatibility but is not loaded). No framework, no build step, no npm.

## File Map

| File | Lines | Functions | Purpose |
|------|-------|-----------|---------|
| `state.js` | 24 | — | `var App = {...}` — single namespace for all shared mutable state |
| `utils.js` | 159 | `scheduleRefresh`, `esc`, `formatTime`, `relativeTime`, `updateClock`, `numVal`, `intVal`, `showError`, `showToast`, `renderResultBanner`, `showStatus`, `statusBadge`, `_renderLLMData`, `copyContextData`, `formatNumber`, `highlightText` | DOM helpers, HTML escaping, formatting, debounced refresh |
| `nav.js` | 58 | `showPage`, `toggleSidebar`, `switchTab` | Page navigation + tab switching |
| `websocket.js` | 81 | `connectWS` | WebSocket connection, reconnect with exponential backoff, message dispatch |
| `dashboard.js` | 281 | `updateMetrics`, `updateEvents`, `_getFilteredInstances`, `renderInstanceGrid`, `instPagePrev`, `instPageNext`, `onInstSearchInput`, `toggleConvEnded`, `onConvSearchInput`, `convPagePrev`, `convPageNext`, `_getFilteredConvs`, `_renderConvTable` | Dashboard page — metric cards, instance grid, conversation table with pagination |
| `conversations.js` | 245 | `showConversationInDrawer`, `drawerBack`, `_smoothScrollToBottom`, `_addTypingIndicator`, `_removeTypingIndicator` | Conversation detail drawer with chat interface and typing indicator |
| `launch.js` | 283 | `showLaunchModal`, `closeLaunchModal`, `toggleLaunchFSMSource`, `renderLaunchPresets`, `filterPresets`, `addStubTool`, `getStubTools`, `onAgentTypeChange` | Launch modal for FSM/workflow/agent instances |
| `control.js` | 572 | `filterControlInstances`, `onCtrlSearchInput`, `_getFilteredCtrlItems`, `renderUnifiedTable`, `ctrlPagePrev`, `ctrlPageNext`, `_updateFilterChipCounts`, `openDrawer`, `closeDrawer`, `navigateToInstance`, `goToConversation`, `_captureTraceState`, `_restoreTraceState`, `_renderToolCalls`, `toggleAllTraceSteps`, `updateRunningAgents` | Control center — unified instance table with expandable drawer, detail panels, actions |
| `graph.js` | 211 | `layoutNodes`, `rectEdgePoint`, `renderGraph` | BFS-based graph layout + SVG rendering (used by visualizer and builder) |
| `visualizer.js` | 155 | `initVizDivider`, `_populatePresetDropdown` | Visualizer page — split-pane FSM editor + graph viewer, preset dropdown, resize handle |
| `logs.js` | 295 | `toggleLogPill`, `getActiveLogLevels`, `getMinLogLevel`, `_updateLogPillCounts`, `_onLogSearchInput`, `_logEntryHtml`, `_isNearBottom`, `_scrollToBottom`, `_updateJumpButton`, `logJumpToLatest`, `_updatePauseButton`, `toggleLogPause`, `clearLogs`, `appendLogs`, `_updateLogSidebarBadge`, `updateLogErrorBadge`, `_attachLogScrollListener` | Log viewer with level pills, text filtering, live/pause toggle, jump-to-latest |
| `settings.js` | 60 | `resetSettings` | Settings page — runtime config and system info display |
| `markdown.js` | 64 | `renderMarkdown` | Markdown rendering utilities for chat bubbles and builder output |
| `builder.js` | 372 | `_isBuilderNearBottom`, `_builderAutoScroll`, `_updateBuilderJump`, `builderJumpToLatest`, `startBuilderSession`, `sendBuilderMessage`, `_onBuilderComplete`, `_buildResultSummary`, `_renderBuilderGraph`, `_appendBuilderBubble`, `copyBuilderResult`, `downloadBuilderResult`, `launchBuilderResult`, `resetBuilder` | Builder page — meta-agent conversational interface for artifact creation |
| `init.js` | 65 | `showShortcutsOverlay`, `closeShortcutsOverlay`, `navigateFromHash` | Keyboard shortcuts (1-6 for pages, Esc, ?) + hash routing + boot sequence |
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
- `graph.js` before `visualizer.js` and `builder.js`
- `markdown.js` before `builder.js` (builder uses `renderMarkdown()`)
- `websocket.js` before `init.js` (boot calls `connectWS()`)
- `init.js` must be last (executes boot sequence immediately)

### WebSocket Message Dispatch
`connectWS()` in `websocket.js` receives JSON messages and dispatches to functions across multiple modules (`updateMetrics`, `renderInstanceGrid`, `renderControlFSMs`, etc.). These functions must all be defined (loaded) before the first WS message arrives — guaranteed because all scripts load synchronously before `init.js` calls `connectWS()`.

## Cross-Module Dependencies

```
state.js ← (everything)
utils.js ← nav.js, dashboard.js, conversations.js, launch.js, control.js, visualizer.js, logs.js, settings.js, graph.js, builder.js
nav.js ← control.js (navigateToInstance calls showPage), launch.js, init.js
dashboard.js ← websocket.js (updateMetrics, updateEvents, renderInstanceGrid, _renderConvTable)
conversations.js ← websocket.js (showConversationInDrawer), control.js (goToConversation)
control.js ← websocket.js (renderUnifiedTable, updateRunningAgents)
graph.js ← visualizer.js (renderGraph), builder.js (_renderBuilderGraph)
markdown.js ← builder.js (_appendBuilderBubble calls renderMarkdown)
logs.js ← websocket.js (appendLogs, updateLogErrorBadge)
```

## Gotchas

- **`control.js` is 572 lines** — largest module. Contains unified instance table + expandable drawer with FSM, workflow, and agent detail renderers + actions. Subsections are marked with `// ---` comments.
- **`builder.js` is 372 lines** — second largest. Contains the full meta-agent builder interface with chat, result display, graph rendering, copy/download/launch actions.
- **`logs.js` is 295 lines** — full log viewer with level pills, search, live/pause, jump-to-latest, and sidebar badge updates.
- **`'use strict'` in every file** — but since these are regular scripts (not modules), `'use strict'` applies per-file, not globally. Function declarations are still hoisted to global scope.
- **`_prefixed` functions** (e.g., `_renderLLMData`, `_renderToolCalls`, `_getFilteredInstances`) are "private by convention" — still global, but not called from HTML templates.
- **`app.js` still exists** — empty comment file kept so existing tests for `/static/app.js` continue to pass. It is NOT loaded by `index.html`.
- **No CSS-in-JS** — all styles are in `style.css`. Exception: `renderLaunchPresets()` sets one inline `style.cssText` on the preset items container.

## Testing

```bash
pytest tests/test_fsm_llm_monitor/test_app.py  # Verifies all 15 modules + app.js serve HTTP 200
```

No JS unit tests — only static file serving is verified. Runtime behavior is validated by structural analysis (function cross-references, handler resolution).

## Adding a New Module

1. Create `static/newmodule.js` with `'use strict';` header
2. Use `App.*` for shared state, plain `function` declarations for globals
3. Add `<script src="/static/newmodule.js"></script>` to `index.html` in the correct load order position
4. Add the module to the test list in `test_app.py` (`test_static_js_modules` and `test_static_files_exist`)
5. Update this CLAUDE.md file map
