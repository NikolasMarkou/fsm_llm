# services

Path: `src/fsm_llm/monitor/static/services`
Purpose: Frontend infrastructure for the FSM-LLM Monitor dashboard: REST client, shared reactive state, WebSocket manager.

## Scope

Four ES modules loaded by the browser. No build step, no framework, no npm. Served as static files by the `fsm_llm.monitor` FastAPI server under `/static/`. Page rendering lives in `static/pages/*.js`; DOM helpers in `static/utils/*.js`. This folder must not render page content beyond the WebSocket status indicator and the API key modal (`#apikey-modal`, markup in `templates/index.html`).

## Architecture

```mermaid
sequenceDiagram
    participant App as app.js
    participant WS as ws.js
    participant S as state.js
    participant P as page modules
    App->>WS: registerHandlers({...})
    App->>WS: connectWS()
    WS->>S: state.ws = new WebSocket(/ws)
    loop each message
        WS->>S: state.instances / agentUpdates / workflowUpdates = ...
        WS->>P: _dispatch.<handler>?.(payload)
        WS->>S: scheduleRefresh(key, fn, ms)
    end
    WS-->>WS: onclose -> setTimeout(connectWS, delay + jitter)
```

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `api.js` | fetch wrapper | Adds `X-API-Key`; on 401 prompts once and retries once; throws `Error(formatted detail)` on non-OK |
| `auth.js` | API key storage and key-entry modal | `sessionStorage['fsmMonitorApiKey']` only (never localStorage, URLs, or logs) |
| `state.js` | Proxy-backed state + event bus | `EventTarget` bus, event name `statechange` |
| `ws.js` | WebSocket lifecycle + dispatch | Imports `hashInstances`, `$` from `../utils/dom.js` |

## Public interface

`api.js`
- `fetchJson(url: string, opts?: RequestInit): Promise<any>` - sends `X-API-Key` when a key is stored (every method, including DELETE); parses JSON; on `!resp.ok` tries `resp.json()`, throws `Error(formatErrorDetail(detail))` with `err.status`. On a 401 it awaits `requestApiKey()` and retries the request once if a key was saved.
- `postJson(url: string, data: any): Promise<any>` - POST with `Content-Type: application/json`.
- `formatErrorDetail(detail, fallback)` - string detail as is; a 422 list of `{loc, msg}` becomes `"field: msg; ..."` (the `body`/`query`/`path` loc prefix is dropped).

`auth.js`
- `getApiKey()`, `setApiKey(key)` (empty clears), `clearApiKey()`, `onApiKeyChange(fn) -> unsubscribe`.
- `checkAuthRequired() -> true|false|null` - `GET /api/auth` (never gated); `isAuthRequired()` returns the last answer.
- `requestApiKey({force, message}) -> Promise<boolean>` - opens `#apikey-modal`; concurrent callers share one pending prompt. After the user cancels, non-forced calls resolve `false` without opening it until a key is saved; `force` (boot check, WS 4401) always opens it.
- `submitApiKeyModal()`, `cancelApiKeyModal()`, `isApiKeyModalOpen()` - wired to `apikey-save`, `apikey-cancel`, `close-apikey-backdrop`, Enter in `#apikey-input`, and Escape by `app.js`.

`state.js`
- `state` - Proxy over `_target`. `set` trap dispatches `CustomEvent('statechange', {detail: {prop, value, old}})` when `old !== value`.
- `onChange(fn: ({prop, value, old}) => void): () => void` - returns unsubscribe.
- `scheduleRefresh(key: string, fn: () => void, delayMs: number)` - no-op if `state.refreshTimers[key]` is set; clears the key before calling `fn`.
- `cancelRefresh(...keys)` - clears pending `scheduleRefresh` timers (the drawer cancels `ctrl-detail`, `ctrl-detail-wf`, `conv-detail` on open/close).
- `WS_MAX_DELAY = 30000`.
- `TOOL_BASED_AGENTS = ['ReactAgent', 'ReflexionAgent', 'PlanExecuteAgent', 'REWOOAgent', 'ADaPTAgent']` - agent types whose launch requires at least one tool (used by `pages/launch.js`, `pages/builder.js`).

`ws.js`
- `registerHandlers(handlers: object)` - shallow-merges into the dispatch table.
- `connectWS()` - opens `ws(s)://<location.host>/ws`; on `open` the first message sent is `{"type": "auth", "api_key": <stored key or "">}`.

## Data shapes

Initial `state` fields: `ws`, `currentPage` (`'dashboard'`), `presets`, `wsRetryDelay` (3000), `capabilities` (`{fsm: true, workflows: false, agents: false}`), `instances` (`[]`), `selectedConvId`, `selectedConvInstanceId`, `selectedDetailId`, `selectedDetailType`, `detailPollTimer`, `agentUpdates`, `workflowUpdates`, `refreshTimers`, `stubToolCount`, `autoScrollLogs` (from `MonitorConfig.auto_scroll_logs`), `_lastContextData`. Pages also add fields ad hoc (for example `state.workflowPresets` in `pages/launch.js`).

WebSocket message fields consumed (any subset may be present in one message):

| Field | Action |
| --- | --- |
| `type === 'metrics'` + `data` | `updateMetrics(data)` |
| `events` | `updateEvents(events)`; activity-type events schedule refreshes |
| `instances` | if `hashInstances` changed: set `state.instances`, `renderInstanceGrid()`, and `renderUnifiedTable()` on the control page |
| `agent_updates` | set `state.agentUpdates`, `updateRunningAgents(updates)` |
| `workflow_updates` | set `state.workflowUpdates`; refresh activity table (dashboard, 3 s) or workflow detail (control, 2 s; the callback reads the current selection when it fires) |
| `logs` (non-empty) | `appendLogs(logs)` |
| `dashboard_config` | `dashboardConfigChanged(cfg)` |

Activity event types that trigger refreshes: `conversation_start`, `conversation_end`, `state_transition`, `post_processing`, `agent_started`, `agent_completed`, `agent_failed`, `workflow_started`, `workflow_completed`, `workflow_cancelled`.

Dispatch handler names: `updateMetrics`, `updateEvents`, `renderInstanceGrid`, `renderUnifiedTable`, `updateRunningAgents`, `refreshActivityTable`, `refreshDetailPanel`, `appendLogs`, `dashboardConfigChanged`, `showConversationDetail`. All are called with optional chaining, so a missing handler is silently skipped.

## Invariants and constraints

- Only one live socket: `connectWS` nulls `onopen/onmessage/onclose/onerror` on the previous `state.ws` before replacing it.
- Backoff: `setTimeout(connectWS, wsRetryDelay + random*1000)`, then `wsRetryDelay = min(2x, WS_MAX_DELAY)`. `onopen` resets to 3000. `onerror` calls `close()`, which triggers `onclose`.
- Close code 4401 (auth failed): no reconnect; status shows "API key required" and `requestApiKey({force: true})` opens the modal. An `onApiKeyChange` listener reconnects once a key is saved.
- Scheduled drawer/conversation refresh callbacks read `state.selectedDetailId/Type` and `state.selectedConvId` when they fire, never values captured at schedule time.
- `onopen/onclose` update `#ws-status` (and the Settings `#conn-status`) text and class and toggle `connected` on `#ws-dot`, guarded for missing elements.
- The API key never appears in URLs or logs; it is read from `sessionStorage` per request (in-memory fallback when storage is unavailable).
- Message parse errors are caught and logged to `console.error`; one bad message does not kill the socket.
- The Proxy only intercepts top-level assignment. Nested mutation does not emit events.

## Dependencies

- `../utils/dom.js`: `hashInstances(items)` (hash over `instance_id:status`), `$` (getElementById), `openDialog`/`closeDialog` (auth modal focus).
- Browser APIs only: `fetch`, `WebSocket`, `EventTarget`, `CustomEvent`, `Proxy`.

## Failure modes

- Server error: `fetchJson` throws; callers show a toast or inline error.
- 401 with the modal cancelled: the request throws `missing or invalid API key`; later background 401s do not reopen the modal until a key is saved (Settings or a forced prompt).
- Non-JSON 2xx body: `resp.json()` rejects with a `SyntaxError`.
- Server down: socket keeps reconnecting up to every ~31 s indefinitely.

## Working here

- New WebSocket payload field: add a branch in `ws.js onmessage`, call `_dispatch.<name>?.(...)`, and register the handler in `static/app.js`.
- New shared state: add the field with its default to `_target` in `state.js` so it is discoverable.
- Keep the `detail` error convention: the Python server raises `HTTPException(detail=...)`.
- No automated JS tests exist for these files. Verify manually by running `fsm-llm-monitor` and watching the browser console.
