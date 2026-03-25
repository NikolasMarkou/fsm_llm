# fsm_llm_monitor -- Claude Code Instructions

## What This Package Does

Real-time web dashboard for monitoring FSM-LLM conversations, agent executions, and workflow instances. FastAPI backend serves a vanilla JS single-page application with a Grafana-inspired dark theme. Captures events from all 8 handler timing points, streams metrics and events via WebSocket, and provides full instance lifecycle management (launch, control, destroy) through 40 REST endpoints.

## File Map

### Python Backend

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app -- 40 REST endpoints + 1 WebSocket, global manager pattern via `configure()` / `get_manager()`, serves SPA, CORS config, lifespan cleanup |
| `bridge.py` | `MonitorBridge` -- connects EventCollector to a live API instance, registers observer handlers, provides query interface (backward compat layer) |
| `collector.py` | `EventCollector` -- thread-safe bounded deques (`maxlen`), 8 handler callbacks, loguru sink, metric counters, `get_events_since()` for incremental streaming |
| `instance_manager.py` | `InstanceManager` -- manages `ManagedFSM`, `ManagedWorkflow`, `ManagedAgent` lifecycle, per-instance + global collectors, `register_monitor_handlers()`, preset scanning, agent background threads |
| `definitions.py` | Pydantic models: `MonitorEvent`, `LogRecord`, `MetricSnapshot`, `ConversationSnapshot`, `FSMSnapshot`, `StateInfo`, `TransitionInfo`, `MonitorConfig`, `InstanceInfo`, request models (`LaunchFSMRequest`, `LaunchAgentRequest`, `LaunchWorkflowRequest`, `StartConversationRequest`, `SendMessageRequest`, `EndConversationRequest`, `WorkflowAdvanceRequest`, `WorkflowCancelRequest`, `StubToolConfig`, `BuilderStartRequest`, `BuilderSendRequest`), `model_to_dict()`, `normalize_message_history()` |
| `constants.py` | Theme colors (11 constants), 17 event type constants, defaults (`DEFAULT_REFRESH_INTERVAL=1.0`, `DEFAULT_MAX_EVENTS=1000`, `DEFAULT_MAX_LOG_LINES=5000`, `DEFAULT_LOG_LEVEL="INFO"`), `MONITOR_HANDLER_NAME="fsm_llm_monitor"`, `MONITOR_HANDLER_PRIORITY=9999` |
| `exceptions.py` | `MonitorError` -> `MonitorInitializationError`, `MetricCollectionError`, `MonitorConnectionError` |
| `__main__.py` | CLI entry point: `fsm-llm-monitor` / `python -m fsm_llm_monitor`, argparse for `--host`, `--port`, `--no-browser`, `--version`, `--info` |
| `__init__.py` | Public exports (56 symbols) |
| `__version__.py` | Version imported from `fsm_llm.__version__` (not independent) |

### Frontend (static/)

| File | Purpose |
|------|---------|
| `state.js` | Global `App` namespace and shared state management |
| `utils.js` | Shared utility functions (formatting, DOM helpers) |
| `websocket.js` | WebSocket connection management and message dispatch |
| `nav.js` | Sidebar navigation and page switching |
| `dashboard.js` | Dashboard page -- metric cards, instance grid, event feed |
| `control.js` | Control Center -- unified instance table with expandable drawer |
| `conversations.js` | Conversation detail view and chat interface |
| `launch.js` | Launch modal for FSMs, agents, workflows |
| `graph.js` | FSM/agent/workflow graph rendering (canvas-based) |
| `visualizer.js` | Visualizer page -- tabbed graph viewer with presets |
| `logs.js` | Logs page -- level-filtered stream with live/pause toggle |
| `settings.js` | Settings page -- runtime config and system info |
| `markdown.js` | Markdown rendering utilities |
| `builder.js` | Builder page -- meta-agent conversational interface |
| `app.js` | Top-level app orchestration |
| `init.js` | App initialization and boot sequence |
| `style.css` | Grafana-inspired dark theme with CSS custom properties |
| `flows.json` | Agent/workflow pattern flow definitions for visualizer |

### Templates

| File | Purpose |
|------|---------|
| `templates/index.html` | Single-page HTML template, loads all 16 JS modules in dependency order |

## Key Patterns

### Global Manager Pattern

`configure(bridge, manager, cors_origins)` sets the module-level `_manager` in `server.py`. `get_manager()` lazy-initializes an empty `InstanceManager()` if none was configured. All endpoints call `get_manager()`. A backward-compat `get_bridge()` wraps the manager's global collector in a `MonitorBridge`.

### Instance Lifecycle Management

`InstanceManager` manages three instance types (`ManagedFSM`, `ManagedWorkflow`, `ManagedAgent`). Each gets a UUID-based instance ID, a label, a status (`running`/`completed`/`failed`/`cancelled`), and a per-instance `EventCollector`. The manager also maintains a global collector that aggregates all events and captures loguru logs via a sink.

### Event Collection

8 handler timing points are mapped to event types via `EventCollector.create_handler_callbacks()`:
- `START_CONVERSATION` -> `conversation_start`
- `PRE_PROCESSING` -> `pre_processing`
- `POST_PROCESSING` -> `post_processing`
- `PRE_TRANSITION` -> `state_transition`
- `POST_TRANSITION` -> (no-op, transition captured at PRE_TRANSITION)
- `CONTEXT_UPDATE` -> `context_update`
- `END_CONVERSATION` -> `conversation_end`
- `ERROR` -> `error`

Handlers are registered at priority 9999 (lowest) via `register_monitor_handlers()`. They return empty dicts -- pure observers that never modify FSM state.

### WebSocket Streaming

The `/ws` endpoint runs a poll loop at `config.refresh_interval` (default 1 second). Each message contains the latest `MetricSnapshot`. Incremental delivery is tracked per connection via `last_event_count` and `last_log_count` -- only new events/logs are pushed (up to 50 per poll). Instance list and running agent status updates are always included.

### Thread Safety

`EventCollector` uses a single `threading.Lock` around all deque operations and metric counter updates. `InstanceManager` uses a `threading.RLock`. Deques have `maxlen` set to prevent unbounded memory growth. Agents run in background threads with a `cancel_event` for graceful shutdown.

### Frontend Architecture

16 vanilla JS modules loaded via `<script>` tags in dependency order. Shared state in `App` namespace (`state.js`). All functions are global for HTML `onclick` handler compatibility. No build step, no framework dependencies.

## REST API Endpoints

### Core

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve SPA |
| GET | `/health` | Health check |
| GET | `/api/metrics` | Current MetricSnapshot |
| GET | `/api/events` | Recent events (query: `limit`) |
| GET | `/api/logs` | Log records (query: `limit`, `level`) |
| GET | `/api/config` | Current MonitorConfig |
| POST | `/api/config` | Update MonitorConfig |
| GET | `/api/info` | Monitor + fsm_llm versions |
| GET | `/api/capabilities` | Feature flags (fsm, workflows, agents) |
| GET | `/api/conversations` | All conversation snapshots (query: `include_ended`) |
| GET | `/api/conversations/{id}` | Single conversation snapshot |
| WS | `/ws` | Real-time metrics + events stream |

### Instance Management

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/instances` | List all instances (query: `type`) |
| GET | `/api/instances/{id}` | Single instance info |
| GET | `/api/instances/{id}/events` | Events for a specific instance |
| DELETE | `/api/instances/{id}` | Destroy instance |

### FSM

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/fsm/launch` | Launch FSM from preset or JSON |
| POST | `/api/fsm/{id}/start` | Start conversation |
| POST | `/api/fsm/{id}/converse` | Send message |
| POST | `/api/fsm/{id}/end` | End conversation |
| GET | `/api/fsm/{id}/conversations` | List conversations |
| POST | `/api/fsm/load` | Load FSM definition (legacy) |
| POST | `/api/fsm/visualize` | Visualize FSM from JSON |
| GET | `/api/fsm/visualize/preset/{id}` | Visualize preset FSM |

### Agents

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/agent/launch` | Launch agent (background thread) |
| GET | `/api/agent/{id}/status` | Agent execution status |
| GET | `/api/agent/{id}/result` | Agent result + trace |
| POST | `/api/agent/{id}/cancel` | Cancel running agent |
| GET | `/api/agent/visualize` | Agent pattern flow graph |

### Workflows

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/workflow/launch` | Launch workflow instance |
| POST | `/api/workflow/{id}/advance` | Advance workflow step |
| POST | `/api/workflow/{id}/cancel` | Cancel workflow |
| GET | `/api/workflow/{id}/status` | Workflow status |
| GET | `/api/workflow/{id}/instances` | Workflow instances |
| GET | `/api/workflow/visualize` | Workflow pattern flow graph |

### Presets

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/presets` | List all presets (scans examples/) |
| GET | `/api/preset/fsm/{id}` | Load FSM preset definition |

### Builder (Meta-Agent)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/builder/start` | Start builder session |
| POST | `/api/builder/send` | Send message to builder session |
| GET | `/api/builder/result/{id}` | Get builder session state |
| DELETE | `/api/builder/{id}` | Delete builder session |

## Dependencies on Core

- `fsm_llm.API` -- FSM execution, conversation management
- `fsm_llm.HandlerTiming` -- handler timing point enum (8 values)
- `fsm_llm.create_handler` -- fluent handler builder (`.at()`, `.with_priority()`, `.do()`)
- `fsm_llm.logging.logger` -- loguru logging (for sink integration and debug logging)
- `fsm_llm.constants.DEFAULT_LLM_MODEL` -- default model for launch requests

## Dependencies

- `fastapi` -- web framework (REST + WebSocket)
- `uvicorn` -- ASGI server
- `jinja2` -- template rendering
- `pydantic` -- request/response models (inherited from core)
- `loguru` -- log capture via sink (inherited from core)

## Exception Hierarchy

```
MonitorError
  MonitorInitializationError  -- monitor setup failures
  MetricCollectionError       -- metric collection failures
  MonitorConnectionError      -- API connection failures
```

## Testing

```bash
pytest tests/test_fsm_llm_monitor/  # 171 tests in 5 files
```

| File | Scope |
|------|-------|
| `test_app.py` | FastAPI endpoint integration tests |
| `test_bridge.py` | MonitorBridge unit tests |
| `test_collector.py` | EventCollector unit tests |
| `test_definitions.py` | Pydantic model tests |
| `test_instance_manager.py` | InstanceManager unit tests |

## Gotchas

- `configure()` must be called before endpoints work properly; `get_manager()` auto-creates an empty `InstanceManager` if none was configured (no API connection, limited functionality)
- Handler priority 9999 means handlers run last -- pure observers that never interfere with FSM processing
- `EventCollector` uses bounded deques (`max_events=1000`, `max_log_lines=5000`) to prevent memory leaks in long-running monitors
- `InstanceManager` caches ended conversations in a bounded `OrderedDict` (max 1,000) since they are no longer queryable from the API after ending
- Frontend is vanilla JS -- no build step, no framework dependencies; all modules share a global `App` namespace
- Version is synced from `fsm_llm.__version__` -- not independently versioned
- Preset endpoints scan `examples/` directory but use path traversal protection (`".."` check + `resolve().relative_to()`) to prevent filesystem exposure
- `MONITOR_HANDLER_NAME = "fsm_llm_monitor"` is used as the handler name prefix; `MONITOR_HANDLER_PRIORITY = 9999` ensures lowest priority
- CLI defaults to port 8420 and auto-opens browser (use `--no-browser` to disable)
- LLM operations (start conversation, send message, end conversation) are wrapped in `asyncio.to_thread` with a 120-second timeout (`_LLM_OPERATION_TIMEOUT`)
- Agent instances run in background threads with a `cancel_event` for graceful cancellation
- Optional imports: `fsm_llm_workflows` and `fsm_llm_agents` are imported conditionally; missing extensions degrade gracefully (capabilities endpoint reports what is available)
- Builder endpoints require `fsm_llm_meta` to be installed; returns HTTP 501 if missing
- CORS defaults to localhost only (`127.0.0.1:8420`, `localhost:8420`); override via `configure(cors_origins=[...])`
