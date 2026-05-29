# fsm_llm_monitor -- Web Monitoring Dashboard

FastAPI-based web dashboard with real-time observability for FSMs, agents, and workflows. Grafana-inspired dark theme. REST + WebSocket APIs.

- **Version**: 0.4.0 (synced from fsm_llm)
- **Extra deps**: fastapi (>=0.100.0), uvicorn (>=0.20.0), jinja2 (>=3.1.0)
- **Install**: `pip install fsm-llm[monitor]`
- **Default URL**: http://127.0.0.1:8420

## File Map

```
fsm_llm_monitor/
├── server.py              # FastAPI app -- 35+ REST endpoints + WebSocket /ws
├── bridge.py              # MonitorBridge -- connects EventCollector to FSM API instances
├── collector.py           # EventCollector -- thread-safe event/log capture with bounded deques + loguru sink
├── instance_manager.py    # InstanceManager -- lifecycle for FSM conversations, workflows (DSL presets), agents (7 launchable types)
├── definitions.py         # 20 Pydantic models: MonitorEvent, MetricSnapshot, MonitorConfig, FSMSnapshot, request/response models
├── constants.py           # EVENT_TYPES (15), THEME colors (11), DEFAULTS (refresh=1.0s, max_events=1000, max_logs=5000)
├── exceptions.py          # MonitorError(Exception) -> MonitorInitializationError, MetricCollectionError, MonitorConnectionError
├── __main__.py            # CLI: fsm-llm-monitor [--host, --port, --no-browser, --version, --info]
├── __version__.py         # Imports from fsm_llm.__version__
├── otel.py                # OTELExporter -- OpenTelemetry adapter, converts events to spans/metrics (requires fsm-llm[otel])
├── __init__.py            # Public exports: EventCollector, MonitorBridge, InstanceManager, MonitorConfig, OTELExporter, configure, app, etc.
├── static/                # Frontend SPA (vanilla JS)
│   ├── app.js             # Main application module
│   ├── style.css          # Grafana-inspired dark theme (bg: #111217, primary: #3274d9)
│   ├── flows.json         # Agent/workflow pattern flow definitions
│   ├── pages/             # Page components
│   │   ├── dashboard.js   # Metric cards, instance grid, event stream
│   │   ├── control.js     # Control Center -- unified instance table with detail drawer
│   │   ├── conversations.js  # Chat interface with state tracking
│   │   ├── launch.js      # Launch modal for FSMs, agents, workflows
│   │   ├── visualizer.js  # Tabbed graph viewer with presets
│   │   ├── logs.js        # Level-filtered log stream (live/pause)
│   │   ├── builder.js     # Meta-agent builder interface
│   │   └── settings.js    # Runtime config and system info
│   ├── services/
│   │   ├── api.js         # REST API client
│   │   ├── state.js       # Global state management
│   │   └── ws.js          # WebSocket client and message dispatch
│   └── utils/
│       ├── dom.js         # DOM manipulation helpers
│       ├── format.js      # Data formatting
│       ├── graph.js       # FSM/agent/workflow graph rendering
│       └── markdown.js    # Markdown rendering
└── templates/
    └── index.html         # Single-page template (Jinja2)
```

## Key Classes

### EventCollector (`collector.py`)

Thread-safe event and log capture with bounded deques.

- Constructor: `__init__(max_events=1000, max_log_lines=5000)`
- `record_event(event_type, data)` -- Record monitoring event
- `get_metrics()` → MetricSnapshot (active conversations, total events, errors, transitions, state visits)
- `get_logs(level=None, limit=None)` → list[LogRecord]
- `get_events(limit=None)` → list[MonitorEvent]
- `create_handler_callbacks()` → dict -- Handler callbacks for FSM API registration at all timing points
- Loguru sink integration for automatic log capture

### MonitorBridge (`bridge.py`)

Connects to FSM API instances, provides snapshot queries.

- `connect(api)` -- Connect to an API instance and register monitor handlers (the collector is supplied at `MonitorBridge(...)` construction, not here)
- `get_conversation_snapshot(conv_id)` → ConversationSnapshot
- `get_all_conversation_snapshots()` → list[ConversationSnapshot]
- `load_fsm_from_dict(fsm_dict)` -- Load FSM definition for visualization

### InstanceManager (`instance_manager.py`)

Lifecycle management for 3 instance types: ManagedFSM, ManagedWorkflow, ManagedAgent.

- **FSM**: `launch_fsm(fsm_source, model)`, `start_conversation(id, context)`, `send_message(id, conv_id, msg)`, `end_conversation(id, conv_id)`
- **Workflow**: `get_workflow_presets()`, `launch_workflow(preset_id)` (registers a built-in DSL preset definition; arbitrary `definition_json` is unsupported), `start_workflow_instance(id, workflow_id, ctx)` (async), `advance_workflow(id)` (async), `cancel_workflow(id)` (async). Built-in presets: `demo_linear`, `demo_branching`.
- **Agent**: `launch_agent(agent_type, model, tools_config)` -- 7 launchable types (React, Reflexion, PlanExecute, REWOO, ADaPT, Debate, SelfConsistency). EvaluatorOptimizer/MakerChecker are intentionally NOT launchable (they need constructor args a web form can't supply) but remain in flows.json for visualization. `cancel_agent(id)`, `get_agent_status(id)`, `get_agent_result(id)`
- Ended conversation cache with bounded size (1000 max)

### FastAPI Server (`server.py`)

35+ endpoints across 7 categories:

| Category | Endpoints |
|----------|-----------|
| Monitoring | GET `/api/metrics`, `/api/events`, `/api/logs` |
| Capabilities | GET `/api/capabilities`, `/api/presets` |
| Instances | GET/POST `/api/instances`, `/api/instances/fsm`, `/api/instances/agent`, `/api/instances/workflow` |
| FSM | GET/POST `/api/conversations`, `/api/conversations/{id}`, `/api/conversations/{id}/message`, `/api/conversations/{id}/end` |
| Workflow | GET `/api/workflow/presets`, POST `/api/workflow/launch`, `/api/workflow/{id}/advance`, `/api/workflow/{id}/cancel`, GET `/api/workflow/{id}/status` |
| Agent | GET/POST `/api/agents/{id}/status`, `/api/agents/{id}/result`, `/api/agents/{id}/cancel` |
| Builder | POST `/api/builder/start`, `/api/builder/send`, GET `/api/builder/result` |

WebSocket `/ws` streams: metrics, events, logs, agent status updates.

- LLM operations: 120-second timeout via `asyncio.to_thread()`
- Builder sessions: TTL-managed MetaBuilderAgent sessions

### OTELExporter (`otel.py`)

OpenTelemetry adapter that wraps EventCollector events into OTEL spans.

- Constructor: `__init__(service_name="fsm-llm", exporter=None)` -- defaults to ConsoleSpanExporter
- `enable(collector)` -- Wraps collector's record_event to also emit OTEL spans (idempotent, stores original for restore)
- `disable()` -- Stop OTEL export, restore original record_event, end active spans
- `shutdown()` -- Flush pending spans and shut down provider
- Properties: `is_enabled`, `active_conversations`
- Static: `generate_trace_id()`, `generate_span_id()`
- Event routing: conversation_start/end → parent spans, state_transition → child spans, processing → attributed spans, errors → status codes
- Requires: `pip install fsm-llm[otel]` (opentelemetry-api, opentelemetry-sdk)

## Data Models (`definitions.py`)

**Events**: MonitorEvent (event_type, data, timestamp), LogRecord (level, message, timestamp)
**Metrics**: MetricSnapshot (active_conversations, total_events, error_count, transition_count, state_visits)
**Snapshots**: ConversationSnapshot, FSMSnapshot (states, transitions, name), StateInfo, TransitionInfo
**Config**: MonitorConfig (refresh_interval, max_events, max_log_lines, log_level, show_internal_keys, auto_scroll_logs) — note: host/port live on the CLI/`MonitorConfig` is not where the server binds them; there is no `open_browser` field on this model
**Instances**: InstanceInfo (id, type, status, created_at)
**Requests**: LaunchFSMRequest, LaunchAgentRequest, LaunchWorkflowRequest, SendMessageRequest, BuilderStartRequest, BuilderSendRequest

## Constants (`constants.py`)

- **Event type constants** (20 `EVENT_*` names, by value): `conversation_start`, `conversation_end`, `state_transition`, `pre_processing`, `post_processing`, `context_update`, `error`, `log` (reserved/unused — logs use `record_log`), `instance_launched`, `instance_destroyed`, `workflow_started`, `workflow_advanced`, `workflow_completed`, `workflow_cancelled`, `agent_started`, `agent_completed`, `agent_failed`, `agent_iteration`, `agent_cancelled`, `agent_tool_call`
- **THEME**: 11 Grafana dark colors (background=#111217, primary=#3274d9, success=#73bf69, error=#f2495c)
- **DEFAULTS**: refresh_interval=1.0s, max_events=1000, max_log_lines=5000, port=8420

## Testing

```bash
pytest tests/test_fsm_llm_monitor/  # 245 tests, 6 test files
```

## Exceptions

Note: `MonitorError` inherits from `Exception`, NOT from `FSMError` -- it's an infrastructure concern.

```
Exception
└── MonitorError
    ├── MonitorInitializationError  # Server/component startup failures
    ├── MetricCollectionError       # Metric gathering failures
    └── MonitorConnectionError      # API/WebSocket connection issues
```
