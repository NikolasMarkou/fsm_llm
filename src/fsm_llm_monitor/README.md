# FSM-LLM Monitor — Web-Based Monitoring Dashboard

A real-time web dashboard for monitoring FSM-LLM conversations, agent executions, and workflow instances. Grafana-inspired dark theme with live WebSocket streaming. Part of the `fsm-llm` package.

```bash
pip install fsm-llm[monitor]
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Programmatic Integration](#programmatic-integration)
- [Dashboard Pages](#dashboard-pages)
- [REST API](#rest-api)
- [WebSocket Protocol](#websocket-protocol)
- [Architecture](#architecture)
- [File Map](#file-map)
- [Exception Hierarchy](#exception-hierarchy)
- [Development](#development)

---

## Quick Start

### Standalone (CLI)

```bash
# Launch the monitor (opens browser at http://127.0.0.1:8420)
fsm-llm-monitor

# Or with Python module syntax
python -m fsm_llm_monitor

# Custom host/port, no auto-open
fsm-llm-monitor --host 0.0.0.0 --port 9000 --no-browser
```

The standalone mode launches an empty dashboard where you can launch FSM instances from presets or custom JSON, run agents, and monitor everything in real time.

### With an Existing API

```python
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge
from fsm_llm_monitor.server import app, configure

# Create your FSM API
api = API.from_file("my_bot.json", model="gpt-4o-mini")

# Wire up the monitor
bridge = MonitorBridge(api=api)
configure(bridge)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8420)
```

---

## Programmatic Integration

### MonitorBridge

The `MonitorBridge` connects an `EventCollector` to a live FSM API and provides a unified query interface for the web server.

```python
from fsm_llm_monitor import MonitorBridge, MonitorConfig, EventCollector

# Default config
bridge = MonitorBridge(api=my_api)

# Custom config
bridge = MonitorBridge(
    api=my_api,
    config=MonitorConfig(
        refresh_interval=0.5,
        max_events=2000,
        max_log_lines=10000,
        log_level="DEBUG",
    ),
)

# Query interface
metrics = bridge.get_metrics()           # MetricSnapshot
events = bridge.get_events(limit=50)     # list[MonitorEvent]
convs = bridge.get_conversations()       # list[ConversationSnapshot]
conv = bridge.get_conversation(conv_id)  # ConversationSnapshot
logs = bridge.get_logs(level="WARNING")  # list[LogRecord]
```

### EventCollector

Low-level event capture with thread-safe bounded deques. Used internally by `MonitorBridge`, but can be used standalone for custom monitoring.

```python
from fsm_llm_monitor import EventCollector

collector = EventCollector(max_events=1000, max_log_lines=5000)

# Create handler callbacks for all 8 timing points
callbacks = collector.create_handler_callbacks()
# Returns: dict mapping HandlerTiming → callback function

# Record events manually
collector.record_event("conversation_start", "Started conv-123", conversation_id="conv-123")

# Query
metrics = collector.get_metrics()    # MetricSnapshot
events = collector.get_events(50)    # list[MonitorEvent]
```

### InstanceManager

Manages the lifecycle of FSM, agent, and workflow instances launched through the dashboard.

```python
from fsm_llm_monitor import InstanceManager

manager = InstanceManager(bridge)

# Launch instances
fsm_info = manager.launch_fsm(preset_id="basic/simple_greeting", model="gpt-4o-mini")
agent_info = manager.launch_agent(agent_type="ReactAgent", task="Search for X", tools=[...])

# Query
instances = manager.list_instances()       # list[InstanceInfo]
instance = manager.get_instance(inst_id)   # InstanceInfo

# Cleanup
manager.destroy_instance(inst_id)
```

---

## Dashboard Pages

| Page | Keyboard | Description |
|------|----------|-------------|
| **Dashboard** | `1` | Live metrics cards (conversations, events, transitions, errors), event log, instance grid |
| **Visualizer** | `2` | Interactive FSM/agent/workflow graph visualization with SVG rendering |
| **Conversations** | `3` | Conversation inspector with live chat, context data, LLM interaction panels (extraction, transition, response) |
| **Control Center** | `4` | Instance management — launch, inspect, start conversations, cancel agents, destroy instances |
| **Logs** | `5` | Filterable log viewer with level and text search |
| **Settings** | `6` | Configuration (refresh interval, max events, log level) + system info |

### Keyboard Shortcuts

- `1`-`6` — Switch pages
- `Escape` — Close launch modal
- `Enter` — Send chat message (in conversation view)

---

## REST API

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/metrics` | Current metric snapshot (active conversations, total events/transitions/errors) |
| `GET` | `/api/events?limit=50&offset=0` | Recent events, newest first |
| `GET` | `/api/logs?limit=500&level=INFO` | Log records filtered by minimum level |
| `GET` | `/api/config` | Current monitor configuration |
| `POST` | `/api/config` | Update configuration (refresh_interval, max_events, max_log_lines, log_level) |
| `GET` | `/api/info` | System info (version, Python version, extensions installed) |
| `GET` | `/api/capabilities` | Feature flags: `{fsm: true, workflows: bool, agents: bool}` |

### Instance Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/instances` | List all managed instances with status |
| `GET` | `/api/instances/{id}` | Single instance details |
| `GET` | `/api/instances/{id}/events?limit=50` | Events filtered by instance |
| `DELETE` | `/api/instances/{id}` | Destroy an instance |

### FSM Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/fsm/launch` | Launch FSM from `preset_id` or `fsm_json` with model/temperature/label |
| `POST` | `/api/fsm/{id}/start` | Start a new conversation on an FSM instance |
| `POST` | `/api/fsm/{id}/converse` | Send a message to an active conversation |
| `POST` | `/api/fsm/{id}/end` | End a conversation |
| `GET` | `/api/fsm/{id}/conversations` | List conversations on an FSM instance |
| `POST` | `/api/fsm/visualize` | Visualize FSM from JSON body → `{nodes, edges, info}` |
| `GET` | `/api/fsm/visualize/preset/{id}` | Visualize a preset FSM |

### Agent Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agent/launch` | Launch agent (type, task, model, tools, max_iterations) |
| `GET` | `/api/agent/{id}/status` | Agent status, iteration count, current state |
| `GET` | `/api/agent/{id}/result` | Final result with execution trace |
| `POST` | `/api/agent/{id}/cancel` | Cancel a running agent |
| `GET` | `/api/agent/visualize?agent_type=X` | Agent pattern flow graph |

### Workflow Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/workflow/launch` | Launch workflow instance |
| `POST` | `/api/workflow/{id}/advance` | Advance workflow to next step |
| `POST` | `/api/workflow/{id}/cancel` | Cancel workflow |
| `GET` | `/api/workflow/{id}/status` | Workflow execution status |
| `GET` | `/api/workflow/{id}/instances` | List workflow instances |
| `GET` | `/api/workflow/visualize?workflow_id=X` | Workflow pattern flow graph |

### Presets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/presets` | List all available presets (scans `examples/` directory) |
| `GET` | `/api/preset/fsm/{id}` | Load an FSM preset definition |

---

## WebSocket Protocol

Connect to `/ws` for real-time streaming. Messages are JSON objects pushed every ~1 second:

```json
{
  "type": "metrics",
  "data": {
    "active_conversations": 2,
    "total_events": 47,
    "total_transitions": 12,
    "total_errors": 0
  },
  "events": [
    {
      "event_type": "state_transition",
      "message": "start → greeting",
      "timestamp": "2026-03-23T10:15:30.000Z",
      "level": "INFO",
      "conversation_id": "conv-abc123"
    }
  ],
  "instances": [...],
  "agent_updates": {}
}
```

Fields:
- `type` — always `"metrics"`
- `data` — latest `MetricSnapshot`
- `events` — new events since last push (up to 50), only included when present
- `instances` — full instance list, included on every message
- `agent_updates` — running agent status updates, included when agents are active

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Browser SPA (vanilla JS)                            │
│  ┌──────────┬──────────┬──────────┬────────────────┐ │
│  │Dashboard │Visualizer│  Convs   │ Control Center │ │
│  └────┬─────┴────┬─────┴────┬─────┴───────┬────────┘ │
│       │          │          │             │          │
│       └──────────┴──────────┴─────────────┘          │
│                      │  WebSocket + REST              │
└──────────────────────┼───────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────┐
│  FastAPI Server      │  (server.py)                  │
│  35 endpoints + /ws  │                               │
│                      ▼                               │
│  ┌──────────────────────────────┐                    │
│  │  MonitorBridge (bridge.py)   │                    │
│  │  Unified query interface     │                    │
│  └──────────┬───────────────────┘                    │
│             │                                        │
│  ┌──────────┼──────────────────────────────┐         │
│  │  InstanceManager (instance_manager.py)  │         │
│  │  FSM / Agent / Workflow lifecycle       │         │
│  └──────────┬──────────────────────────────┘         │
│             │                                        │
│  ┌──────────┼──────────────────────────────┐         │
│  │  EventCollector (collector.py)          │         │
│  │  Thread-safe bounded deques             │         │
│  │  Handler callbacks at priority 9999     │         │
│  └──────────┬──────────────────────────────┘         │
└─────────────┼────────────────────────────────────────┘
              │
    ┌─────────┴──────────┐
    │  fsm_llm.API       │
    │  8 handler timing  │
    │  points observed   │
    └────────────────────┘
```

### Handler Timing Points Observed

| Timing | Event Type | What's Captured |
|--------|-----------|-----------------|
| `START_CONVERSATION` | `conversation_start` | Conversation ID, initial state |
| `END_CONVERSATION` | `conversation_end` | Conversation ID, final state |
| `PRE_PROCESSING` | `pre_processing` | User message received |
| `POST_PROCESSING` | `post_processing` | Assistant response generated |
| `PRE_TRANSITION` | `state_transition` | From state → to state, trigger |
| `POST_TRANSITION` | *(no event)* | Registered but silent (captured at PRE) |
| `CONTEXT_UPDATE` | `context_update` | Context key changes |
| `ERROR` | `error` | Exception type, message |

---

## File Map

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app — 35 REST endpoints + WebSocket, serves SPA |
| `bridge.py` | **MonitorBridge** — EventCollector ↔ API connector, unified query interface |
| `collector.py` | **EventCollector** — thread-safe event/log capture with bounded deques |
| `instance_manager.py` | **InstanceManager** — FSM/agent/workflow instance lifecycle |
| `definitions.py` | Pydantic models for events, metrics, snapshots, configs, requests |
| `constants.py` | Theme colors, 17 event types, defaults, handler config |
| `exceptions.py` | MonitorError → MonitorInitializationError, MetricCollectionError, MonitorConnectionError |
| `__main__.py` | CLI: `fsm-llm-monitor` / `python -m fsm_llm_monitor` |
| `__init__.py` | Public API exports |
| `__version__.py` | Version from fsm_llm |
| `static/` | Modular vanilla JS frontend (13 modules — see `static/README.md`) |
| `templates/index.html` | Single-page HTML template |

---

## Exception Hierarchy

```
MonitorError (extends FSMError)
├── MonitorInitializationError    # Bridge/collector setup failures
├── MetricCollectionError         # Metric snapshot collection failures
└── MonitorConnectionError        # WebSocket/API connection failures
```

---

## Development

```bash
# Install with monitor extra
pip install -e ".[dev,monitor]"

# Run tests
pytest tests/test_fsm_llm_monitor/  # 86 tests

# Launch in development
python -m fsm_llm_monitor --port 8420
```

---

## License

GPL-3.0-or-later. See the [main project](https://github.com/NikolasMarkou/fsm_llm) for details.
