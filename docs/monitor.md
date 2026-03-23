# Monitor Dashboard - Real-Time FSM Monitoring

Web-based monitoring dashboard for FSM-LLM conversations, agents, and workflows. Features a retro 90s CRT terminal aesthetic with real-time event streaming via WebSocket.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Dashboard Pages](#dashboard-pages)
4. [Programmatic Integration](#programmatic-integration)
5. [REST API Reference](#rest-api-reference)
6. [WebSocket API](#websocket-api)
7. [Configuration](#configuration)
8. [Data Models](#data-models)
9. [Event Types](#event-types)
10. [Exceptions](#exceptions)
11. [CLI Reference](#cli-reference)

---

## Quick Start

### Install

```bash
pip install fsm-llm[monitor]
```

### Path 1: Standalone CLI

Launch the dashboard server directly from the command line. Opens a browser at `http://127.0.0.1:8420` with the FSM visualizer and preset loader — no live API connection needed.

```bash
# Launch with auto-open browser
fsm-llm-monitor

# Custom host/port, no auto-open
fsm-llm-monitor --host 0.0.0.0 --port 9000 --no-browser
```

### Path 2: Programmatic with Live API

Connect the monitor to a live `API` instance to capture real-time events, metrics, and conversation snapshots.

```python
import threading
import uvicorn
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge
from fsm_llm_monitor.server import app, configure

# Create your FSM API
api = API.from_file("my_bot.json")

# Wire up the monitor
bridge = MonitorBridge(api=api)
configure(bridge)

# Run the web server in a background thread
server_thread = threading.Thread(
    target=uvicorn.run,
    kwargs={"app": app, "host": "127.0.0.1", "port": 8420, "log_level": "warning"},
    daemon=True,
)
server_thread.start()

# Now use the API as normal — events are captured automatically
conversation_id, response = api.start_conversation()
response = api.converse("Hello!", conversation_id)
```

### Path 3: Standalone Visualization Only

Use the dashboard without a live API to visualize FSM definitions from JSON files via the preset browser or by pasting JSON directly.

```bash
fsm-llm-monitor
# Navigate to the Visualizer page in the browser
```

---

## Architecture Overview

```
┌──────────────┐     Handler Callbacks     ┌──────────────────┐
│   FSM API    │ ──────────────────────────>│  EventCollector   │
│  (fsm_llm)   │   (8 timing points,       │  (bounded deques, │
│              │    priority 9999)          │   thread-safe)    │
└──────────────┘                            └────────┬─────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  MonitorBridge    │
                                            │  (query interface,│
                                            │   FSM parsing)    │
                                            └────────┬─────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  FastAPI Server   │
                                            │  (REST + WS,      │
                                            │   static files)   │
                                            └────────┬─────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  Browser SPA      │
                                            │  (vanilla JS,     │
                                            │   retro CRT theme)│
                                            └──────────────────┘
```

### Key Design Decisions

- **Thread-safe bounded deques** — `EventCollector` uses `collections.deque(maxlen=N)` with a threading lock to prevent memory leaks while allowing concurrent event capture.
- **Priority 9999 observe-only handlers** — Monitor handlers are registered at the lowest priority and return empty dicts, ensuring they never interfere with FSM processing.
- **Global bridge pattern** — `configure(bridge)` sets a module-level `_bridge` in `server.py`; `get_bridge()` lazy-initializes an empty bridge if none was configured.
- **No build step** — The frontend is vanilla JavaScript with no framework dependencies or build tooling.

---

## Dashboard Pages

The SPA contains 5 pages, accessible via the navigation bar:

### 1. Dashboard

Real-time metrics overview and live event stream.

- **Metrics cards** — Active conversations, total events, total errors, total transitions
- **Event stream** — Live feed of FSM lifecycle events, auto-scrolling
- **Events per type** — Breakdown by event type

### 2. Visualizer

Interactive FSM, agent, and workflow graph rendering.

- **FSM visualization** — Paste JSON or load from presets. Renders state nodes and transition edges.
- **Agent patterns** — Visualize built-in agent flow patterns (ReactAgent, PlanExecuteAgent, etc.)
- **Workflow patterns** — Visualize workflow step flows from `flows.json`
- **Preset browser** — Browse and load FSM definitions from the `examples/` directory

### 3. Conversations

Inspector for active FSM conversations.

- **Conversation list** — All active conversations with current state and terminal status
- **Detail view** — Context data, message history, stack depth, last extraction/transition/response

### 4. Logs

Level-filtered loguru log stream.

- **Level filter** — DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Auto-scroll** — Configurable via Settings
- **Per-conversation filtering** — Logs tagged with conversation IDs

### 5. Settings

Runtime configuration panel.

- **Refresh interval** — WebSocket push frequency
- **Log level** — Minimum log level to capture
- **Max events / max log lines** — Buffer sizes
- **Show internal keys** — Toggle display of internal context keys
- **Auto-scroll logs** — Toggle auto-scroll behavior

---

## Programmatic Integration

### Monitor Alongside Interactive Runner

```python
import threading
import uvicorn
from fsm_llm import API
from fsm_llm.runner import run_interactive
from fsm_llm_monitor import MonitorBridge
from fsm_llm_monitor.server import app, configure

api = API.from_file("my_bot.json")
bridge = MonitorBridge(api=api)
configure(bridge)

# Start monitor server in background
threading.Thread(
    target=uvicorn.run,
    kwargs={"app": app, "host": "127.0.0.1", "port": 8420, "log_level": "warning"},
    daemon=True,
).start()

# Run the interactive conversation (monitor captures events in real time)
run_interactive(api)
```

### Dynamic Connect/Disconnect

```python
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge

bridge = MonitorBridge()  # No API yet

# Later, connect to an API instance
api = API.from_file("bot.json")
bridge.connect(api)
print(bridge.connected)  # True

# Disconnect when done
bridge.disconnect()
print(bridge.connected)  # False
```

### Log Capture via Loguru Sink

```python
from fsm_llm.logging import logger
from fsm_llm_monitor import EventCollector

collector = EventCollector()
sink = collector.create_loguru_sink()

# Add the sink to loguru — all fsm_llm log messages are captured
sink_id = logger.add(sink, level="DEBUG")

# Logs are now available via collector.get_logs()
logs = collector.get_logs(limit=50, level="INFO")
```

---

## REST API Reference

All endpoints are served by the FastAPI application. Interactive API docs are available at `/api/docs`.

### Monitoring

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `GET` | `/api/metrics` | — | Current metric snapshot (active conversations, event counts, state visit counts) |
| `GET` | `/api/conversations` | — | List all active conversation snapshots |
| `GET` | `/api/conversations/{conversation_id}` | — | Single conversation snapshot (context, history, stack depth) |
| `GET` | `/api/events` | `limit` (int, default: 50) | Recent events, newest first |
| `GET` | `/api/logs` | `limit` (int, default: 100), `level` (str, default: "INFO") | Recent logs filtered by minimum level |
| `GET` | `/api/config` | — | Current monitor configuration |
| `POST` | `/api/config` | — | Update monitor configuration (JSON body: `MonitorConfig`) |
| `GET` | `/api/info` | — | Monitor and fsm_llm version info |

### FSM Visualization

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `POST` | `/api/fsm/load` | — | Load FSM definition from JSON body, return `FSMSnapshot` |
| `POST` | `/api/fsm/visualize` | — | Load FSM definition from JSON body, return nodes + edges for rendering |
| `GET` | `/api/fsm/visualize/preset/{preset_id}` | — | Load FSM preset by ID and return visualization data |

### Presets

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `GET` | `/api/presets` | — | Scan `examples/` directory for FSM presets (metadata only, no paths) |
| `GET` | `/api/preset/fsm/{preset_id}` | — | Load FSM preset JSON content by ID |

### Pattern Visualization

| Method | Path | Query Params | Description |
|--------|------|-------------|-------------|
| `GET` | `/api/agent/visualize` | `agent_type` (str, default: "ReactAgent") | Agent pattern flow visualization |
| `GET` | `/api/workflow/visualize` | `workflow_id` (str, default: "order_processing") | Workflow pattern flow visualization |

### Example

```bash
# Get current metrics
curl http://127.0.0.1:8420/api/metrics

# Get recent events
curl http://127.0.0.1:8420/api/events?limit=10

# Load and visualize an FSM definition
curl -X POST http://127.0.0.1:8420/api/fsm/visualize \
  -H "Content-Type: application/json" \
  -d @my_bot.json

# List available presets
curl http://127.0.0.1:8420/api/presets
```

---

## WebSocket API

### Endpoint

```
ws://host:port/ws
```

Default: `ws://127.0.0.1:8420/ws`

### Message Format

The server pushes messages every 1 second. Each message is a JSON object:

```json
{
  "type": "metrics",
  "data": {
    "timestamp": "2025-01-15T10:30:00",
    "active_conversations": 2,
    "total_events": 47,
    "total_errors": 0,
    "total_transitions": 12,
    "events_per_type": {"state_transition": 12, "pre_processing": 15, "...": "..."},
    "states_visited": {"greeting": 3, "collect_info": 5, "...": "..."}
  },
  "events": [
    {
      "event_type": "state_transition",
      "timestamp": "2025-01-15T10:30:00",
      "conversation_id": "abc-123",
      "source_state": "greeting",
      "target_state": "collect_info",
      "message": "Transition: greeting -> collect_info",
      "level": "INFO",
      "data": {}
    }
  ]
}
```

- The `data` field (metrics) is always present.
- The `events` field is only present when new events have occurred since the last push.
- At most 50 new events are included per message.

---

## Configuration

The `MonitorConfig` Pydantic model controls runtime behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `refresh_interval` | `float` | `1.0` | WebSocket push interval in seconds |
| `max_events` | `int` | `1000` | Maximum events stored in the bounded deque |
| `max_log_lines` | `int` | `5000` | Maximum log records stored in the bounded deque |
| `log_level` | `str` | `"INFO"` | Minimum log level for display |
| `show_internal_keys` | `bool` | `False` | Show internal context keys (prefixed with `_`) |
| `auto_scroll_logs` | `bool` | `True` | Auto-scroll log view to latest entries |

Configuration can be updated at runtime via the Settings page or the REST API:

```python
from fsm_llm_monitor import MonitorConfig

config = MonitorConfig(
    refresh_interval=2.0,
    max_events=5000,
    log_level="DEBUG",
    show_internal_keys=True,
)
bridge = MonitorBridge(api=api, config=config)
```

```bash
# Update via REST API
curl -X POST http://127.0.0.1:8420/api/config \
  -H "Content-Type: application/json" \
  -d '{"refresh_interval": 2.0, "log_level": "DEBUG"}'
```

---

## Data Models

All models are Pydantic v2 `BaseModel` subclasses defined in `fsm_llm_monitor.definitions`.

### MonitorEvent

A single observable event captured from the FSM system.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `event_type` | `str` | *required* | Event type constant (see [Event Types](#event-types)) |
| `timestamp` | `datetime` | `datetime.now()` | When the event occurred |
| `conversation_id` | `str \| None` | `None` | Associated conversation ID |
| `source_state` | `str \| None` | `None` | Source state (for transitions) |
| `target_state` | `str \| None` | `None` | Target state (for transitions) |
| `data` | `dict[str, Any]` | `{}` | Additional event data |
| `level` | `str` | `"INFO"` | Log level |
| `message` | `str` | `""` | Human-readable event description |

### LogRecord

A captured log record from loguru.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timestamp` | `datetime` | `datetime.now()` | When the log was recorded |
| `level` | `str` | `"INFO"` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `message` | `str` | `""` | Log message text |
| `module` | `str` | `""` | Python module that generated the log |
| `function` | `str` | `""` | Function name |
| `line` | `int` | `0` | Source line number |
| `conversation_id` | `str \| None` | `None` | Associated conversation ID (from loguru extras) |

### MetricSnapshot

Point-in-time metric snapshot of the FSM system.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timestamp` | `datetime` | `datetime.now()` | Snapshot time |
| `active_conversations` | `int` | `0` | Number of currently active conversations |
| `total_events` | `int` | `0` | Total events captured since start |
| `total_errors` | `int` | `0` | Total error events |
| `total_transitions` | `int` | `0` | Total state transitions |
| `events_per_type` | `dict[str, int]` | `{}` | Event count breakdown by type |
| `states_visited` | `dict[str, int]` | `{}` | Visit count per state ID |

### ConversationSnapshot

Snapshot of a single conversation's state.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conversation_id` | `str` | *required* | Conversation identifier |
| `current_state` | `str` | `""` | Current FSM state ID |
| `state_description` | `str` | `""` | Description of the current state |
| `is_terminal` | `bool` | `False` | Whether the current state is terminal |
| `context_data` | `dict[str, Any]` | `{}` | Collected context data |
| `message_history` | `list[dict[str, str]]` | `[]` | Conversation message history |
| `stack_depth` | `int` | `1` | FSM stack depth (>1 means pushed sub-FSMs) |
| `last_extraction` | `dict[str, Any] \| None` | `None` | Last Pass 1 extraction response |
| `last_transition` | `dict[str, Any] \| None` | `None` | Last transition decision |
| `last_response` | `dict[str, Any] \| None` | `None` | Last Pass 2 response generation |

### FSMSnapshot

Snapshot of an FSM definition for display.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `""` | FSM name |
| `description` | `str` | `""` | FSM description |
| `version` | `str` | `""` | FSM version string |
| `initial_state` | `str` | `""` | Initial state ID |
| `persona` | `str \| None` | `None` | FSM persona |
| `state_count` | `int` | `0` | Number of states |
| `states` | `list[StateInfo]` | `[]` | State details |

### StateInfo

Information about a single FSM state.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `state_id` | `str` | *required* | State identifier |
| `description` | `str` | `""` | State description |
| `purpose` | `str` | `""` | State purpose |
| `is_initial` | `bool` | `False` | Whether this is the initial state |
| `is_terminal` | `bool` | `False` | Whether this is a terminal state (no outgoing transitions) |
| `transition_count` | `int` | `0` | Number of outgoing transitions |
| `transitions` | `list[TransitionInfo]` | `[]` | Transition details |

### TransitionInfo

Information about a single FSM transition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_state` | `str` | *required* | Target state ID |
| `description` | `str` | `""` | Transition description |
| `priority` | `int` | `0` | Transition priority |
| `condition_count` | `int` | `0` | Number of conditions |
| `has_logic` | `bool` | `False` | Whether any condition uses JsonLogic |

---

## Event Types

Events are captured at 8 handler timing points. Each handler returns an empty dict (pure observation).

| Event Type Constant | Value | Handler Timing | Captured Data |
|---------------------|-------|---------------|---------------|
| `EVENT_CONVERSATION_START` | `"conversation_start"` | `START_CONVERSATION` | Conversation ID |
| `EVENT_PRE_PROCESSING` | `"pre_processing"` | `PRE_PROCESSING` | Conversation ID |
| `EVENT_POST_PROCESSING` | `"post_processing"` | `POST_PROCESSING` | Conversation ID |
| `EVENT_STATE_TRANSITION` | `"state_transition"` | `PRE_TRANSITION` | Conversation ID, source state, target state |
| `EVENT_CONTEXT_UPDATE` | `"context_update"` | `CONTEXT_UPDATE` | Conversation ID |
| `EVENT_CONVERSATION_END` | `"conversation_end"` | `END_CONVERSATION` | Conversation ID |
| `EVENT_ERROR` | `"error"` | `ERROR` | Conversation ID, error message |
| `EVENT_LOG` | `"log"` | *(loguru sink)* | Log level, message, module, function, line |

Note: `POST_TRANSITION` has a registered handler but does not emit events (the transition is already captured at `PRE_TRANSITION`).

---

## Exceptions

| Exception | Parent | Description |
|-----------|--------|-------------|
| `MonitorError` | `Exception` | Base exception for all monitor errors |
| `MonitorInitializationError` | `MonitorError` | Monitor initialization failed |
| `MetricCollectionError` | `MonitorError` | Metric collection failed |
| `MonitorConnectionError` | `MonitorError` | Cannot connect to the API instance |

---

## CLI Reference

```
usage: fsm-llm-monitor [--version] [--info] [--host HOST] [--port PORT] [--no-browser]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--version` | — | Show version and exit |
| `--info` | — | Show feature summary and exit |
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8420` | Port to bind to |
| `--no-browser` | `False` | Don't auto-open the browser |

The CLI can also be invoked as a Python module:

```bash
python -m fsm_llm_monitor [--host HOST] [--port PORT] [--no-browser]
```

Requires: `fastapi`, `uvicorn`, `jinja2`. Install with `pip install fsm-llm[monitor]`.
