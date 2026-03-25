# fsm_llm_monitor

Web-based real-time monitoring dashboard for FSM-LLM. Tracks FSM conversations, agent executions, and workflow instances through a Grafana-inspired dark-themed single-page application. Built on FastAPI with vanilla JavaScript -- no build step required.

## Features

- Real-time WebSocket streaming of metrics, events, and logs (configurable poll interval, default 1 second)
- Grafana-inspired dark dashboard theme (charcoal backgrounds, blue primary, orange accents)
- 5 dashboard pages: Dashboard, Control Center, Visualizer, Logs, Settings
- 40 REST endpoints covering monitoring, instance management, FSM/agent/workflow control, presets, and visualization
- Launch, control, and destroy FSM, agent, and workflow instances from the browser
- Event collection from all 8 handler timing points (pure observer pattern)
- Preset scanning from the `examples/` directory with path traversal protection
- Builder integration with `fsm_llm_meta` for conversational artifact creation

## Installation

```bash
pip install fsm-llm[monitor]
```

This installs the additional dependencies: `fastapi`, `uvicorn`, `jinja2`.

## Quick Start

### Standalone CLI

Launch the dashboard with no connected API -- useful for exploring presets and the UI:

```bash
# Opens browser at http://127.0.0.1:8420
fsm-llm-monitor

# Or with custom host/port
fsm-llm-monitor --host 0.0.0.0 --port 9000

# Suppress auto-open browser
fsm-llm-monitor --no-browser
```

### Programmatic

Connect the monitor to a live FSM API instance:

```python
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge, configure, app

# Create your API and bridge
api = API.from_file("my_fsm.json", model="gpt-4o-mini")
bridge = MonitorBridge(api=api)

# Configure the server with the bridge
configure(bridge)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8420)
```

Or use the `InstanceManager` directly for multi-instance management:

```python
from fsm_llm_monitor import InstanceManager, configure, app

manager = InstanceManager()
configure(manager=manager)

# Launch FSMs, agents, and workflows via REST API or programmatically
```

## Dashboard Pages

### Dashboard

Overview page with metric cards (active conversations, total events, errors, transitions), an instance grid showing all managed FSM/agent/workflow instances, and a live event feed.

### Control Center

Unified instance management table with an expandable drawer for each instance. Launch new FSM, agent, or workflow instances. Start conversations, send messages, and view real-time status. Supports launching from presets or custom JSON definitions.

### Visualizer

Tabbed graph viewer for FSM state diagrams, agent pattern flows, and workflow step graphs. Supports loading from presets, custom JSON input, or active instances. Renders interactive node-edge graphs with state metadata.

### Logs

Live log stream captured from loguru. Supports level filtering (DEBUG, INFO, WARNING, ERROR, CRITICAL), live/pause toggle, and automatic scrolling. Logs are stored in a bounded deque (default 5,000 entries).

### Settings

Runtime configuration editor for refresh interval, max events, max log lines, log level, and display options. Also shows system info (monitor version, fsm_llm version) and installed capabilities (FSM, workflows, agents).

## Architecture

```
Handler Callbacks (8 timing points)
        |
        v
EventCollector (bounded deques, thread-safe)
        |
        v
InstanceManager (per-instance + global collectors)
        |
        v
FastAPI Server (REST + WebSocket)
        |
        v
Browser SPA (vanilla JS, 16 modules)
```

### Data Flow

1. **Event capture**: Monitor handlers are registered at priority 9999 (lowest) on each API instance. They observe all 8 handler timing points and record `MonitorEvent` objects into bounded deques.
2. **Log capture**: A loguru sink feeds `LogRecord` objects into the global collector.
3. **Query**: REST endpoints query the `InstanceManager` and its collectors for metrics, events, logs, conversations, and instance status.
4. **Streaming**: The WebSocket endpoint polls at the configured refresh interval (default 1 second) and pushes new metrics, events, logs, instance lists, and agent status updates to connected clients.

## REST API

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the single-page application |
| GET | `/health` | Health check |
| GET | `/api/metrics` | Current MetricSnapshot |
| GET | `/api/events` | Recent events (query: `limit`) |
| GET | `/api/logs` | Log records (query: `limit`, `level`) |
| GET | `/api/config` | Current MonitorConfig |
| POST | `/api/config` | Update MonitorConfig |
| GET | `/api/info` | Monitor and fsm_llm versions |
| GET | `/api/capabilities` | Installed extension flags |
| GET | `/api/conversations` | All conversation snapshots (query: `include_ended`) |
| GET | `/api/conversations/{id}` | Single conversation snapshot |

### Instance Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/instances` | List all managed instances (query: `type`) |
| GET | `/api/instances/{id}` | Single instance detail |
| GET | `/api/instances/{id}/events` | Events for a specific instance |
| DELETE | `/api/instances/{id}` | Destroy an instance |

### FSM Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/fsm/launch` | Launch FSM from preset or JSON |
| POST | `/api/fsm/{id}/start` | Start a conversation |
| POST | `/api/fsm/{id}/converse` | Send a message |
| POST | `/api/fsm/{id}/end` | End a conversation |
| GET | `/api/fsm/{id}/conversations` | List conversations on an instance |
| POST | `/api/fsm/load` | Load FSM definition (returns FSMSnapshot) |
| POST | `/api/fsm/visualize` | Visualize FSM from JSON (returns nodes + edges) |
| GET | `/api/fsm/visualize/preset/{id}` | Visualize a preset FSM |

### Agent Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/agent/launch` | Launch an agent in a background thread |
| GET | `/api/agent/{id}/status` | Agent execution status |
| GET | `/api/agent/{id}/result` | Final agent result and trace |
| POST | `/api/agent/{id}/cancel` | Cancel a running agent |
| GET | `/api/agent/visualize` | Agent pattern flow graph (query: `agent_type`) |

### Workflow Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/workflow/launch` | Launch a workflow instance |
| POST | `/api/workflow/{id}/advance` | Advance a workflow step |
| POST | `/api/workflow/{id}/cancel` | Cancel a workflow |
| GET | `/api/workflow/{id}/status` | Workflow instance status |
| GET | `/api/workflow/{id}/instances` | List workflow instances |
| GET | `/api/workflow/visualize` | Workflow pattern flow graph (query: `workflow_id`) |

### Presets

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/presets` | Scan examples/ for FSM presets |
| GET | `/api/preset/fsm/{id}` | Load a preset FSM definition |

### Builder (Meta-Agent)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/builder/start` | Start a new builder session |
| POST | `/api/builder/send` | Send a message to a builder session |
| GET | `/api/builder/result/{id}` | Get builder session state |
| DELETE | `/api/builder/{id}` | Delete a builder session |

## WebSocket Protocol

Connect to `ws://host:port/ws` for real-time updates.

### Message Format

Each message is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"metrics"` |
| `data` | object | Current `MetricSnapshot` |
| `events` | array | New events since last push (up to 50), present only when there are new events |
| `logs` | array | New log records since last push (up to 50), present only when there are new logs |
| `log_count` | int | Total log count (present with `logs`) |
| `instances` | array | Full instance list (always present) |
| `agent_updates` | object | Status updates for running agents (present only when agents are running) |

Messages are pushed at the configured refresh interval (default 1 second). The server tracks `last_event_count` and `last_log_count` per connection to send only incremental updates.

## Frontend Architecture

The frontend is a single-page application built with vanilla JavaScript -- no framework, no build step. All 16 JS modules are loaded via `<script>` tags in dependency order from `templates/index.html`.

| Module | Purpose |
|--------|---------|
| `state.js` | Global `App` namespace and shared state |
| `utils.js` | Shared utility functions (formatting, DOM helpers) |
| `websocket.js` | WebSocket connection management and message dispatch |
| `nav.js` | Sidebar navigation and page switching |
| `dashboard.js` | Dashboard page -- metric cards, instance grid, events |
| `control.js` | Control Center -- instance table with expandable drawer |
| `conversations.js` | Conversation detail view and chat interface |
| `launch.js` | Launch modal for FSMs, agents, workflows |
| `graph.js` | FSM/agent/workflow graph rendering (canvas-based) |
| `visualizer.js` | Visualizer page -- tabbed graph viewer with presets |
| `logs.js` | Logs page -- level-filtered stream with live/pause |
| `settings.js` | Settings page -- runtime config and system info |
| `markdown.js` | Markdown rendering utilities |
| `builder.js` | Builder page -- meta-agent conversational interface |
| `app.js` | Top-level app orchestration |
| `init.js` | App initialization and boot sequence |

All functions are exposed globally on the `App` namespace or as top-level functions for HTML `onclick` handler compatibility.

## File Map

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app -- 40 REST endpoints + WebSocket, serves SPA |
| `bridge.py` | MonitorBridge -- connects EventCollector to live API |
| `collector.py` | EventCollector -- thread-safe bounded deques, handler callbacks, loguru sink |
| `instance_manager.py` | InstanceManager -- FSM/agent/workflow lifecycle management |
| `definitions.py` | Pydantic models for events, metrics, snapshots, requests |
| `constants.py` | Theme colors, event type constants, defaults |
| `exceptions.py` | MonitorError hierarchy |
| `__main__.py` | CLI entry point |
| `__init__.py` | Public exports (56 symbols) |
| `__version__.py` | Version synced from fsm_llm |
| `static/` | 16 vanilla JS modules + CSS + flows.json |
| `static/style.css` | Grafana-inspired dark theme with CSS custom properties |
| `static/flows.json` | Agent/workflow pattern flow definitions |
| `templates/index.html` | Single-page HTML template |

## Development

### Running Tests

```bash
pytest tests/test_fsm_llm_monitor/  # 171 tests
```

Tests are spread across 5 files:

| File | Scope |
|------|-------|
| `test_app.py` | FastAPI endpoint integration tests |
| `test_bridge.py` | MonitorBridge unit tests |
| `test_collector.py` | EventCollector unit tests |
| `test_definitions.py` | Pydantic model tests |
| `test_instance_manager.py` | InstanceManager unit tests |

### CLI Options

```
fsm-llm-monitor [OPTIONS]

  --host HOST        Host to bind to (default: 127.0.0.1)
  --port PORT        Port to bind to (default: 8420)
  --no-browser       Don't auto-open the browser
  --version          Show version and exit
  --info             Show monitor info and exit
```
