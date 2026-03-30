# FSM-LLM Monitor

> Web-based monitoring dashboard with real-time observability for FSMs, agents, and workflows.

---

## Overview

`fsm_llm_monitor` is a web-based monitoring dashboard for FSM-LLM systems. It provides real-time visibility into running conversations, agent executions, and workflow instances through a Grafana-inspired dark-themed UI.

Key capabilities:
- **Real-time dashboard** with metric cards, instance grid, and event stream
- **Conversation viewer** with chat interface and state tracking
- **Control center** for managing FSM, agent, and workflow instances
- **FSM visualizer** with interactive graph rendering
- **Log viewer** with level filtering and live/pause streaming
- **Builder page** for interactive artifact construction via the meta-agent
- **WebSocket** for live updates (metrics, events, logs, agent status)
- **OpenTelemetry export** via `OTELExporter` -- send spans to Jaeger, Datadog, Langfuse, etc.

## Installation

```bash
pip install fsm-llm[monitor]
```

**Requirements**: Python 3.10+ | Additional dependencies: fastapi, uvicorn, jinja2

## Quick Start

**1. Launch the dashboard**:

```bash
fsm-llm-monitor                          # http://127.0.0.1:8420
fsm-llm-monitor --host 0.0.0.0 --port 9000  # Custom host/port
fsm-llm-monitor --no-browser             # Without auto-opening browser
```

**2. Connect programmatically**:

```python
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge, EventCollector, create_server

api = API.from_file("my_fsm.json", model="gpt-4o-mini")
collector = EventCollector()
bridge = MonitorBridge()
bridge.connect(api, collector)

app = create_server(bridge, collector)

import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8420)
```

**3. Monitor agents and workflows**:

```python
from fsm_llm_monitor import InstanceManager

manager = InstanceManager(collector=collector)
instance_id = manager.launch_agent(agent_type="react", model="gpt-4o-mini",
                                   tools_config=[{"name": "search", "type": "builtin"}])
status = manager.get_agent_status(instance_id)
```

## Architecture

```
Browser (SPA) ──── REST API ──────┐
                                  │
              ──── WebSocket ─┐   │
                              ▼   ▼
                    FastAPI Server
                    ├── MonitorBridge    → FSM API instances
                    ├── EventCollector   → Events, logs, metrics
                    └── InstanceManager  → FSM, agent, workflow lifecycle
```

### Core Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `EventCollector` | `collector.py` | Thread-safe event/log capture with bounded deques and loguru integration |
| `MonitorBridge` | `bridge.py` | Connects to FSM API instances, provides query interface for snapshots |
| `InstanceManager` | `instance_manager.py` | Manages lifecycle of FSM conversations, workflows, and agents |
| FastAPI Server | `server.py` | 35+ REST endpoints + WebSocket for real-time updates |

## Dashboard Pages

- **Dashboard** -- Metric cards, instance grid, live event stream
- **Control Center** -- Unified instance table with detail drawer for FSMs, agents, workflows
- **Conversations** -- Chat interface with state tracking and context view
- **Visualizer** -- Interactive FSM graph rendering with presets
- **Logs** -- Level-filtered stream (DEBUG/INFO/WARNING/ERROR) with live/pause
- **Builder** -- Interactive artifact construction via MetaBuilderAgent
- **Settings** -- Runtime config and system info

## REST API Endpoints

### Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/metrics` | Current metric snapshot |
| GET | `/api/events` | Recent events (with limit param) |
| GET | `/api/logs` | Recent logs (with limit and level filter) |

### Instance Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/instances` | List all instances |
| POST | `/api/instances/fsm` | Launch new FSM instance |
| POST | `/api/instances/agent` | Launch new agent instance |
| POST | `/api/instances/workflow` | Launch new workflow instance |

### FSM / Agent / Workflow Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/api/conversations/{id}/*` | Conversation CRUD + messaging |
| GET/POST | `/api/agents/{id}/*` | Agent status, result, cancel |
| GET/POST | `/api/workflows/{id}/*` | Workflow status, advance, cancel |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `/ws` | Real-time updates: metrics, events, logs, agent status |

## Key API Reference

### EventCollector

```python
collector = EventCollector(max_events=1000, max_logs=5000)
collector.record_event("conversation_started", {"conversation_id": "abc"})
metrics = collector.get_metrics()
logs = collector.get_logs(level="ERROR", limit=50)
callbacks = collector.create_handler_callbacks()
```

### MonitorBridge

```python
bridge = MonitorBridge()
bridge.connect(api, collector)
snapshot = bridge.get_conversation_snapshot(conversation_id)
all_snapshots = bridge.get_all_conversation_snapshots()
bridge.load_fsm_from_dict(fsm_dict)
```

### MonitorConfig

```python
from fsm_llm_monitor import MonitorConfig
config = MonitorConfig(host="127.0.0.1", port=8420, refresh_interval=1.0,
                       max_events=1000, max_logs=5000, open_browser=True)
```

### OTELExporter (OpenTelemetry)

```python
from fsm_llm_monitor import EventCollector
from fsm_llm_monitor.otel import OTELExporter

collector = EventCollector()
otel = OTELExporter(service_name="my-chatbot")
otel.enable(collector)

# Events are now also exported as OTEL spans
# Supports: Jaeger, Datadog, Langfuse, or any OTEL-compatible backend
```

Requires: `pip install fsm-llm[otel]`

## Exception Hierarchy

```
Exception
└── MonitorError
    ├── MonitorInitializationError  # Server/component startup failures
    ├── MetricCollectionError       # Metric gathering failures
    └── MonitorConnectionError      # API/WebSocket connection issues
```

Note: `MonitorError` inherits from `Exception` (not `FSMError`) since it's an infrastructure concern.

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
