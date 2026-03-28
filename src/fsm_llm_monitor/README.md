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
- **Launch modal** to start FSMs, agents, and workflows from the UI
- **Builder page** for interactive artifact construction via the meta-agent
- **WebSocket** for live updates (metrics, events, logs, agent status)

## Installation

```bash
pip install fsm-llm[monitor]
```

**Requirements**: Python 3.10+ | Additional dependencies: fastapi, uvicorn, jinja2

## Quick Start

**1. Launch the dashboard**:

```bash
# Start with defaults (http://127.0.0.1:8420)
fsm-llm-monitor

# Custom host and port
fsm-llm-monitor --host 0.0.0.0 --port 9000

# Without auto-opening browser
fsm-llm-monitor --no-browser
```

**2. Connect programmatically**:

```python
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge, EventCollector, create_server

# Create your FSM API
api = API.from_file("my_fsm.json", model="gpt-4o-mini")

# Set up monitoring
collector = EventCollector()
bridge = MonitorBridge()
bridge.connect(api, collector)

# Launch server
app = create_server(bridge, collector)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8420)
```

**3. Monitor agents and workflows**:

```python
from fsm_llm_monitor import InstanceManager

manager = InstanceManager(collector=collector)

# Launch and monitor an agent
instance_id = manager.launch_agent(
    agent_type="react",
    model="gpt-4o-mini",
    tools_config=[{"name": "search", "type": "builtin"}],
)

# Get real-time status
status = manager.get_agent_status(instance_id)
print(status)
```

## Architecture

```
Browser (SPA)
  │
  ├── REST API ──────────┐
  │                      │
  └── WebSocket ─────┐   │
                     │   │
              ┌──────▼───▼──────────────┐
              │  FastAPI Server          │
              │                         │
              │  ┌───────────────────┐  │
              │  │  MonitorBridge    │──│──→ FSM API instances
              │  └───────────────────┘  │
              │  ┌───────────────────┐  │
              │  │  EventCollector   │──│──→ Events, logs, metrics
              │  └───────────────────┘  │
              │  ┌───────────────────┐  │
              │  │ InstanceManager   │──│──→ FSM, agent, workflow lifecycle
              │  └───────────────────┘  │
              └─────────────────────────┘
```

### Core Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `EventCollector` | `collector.py` | Thread-safe event/log capture with bounded deques and loguru integration |
| `MonitorBridge` | `bridge.py` | Connects to FSM API instances, provides query interface for snapshots |
| `InstanceManager` | `instance_manager.py` | Manages lifecycle of FSM conversations, workflows, and agents |
| FastAPI Server | `server.py` | 35+ REST endpoints + WebSocket for real-time updates |

## Dashboard Pages

### Dashboard
Real-time overview with:
- Metric cards (active conversations, events, errors, transitions)
- Instance grid showing all running FSMs, agents, and workflows
- Live event stream

### Control Center
Unified instance table with detail drawer:
- Start, stop, and manage FSM conversations
- Launch and cancel agents (9 supported agent types)
- Start, advance, and cancel workflows
- View instance details, context, and history

### Conversations
Chat interface for FSM conversations:
- Send messages and view responses
- Track current state and context data
- View conversation history

### Visualizer
Interactive FSM graph rendering:
- Node and edge visualization with state highlighting
- Preset scanning from examples directory
- Tabbed viewer for multiple graphs

### Logs
Level-filtered log viewer:
- Filter by DEBUG, INFO, WARNING, ERROR
- Live streaming with pause/resume
- Log sink connected to loguru

### Builder
Interactive artifact construction:
- Build FSMs, workflows, and agents via conversation
- Powered by the MetaBuilderAgent
- Session management with TTL

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
| GET | `/api/instances` | List all instances (FSM, agent, workflow) |
| POST | `/api/instances/fsm` | Launch new FSM instance |
| POST | `/api/instances/agent` | Launch new agent instance |
| POST | `/api/instances/workflow` | Launch new workflow instance |

### FSM Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List active conversations |
| GET | `/api/conversations/{id}` | Conversation detail |
| POST | `/api/conversations/{id}/message` | Send message |
| POST | `/api/conversations/{id}/end` | End conversation |

### Agent Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agents/{id}/status` | Agent execution status |
| GET | `/api/agents/{id}/result` | Agent result (when complete) |
| POST | `/api/agents/{id}/cancel` | Cancel agent execution |

### Workflow Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workflows/{id}/status` | Workflow status |
| POST | `/api/workflows/{id}/advance` | Advance workflow |
| POST | `/api/workflows/{id}/cancel` | Cancel workflow |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `/ws` | Real-time updates: metrics, events, logs, agent status |

## Key API Reference

### EventCollector

```python
from fsm_llm_monitor import EventCollector

collector = EventCollector(max_events=1000, max_logs=5000)

# Record events manually
collector.record_event("conversation_started", {"conversation_id": "abc"})

# Get metrics
metrics = collector.get_metrics()
print(metrics.active_conversations, metrics.total_events)

# Get logs
logs = collector.get_logs(level="ERROR", limit=50)

# Create handler callbacks for FSM API registration
callbacks = collector.create_handler_callbacks()
```

### MonitorBridge

```python
from fsm_llm_monitor import MonitorBridge

bridge = MonitorBridge()

# Connect to an FSM API instance
bridge.connect(api, collector)

# Query snapshots
snapshot = bridge.get_conversation_snapshot(conversation_id)
all_snapshots = bridge.get_all_conversation_snapshots()

# Load FSM from dict
bridge.load_fsm_from_dict(fsm_dict)
```

### MonitorConfig

```python
from fsm_llm_monitor import MonitorConfig

config = MonitorConfig(
    host="127.0.0.1",
    port=8420,
    refresh_interval=1.0,    # seconds
    max_events=1000,
    max_logs=5000,
    open_browser=True,
)
```

## Frontend Architecture

The dashboard is a single-page application built with vanilla JavaScript:

```
static/
├── app.js              # Main application module
├── style.css           # Grafana-inspired dark theme
├── flows.json          # Agent/workflow pattern flow definitions
├── pages/              # Page components (dashboard, control, conversations, ...)
├── services/           # API client, state management, WebSocket
└── utils/              # DOM helpers, formatting, graph rendering, markdown
```

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
