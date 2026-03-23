# fsm_llm_monitor — Web-Based Monitoring Dashboard

## What This Package Does

Web-based real-time monitoring dashboard for FSM-LLM conversations, agents, and workflows. Captures events via handler callbacks at all 8 timing points, streams metrics and events to the browser via WebSocket, and serves a single-page application with a Grafana-inspired dark dashboard theme (charcoal backgrounds, blue primary, orange accents).

## File Map

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app — 35 REST endpoints + WebSocket, serves SPA, global bridge pattern via `configure()` / `get_bridge()` |
| `bridge.py` | **MonitorBridge** — connects EventCollector to live API, registers observer handlers, provides unified query interface |
| `collector.py` | **EventCollector** — thread-safe bounded deques, handler callbacks, loguru sink, metric counters |
| `instance_manager.py` | **InstanceManager** — manages FSM/agent/workflow instance lifecycle, registers monitor handlers |
| `definitions.py` | Pydantic models: MonitorEvent, LogRecord, MetricSnapshot, ConversationSnapshot, FSMSnapshot, StateInfo, TransitionInfo, MonitorConfig, request/response models |
| `constants.py` | Theme colors, 17 event type constants, defaults (refresh interval, max events, max log lines), handler config |
| `exceptions.py` | MonitorError hierarchy: MonitorInitializationError, MetricCollectionError, MonitorConnectionError |
| `__main__.py` | CLI entry point: `fsm-llm-monitor` / `python -m fsm_llm_monitor` |
| `__init__.py` | Public exports (50 items) |
| `__version__.py` | Version import from fsm_llm |
| `static/` | Modular vanilla JS frontend (13 modules) — see `static/CLAUDE.md` for full file map |
| `static/style.css` | Grafana-inspired dark dashboard theme with CSS custom properties |
| `static/flows.json` | Agent/workflow pattern flow definitions for the visualizer |
| `templates/index.html` | Single-page template (6 pages: Dashboard, Visualizer, Conversations, Control Center, Logs, Settings) |

## Key Patterns

### Data Flow
```
Handler callbacks → EventCollector (bounded deques) → MonitorBridge (query interface)
    → FastAPI (REST + WebSocket) → Browser SPA (vanilla JS)
```

### Global Bridge Pattern
`configure(bridge)` sets module-level `_bridge` in `server.py`. `get_bridge()` lazy-initializes an empty `MonitorBridge()` if none was configured. All endpoints call `get_bridge()` to access data.

### Instance Management
`InstanceManager` manages the full lifecycle of FSM, agent, and workflow instances. Each launched instance gets a unique ID, label, and type. The manager tracks status, registers monitor handlers per-instance, and provides query methods for the REST API.

### Event Collection
8 handler timing points are mapped to event types via `EventCollector.create_handler_callbacks()`. Handlers are registered at priority 9999 (lowest) and return empty dicts — pure observers that never modify FSM state. `POST_TRANSITION` handler is registered but does not emit events (transition captured at `PRE_TRANSITION`).

### WebSocket Streaming
1-second poll loop in `websocket_endpoint()`. Each message contains the latest `MetricSnapshot`. If new events exist since the last push, up to 50 events are included in the `events` field. Instance list and agent updates are also pushed.

### Thread Safety
`EventCollector` uses a single `threading.Lock` around all deque operations and metric counter updates. Deques have `maxlen` set to prevent unbounded memory growth.

### Frontend Architecture
13 vanilla JS modules loaded via `<script>` tags in dependency order. Shared state in `App` namespace (`state.js`). All functions are global for HTML `onclick` handler compatibility. See `static/CLAUDE.md` for details.

## REST API Endpoints

### Core
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve SPA |
| GET | `/api/metrics` | Current MetricSnapshot |
| GET | `/api/events` | Recent events (limit, offset) |
| GET | `/api/logs` | Log records (level filter) |
| GET | `/api/config` | Current MonitorConfig |
| POST | `/api/config` | Update MonitorConfig |
| GET | `/api/info` | System info + version |
| GET | `/api/capabilities` | Feature flags (fsm, workflows, agents) |
| WS | `/ws` | Real-time metrics + events stream |

### Instance Management
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/instances` | List all managed instances |
| GET | `/api/instances/{id}` | Single instance info |
| GET | `/api/instances/{id}/events` | Events filtered by instance |
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
| POST | `/api/agent/launch` | Launch agent instance |
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

## Dependencies on Core
- `fsm_llm.API` — FSM execution, conversation queries
- `fsm_llm.HandlerTiming` — handler timing point enum
- `fsm_llm.create_handler` — fluent handler builder
- `fsm_llm.logging.logger` — loguru logging (for sink integration)
- External: `fastapi`, `uvicorn`, `jinja2`, `pydantic`, `loguru`

## Testing
```bash
pytest tests/test_fsm_llm_monitor/  # 86 tests
```

## Gotchas
- `configure()` must be called before endpoints work, or `get_bridge()` auto-creates an empty bridge with no API connection
- Handlers registered at priority 9999 and return empty dicts — pure observers, never interfere with FSM processing
- EventCollector uses bounded deques (`max_events=1000`, `max_log_lines=5000`) to prevent memory leaks
- Frontend is vanilla JS — no build step, no framework dependencies. See `static/CLAUDE.md` for module details
- Version synced from `fsm_llm.__version__` — not independent
- Preset endpoints scan `examples/` directory but never expose filesystem paths to clients (path traversal protection)
- CLI defaults to port 8420 and auto-opens browser (use `--no-browser` to disable)
