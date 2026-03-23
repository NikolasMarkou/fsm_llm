# fsm_llm_monitor — Web-Based Monitoring Dashboard

## What This Package Does

Web-based real-time monitoring dashboard for FSM-LLM conversations, agents, and workflows. Captures events via handler callbacks at all 8 timing points, streams metrics and events to the browser via WebSocket, and serves a single-page application with a Grafana-inspired dark dashboard theme (charcoal backgrounds, blue primary, orange accents).

## File Map

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app — 16 REST endpoints + WebSocket, serves SPA, global bridge pattern via `configure()` / `get_bridge()` |
| `bridge.py` | **MonitorBridge** — connects EventCollector to live API, registers 8 observer handlers at priority 9999 |
| `collector.py` | **EventCollector** — thread-safe bounded deques, handler callbacks, loguru sink, metric counters |
| `definitions.py` | Pydantic models: MonitorEvent, LogRecord, MetricSnapshot, ConversationSnapshot, FSMSnapshot, StateInfo, TransitionInfo, MonitorConfig |
| `constants.py` | Theme colors, 8 event type constants, defaults (refresh interval, max events, max log lines), handler config |
| `exceptions.py` | MonitorError hierarchy: MonitorInitializationError, MetricCollectionError, MonitorConnectionError |
| `__main__.py` | CLI entry point: `fsm-llm-monitor` / `python -m fsm_llm_monitor` |
| `__init__.py` | Public exports (36 items) |
| `__version__.py` | Version import from fsm_llm |
| `static/app.js` | SPA logic — navigation, WebSocket client, graph rendering (vanilla JS, no framework) |
| `static/style.css` | Grafana-inspired dark dashboard theme |
| `static/flows.json` | Agent/workflow pattern flow definitions for the visualizer |
| `templates/index.html` | Single-page template (5 pages: Dashboard, Visualizer, Conversations, Logs, Settings) |

## Key Patterns

### Data Flow
Handler callbacks → EventCollector (bounded deques) → MonitorBridge (query interface) → FastAPI (REST + WebSocket) → Browser SPA

### Global Bridge Pattern
`configure(bridge)` sets module-level `_bridge` in `server.py`. `get_bridge()` lazy-initializes an empty `MonitorBridge()` if none was configured. All endpoints call `get_bridge()` to access data.

### Event Collection
8 handler timing points are mapped to event types via `EventCollector.create_handler_callbacks()`. Handlers are registered at priority 9999 (lowest) and return empty dicts — pure observers that never modify FSM state. `POST_TRANSITION` handler is registered but does not emit events (transition captured at `PRE_TRANSITION`).

### WebSocket Streaming
1-second poll loop in `websocket_endpoint()`. Each message contains the latest `MetricSnapshot`. If new events exist since the last push, up to 50 events are included in the `events` field.

### Thread Safety
`EventCollector` uses a single `threading.Lock` around all deque operations and metric counter updates. Deques have `maxlen` set to prevent unbounded memory growth.

## Dependencies on Core
- `fsm_llm.API` — FSM execution, conversation queries
- `fsm_llm.HandlerTiming` — handler timing point enum
- `fsm_llm.create_handler` — fluent handler builder
- `fsm_llm.logging.logger` — loguru logging (for sink integration)
- External: `fastapi`, `uvicorn`, `jinja2`, `pydantic`, `loguru`

## Testing
```bash
pytest tests/test_fsm_llm_monitor/  # 68 tests
```

## Gotchas
- `configure()` must be called before endpoints work, or `get_bridge()` auto-creates an empty bridge with no API connection
- Handlers registered at priority 9999 and return empty dicts — pure observers, never interfere with FSM processing
- EventCollector uses bounded deques (`max_events=1000`, `max_log_lines=5000`) to prevent memory leaks
- Frontend is vanilla JS — no build step, no framework dependencies
- Version synced from `fsm_llm.__version__` — not independent
- Preset endpoints scan `examples/` directory but never expose filesystem paths to clients (path traversal protection)
