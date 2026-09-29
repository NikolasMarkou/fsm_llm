# fsm_llm.monitor

Path: `src/fsm_llm/monitor`
Purpose: FastAPI web dashboard (REST + WebSocket + vanilla-JS SPA) that launches, observes, and drives FSM-LLM FSMs, agents, and workflows; optional OpenTelemetry span export.

## Scope

Python backend (`server.py`, `instance_manager.py`, `collector.py`, `bridge.py`, `otel.py`, models, constants), the browser app in `static/`, and the Jinja2 shell `templates/index.html`. Part of the `fsm-llm` distribution (version shared with `fsm_llm`, currently 0.9.0); its deps come from extra `monitor` (fastapi >=0.100, uvicorn >=0.20, jinja2 >=3.1); OTEL via extra `otel`. Not here: FSM runtime (`fsm_llm`), agent implementations (`fsm_llm.agents`), workflow engine (`fsm_llm.workflows`).

## Architecture

```mermaid
flowchart TD
    CLI[__main__.main_cli] --> Cfg[server.configure]
    Cfg --> IM[InstanceManager]
    IM --> GC[global EventCollector + loguru sink at config.log_level]
    IM --> MF[ManagedFSM: API.from_definition]
    IM --> MA[ManagedAgent: daemon thread, agent.run]
    IM --> MW[ManagedWorkflow: WorkflowEngine + DSL preset + engine hook]
    MF -- register_monitor_handlers x2 --> PC[per-instance EventCollector]
    MF --> GC
    MA -- handlers kwarg: monitor + global + context-capture --> PC
    MW -- add_hook: steps + run status --> PC
    MW --> GC
    Server[server.py app] --> IM
    Server -- /ws loop --> Browser
    OTEL[OTELExporter.enable] -. wraps record_event .-> GC
```

Handler hooks are `_MonitorHandler(BaseHandler)` objects (observe-only, priority 9999; context capture 9998). `should_execute` keeps the core's `current_state`/`target_state` per thread and `execute` passes them to the collector callback, because the core never puts `_target_state` into the handler context (D-003). Registration is idempotent per (api, collector); `unregister_monitor_handlers` switches them off (the core has no unregister). POST_TRANSITION (a no-op) is not registered.

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `server.py` | FastAPI `app`, `configure`, all routes, `/ws`, builder sessions, security middleware | Module globals: `_manager`, `_api_key`, `_CORS_ORIGINS`, `_TRUSTED_HOSTS`, `_flows`, `_builder_sessions`, `_builder_busy`, `_preset_cache` |
| `instance_manager.py` | `InstanceManager`, `ManagedFSM/Agent/Workflow`, `_MonitorHandler`, `register_monitor_handlers`, `unregister_monitor_handlers`, `snapshot_from_api`, `validate_preset_id`, `_find_examples_dir` | Optional imports set `_HAS_WORKFLOWS`, `_HAS_AGENTS` |
| `collector.py` | `EventCollector`, `redact_context` | `threading.Lock`, `deque(maxlen)` for events and logs |
| `bridge.py` | `MonitorBridge`, `_fsm_dict_to_snapshot` | Backward-compat wrapper |
| `otel.py` | `OTELExporter` | Own `TracerProvider`, never set as global |
| `definitions.py` | Pydantic models, `normalize_message_history`, `model_to_dict` | Request/config bounds live here |
| `constants.py` | Event names, defaults, bounds, theme colors, handler name/priority | Theme colors are not used by `static/style.css` |
| `__main__.py` | CLI `fsm-llm-monitor` | enables `fsm_llm` library logging (D-008; embedders running `uvicorn fsm_llm.monitor.server:app` call `setup_logging()` themselves), `configure(api_key, trusted_hosts)`, then `uvicorn.run(app, log_level="warning")` |
| `static/`, `templates/index.html` | Frontend | Element ids and `data-action` names are the frontend contract (see `static/pages/CLAUDE.md`) |

## Public interface

Exports (`__init__.__all__`): `MonitorBridge`, `EventCollector`, `InstanceManager`, `app`, `configure`, `OTELExporter`, all models listed under Data shapes, `model_to_dict`, constants (`THEME_NAME`, `COLOR_PRIMARY`, `DEFAULT_*`, all `EVENT_*`, `MONITOR_HANDLER_NAME`, `MONITOR_HANDLER_PRIORITY`), exceptions.

- `configure(bridge=None, manager=None, cors_origins=None, api_key=None, trusted_hosts=None)`: sets global `_manager` (explicit manager > new manager wrapping `bridge.api` via `connect_bridge`, after `bridge.disconnect()` > fresh `InstanceManager()`), reloads `flows.json`, and calls `shutdown()` on the previous manager unless it is the one being installed (so `configure(manager=current, api_key=...)` rotates a key without killing the log feed). `api_key` is re-derived every call: arg, else env `FSM_LLM_MONITOR_API_KEY`, else `None` (warns when a previous key is cleared); the module also reads the env var at import, so `uvicorn fsm_llm.monitor.server:app` without `configure()` honours it. An empty or whitespace-only `api_key` argument raises `ValueError` before any global changes; an empty env var still means no key. `cors_origins` (env `FSM_LLM_MONITOR_CORS_ORIGINS`) mutates middleware kwargs and only works before the first request; `trusted_hosts` (env `FSM_LLM_MONITOR_TRUSTED_HOSTS`, `["*"]` disables) takes effect immediately.
- CLI: `fsm-llm-monitor [--host 127.0.0.1] [--port 8420] [--api-key KEY] [--otel] [--no-browser] [--version] [--info]`; also `python -m fsm_llm.monitor`. A wildcard bind (`0.0.0.0`, `::`) sets `trusted_hosts=["*"]`, another non-loopback host adds it to the allow-list; a non-loopback bind without a key prints a warning. The browser opens only once the port accepts connections.
- `EventCollector(max_events=1000, max_log_lines=5000)`: `record_event`, `record_log`, `get_events(limit=0)`, `get_events_since(after_total, limit=50)` (newest first), `events_after(cursor, limit=50) -> (oldest-first events, next_cursor)`, `logs_after(cursor, limit=50)`, `get_logs(limit=0, level=None)` (min-level filter, case-insensitive, TRACE/SUCCESS ranked), `get_logs_since`, `get_metrics() -> MetricSnapshot` (raises `MetricCollectionError`), `get_events_by_conversation`, `resize(max_events, max_log_lines)`, `clear()`, `cleanup()` (remove loguru sink), `create_loguru_sink()`, `create_handler_callbacks() -> {TIMING_NAME: cb(context, current_state=None, target_state=None)}` (8 timings), `create_context_capture_callbacks(sink)` (6 timings), static `snapshot_context(ctx, max_value_len=1500)`.
- `redact_context(data, drop_internal=True)`: the one dashboard redaction path: drops `is_forbidden_context_entry` entries at every depth, internal-prefix keys unless `drop_internal=False`, and turns non-JSON leaves into `<redacted:Type>` without calling `__str__` (D-002).
- `MonitorBridge(api=None, config=None)`: `connect(api)` (idempotent for the same API; raises `MonitorConnectionError` and stays disconnected on failure), `disconnect()` (its handlers stop recording), `set_collector()`, `get_metrics()`, `get_active_conversations()`, `get_conversation_snapshot(id)`, `get_all_conversation_snapshots()`, `get_recent_events(limit)`, `load_fsm_from_file(path)`, `load_fsm_from_dict(data)` (None on error).
- `InstanceManager(config=None)` (raises `MonitorInitializationError` if the loguru sink fails): `launch_fsm(preset_id|fsm_json, model, temperature, label)`, `start_conversation(id, ctx) -> (conv_id, response)` (reopens a completed instance), `send_message(id, conv_id, msg) -> {response, current_state, is_terminal}` (a terminal conversation is cached and ended), `end_conversation` (KeyError for an unknown conversation, no-op if already ended), `get_fsm_conversations`; `get_workflow_presets()`, `launch_workflow(preset_id, ...)`, async `start_workflow_instance`, async `advance_workflow`, async `cancel_workflow`, async `send_workflow_event(id, event_type, payload, wf_instance_id=None)`, `get_workflow_status` (KeyError for an unknown run), `get_workflow_instances`; `launch_agent(...)` (raises `MonitorCapacityError` at `max_running_agents`), `cancel_agent -> bool` (False for a finished agent), `get_agent_status`, `get_agent_result`; `destroy_instance`; `shutdown()`; `list_instances(type_filter)`, `get_instance`, `get_instance_collector`, `get_metrics`, `get_events`, `get_active_conversations`, `find_instance_for_conversation`, `get_conversation_snapshot`, `get_all_conversation_snapshots(include_ended)`, `get_all_activity_snapshots(include_ended)`, `get_capabilities()`, `connect_bridge(api)`, properties `config` (setter resizes the global buffers and re-registers the sink at `log_level`), `global_collector`, `dashboard_config`, `dashboard_config_version`.
- `OTELExporter(service_name="fsm-llm", exporter=None)` (raises `ImportError` without opentelemetry; default `ConsoleSpanExporter` in `BatchSpanProcessor`): `enable(collector)`, `disable()` (restores `record_event` only if its wrapper is still outermost), `shutdown()`, `is_enabled`, `active_conversations`, static `generate_trace_id()`, `generate_span_id()`. Span names are fixed (`conversation`, `transition`, `processing.<phase>`, `lifecycle.<event_type>`); ids are attributes. Spans without a conversation parent start from an empty context. Error text is the first line, max 200 chars; lifecycle spans carry ids only (no task text). At most 10,000 open conversation spans (oldest ended first).

## HTTP routes (server.py)

`[K]` = requires the API key when one is configured (`Authorization: Bearer <key>` or `X-API-Key`, `hmac.compare_digest` on UTF-8 surrogateescape bytes, 401 on mismatch). Every request passes the security middleware (D-001): Host must be in `_TRUSTED_HOSTS` (400), POST/PUT/PATCH/DELETE with an `Origin` must come from the monitor's own origin or `_CORS_ORIGINS` (403), bodies over 1 MiB are refused (413), and responses carry CSP (`frame-ancestors 'none'`), `X-Frame-Options`, `nosniff`, `Referrer-Policy`.

- Pages: `GET /` (index.html), `GET /health`, `GET /api/docs` (OpenAPI UI), `/static/*` (no-cache headers), `GET /api/auth` -> `{auth_required}` (never gated).
- Monitoring: `GET /api/metrics`, `/api/conversations?include_ended [K]`, `/api/conversations/{id} [K]` (404), `/api/activity?include_ended [K]`, `/api/events?limit [K]`, `/api/logs?limit&level [K]`, `/api/info`, `/api/capabilities`. Conversation snapshot routes run in a worker thread (they wait on the conversation's turn lock; D-005).
- Config: `GET /api/config`, `POST /api/config [K]` (422 outside the bounds); `GET /api/dashboard/config`, `POST [K]` (MonitorBuilder `to_dict()` output, bare or wrapped as `{"config": ...}`; 400 when malformed), `DELETE [K]`.
- Instances: `GET /api/instances?type`, `GET /api/instances/{id}`, `GET /api/instances/{id}/events?limit [K]` (404 unknown), `DELETE /api/instances/{id} [K]`.
- FSM: `POST /api/fsm/launch [K]`, `POST /api/fsm/{id}/start [K]`, `/converse [K]`, `/end [K]`, `GET /api/fsm/{id}/conversations [K]`; `POST /api/fsm/load`, `POST /api/fsm/visualize`, `GET /api/fsm/visualize/preset/{preset_id:path}`; `GET /api/presets` (60 s cache), `GET /api/preset/fsm/{preset_id:path}`.
- Workflow: `GET /api/workflow/presets`, `POST /api/workflow/launch [K]` (starts one run, 120 s timeout, the instance is destroyed if the start fails), `POST /api/workflow/{id}/advance [K]`, `/cancel [K]`, `/event [K]` (`WorkflowEventRequest`; empty `workflow_instance_id` = broadcast), `GET /api/workflow/{id}/status?workflow_instance_id [K]` (400 if missing), `GET /api/workflow/{id}/instances [K]`.
- Agent: `POST /api/agent/launch [K]`, `GET /api/agent/{id}/status [K]`, `GET /api/agent/{id}/result [K]`, `POST /api/agent/{id}/cancel [K]` -> `{status, cancelled}`; `GET /api/agent/visualize?agent_type`, `GET /api/workflow/visualize?workflow_id` (from `flows.json`, 404 unknown).
- Builder: `POST /api/builder/start [K]` (501 if `fsm_llm.agents.meta_builder` missing, 429 over 50 sessions), `POST /api/builder/send [K]` (404 unknown, 409 while a send is in flight, including one that timed out and is still running; session removed on completion), `GET /api/builder/result/{sid} [K]` (`{busy: true}` while a send runs), `DELETE /api/builder/{sid} [K]` (409 while busy). Sessions expire 1 h after last use.
- `WS /ws`: Origin/Host checked (close 4403); with a key the first client message must be `{"type": "auth", "api_key": "..."}` within 5 s (close 4401). Each `refresh_interval` sends `{type:"metrics", data, events?, logs?, log_count, instances, agent_updates?, workflow_updates?, dashboard_config?}`; events and logs stream from cursors (oldest first, sent newest first, at most 50 per cycle, the rest next cycle; D-004); the manager is re-read each cycle.

Error mapping (`_raise_http`, all routes): `KeyError`/`FileNotFoundError` -> 404, `TypeError`/`ValueError` -> 400, `ConversationBusyError` -> 409, `MonitorCapacityError` -> 429, `NotImplementedError` (extension missing) -> 501, other -> 500 `"Internal server error"` (message logged, not returned), timeouts -> 504.

## Data shapes

- `MonitorEvent{event_type, timestamp (UTC), conversation_id, source_state, target_state, data, level="INFO", message}`; `LogRecord{timestamp, level, message, module, function, line, conversation_id}`.
- `MetricSnapshot{active_conversations, total_events, total_errors, total_transitions, events_per_type, states_visited, active_agents, active_workflows, total_agent_iterations, total_tool_calls, total_workflow_steps}` (`total_workflow_steps` counts executed steps; `states_visited` includes each conversation's start state).
- `ConversationSnapshot{conversation_id, instance_id, current_state, state_description, is_terminal, context_data, message_history[{role, content}], stack_depth, last_extraction, last_transition, last_response}` (context and `last_*` pass `redact_context`); `ActivityItem{item_id, item_type fsm_conversation|agent_task|workflow_instance, instance_id, label, status, current_step, detail, message_count, created_at, is_terminal}` (workflow items cover every run; ended ones only with `include_ended`).
- `FSMSnapshot{name, description, version, initial_state, persona, state_count, states[StateInfo{state_id, description, purpose, is_initial, is_terminal (no transitions), transition_count, transitions[TransitionInfo{target_state, description, priority=100, condition_count, has_logic}]}]}`.
- `MonitorConfig{refresh_interval=1.0 (0.5..60), max_events=1000 (10..100000), max_log_lines=5000 (10..100000), log_level="INFO" (TRACE..CRITICAL), show_internal_keys=False, auto_scroll_logs=True, max_instances=200, max_running_agents=8}`; `InstanceInfo{instance_id, instance_type fsm|workflow|agent, label, status, created_at, source, conversation_count, active_workflows, agent_type, task}`.
- The dashboard websocket push serializes with `fsm_llm.utilities.redacting_json_default`: a context value that is not JSON-native is sent as `"<redacted:TypeName>"`, never its `__str__`.
- Requests: `LaunchFSMRequest{preset_id, fsm_json, model=DEFAULT_LLM_MODEL, temperature 0..2 =0.5, label}`, `StartConversationRequest{initial_context}`, `SendMessageRequest{message (max 10000), conversation_id}`, `EndConversationRequest`, `StubToolConfig{name, description, stub_response}`, `LaunchAgentRequest{agent_type="ReactAgent", task (1..20000), model, max_iterations=10 (1..100), timeout_seconds=120 (0<..3600), tools (max 50), label}`, `LaunchWorkflowRequest{preset_id, definition_json, initial_context, label}`, `WorkflowAdvanceRequest{workflow_instance_id, user_input}`, `WorkflowCancelRequest{workflow_instance_id, reason}`, `WorkflowEventRequest{event_type, payload, workflow_instance_id=""}`, `BuilderStartRequest{artifact_type, model, temperature=0.7, max_tokens=2000}`, `BuilderSendRequest{session_id, message (max 10000)}`, `DashboardConfig{name, description, panels[DashboardPanel], alerts[DashboardAlert], refresh_interval_seconds=30, retention_hours=24}`.
- Event types: `conversation_start|end` (end `data.state` = final state), `state_transition`, `pre_processing`, `post_processing`, `context_update`, `error`, `log` (reserved, never emitted), `instance_launched|destroyed`, `workflow_started|advanced (one per executed step)|completed|failed|cancelled|event_delivered`, `agent_started|completed|failed|iteration|cancelled|tool_call`.

## Invariants and constraints

- Internal-key hiding uses `fsm_llm.constants.has_internal_prefix`; secret hiding uses `is_forbidden_context_entry`; both only through `redact_context` (D-002). Never re-inline `k.startswith("_")` or call `str(value)` before redaction. `snapshot_from_api(show_internal_keys=False)` default is fail-closed; keep it. `show_internal_keys=True` shows internal keys but never secret-looking ones.
- `snapshot_context` also drops noise keys `task`, `should_terminate` and empty-looking values, truncating to 1500 chars.
- `InstanceManager` uses one `RLock`; LLM calls (`start_conversation`, `converse`) run outside it. Server wraps blocking calls in `asyncio.to_thread` with 120 s timeout (504), builder send 300 s. A timed-out worker thread keeps running (a started conversation may still appear).
- Instance ids are the first 12 chars of a uuid4. Ended conversation snapshots cached in an `OrderedDict`, max 1000, oldest evicted. At most `max_instances` instances: a launch evicts the oldest finished ones and raises `MonitorCapacityError` when all are active.
- An FSM instance becomes `completed` when every started conversation has ended (terminal via `send_message`, or `end_conversation`) and returns to `running` when a new conversation starts. `destroy_instance` unregisters the global handlers and calls `api.close()`.
- Agents: launchable set `_AGENT_CLASSES` = React, Reflexion, PlanExecute, REWOO, ADaPT, Debate, SelfConsistency. EvaluatorOptimizer and MakerChecker are deliberately excluded (constructor args a form cannot supply). `_TOOL_BASED_AGENTS` require non-empty tools. Status goes `running -> cancelling -> cancelled` or `completed|failed`; a cancel after the agent finished changes nothing; a run that finishes after a cancel keeps its result (status `cancelled`); `get_agent_status` reconciles a dead thread.
- `destroy_instance` for agents joins the thread with timeout 1.5 s and logs a warning if still alive; there is no mid-run cancellation. For workflows it schedules `engine.shutdown()` on the loop that owns the engine.
- Workflows: only `_WORKFLOW_PRESETS` (`demo_linear`, `demo_branching`) built with the workflows DSL; `definition_json` alone raises `ValueError`. Run bookkeeping (active runs, `completed|failed|cancelled` events, `ManagedWorkflow.status`, step events) comes from the engine hook, whatever ended the run (D-006).
- `validate_preset_id` rejects `..`, absolute paths, and anything resolving outside `examples/`. `_find_examples_dir` looks at `<repo>/examples` (`Path(__file__).resolve().parents[3]`); None in a wheel install.
- `OTELExporter` must not register a global tracer provider. Callers must hold a reference to it while enabled.
- Known core interaction: with the monitor attached, the core deep-copies the conversation context for handler timings, so a context value that cannot be deep-copied raises `TypeError` (core D-025) only when the monitor is attached.

## Dependencies

- `fsm_llm`: `API`, `HandlerTiming`, `handlers.BaseHandler`, `definitions.ConversationBusyError`, `constants` (`DEFAULT_LLM_MODEL`, `has_internal_prefix`, `is_forbidden_context_entry`, `MAX_CONTEXT_FILTER_DEPTH`), `utilities` (`filter_context_tree`, `redact_non_json_leaf`, `redacting_json_default`), `logging.logger`, `__version__`. Uses `api.fsm_manager.get_complete_conversation`, `api.get_stack_depth`, `api.close`.
- Optional `fsm_llm.agents`: agent classes, `AgentConfig`, `ToolRegistry.register_function`; `meta_builder.MetaBuilderAgent/MetaBuilderConfig` for the Builder page.
- Optional `fsm_llm.workflows`: `WorkflowEngine` (`add_hook`, `shutdown`), `WorkflowEvent`, `WorkflowStep`, `WorkflowStepResult`, `create_workflow`, `auto_step`, `condition_step`.
- External: fastapi, starlette, uvicorn, jinja2, pydantic v2, loguru; optional opentelemetry-api/sdk.

## Failure modes

- See the error mapping above. Agent type not launchable -> 400 from `/api/agent/launch`. The Builder page maps some artifact agent types (for example `evaluator_optimizer`, `prompt_chain`, `orchestrator`) to classes that are not launchable, so those launches fail with 400; workflow artifacts cannot be launched from the Builder.
- `get_conversation_snapshot` returns `None` on any exception (logged at DEBUG).
- Calling `configure()` again without `api_key` and without the env var silently disables auth (warning logged).

## Working here

- New REST route: add in `server.py`; gate mutating and sensitive-read routes with `dependencies=_GATED`; wrap blocking or LLM calls in `asyncio.to_thread` + `asyncio.wait_for`; map errors with `_raise_http`.
- Anything shown on the dashboard that comes from a context or trace: pass it through `redact_context`.
- New event type: add `EVENT_*` in `constants.py`, export it in `__init__.__all__`, emit via `_emit_global_event`, and handle it in `collector.record_event` if it drives a counter and in `otel._route_event` if it should be a span.
- New launchable agent: add to `_AGENT_CLASSES` (and `_TOOL_BASED_AGENTS` if needed), mirror in `static/services/state.js TOOL_BASED_AGENTS` and the launch `<select>` in `templates/index.html`; add its graph to `static/flows.json`.
- Keep `__init__.__all__` as a single static list.
- Tests: `pytest tests/test_fsm_llm_monitor/` (files: `test_app.py`, `test_audit_2026_09_28.py`, `test_bridge.py`, `test_collector.py`, `test_definitions.py`, `test_instance_manager.py`, `test_otel.py`, `test_server_security.py`). Lint and types: `make lint`, `make type-check`.
