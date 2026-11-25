# test_fsm_llm_monitor

The pytest suite for `fsm_llm.monitor`, the web dashboard package of FSM-LLM (source in `src/fsm_llm/monitor/`). It lives at `tests/test_fsm_llm_monitor/`.

## What it is for

The monitor is a FastAPI server (a Python web framework) with a browser dashboard. It watches FSM conversations, agents and workflows, lets you launch them, and streams metrics over a WebSocket. These tests check that the pieces behind it behave correctly: the event collector, the data models, the instance manager (including `attach_api` for an `API` you created) that launches and tracks FSMs, agents and workflows, the optional OpenTelemetry exporter, and the HTTP server itself, including its security checks. No test calls a real LLM.

## How it works

Each test file targets one source module in `src/fsm_llm/monitor/`. Server tests call `configure(...)` to install a fresh `InstanceManager`, then drive the FastAPI `app` through `fastapi.testclient.TestClient`. The other tests build objects directly and replace the FSM `API` with `unittest.mock.MagicMock` or small fake classes.

```mermaid
flowchart LR
    conftest[conftest.py: clears FSM_LLM_MONITOR_API_KEY] --> all[every test]
    app[test_app.py, test_server_security.py] --> server[server/ via TestClient]
    im[test_instance_manager.py] --> manager[instance_manager/]
    col[test_collector.py] --> collector[collector.py]
    defs[test_definitions.py] --> definitions[definitions.py]
    ot[test_otel.py] --> otel[otel.py with mocked opentelemetry]
```

## Files

- `conftest.py` - autouse fixture that removes `FSM_LLM_MONITOR_API_KEY` from the environment for every test.
- `support.py` - helpers that more than one test file uses (`_manager()`, `server_modules()`, `server_package_modules()`). Not collected: the name has no `test_` prefix.
- `__init__.py` - empty package marker.
- `test_app.py` - HTTP routes, static files, public exports, version, API-key gate, WebSocket redaction hook, and that the UI log levels, Max Iterations limit and log CSS match the server constants.
- `test_server_security.py` - Origin and Host checks, body size limit, security headers, key-gated reads, WebSocket auth, error-to-status mapping, request bounds, dashboard config parsing, builder guards, preset path validation.
- `test_instance_manager.py` - `ManagedFSM`/`ManagedWorkflow`/`ManagedAgent`, `InstanceManager` lookups, ending an instance, activity, workflow presets, disabled agent types, agent status resolution, stub tools, a failed launch keeping a finished instance, a terminal send when `end_conversation` fails, one span per exporter after an OTEL re-enable (real `opentelemetry` SDK, skipped without it), the instance event feed, `close()` teardown of every instance type, the `examples/` directory read from its module's depth, the package layout of `instance_manager/`.
- `test_collector.py` - `EventCollector` buffers, metrics, log filtering, handler callbacks, loguru sink, cursors, thread safety.
- `test_definitions.py` - Pydantic models and helpers in `definitions.py`.
- `test_otel.py` - `OTELExporter` enable, disable, shutdown and event routing.
- `test_lifecycle_remainders.py`, `test_usage_metrics.py`, `test_conversation_signals.py`, `test_event_log_browse.py` - the closed manager and the run list cap, LLM usage on the metrics and the workflow spans, the classification and cut-off fields of a conversation snapshot, and the read-only event log browse.
- `test_workflow_presets.py` - the workflow presets as documents: the registry loads lazily once off the lock and the loop, bad documents, exact ids, a fresh definition per launch, runs with no model.
- `test_workflow_preset_golden.py`, `workflow_preset_corpus.py` - the preset runs and the HTTP wire against a golden written before the presets became documents (two named differences of the `finish` step).
- `test_agent_types.py`, `test_agent_launch.py`, `test_agent_draw.py`, `test_agent_selects.py` - the monitor lists, launches and draws agents by executable id (plan 659c07c3): the launchable rule and the lazy registry load, launch by exact id and the `create_agent` arguments, the preview graph per id and its hostile-id refusals, and the page that holds no agent list. Doubles only.
- `test_builder_improvements.py` - the Builder improvements: the launch form limits, the launchable agent type list, the `max_tokens` default and the forced delete of a busy session, with a fake builder agent and no model call.
- `test_static_js.py` - runs the node probes in `static_probes/*.mjs` against a copy of the monitor's JavaScript on a small DOM stand-in (the Markdown renderer, the chat switch, the Send Event targets, the custom panels, the launch modal, the builder launch, cancel, edit and start requests, and the JSON colouring of `utils/json_highlight.js` and of every page that shows JSON; the colour contrast is a plain Python test on `style.css`). It skips when `node` 18 or newer is not on PATH, and a test fails on a Linux GitHub Actions leg without it.
- `test_server_package.py` - the `monitor/server/` package (plan d454fd62 step 94): the static and template directories the server really uses, route and middleware registration order, one home for each global `configure()` rebinds, the package layout.
- `test_server_consistency.py` - the 2026-11-05 server and SPA fixes (plan 5980fbb7 step 17): the `/ws` tick, visualize routes, guard headers, worker-thread launch, Ctrl-C exit, string-level SPA pins, preset JSON read as UTF-8 under an ASCII locale (subprocess), the route-launched workflow engine stopped by `end_instance` and `close`.

## How to use it

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_monitor/
.venv/bin/python -m pytest tests/test_fsm_llm_monitor/test_server_security.py -v
.venv/bin/python -m pytest tests/test_fsm_llm_monitor --collect-only -q | tail -1
```

The monitor extra must be installed (`fastapi`, `uvicorn`, `jinja2`), for example through `make install-dev`.

## Things to know

- Any exported `FSM_LLM_MONITOR_API_KEY` is removed by `conftest.py`. Without it, many route tests would fail with 401.
- `server/` keeps module-level state (`_api_key` in `server.application`, builder sessions in `server.routes_builder`). Tests reset it with `configure(...)` in `setup_method`, `teardown_method` or an autouse fixture.
- `test_otel.py` does not need `opentelemetry` installed. It puts fake modules into `sys.modules` and re-imports `fsm_llm.monitor.otel` for each test.
- `TestWorkflowTracking` (in `test_instance_manager.py`) skips itself when `fsm_llm.workflows` cannot be imported. `TestStubToolExecution` is skipped without `fsm_llm.agents`.
- Async tests have no marker. They rely on `asyncio_mode = "auto"` in `pyproject.toml`.
- `test_app.py` asserts the version string `"0.13.0"` for `/api/info` `monitor_version` and for `fsm_llm.monitor.__version__`. Update both on a version bump.
- The server test files import `fastapi.testclient` at module level, so they fail at collection (they do not skip) when the monitor extra is missing.
- `test_env_key_applies_without_configure` (in `test_server_security.py`) starts a Python subprocess (120 s timeout).
- Preset tests need the repo `examples/` directory. Some of them only assert when presets are found.
