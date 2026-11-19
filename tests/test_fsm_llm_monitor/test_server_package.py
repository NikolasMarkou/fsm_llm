"""``monitor/server.py`` is a package at the same dotted path (plan d454fd62, step 94).

The module became ``monitor/server/``: ``application`` (the ``app`` object, the
state ``configure()`` rebinds, the middleware, ``configure``), ``support``, six
route modules and ``websocket``. Three things can break silently in a split like
this, and each has tests here:

* a ``Path(__file__).parent`` read that counts directories from where its module
  sits (the static and template directories, ``flows.json``);
* a piece of module-level state that ``configure()`` rebinds, read through a
  copy another module imported (``configure()`` then no longer reaches it);
* the registration order of routes and middleware (FastAPI matches in order, and
  the OpenAPI schema lists paths in it).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from starlette.routing import Mount
from starlette.websockets import WebSocketDisconnect

import fsm_llm.monitor
import fsm_llm.monitor.server as facade
from fsm_llm.monitor.definitions import DashboardConfig, MonitorConfig
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.monitor.server import application, configure
from tests.test_fsm_llm_monitor.support import server_modules

# The monitor package directory as the INSTALLED package reports it: independent
# of where any module of the server sits.
MONITOR = Path(fsm_llm.monitor.__file__).resolve().parent

_ROUTE_MODULES = (
    "routes_monitoring",
    "routes_fsm",
    "routes_workflow",
    "routes_agent",
    "routes_visualize",
    "routes_builder",
    "websocket",
)
_ALL_MODULES = ("application", "support", *_ROUTE_MODULES)


@pytest.fixture(autouse=True)
def _reset_key():
    yield
    with mock.patch.dict(os.environ, {"FSM_LLM_MONITOR_API_KEY": ""}):
        configure(manager=InstanceManager())


def _client(api_key: str | None = None, manager=None) -> TestClient:
    mgr = manager or InstanceManager(config=MonitorConfig(refresh_interval=0.5))
    configure(manager=mgr, api_key=api_key)
    return TestClient(facade.app)


# --------------------------------------------------------------------------
# The directories the server module itself uses (silent failure guard)
# --------------------------------------------------------------------------


class TestStaticAndTemplateDirectories:
    """``STATIC_DIR`` and ``TEMPLATE_DIR`` count one directory up from the module
    that defines them. A wrong depth is loud for the static mount (``StaticFiles``
    checks its directory) and SILENT for the template (a 500 on ``/``) and for
    ``flows.json`` (empty flows). These tests resolve the directories the server
    really uses (the mounted app, the Jinja loader, the constants) and compare
    them with the package's own ``static/`` and ``templates/``, never with a
    constant a test module computes for itself."""

    def test_the_constants_name_the_monitor_trees(self):
        assert application.STATIC_DIR.resolve() == MONITOR / "static"
        assert application.TEMPLATE_DIR.resolve() == MONITOR / "templates"
        assert facade.STATIC_DIR is application.STATIC_DIR
        assert facade.TEMPLATE_DIR is application.TEMPLATE_DIR

    def test_the_static_mount_serves_the_real_tree(self):
        mounts = [
            r for r in facade.app.routes if isinstance(r, Mount) and r.path == "/static"
        ]
        assert len(mounts) == 1
        served = Path(mounts[0].app.directory).resolve()
        assert served == MONITOR / "static"
        for name in ("app.js", "style.css", "flows.json", "services/api.js"):
            assert (served / name).is_file(), name

    def test_the_template_loader_searches_the_real_directory(self):
        searched = [
            Path(p).resolve() for p in application.templates.env.loader.searchpath
        ]
        assert searched == [MONITOR / "templates"]
        assert (searched[0] / "index.html").is_file()

    def test_index_returns_the_dashboard_shell(self):
        resp = _client().get("/")
        assert resp.status_code == 200, resp.text[:200]
        assert resp.headers["content-type"].startswith("text/html")
        shell = (MONITOR / "templates" / "index.html").read_text(encoding="utf-8")
        assert resp.text.strip() == shell.strip()
        assert 'id="page-dashboard"' in resp.text

    @pytest.mark.parametrize(
        "name", ["app.js", "style.css", "flows.json", "services/api.js"]
    )
    def test_static_assets_are_served_byte_for_byte(self, name):
        resp = _client().get(f"/static/{name}")
        assert resp.status_code == 200, name
        assert resp.content == (MONITOR / "static" / name).read_bytes()
        assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"

    def test_the_flows_come_from_the_real_file(self, monkeypatch):
        monkeypatch.setattr(application, "_flows", {})
        flows = facade.get_flows()
        expected = json.loads((MONITOR / "static" / "flows.json").read_text("utf-8"))
        assert flows == expected
        assert flows["agents"] and flows["workflows"]
        assert application._load_flows() == expected

    def test_configure_loads_the_flows_from_the_real_file(self):
        _client()
        assert facade.get_flows()["agents"]
        resp = TestClient(facade.app).get("/api/agent/visualize?agent_type=ReactAgent")
        assert resp.status_code == 200
        assert resp.json()["nodes"]

    def test_only_the_application_module_reads_its_own_file_location(self):
        users = {
            name
            for name, text in server_modules().items()
            if name != "__init__"
            and "__file__" in text.replace("``__file__``", "").replace("`__file__`", "")
        }
        assert users == {"application"}


# --------------------------------------------------------------------------
# Registration order
# --------------------------------------------------------------------------

_ROUTES = [
    ("Route", "/openapi.json"),
    ("Route", "/api/docs"),
    ("Route", "/docs/oauth2-redirect"),
    ("Route", "/redoc"),
    ("Mount", "/static"),
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/api/auth"),
    ("GET", "/api/metrics"),
    ("GET", "/api/conversations"),
    ("GET", "/api/conversations/{conversation_id}"),
    ("GET", "/api/activity"),
    ("GET", "/api/events"),
    ("GET", "/api/logs"),
    ("GET", "/api/config"),
    ("POST", "/api/config"),
    ("GET", "/api/dashboard/config"),
    ("POST", "/api/dashboard/config"),
    ("DELETE", "/api/dashboard/config"),
    ("GET", "/api/info"),
    ("GET", "/api/capabilities"),
    ("GET", "/api/instances"),
    ("GET", "/api/instances/{instance_id}"),
    ("GET", "/api/instances/{instance_id}/events"),
    ("DELETE", "/api/instances/{instance_id}"),
    ("POST", "/api/fsm/launch"),
    ("POST", "/api/fsm/{instance_id}/start"),
    ("POST", "/api/fsm/{instance_id}/converse"),
    ("POST", "/api/fsm/{instance_id}/end"),
    ("GET", "/api/fsm/{instance_id}/conversations"),
    ("GET", "/api/workflow/presets"),
    ("POST", "/api/workflow/launch"),
    ("POST", "/api/workflow/{instance_id}/advance"),
    ("POST", "/api/workflow/{instance_id}/cancel"),
    ("POST", "/api/workflow/{instance_id}/event"),
    ("GET", "/api/workflow/{instance_id}/status"),
    ("GET", "/api/workflow/{instance_id}/instances"),
    ("POST", "/api/agent/launch"),
    ("GET", "/api/agent/{instance_id}/status"),
    ("GET", "/api/agent/{instance_id}/result"),
    ("POST", "/api/agent/{instance_id}/cancel"),
    ("POST", "/api/fsm/load"),
    ("POST", "/api/fsm/visualize"),
    ("GET", "/api/fsm/visualize/preset/{preset_id:path}"),
    ("GET", "/api/presets"),
    ("GET", "/api/preset/fsm/{preset_id:path}"),
    ("GET", "/api/agent/visualize"),
    ("GET", "/api/workflow/visualize"),
    ("POST", "/api/builder/start"),
    ("POST", "/api/builder/send"),
    ("GET", "/api/builder/result/{session_id}"),
    ("DELETE", "/api/builder/{session_id}"),
    ("WebSocket", "/ws"),
]

# The module that defines the handler of each route, in registration order.
_ROUTE_HOMES = (
    ["fastapi"] * 4
    + ["starlette"]
    + ["application"] * 3
    + ["routes_monitoring"] * 17
    + ["routes_fsm"] * 5
    + ["routes_workflow"] * 7
    + ["routes_agent"] * 4
    + ["routes_visualize"] * 7
    + ["routes_builder"] * 4
    + ["websocket"]
)


def _route_key(route) -> tuple[str, str]:
    kind = type(route).__name__
    if kind == "APIRoute":
        (method,) = route.methods
        return method, route.path
    if kind == "APIWebSocketRoute":
        return "WebSocket", route.path
    return kind, route.path


class TestRegistrationOrder:
    """FastAPI matches routes in registration order and ``app.openapi()`` lists
    paths in it; middleware wraps in the reverse of its registration order. The
    old single module registered them in file order; the package imports its
    modules in that order."""

    def test_the_routes_are_registered_in_the_old_order(self):
        assert [_route_key(r) for r in facade.app.routes] == _ROUTES

    def test_each_module_registers_its_own_contiguous_block(self):
        homes = [
            (getattr(r, "endpoint", None) or r.app).__module__.rpartition(".")[2]
            if type(r).__name__ in {"APIRoute", "APIWebSocketRoute"}
            else "fastapi"
            if type(r).__name__ == "Route"
            else "starlette"
            for r in facade.app.routes
        ]
        assert homes == _ROUTE_HOMES

    def test_the_middleware_is_registered_in_the_old_order(self):
        names = []
        for m in facade.app.user_middleware:
            dispatch = m.kwargs.get("dispatch")
            names.append(dispatch.__name__ if dispatch else m.cls.__name__)
        # user_middleware[0] is the outermost: the last one registered
        assert names == [
            "_BodyCapMiddleware",  # F5: outermost, judges a chunked body first
            "no_cache_static",
            "_count_requests",
            "_security_guard",
            "CORSMiddleware",
        ]

    def test_the_openapi_paths_follow_the_registration_order(self):
        expected: list[str] = []
        for kind, path in _ROUTES:
            if kind in {"GET", "POST", "DELETE"}:
                # OpenAPI writes a `:path` converter as the bare parameter name
                path = path.replace(":path}", "}")
                if path not in expected:
                    expected.append(path)
        assert list(facade.app.openapi()["paths"]) == expected

    def test_the_facade_imports_the_route_modules_in_that_order(self):
        tree = ast.parse(Path(facade.__file__).read_text(encoding="utf-8"))
        imported = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is None
            for alias in node.names
        ]
        assert imported == list(_ROUTE_MODULES)


# --------------------------------------------------------------------------
# One home for each piece of shared state
# --------------------------------------------------------------------------

_STATE_HOMES = {
    # rebound by configure() or by a route: read ONLY in the defining module
    "_manager": "application",
    "_api_key": "application",
    "_flows": "application",
    "_CORS_ORIGIN_REGEX": "application",
    "_requests_processed": "application",
    "_builder_sessions_lock": "routes_builder",
    "_preset_cache": "routes_visualize",
    # mutated in place: still one object, so one binding
    "_CORS_ORIGINS": "application",
    "_TRUSTED_HOSTS": "application",
    "_builder_sessions": "routes_builder",
    "_builder_busy": "routes_builder",
    "_background_tasks": "routes_builder",
}


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


class TestSharedStateHasOneHome:
    def test_each_global_is_assigned_in_exactly_one_module(self):
        defined: dict[str, list[str]] = {}
        for module, text in server_modules().items():
            for name in _module_level_names(ast.parse(text)):
                if name in _STATE_HOMES:
                    defined.setdefault(name, []).append(module)
        assert defined == {name: [home] for name, home in _STATE_HOMES.items()}

    def test_no_module_imports_or_rebinds_a_copy_of_a_global(self):
        for module, text in server_modules().items():
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        bound = alias.asname or alias.name
                        assert bound not in _STATE_HOMES, (module, bound)
                elif isinstance(node, ast.Global):
                    for name in node.names:
                        assert _STATE_HOMES.get(name) == module, (module, name)

    def test_every_module_reads_a_global_through_its_home_only(self):
        # a name that is read in a module without being defined there must be
        # one of the constants and helpers a module imports, never a global
        for module, text in server_modules().items():
            tree = ast.parse(text)
            assigned = _module_level_names(tree)
            loads = {
                n.id
                for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            for name, home in _STATE_HOMES.items():
                if name in loads:
                    assert module == home or name in assigned, (module, name)

    def test_the_gate_dependency_is_the_application_function(self):
        from fsm_llm.monitor.server import support

        (dependency,) = support._GATED
        assert dependency.dependency is application._require_api_key

    def test_configure_reaches_the_manager_of_every_router_module(self):
        mgr = mock.MagicMock(spec=InstanceManager)
        mgr.get_metrics.return_value.model_dump.return_value = {"m": 1}
        mgr.get_fsm_conversations.return_value = []
        mgr.get_workflow_instances.return_value = []
        mgr.get_agent_status.return_value = {"status": "x"}
        mgr.get_workflow_presets.return_value = []
        client = _client(manager=mgr)
        assert client.get("/api/metrics").json() == {"m": 1}
        assert client.get("/api/fsm/f1/conversations").json() == []
        assert client.get("/api/workflow/w1/instances").json() == []
        assert client.get("/api/workflow/presets").json() == {"workflows": []}
        assert client.get("/api/agent/a1/status").json() == {"status": "x"}
        mgr.get_metrics.assert_called_once()
        mgr.get_fsm_conversations.assert_called_once_with("f1")
        mgr.get_workflow_instances.assert_called_once_with("w1")
        mgr.get_workflow_presets.assert_called_once()
        mgr.get_agent_status.assert_called_once_with("a1")
        assert application._manager is mgr
        assert facade.get_manager() is mgr

    def test_the_websocket_reads_the_configured_manager(self):
        mgr = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
        managed = mgr.launch_workflow(preset_id="demo_linear")
        client = _client(manager=mgr)
        with client.websocket_connect("/ws") as ws:
            frame = ws.receive_json()
        assert [i["instance_id"] for i in frame["instances"]] == [managed.instance_id]
        other = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
        client = _client(manager=other)
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["instances"] == []

    def test_the_websocket_starts_from_the_configured_managers_dashboard_version(self):
        # the push remembers the dashboard version of the manager it started
        # with: a stale manager there would add a spurious change signal
        mgr = InstanceManager(config=MonitorConfig(refresh_interval=0.5))
        mgr.dashboard_config = DashboardConfig(name="d")
        assert mgr.dashboard_config_version > 0
        client = _client(manager=mgr)
        with client.websocket_connect("/ws") as ws:
            frame = ws.receive_json()
        assert "dashboard_config_version" not in frame

    def test_the_request_counter_reaches_configure_across_modules(self, monkeypatch):
        # _count_requests (middleware) increments the counter configure() reads
        monkeypatch.setattr(application, "_requests_processed", 0)
        records: list[str] = []
        logger.enable("fsm_llm")
        sink = logger.add(
            lambda m: records.append(m.record["message"]),
            level="WARNING",
            format="{message}",
        )
        try:
            client = _client()
            assert not any("processed" in r for r in records)
            client.get("/health")
            client.get("/health")
            assert application._requests_processed == 2
            configure(manager=application._manager)
        finally:
            logger.remove(sink)
        assert any(
            "configure() called after server has processed 2 request(s)" in r
            for r in records
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/api/events",  # routes_monitoring
            "/api/fsm/x/conversations",  # routes_fsm
            "/api/workflow/x/instances",  # routes_workflow
            "/api/agent/x/status",  # routes_agent
            "/api/builder/result/x",  # routes_builder
        ],
    )
    def test_the_api_key_set_by_configure_gates_a_route_of_every_module(self, path):
        client = _client(api_key="k1")
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "k1"}).status_code != 401
        # re-derived on every call: the next configure() clears it everywhere
        configure(manager=application._manager)
        assert client.get(path).status_code != 401

    def test_api_auth_and_the_websocket_see_a_rebound_key(self):
        client = _client(api_key="k1")
        assert client.get("/api/auth").json() == {"auth_required": True}
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "api_key": "bad"}))
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
        assert exc.value.code == 4401
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "auth", "api_key": "k1"}))
            assert ws.receive_json()["type"] == "metrics"
        configure(manager=application._manager)
        assert client.get("/api/auth").json() == {"auth_required": False}

    def test_the_builder_store_is_one_object_for_the_routes_and_the_sweep(self):
        from fsm_llm.monitor.server import routes_builder, websocket

        assert websocket._cleanup_stale_builder_sessions is (
            routes_builder._cleanup_stale_builder_sessions
        )
        routes_builder._builder_sessions["stale"] = (mock.MagicMock(), 0.0)
        try:
            import asyncio

            asyncio.run(websocket._cleanup_stale_builder_sessions())
            assert "stale" not in routes_builder._builder_sessions
        finally:
            routes_builder._builder_sessions.pop("stale", None)

    def test_the_builder_lock_is_created_once_and_kept(self):
        # a lock made anew on every call would guard nothing
        from fsm_llm.monitor.server import routes_builder

        routes_builder._builder_sessions_lock = None
        try:
            first = routes_builder._get_builder_lock()
            assert routes_builder._builder_sessions_lock is first
            assert routes_builder._get_builder_lock() is first
        finally:
            routes_builder._builder_sessions_lock = None

    def test_the_preset_cache_is_one_binding_across_requests(self):
        from fsm_llm.monitor.server import routes_visualize

        client = _client()
        routes_visualize._preset_cache = None
        first = client.get("/api/presets").json()
        cached = routes_visualize._preset_cache
        assert cached is not None and cached[0] == first
        assert client.get("/api/presets").json() == first
        assert routes_visualize._preset_cache is cached


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

_SIBLING_IMPORTS = {
    "application": set(),
    "support": {"application"},
    "routes_monitoring": {"application", "support"},
    "routes_fsm": {"application", "support"},
    "routes_workflow": {"application", "support"},
    "routes_agent": {"application", "support"},
    "routes_visualize": {"application", "support"},
    "routes_builder": {"application", "support"},
    "websocket": {"application", "routes_builder"},
}


class TestServerPackageLayout:
    def test_the_facade_exports_exactly_the_documented_names(self):
        assert sorted(facade.__all__) == [
            "STATIC_DIR",
            "TEMPLATE_DIR",
            "app",
            "configure",
            "get_flows",
            "get_manager",
        ]
        namespace: dict = {}
        exec("from fsm_llm.monitor.server import *", namespace)
        assert set(facade.__all__) <= set(namespace)
        assert facade.app is fsm_llm.monitor.app
        assert facade.configure is fsm_llm.monitor.configure

    def test_a_private_name_lives_in_its_defining_module_only(self):
        for private in ("_api_key", "_manager", "_flows", "_require_api_key"):
            assert not hasattr(facade, private), private
            assert hasattr(application, private), private

    def test_modules_import_only_their_designated_siblings(self):
        assert set(server_modules()) == {"__init__", *_ALL_MODULES}
        for module, text in server_modules().items():
            if module == "__init__":
                continue
            siblings = set()
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 1:
                        siblings.add(node.module or "")
                    else:
                        # an absolute import of the package or of a module of it
                        # reaches the facade (a chain to every route module)
                        assert not (node.module or "").startswith(
                            "fsm_llm.monitor.server"
                        ), (module, node.module)
                elif isinstance(node, ast.Import):
                    assert not any(
                        a.name.startswith("fsm_llm.monitor.server") for a in node.names
                    ), module
            assert siblings == _SIBLING_IMPORTS[module], module

    def test_no_module_is_over_the_size_limit(self):
        for module, text in server_modules().items():
            assert len(text.splitlines()) <= 800, module

    def test_the_old_module_file_is_gone(self):
        assert not (MONITOR / "server.py").exists()
        assert (MONITOR / "server" / "__init__.py").is_file()

    def test_a_log_record_is_named_after_the_module_that_logs_it(self):
        records: list[tuple[str, str, str]] = []
        logger.enable("fsm_llm")
        sink = logger.add(
            lambda m: records.append(
                (m.record["name"], m.record["module"], m.record["message"])
            ),
            level="DEBUG",
            format="{message}",
        )
        try:
            # application: re-configuring without a key clears a previous one
            configure(manager=InstanceManager(), api_key="k")
            with mock.patch.dict(os.environ, {"FSM_LLM_MONITOR_API_KEY": ""}):
                configure(manager=application._manager)
            # support: an unmapped failure is logged by the error mapping
            mgr = mock.MagicMock(spec=InstanceManager)
            mgr.get_agent_status.side_effect = RuntimeError("boom")
            resp = _client(manager=mgr).get("/api/agent/a1/status")
            assert resp.status_code == 500
            assert resp.json() == {"detail": "Internal server error"}
        finally:
            logger.remove(sink)
        by_message = {m: (n, mod) for n, mod, m in records}
        cleared = next(v for k, v in by_message.items() if "is being cleared" in k)
        assert cleared == ("fsm_llm.monitor.server.application", "application")
        failed = next(v for k, v in by_message.items() if k.startswith("Failed to get"))
        assert failed == ("fsm_llm.monitor.server.support", "support")
