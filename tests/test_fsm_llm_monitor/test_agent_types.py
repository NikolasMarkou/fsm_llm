"""Plan 659c07c3 step 3: the monitor lists agents from the executable registry (roadmap E4.2 and E4.3).

One rule (``_launchable``) picks the launchable rows (the ids of ``LAUNCHABLE``, user files included), the
registry loads once, lazily, off the event loop, under a leaf lock never held with the manager lock,
``GET /api/agent/types`` is open (launch stays gated) and ``fsm-llm-monitor --executables DIR`` loads
a directory with the one-warning-per-file rule. Doubles only: no model, no network.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.executables import builtin_executables, preset_executables
from fsm_llm.core.definitions.executable import Capability, Contract, Requires
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable
from fsm_llm.engine.executables.directory import load_directory_registry
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.monitor.instance_manager import agent as agent_mod
from fsm_llm.monitor.server import app, application, configure
from tests.test_fsm_llm_monitor.support import _manager

LAUNCHABLE = [
    "adapt",
    "auto_memory",
    "code-review",
    "data-extraction",
    "debate",
    "native_fc",
    "parallel_react",
    "plan_execute",
    "react",
    "reasoning_react",
    "reflexion",
    "rewoo",
    "self_consistency",
    "summarize",
    "verified_react",
]
NOT_LAUNCHABLE = [
    "evaluator_optimizer",
    "maker_checker",
    "meta_builder",
    "orchestrator",
    "prompt_chain",
    "reasoning.solve",
    "swarm",
]
ROW_FIELDS = {
    "id",
    "name",
    "launchable",
    "needs_tools",
    "takes_tools",
    "tool_names",
    "drawable",
}
_LOAD = "fsm_llm.engine.executables.registry.ExecutableRegistry.load_sources"

pytestmark = pytest.mark.usefixtures("reset_app")


def _doc(doc_id: str, base: str) -> dict[str, Any]:
    doc = pf.load_document(base)
    doc["id"] = doc_id
    return doc


def _write(directory: Path, name: str, doc: dict[str, Any]) -> None:
    (directory / name).write_text(json.dumps(doc), encoding="utf-8")


def _user_registry(directory: Path) -> ExecutableRegistry:
    builtins = [*builtin_executables(), *preset_executables()]
    return load_directory_registry(directory, builtins, prog="t")  # type: ignore[arg-type]


def _ids(rows: list[dict[str, Any]], **flags: bool) -> list[str]:
    return [r["id"] for r in rows if all(r[k] is v for k, v in flags.items())]


def _loader_probe() -> tuple[Any, dict[str, Any]]:
    """A ``load_sources`` double: counts calls and records whether a loop ran on its thread."""
    seen: dict[str, Any] = {"calls": 0, "on_loop": []}
    real = ExecutableRegistry.load_sources

    def load(self: ExecutableRegistry) -> Any:
        seen["calls"] += 1
        try:
            asyncio.get_running_loop()
            seen["on_loop"].append(True)
        except RuntimeError:
            seen["on_loop"].append(False)
        return real(self)

    return load, seen


class TestTheLaunchableRule:
    def test_the_rule_gives_exactly_the_fifteen_ids(self):
        rows = _manager().agent_types()
        assert _ids(rows, launchable=True) == LAUNCHABLE

    def test_the_seven_exclusions_are_listed_and_not_launchable(self):
        rows = _manager().agent_types()
        assert _ids(rows, launchable=False) == NOT_LAUNCHABLE

    def test_rows_are_sorted_have_exactly_the_fields_and_no_document_text(self):
        rows = _manager().agent_types()
        assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
        for row in rows:
            assert set(row) == ROW_FIELDS
            assert row["name"] == row["id"]
            assert all(isinstance(row[k], bool) for k in ("launchable", "drawable"))
            assert isinstance(row["tool_names"], list)

    def test_the_tool_flags_follow_the_contract(self):
        rows = {r["id"]: r for r in _manager().agent_types()}
        assert rows["react"]["needs_tools"] and rows["react"]["takes_tools"]
        assert not rows["debate"]["needs_tools"] and not rows["debate"]["takes_tools"]
        # optional tools: takes them, does not need them
        assert rows["plan_execute"]["takes_tools"]
        assert not rows["plan_execute"]["needs_tools"]

    def test_swarm_is_listed_and_not_drawable_the_rest_of_the_launchable_draw(self):
        rows = {r["id"]: r for r in _manager().agent_types()}
        assert rows["swarm"]["drawable"] is False
        assert all(rows[i]["drawable"] for i in LAUNCHABLE)

    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            ({}, True),
            ({"inputs": ["problem"]}, False),
            ({"inputs": ["task", "extra"]}, False),
            ({"callables": ["tools", "chain"]}, False),
            ({"callables": ["evaluation_fn"]}, False),
            ({"capabilities": [Capability.CONVERSATIONAL]}, False),
            ({"id": "orchestrator"}, False),
            ({"kind": "fsm"}, False),
        ],
    )
    def test_each_clause_of_the_rule_decides(self, change, expected):
        fields = {
            "id": "x",
            "kind": "pattern",
            "inputs": ["task"],
            "callables": ["tools"],
            "capabilities": [Capability.TOOLS],
            **change,
        }
        executable = SimpleNamespace(
            id=fields["id"],
            kind=fields["kind"],
            contract=Contract(inputs=fields["inputs"]),
            requires=Requires(callables=fields["callables"]),
            capabilities=fields["capabilities"],
        )
        assert agent_mod._launchable(executable) is expected

    def test_the_registry_filter_is_not_the_rule(self):
        # ``registry.list(launchable=True)`` excludes every tools pattern
        registry = ExecutableRegistry()
        registry.load_sources()
        wrong = {e.id for e in registry.list(kind="pattern", launchable=True)}
        assert "react" not in wrong
        assert "react" in _ids(_manager().agent_types(), launchable=True)

    def test_a_third_party_executable_with_no_preview_is_listed_not_drawable(self):
        class Plain(CodeBackedExecutable):
            async def execute(self, inputs, objects, context, identity):
                raise NotImplementedError

        registry = ExecutableRegistry()
        registry.register(
            Plain(
                id="plain",
                kind="pattern",
                impl="x.Plain",
                contract=Contract(inputs=["task"]),
            )
        )
        mgr = _manager()
        mgr.attach_executables(registry)
        (row,) = mgr.agent_types()
        assert row["id"] == "plain"
        assert (row["launchable"], row["drawable"]) == (True, False)
        assert (row["needs_tools"], row["takes_tools"]) == (False, False)


class TestAUserDirectory:
    def _directory(self, tmp_path: Path) -> Path:
        _write(tmp_path, "user_loop.json", _doc("user.loop", "react"))
        _write(tmp_path, "user_static.json", _doc("user.static", "evaluator_optimizer"))
        _write(tmp_path, "shadow.json", _doc("react", "rewoo"))
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        return tmp_path

    def test_a_conforming_file_launches_a_non_conforming_one_is_listed_only(
        self, tmp_path, capsys
    ):
        mgr = _manager()
        mgr.attach_executables(_user_registry(self._directory(tmp_path)))
        rows = {r["id"]: r for r in mgr.agent_types()}
        assert rows["user.loop"]["launchable"] and rows["user.loop"]["needs_tools"]
        assert rows["user.loop"]["drawable"]
        assert rows["user.static"]["launchable"] is False
        assert _ids(list(rows.values()), launchable=True) == sorted(
            [*LAUNCHABLE, "user.loop"]
        )

    def test_a_file_named_like_a_builtin_is_refused_with_one_warning_and_changes_nothing(
        self, tmp_path, capsys
    ):
        reference = {r["id"]: r for r in _manager().agent_types()}["react"]
        mgr = _manager()
        mgr.attach_executables(_user_registry(self._directory(tmp_path)))
        err = capsys.readouterr().err
        rows = {r["id"]: r for r in mgr.agent_types()}
        assert rows["react"] == reference
        warnings = [ln for ln in err.splitlines() if "warning" in ln]
        assert (
            len(warnings) == 2
        )  # the shadowing file and the broken one, one line each
        assert all(ln.startswith("t: warning: ") for ln in warnings)

    def test_a_file_that_differs_from_a_builtin_only_in_case_is_not_listed(
        self, tmp_path, capsys
    ):
        """Review of plan 659c07c3 (probe1, probe3): a user file ``React`` was listed, drawn from
        the file and launched as the built-in ``react``; the loader now refuses it."""
        _write(tmp_path, "React.json", _doc("React", "react"))
        mgr = _manager()
        mgr.attach_executables(_user_registry(tmp_path))
        assert len(capsys.readouterr().err.splitlines()) == 1
        assert _ids(mgr.agent_types(), launchable=True) == LAUNCHABLE
        with pytest.raises(KeyError):
            mgr.agent_preview("React")

    def test_the_rows_never_list_two_ids_equal_after_casefold(self):
        """Belt and braces: a registry the monitor did not load (the loader's rule is not there)."""
        registry = ExecutableRegistry()
        registry.load_sources()
        by_id = {e.id: e for e in registry.list(kind="pattern")}
        clones = []
        for new_id, base in (("React", "react"), ("Zed", "debate"), ("ZED", "debate")):
            clone = copy.copy(by_id[base])
            clone.id = new_id
            clones.append(clone)

        class Shadowed(ExecutableRegistry):
            def list(self, *args, **kwargs):
                return [*super().list(*args, **kwargs), *clones]

            def get(self, id, version=None):
                return next((c for c in clones if c.id == id), None) or super().get(
                    id, version
                )

        shadowed = Shadowed()
        shadowed.load_sources()
        mgr = _manager()
        mgr.attach_executables(shadowed)
        ids = _ids(mgr.agent_types())
        assert "react" in ids and "React" not in ids
        assert [i for i in ids if i.casefold() == "zed"] == ["ZED"]
        assert len({i.casefold() for i in ids}) == len(ids)

    def test_attach_replaces_the_rows_and_loads_no_default_registry(self, tmp_path):
        mgr = _manager()
        with mock.patch(_LOAD) as load:
            mgr.attach_executables(_user_registry(self._directory(tmp_path)))
            assert "user.loop" in _ids(mgr.agent_types())
        load.assert_not_called()
        mgr.attach_executables(ExecutableRegistry())
        assert mgr.agent_types() == []


class TestTheRegistryLoadsOnceAndOffTheLoop:
    def test_a_manager_loads_nothing_when_built(self):
        with mock.patch(_LOAD) as load:
            InstanceManager().close()
        load.assert_not_called()

    def test_the_first_call_loads_once_and_the_second_never(self):
        mgr = _manager()
        load, seen = _loader_probe()
        with mock.patch(_LOAD, load):
            first = mgr.agent_types()
            second = mgr.agent_types()
            mgr.get_capabilities()
        assert seen["calls"] == 1
        assert first == second

    def test_a_returned_row_is_a_copy(self):
        mgr = _manager()
        mgr.agent_types()[0]["id"] = "mutated"
        mgr.agent_types()[0]["tool_names"].append("x")
        assert mgr.agent_types()[0]["id"] != "mutated"
        assert "x" not in mgr.agent_types()[0]["tool_names"]

    def test_two_concurrent_first_calls_load_once(self):
        mgr = _manager()
        barrier = threading.Barrier(2)
        calls: list[int] = []
        real = ExecutableRegistry.load_sources

        def load(self: ExecutableRegistry) -> Any:
            calls.append(1)
            try:  # a second loader would meet this one here and pass at once
                barrier.wait(timeout=0.3)
            except threading.BrokenBarrierError:
                pass
            return real(self)

        results: list[Any] = []
        with mock.patch(_LOAD, load):
            threads = [
                threading.Thread(target=lambda: results.append(mgr.agent_types()))
                for _ in range(2)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        assert len(calls) == 1
        assert len(results) == 2 and results[0] == results[1]

    def test_the_load_does_not_hold_the_manager_lock(self):
        mgr = _manager()
        entered, release = threading.Event(), threading.Event()
        real = ExecutableRegistry.load_sources

        def load(self: ExecutableRegistry) -> Any:
            entered.set()
            release.wait(timeout=30)
            return real(self)

        with mock.patch(_LOAD, load):
            loader = threading.Thread(target=mgr.agent_types)
            loader.start()
            assert entered.wait(timeout=30)
            reader: list[Any] = []
            probe = threading.Thread(target=lambda: reader.append(mgr.list_instances()))
            probe.start()
            probe.join(timeout=10)  # the reader holds nothing the loader waits for
            blocked = probe.is_alive()
            release.set()
            loader.join(timeout=30)
            probe.join(timeout=30)
        assert not blocked
        assert reader == [[]]

    def test_a_load_that_raises_caches_nothing_and_the_next_call_retries(self):
        mgr = _manager()
        with mock.patch(_LOAD, side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                mgr.agent_types()
        assert _ids(mgr.agent_types(), launchable=True) == LAUNCHABLE

    def test_a_load_that_keeps_raising_is_a_500_whose_body_has_no_exception_text(self):
        """Review of plan 659c07c3 (blind spot): ``/api/capabilities`` can now fail, and the load is
        retried on every request with no backoff (a stated limit, not fixed); the body stays fixed."""
        configure(manager=_manager())
        client = TestClient(app, raise_server_exceptions=False)
        with mock.patch(_LOAD, side_effect=RuntimeError("SECRET-boom-text")) as load:
            first = client.get("/api/capabilities")
            second = client.get("/api/agent/types")
        assert (first.status_code, second.status_code) == (500, 500)
        assert "SECRET" not in first.text + second.text
        assert load.call_count == 2  # retried each time: no backoff

    def test_capabilities_and_types_load_off_the_loop_thread(self):
        mgr = _manager()
        configure(manager=mgr)
        load, seen = _loader_probe()
        with mock.patch(_LOAD, load):
            client = TestClient(app)
            assert client.get("/api/agent/types").status_code == 200
            mgr2 = _manager()
            configure(manager=mgr2)
            assert client.get("/api/capabilities").status_code == 200
        assert seen["calls"] == 2
        assert seen["on_loop"] == [False, False]


class TestAPreviewThatRaisesLeavesATrace:
    def test_the_draw_failure_is_one_debug_line_with_the_id_and_the_error_type(self):
        """Review of plan 659c07c3 (concern 8): ``_preview`` swallowed every exception silently."""
        from loguru import logger

        class Boom:
            id = "boom"

            def preview_definition(self):
                raise KeyError("DOCUMENT-TEXT")

        lines: list[str] = []
        logger.enable("fsm_llm")
        sink = logger.add(lines.append, level="DEBUG", format="{message}")
        try:
            assert agent_mod._preview(Boom()) is None
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")
        (line,) = [ln for ln in lines if "boom" in ln]
        assert "KeyError" in line and "DOCUMENT-TEXT" not in line


class TestTheRoutes:
    def test_the_types_route_answers_the_rows_and_is_open_while_launch_is_gated(self):
        configure(manager=_manager(), api_key="secret")
        client = TestClient(app)
        resp = client.get("/api/agent/types")
        assert resp.status_code == 200
        body = resp.json()
        assert list(body) == ["agent_types"]
        assert _ids(body["agent_types"], launchable=True) == LAUNCHABLE
        assert all(set(r) == ROW_FIELDS for r in body["agent_types"])
        launch = client.post("/api/agent/launch", json={"task": "t"})
        assert launch.status_code == 401

    def test_capabilities_list_the_sorted_launchable_ids(self):
        configure(manager=_manager())
        data = TestClient(app).get("/api/capabilities").json()
        assert data["agent_types"] == LAUNCHABLE
        assert (data["fsm"], data["workflows"], data["agents"]) == (True,) * 3

    def test_a_failed_load_is_a_generic_500_then_the_next_request_works(self):
        configure(manager=_manager())
        client = TestClient(app, raise_server_exceptions=False)
        with mock.patch(_LOAD, side_effect=RuntimeError("secret detail")):
            resp = client.get("/api/agent/types")
        assert resp.status_code == 500
        assert "secret detail" not in resp.text
        assert client.get("/api/agent/types").status_code == 200

    def test_a_directory_with_bad_files_keeps_the_builtins(self, tmp_path):
        (tmp_path / "broken.json").write_text("[]", encoding="utf-8")
        mgr = _manager()
        mgr.attach_executables(_user_registry(tmp_path))
        configure(manager=mgr)
        rows = TestClient(app).get("/api/agent/types").json()["agent_types"]
        assert _ids(rows, launchable=True) == LAUNCHABLE


class TestTheCarryAcrossTheLifespan:
    def test_a_closed_manager_replaced_by_the_lifespan_keeps_the_registry(self):
        registry = ExecutableRegistry()
        mgr = _manager()
        mgr.attach_executables(registry)
        configure(manager=mgr)
        with TestClient(app):
            pass
        assert mgr.closed
        with TestClient(app) as client:
            fresh = application.get_manager()
            try:
                assert fresh is not mgr
                assert fresh.executables is registry
                assert client.get("/api/agent/types").json() == {"agent_types": []}
            finally:
                fresh.close()


class TestTheExecutablesFlag:
    def _run(self, monkeypatch, *argv: str) -> pytest.ExceptionInfo[SystemExit]:
        import uvicorn

        from fsm_llm.monitor import __main__ as monitor_main

        monkeypatch.setattr(sys, "argv", ["fsm-llm-monitor", "--no-browser", *argv])
        monkeypatch.setattr(
            uvicorn.Server, "run", mock.Mock(side_effect=KeyboardInterrupt)
        )
        with pytest.raises(SystemExit) as exc:
            monitor_main.main_cli()
        return exc

    def test_a_missing_directory_is_one_stderr_line_exit_1_and_no_server(
        self, monkeypatch, capsys, tmp_path
    ):
        import uvicorn

        spy = mock.Mock(side_effect=AssertionError("the server was built"))
        monkeypatch.setattr(uvicorn, "Server", spy)
        monkeypatch.setattr(
            sys, "argv", ["fsm-llm-monitor", "--executables", str(tmp_path / "nope")]
        )
        from fsm_llm.monitor import __main__ as monitor_main

        with pytest.raises(SystemExit) as exc:
            monitor_main.main_cli()
        assert exc.value.code == 1
        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("fsm-llm-monitor: error: --executables ")
        spy.assert_not_called()

    def test_a_file_is_an_error_too(self, monkeypatch, capsys, tmp_path):
        target = tmp_path / "a.json"
        target.write_text("{}", encoding="utf-8")
        exc = self._run(monkeypatch, "--executables", str(target))
        assert exc.value.code == 1
        assert "not an existing directory" in capsys.readouterr().err

    def test_a_bad_file_is_one_warning_and_the_start_goes_on(
        self, monkeypatch, capsys, tmp_path
    ):
        _write(tmp_path, "user_loop.json", _doc("user.loop", "react"))
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        exc = self._run(monkeypatch, "--executables", str(tmp_path))
        assert exc.value.code == 130  # reached server.run (Ctrl-C)
        err = capsys.readouterr().err
        assert len([ln for ln in err.splitlines() if "warning" in ln]) == 1
        rows = application.get_manager().agent_types()
        assert "user.loop" in _ids(rows, launchable=True)
        assert _ids(rows, launchable=True) == sorted([*LAUNCHABLE, "user.loop"])

    def test_no_flag_leaves_the_default_registry_lazy(self, monkeypatch):
        with mock.patch(_LOAD) as load:
            exc = self._run(monkeypatch)
        assert exc.value.code == 130
        load.assert_not_called()
        assert application.get_manager().executables is None

    def test_help_names_the_flag(self, monkeypatch, capsys):
        from fsm_llm.monitor import __main__ as monitor_main

        monkeypatch.setattr(sys, "argv", ["fsm-llm-monitor", "--help"])
        with pytest.raises(SystemExit) as exc:
            monitor_main.main_cli()
        assert exc.value.code == 0
        assert "--executables DIR" in capsys.readouterr().out
