"""The workflow presets of the dashboard are documents (plan 7d9c0955 step 7, D-011).

``monitor/workflows/demo_linear.json`` and ``demo_branching.json`` are ``workflow`` envelopes; the
monitor offers those two files and nothing else, built from its own named callables, read through a
private ``ExecutableRegistry`` that loads once, lazily, off the loop, under the leaf
``_catalog_lock``. The wire of the two routes is held by ``test_workflow_preset_golden.py`` (G2: the
bytes of a8ad5d31 plus the two named deltas). These rows guard the rest.

Invariants (plan 7d9c0955): 2 (a document names objects and holds no code; a request never names a
file, a callable or a step), 5 (a refusal names a label, never a value), 8 (a workflow body never
loads as an FSM), and the census rows D6 (document text and paths), D9 (the launch key gate) and D11
(a preset run uses no model).
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fsm_llm.core.verify.executable import load_executable
from fsm_llm.monitor.definitions import MonitorConfig
from fsm_llm.monitor.instance_manager import InstanceManager, workflow, workflow_presets
from fsm_llm.monitor.instance_manager.workflow_presets import (
    build_preset,
    load_preset_registry,
    preset_callables,
)
from fsm_llm.monitor.server import app, configure
from fsm_llm.workflows import AutoStep, WorkflowDefinition
from fsm_llm.workflows.loader import builtin_step_classes

PRESETS = ("demo_branching", "demo_linear")
DIRECTORY = Path(str(resources.files("fsm_llm.monitor").joinpath("workflows")))


def _document(preset_id: str) -> dict[str, Any]:
    return json.loads((DIRECTORY / f"{preset_id}.json").read_text(encoding="utf-8"))


def _manager() -> InstanceManager:
    manager = InstanceManager(config=MonitorConfig())
    manager.global_collector.cleanup()
    return manager


def _on_a_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _steps(preset_id: str) -> dict[str, dict[str, Any]]:
    return _document(preset_id)["body"]["steps"]


# ---- the documents ------------------------------------------------------------------------


class TestTheDocuments:
    def test_the_directory_ships_exactly_the_two_documents(self) -> None:
        assert sorted(p.name for p in DIRECTORY.iterdir()) == [
            "demo_branching.json",
            "demo_linear.json",
        ]

    @pytest.mark.parametrize("preset_id", PRESETS)
    def test_a_document_passes_the_whole_load_sequence(self, preset_id: str) -> None:
        envelope, result = load_executable(
            _document(preset_id),
            resolver=lambda category, name: (
                category != "callables" or preset_callables().get(name) is not None
            ),
        )
        assert result.is_valid, result.errors
        assert envelope is not None and envelope.kind == "workflow"
        assert envelope.id == preset_id

    @pytest.mark.parametrize("preset_id", PRESETS)
    def test_a_document_holds_names_and_no_code_or_host_object(
        self, preset_id: str
    ) -> None:
        document = _document(preset_id)
        steps = document["body"]["steps"]
        allowed = set(builtin_step_classes())
        assert {s["type"] for s in steps.values()} <= {"AutoStep", "ConditionStep"}
        assert {s["type"] for s in steps.values()} <= allowed
        assert not any("registry" in s or "callables" in s for s in steps.values())
        assert document["requires"]["tools"] == "none"
        used = {s[k] for s in steps.values() for k in ("action", "condition") if k in s}
        assert used == set(document["requires"]["callables"])
        assert used <= {name for name, _ in preset_callables().items()}

    @pytest.mark.parametrize("preset_id", PRESETS)
    def test_finish_is_a_plain_auto_step_and_not_a_custom_class(
        self, preset_id: str
    ) -> None:
        finish = _steps(preset_id)["finish"]
        assert finish["type"] == "AutoStep" and finish["next_step_id"] == ""
        assert "action" not in finish
        definition = build_preset(load_preset_registry(), preset_id)
        assert type(definition.steps["finish"]) is AutoStep

    def test_the_header_texts_are_the_base_ones(self) -> None:
        rows = {p["id"]: p for p in _manager().get_workflow_presets()}
        assert rows["demo_linear"] == {
            "id": "demo_linear",
            "name": "Demo: Linear Pipeline",
            "description": "Two automatic processing steps followed by a terminal step.",
        }
        assert rows["demo_branching"]["name"] == "Demo: Conditional Branch"

    def test_the_callables_are_the_five_named_by_the_documents(self) -> None:
        assert sorted(n for n, _ in preset_callables().items()) == [
            "amount_is_high",
            "ingest",
            "mark_high",
            "mark_low",
            "process",
        ]

    def test_the_package_data_glob_covers_the_documents(self) -> None:
        body = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        assert '"workflows/*.json"' in body
        covered = {p.name for p in (DIRECTORY.parent).glob("workflows/*.json")}
        assert covered == {f"{p}.json" for p in PRESETS}


# ---- the registry -------------------------------------------------------------------------


class TestTheRegistryLoadsLazilyOnce:
    def test_the_constructor_loads_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(workflow, "load_preset_registry", lambda: calls.append(1))
        manager = _manager()
        assert calls == [] and manager._workflow_registry is None

    def test_the_first_read_loads_once_under_the_catalog_lock_and_not_under_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager()
        seen: list[tuple[bool, bool]] = []
        real = load_preset_registry

        def spy() -> Any:
            seen.append((manager._catalog_lock.locked(), manager._lock._is_owned()))
            return real()

        monkeypatch.setattr(workflow, "load_preset_registry", spy)
        for _ in range(3):
            manager.get_workflow_presets()
            manager.launch_workflow(preset_id="demo_linear")
        assert seen == [(True, False)]

    def test_concurrent_first_calls_load_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager()
        loads: list[int] = []
        real = load_preset_registry
        monkeypatch.setattr(
            workflow,
            "load_preset_registry",
            lambda: (loads.append(1), real())[1],
        )
        start = threading.Barrier(8)
        results: list[list[dict[str, str]]] = []

        def read() -> None:
            start.wait(timeout=30)
            results.append(manager.get_workflow_presets())

        threads = [threading.Thread(target=read) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert len(loads) == 1
        assert len(results) == 8 and all(r == results[0] for r in results)

    def test_a_load_that_raises_caches_nothing_and_the_next_call_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager()
        real = load_preset_registry
        calls: list[int] = []

        def flaky() -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("the first load fails")
            return real()

        monkeypatch.setattr(workflow, "load_preset_registry", flaky)
        with pytest.raises(RuntimeError, match="first load"):
            manager.get_workflow_presets()
        assert manager._workflow_registry is None
        assert [p["id"] for p in manager.get_workflow_presets()] == list(PRESETS)
        assert len(calls) == 2
        manager.get_workflow_presets()
        assert len(calls) == 2

    def test_two_managers_hold_two_registries(self) -> None:
        first, second = _manager(), _manager()
        first.get_workflow_presets()
        assert second._workflow_registry is None
        second.get_workflow_presets()
        assert first._workflow_registry is not second._workflow_registry

    def test_the_route_reads_the_catalog_off_the_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager()
        on_a_loop: list[bool] = []
        real = load_preset_registry

        def spy() -> Any:
            on_a_loop.append(_on_a_loop())
            return real()

        monkeypatch.setattr(workflow, "load_preset_registry", spy)
        configure(manager=manager)
        try:
            response = TestClient(app).get("/api/workflow/presets")
        finally:
            manager.close()
            configure()
        assert response.status_code == 200
        assert [w["id"] for w in response.json()["workflows"]] == list(PRESETS)
        assert on_a_loop == [False]


# ---- a directory that does not load -------------------------------------------------------


def _write(directory: Path, name: str, document: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(document), encoding="utf-8")


def _point_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        workflow_presets,
        "resources",
        SimpleNamespace(files=lambda package: root),
    )


def _good(preset_id: str = "demo_linear") -> dict[str, Any]:
    return _document(preset_id)


class TestADirectoryThatDoesNotLoad:
    """The registry's own refusals (V1, V3, the V2 probe) hold for the operator's package files."""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda d: d.update(body=[]), id="body-not-a-mapping"),
            pytest.param(
                lambda d: d["body"]["steps"]["finish"].update(type="TimerStep"),
                id="a-pausing-step",
            ),
            pytest.param(
                lambda d: d["body"]["steps"]["finish"].update(type="NoSuchStep"),
                id="an-unknown-step-type",
            ),
            pytest.param(
                lambda d: d["requires"].update(callables=[]), id="unclaimed-callables"
            ),
            pytest.param(
                lambda d: d["body"]["steps"]["ingest"].update(action="not_a_callable"),
                id="an-unknown-callable-name",
            ),
            pytest.param(
                lambda d: d["body"]["steps"]["finish"].update(registry="reg"),
                id="a-host-bound-field",
            ),
        ],
    )
    def test_a_bad_document_fails_the_load_and_nothing_is_cached(
        self,
        mutate: Callable[[dict[str, Any]], Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        document = _good()
        mutate(document)
        _write(tmp_path / "workflows", "demo_linear.json", document)
        _point_at(monkeypatch, tmp_path)
        manager = _manager()
        with pytest.raises(Exception) as caught:
            manager.get_workflow_presets()
        assert manager._workflow_registry is None
        assert "demo_linear" in str(caught.value) or "workflow" in str(caught.value)

    def test_an_empty_or_missing_directory_is_a_failed_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _point_at(monkeypatch, tmp_path)  # no `workflows` directory at all
        with pytest.raises(RuntimeError, match="did not load"):
            load_preset_registry()
        (tmp_path / "workflows").mkdir()
        with pytest.raises(RuntimeError, match="did not load"):
            load_preset_registry()

    def test_a_document_of_another_kind_is_no_preset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests import executables_corpus as corpus

        case = next(c for c in corpus.fsm_cases() if not c.startswith("reasoning/"))
        _write(tmp_path / "workflows", "demo_linear.json", _good())
        _write(tmp_path / "workflows", "an_fsm.json", corpus.wrap(case))
        _point_at(monkeypatch, tmp_path)
        registry = load_preset_registry()
        assert len(registry.list()) == 2
        assert [e.id for e in registry.list(kind="workflow")] == ["demo_linear"]
        manager = _manager()
        manager._workflow_registry = registry
        assert [p["id"] for p in manager.get_workflow_presets()] == ["demo_linear"]
        with pytest.raises(ValueError, match=r"^Unknown workflow preset: "):
            manager.launch_workflow(preset_id=case)

    def test_the_load_error_names_a_label_and_never_a_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document = _good()
        document["body"]["steps"]["finish"]["name"] = "SECRET-VALUE-4711"
        document["body"]["steps"]["finish"]["type"] = "NoSuchStep"
        _write(tmp_path / "workflows", "demo_linear.json", document)
        _point_at(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError) as caught:
            load_preset_registry()
        assert "SECRET-VALUE-4711" not in str(caught.value)


# ---- the launch ---------------------------------------------------------------------------


class TestTheLaunch:
    @pytest.mark.parametrize(
        "text",
        [
            "demo_linear\n",
            "\tdemo_linear",
            "DEMO_LINEAR",
            "Demo_Linear",
            "demo‑linear",
            "demo_linear.json",
            "workflows/demo_linear",
            "../workflows/demo_linear",
            "demo_linear\x00",
            "demo_linear, demo_branching",
        ],
    )
    def test_the_id_is_matched_exactly(self, text: str) -> None:
        manager = _manager()
        with pytest.raises(ValueError, match=r"^Unknown workflow preset: "):
            manager.launch_workflow(preset_id=text)

    def test_definition_json_alone_is_refused(self) -> None:
        manager = _manager()
        with pytest.raises(ValueError, match="Custom workflow JSON is not supported"):
            manager.launch_workflow(definition_json=_document("demo_linear"))
        with pytest.raises(ValueError, match="Custom workflow JSON is not supported"):
            manager.launch_workflow(
                definition_json={"workflow_id": "x", "initial_step_id": "a"}
            )

    async def test_definition_json_beside_a_preset_id_is_ignored(self) -> None:
        manager = _manager()
        bogus = _document("demo_linear")
        bogus["body"]["steps"]["ingest"]["action"] = "mark_low"
        managed = manager.launch_workflow(
            preset_id="demo_linear", definition_json=bogus
        )
        run = await manager.start_workflow_instance(
            managed.instance_id, managed.workflow_id, {}
        )
        context = manager.get_workflow_status(managed.instance_id, run)["context"]
        assert context == {"ingested": True, "processed": True}

    def test_every_launch_builds_a_fresh_definition(self) -> None:
        registry = load_preset_registry()
        first = build_preset(registry, "demo_branching")
        second = build_preset(registry, "demo_branching")
        assert isinstance(first, WorkflowDefinition) and first is not second
        assert first.steps["route"] is not second.steps["route"]
        manager = _manager()
        one = manager.launch_workflow(preset_id="demo_branching")
        two = manager.launch_workflow(preset_id="demo_branching")
        assert one.engine is not two.engine
        assert (
            one.engine.workflow_definitions["demo_branching"]
            is not two.engine.workflow_definitions["demo_branching"]
        )

    @pytest.mark.parametrize(
        ("preset_id", "context", "expected"),
        [
            ("demo_linear", {}, {"ingested": True, "processed": True}),
            (
                "demo_branching",
                {"amount": 1000},
                {"amount": 1000, "tier": "high", "condition_result": True},
            ),
            (
                "demo_branching",
                {"amount": 999},
                {"amount": 999, "tier": "low", "condition_result": False},
            ),
            ("demo_branching", {}, {"tier": "low", "condition_result": False}),
        ],
    )
    async def test_a_launched_preset_runs_with_no_model(
        self,
        preset_id: str,
        context: dict[str, Any],
        expected: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def no_model(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a preset run made a model call")

        monkeypatch.setattr("fsm_llm.llm.providers.litellm.completion", no_model)
        monkeypatch.setenv("LLM_MODEL", "")
        manager = _manager()
        managed = manager.launch_workflow(preset_id=preset_id)
        run = await manager.start_workflow_instance(
            managed.instance_id, managed.workflow_id, context
        )
        status = manager.get_workflow_status(managed.instance_id, run)
        assert status["status"] == "completed"
        assert status["context"] == expected

    def test_the_run_names_no_executable_step_and_no_host_object(self) -> None:
        for preset_id in PRESETS:
            definition = build_preset(load_preset_registry(), preset_id)
            assert {type(s).__name__ for s in definition.steps.values()} <= {
                "AutoStep",
                "ConditionStep",
            }
