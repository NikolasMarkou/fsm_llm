"""The executable registry and weighted aliases (plan ace7a8c8 step 22.2, roadmap E2.1, E2.2, D-059).

The registry serves executables by id, by version and by template identity, reloads atomically
(validate every source with no lock held, swap one immutable snapshot under one leaf lock),
retains TEMPLATES only (a bounded LRU, pinned in-flight ones exempt) and never an instance.
Rows with threads use events and bounded waits, never sleeps. Real core, scripted model.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions.executable import (
    Contract,
    ExecutableError,
    InstanceIdentityMismatchError,
    Requires,
)
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import (
    MAX_DOCUMENT_BYTES,
    MAX_RETAINED_TEMPLATES,
    ExecutableCollisionError,
    ExecutableEvictedError,
    ExecutableNotFoundError,
    ExecutableRegistry,
    FsmBodyExecutable,
    FsmInstance,
    is_template_identity,
    make_alias,
    pick_route,
)
from fsm_llm.engine.executables import registry as registry_module
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.test_async_engine import _Gated

_BOUND = 10.0


def _fsm(response: str = "A.") -> dict[str, Any]:
    return kc._fsm(
        "Demo",
        "a",
        kc._state(
            "a",
            extraction_instructions="Extract the result.",
            required_context_keys=["result"],
            response_instructions=response,
            transitions=[kc._go("b")],
        ),
        kc._state("b", response_instructions="B."),
    )


def _doc(
    ident: str = "demo",
    version: str = "1",
    *,
    response: str = "A.",
    capabilities: tuple[str, ...] = (),
    callables: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "schema": "fsm-llm/executable",
        "kind": "fsm",
        "id": ident,
        "version": version,
        "contract": {"inputs": ["topic"], "outputs": ["result"]},
        "requires": {"callables": list(callables)},
        "capabilities": list(capabilities),
        "body": {"fsm": _fsm(response)},
    }


def _write(path: Path, document: Any) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _context() -> RunContext:
    return RunContext(
        llm_interface=ScriptedLLM(extraction_data={"result": "1"}, default_response="r")
    )


class _CodeBacked:
    """A minimal code-backed executable: the protocol's attributes and nothing else."""

    kind = "fsm"
    contract = Contract()
    capabilities: ClassVar[list[Any]] = []
    lifecycle = Lifecycle.BUILTIN

    def __init__(
        self, ident: str, version: str = "1", callables: list[str] | None = None
    ):
        self.id, self.version = ident, version
        self.requires = Requires(callables=callables or [])
        self.template_identity = (ident + ":" + version).encode().hex().ljust(64, "0")

    def instantiate(self, options: Any = None, variables: Any = None) -> Any:
        raise NotImplementedError

    def run(self, inputs: Any, *, context: Any = None) -> RunResult:
        raise NotImplementedError

    async def arun(self, inputs: Any, *, context: Any = None) -> RunResult:
        raise NotImplementedError


# ---- lookup -----------------------------------------------------------------------------------


class TestLookup:
    def test_get_by_id_version_and_template_identity(self) -> None:
        registry = ExecutableRegistry()
        one = registry.add_document(_doc(version="1"))
        two = registry.add_document(_doc(version="2"))
        assert registry.get("demo") is two  # the greatest version
        assert registry.get("demo", "1") is one
        assert registry.get_by_template(one.template_identity) is one
        with pytest.raises(ExecutableNotFoundError, match="'nope'"):
            registry.get("nope")
        with pytest.raises(ExecutableNotFoundError, match="version '9'"):
            registry.get("demo", "9")

    def test_versions_order_by_integer_tuple(self) -> None:
        registry = ExecutableRegistry()
        for label in ("9", "10", "1.2", "1.10"):
            registry.add_document(_doc("v", label))
        assert registry.get("v").version == "10"  # 10 above 9, not text order
        assert [e.version for e in registry.list()] == ["1.2", "1.10", "9", "10"]

    @pytest.mark.parametrize("order", [("alpha", "1"), ("1", "alpha")])
    def test_an_unparseable_label_is_allowed_only_alone(
        self, order: tuple[str, str]
    ) -> None:
        registry = ExecutableRegistry()
        registry.add_document(_doc("v", order[0]))
        with pytest.raises(ExecutableCollisionError, match="sole version"):
            registry.add_document(_doc("v", order[1]))
        assert [e.version for e in registry.list()] == [order[0]]

    def test_a_duplicate_id_and_version_is_refused_and_the_first_stays(self) -> None:
        registry = ExecutableRegistry()
        first = registry.add_document(_doc(response="first"))
        with pytest.raises(ExecutableCollisionError, match="already served"):
            registry.add_document(_doc(response="second"))
        assert registry.get("demo") is first

    def test_an_invalid_document_raises_with_the_errors_and_adds_nothing(self) -> None:
        registry = ExecutableRegistry()
        bad = _doc()
        bad["body"]["fsm"]["initial_state"] = "nowhere"
        with pytest.raises(ExecutableError, match="invalid executable document"):
            registry.add_document(bad)
        with pytest.raises(ExecutableError):
            registry.add_document({"kind": "fsm"})
        assert registry.list() == []

    def test_list_filters_by_kind_capability_and_launchable(self) -> None:
        registry = ExecutableRegistry()
        registry.add_document(_doc("a", capabilities=("tools",)))
        registry.add_document(_doc("b", capabilities=("hitl", "tools")))
        registry.register(_CodeBacked("c", callables=["hitl_policy"]))  # type: ignore[arg-type]
        assert [e.id for e in registry.list()] == ["a", "b", "c"]
        assert [e.id for e in registry.list(kind="fsm")] == ["a", "b", "c"]
        assert registry.list(kind="strategy") == []
        assert [e.id for e in registry.list(capability="tools")] == ["a", "b"]
        assert [e.id for e in registry.list(capability="hitl")] == ["b"]
        assert [e.id for e in registry.list(launchable=True)] == ["a", "b"]
        assert [e.id for e in registry.list(launchable=False)] == ["c"]
        with pytest.raises(ValueError, match="capability"):
            registry.list(capability="flying")

    def test_register_takes_the_same_collision_rule(self) -> None:
        registry = ExecutableRegistry()
        registry.register(_CodeBacked("x"))  # type: ignore[arg-type]
        with pytest.raises(ExecutableCollisionError):
            registry.register(_CodeBacked("x"))  # type: ignore[arg-type]
        assert registry.get_by_template(_CodeBacked("x").template_identity).id == "x"


# ---- paths and reload ---------------------------------------------------------------------------


class TestPaths:
    def test_a_directory_contributes_sorted_json_members_each_failure_reported(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / "a.json", _doc("a"))
        _write(tmp_path / "b.json", _doc("b"))
        (tmp_path / "notes.txt").write_text("not a document")
        (tmp_path / "bad_json.json").write_text("{not json")
        (tmp_path / "list.json").write_text("[1, 2]")
        (tmp_path / "latin.json").write_bytes(b'{"x": "\xe9"}')
        (tmp_path / "huge.json").write_text(" " * (MAX_DOCUMENT_BYTES + 1))
        _write(tmp_path / "invalid.json", {"kind": "fsm"})
        outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
        _write(outside, _doc("outside"))
        (tmp_path / "link.json").symlink_to(outside)
        registry = ExecutableRegistry.from_paths([tmp_path])
        assert [e.id for e in registry.list()] == ["a", "b"]
        report = registry.reload()
        failed = {Path(k).name for k in report.failed}
        assert failed == {
            "bad_json.json",
            "list.json",
            "latin.json",
            "huge.json",
            "invalid.json",
            "link.json",
        }
        assert dict((Path(k).name, v) for k, v in report.failed.items())[
            "link.json"
        ] == ("a symbolic link is skipped",)
        assert report.ids == (("a", "1"), ("b", "1"))

    def test_a_file_path_loads_and_a_missing_path_contributes_nothing(
        self, tmp_path: Path
    ) -> None:
        one = _write(tmp_path / "one.json", _doc("one"))
        registry = ExecutableRegistry.from_paths([one, tmp_path / "missing"])
        assert [e.id for e in registry.list()] == ["one"]

    def test_the_first_source_wins_a_duplicate_and_the_refusal_is_reported(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / "1_first.json", _doc("same", response="first"))
        _write(tmp_path / "2_second.json", _doc("same", response="second"))
        registry = ExecutableRegistry(paths=[tmp_path])
        report = registry.reload()
        assert len(report.collisions) == 1 and "2_second.json" in report.collisions[0]
        winner = registry.get("same")
        assert (
            winner.envelope.body.fsm["states"]["a"]["response_instructions"] == "first"
        )  # type: ignore[attr-defined]

    def test_a_reload_serves_the_new_template_and_keeps_the_old_one_retained(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path / "x.json", _doc("x", response="one"))
        registry = ExecutableRegistry.from_paths([tmp_path])
        old = registry.get("x")
        _write(path, _doc("x", response="two"))
        report = registry.reload()
        new = registry.get("x")
        assert new is not old and new.template_identity != old.template_identity
        assert report.failed == {}
        # a running conversation (or a resume) still resolves the old identity
        assert registry.get_by_template(old.template_identity) is old
        path.unlink()
        registry.reload()
        with pytest.raises(ExecutableNotFoundError):
            registry.get("x")  # the id is gone with the file
        assert (
            registry.get_by_template(old.template_identity) is old
        )  # the identity is not

    def test_a_failing_file_keeps_its_previous_entries_and_is_reported(
        self, tmp_path: Path
    ) -> None:
        good = _write(tmp_path / "x.json", _doc("x", response="one"))
        _write(tmp_path / "y.json", _doc("y"))
        registry = ExecutableRegistry.from_paths([tmp_path])
        before = registry.get("x")
        good.write_text("{broken", encoding="utf-8")
        report = registry.reload()
        assert registry.get("x") is before  # kept
        assert [Path(k).name for k in report.failed] == ["x.json"]
        assert registry.get("y")

    def test_memory_entries_survive_a_reload(self, tmp_path: Path) -> None:
        registry = ExecutableRegistry(paths=[tmp_path])
        mine = registry.add_document(_doc("mine"))
        _write(tmp_path / "f.json", _doc("f"))
        registry.reload()
        assert registry.get("mine") is mine and registry.get("f")


class TestAtomicReload:
    def test_a_run_started_before_a_reload_finishes_on_the_old_template(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path / "x.json", _doc("x", response="one"))
        registry = ExecutableRegistry.from_paths([tmp_path])
        old = registry.get("x")

        async def scenario() -> tuple[RunResult, Any]:
            llm = _Gated(extraction_data={"result": "1"}, default_response="r")
            llm.arm()
            run = asyncio.create_task(
                old.arun({"topic": "t"}, context=RunContext(llm_interface=llm))
            )
            await asyncio.wait_for(llm.entered.wait(), _BOUND)
            # the run is in flight: it holds a pin on the OLD template
            assert registry._pins == {old.template_identity: 1}
            _write(path, _doc("x", response="two"))
            registry.reload()
            new = registry.get("x")
            llm.release.set()
            return await asyncio.wait_for(run, _BOUND), new

        result, new = asyncio.run(scenario())
        assert result.identity.template_identity == old.template_identity
        assert new.template_identity != old.template_identity
        assert result.success
        assert registry._pins == {}

    def test_a_reader_thread_sees_only_wholly_old_or_wholly_new_snapshots(
        self, tmp_path: Path
    ) -> None:
        x = tmp_path / "x.json"
        y = tmp_path / "y.json"
        _write(x, _doc("x", "1"))
        _write(y, _doc("y", "1"))
        registry = ExecutableRegistry.from_paths([tmp_path])
        stop = threading.Event()
        started = threading.Event()
        seen: set[tuple[tuple[str, str], ...]] = set()
        samples = 0

        def reader() -> None:
            nonlocal samples
            started.set()
            while not stop.is_set():
                seen.add(tuple((e.id, e.version) for e in registry.list()))
                samples += 1

        thread = threading.Thread(target=reader)
        thread.start()
        assert started.wait(_BOUND)
        for generation in range(30):
            label = "2" if generation % 2 == 0 else "1"
            _write(x, _doc("x", label))
            _write(y, _doc("y", label))
            registry.reload()
        stop.set()
        thread.join(_BOUND)
        assert not thread.is_alive()
        assert samples > 0
        allowed = {(("x", "1"), ("y", "1")), (("x", "2"), ("y", "2"))}
        assert seen <= allowed, f"a torn snapshot was visible: {seen - allowed}"

    def test_a_reload_never_edits_an_installed_snapshot_and_swaps_once(
        self, tmp_path: Path
    ) -> None:
        """Kills: an in-place edit of the live snapshot, a swap in two steps (a half-new view)."""
        import ast
        import inspect
        import textwrap

        path = _write(tmp_path / "x.json", _doc("x", "1"))
        registry = ExecutableRegistry.from_paths([tmp_path])
        installed = registry._snapshot
        frozen = dict(installed.entries)
        _write(path, _doc("x", "2"))
        registry.reload()
        assert registry._snapshot is not installed
        assert installed.entries == frozen  # the old view is whole
        tree = ast.parse(textwrap.dedent(inspect.getsource(ExecutableRegistry.reload)))
        swaps = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_snapshot"
                for t in n.targets
            )
        ]
        assert len(swaps) == 1

    def test_no_read_validation_or_log_runs_under_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills: a loader, a file read or a logger call moved under ``_lock``."""
        _write(tmp_path / "x.json", _doc("x"))
        (tmp_path / "broken.json").write_text("{broken")
        registry = ExecutableRegistry(paths=[tmp_path])

        class _SpyLock:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self.held = False

            def __enter__(self) -> None:
                self._lock.acquire()
                self.held = True

            def __exit__(self, *exc: object) -> None:
                self.held = False
                self._lock.release()

        spy = _SpyLock()
        registry._lock = spy  # type: ignore[assignment]
        violations: list[str] = []

        def guard(name: str, real: Any) -> Any:
            def wrapper(*a: Any, **k: Any) -> Any:
                if spy.held:
                    violations.append(name)
                return real(*a, **k)

            return wrapper

        monkeypatch.setattr(
            registry_module,
            "load_executable",
            guard("load", registry_module.load_executable),
        )
        monkeypatch.setattr(
            registry_module.json, "loads", guard("json", registry_module.json.loads)
        )
        monkeypatch.setattr(registry_module.os, "listdir", guard("listdir", os.listdir))
        monkeypatch.setattr(
            registry_module,
            "resolve_within",
            guard("confine", registry_module.resolve_within),
        )
        monkeypatch.setattr(
            registry_module.logger,
            "warning",
            guard("log", registry_module.logger.warning),
        )
        report = registry.reload()
        registry.add_document(_doc("y"))
        registry.get("x")
        registry.pin(registry.get("x"))
        registry.unpin(registry.get("x"))
        assert report.failed  # the broken file made the log call happen
        assert violations == []


# ---- retention -------------------------------------------------------------------------------------


class TestRetention:
    def test_many_runs_with_distinct_inputs_retain_one_template_and_no_instance(
        self,
    ) -> None:
        registry = ExecutableRegistry()
        registry.add_document(_doc())
        exe = registry.get("demo")
        for n in range(200):  # distinct task text on every run
            result = exe.run({"topic": f"task number {n}"}, context=_context())
            assert result.success
        for _ in range(10_000):
            registry.get("demo").instantiate()
        gc.collect()
        assert list(registry._retained) == [exe.template_identity]
        assert registry._pins == {}
        assert [o for o in gc.get_objects() if isinstance(o, FsmInstance)] == []
        assert len(registry._snapshot.entries) == 1

    def test_more_than_the_bound_evicts_the_oldest_unpinned_only(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path / "x.json", _doc("x", response="r0"))
        registry = ExecutableRegistry.from_paths([tmp_path], max_retained_templates=3)
        identities = [registry.get("x").template_identity]
        for n in range(1, 6):
            _write(path, _doc("x", response=f"r{n}"))
            registry.reload()
            identities.append(registry.get("x").template_identity)
        assert len(set(identities)) == 6
        assert len(registry._retained) == 3
        for old in identities[:3]:
            with pytest.raises(ExecutableEvictedError, match=old):
                registry.get_by_template(old)
        for kept in identities[3:]:
            assert registry.get_by_template(kept).template_identity == kept

    def test_a_pinned_template_survives_pressure_and_is_evictable_after_unpin(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path / "x.json", _doc("x", response="r0"))
        registry = ExecutableRegistry.from_paths([tmp_path], max_retained_templates=2)
        old = registry.get("x")
        registry.pin(old)
        registry.pin(old)  # counted: two runs in flight
        for n in range(1, 8):
            _write(path, _doc("x", response=f"r{n}"))
            registry.reload()
        assert registry.get_by_template(old.template_identity) is old
        registry.unpin(old)
        assert registry._pins == {old.template_identity: 1}  # one run left
        registry.unpin(old)
        assert registry._pins == {}
        for n in range(8, 12):  # pressure again: now the oldest unpinned goes
            _write(path, _doc("x", response=f"r{n}"))
            registry.reload()
        with pytest.raises(ExecutableEvictedError):
            registry.get_by_template(old.template_identity)
        registry.unpin(old)  # a stray unpin changes nothing
        assert registry._pins == {}

    def test_the_bound_is_validated_and_the_default_is_a_constant(self) -> None:
        assert MAX_RETAINED_TEMPLATES >= 16
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="max_retained_templates"):
                ExecutableRegistry(max_retained_templates=bad)  # type: ignore[arg-type]

    def test_a_template_pinned_after_eviction_is_retained_again(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path / "x.json", _doc("x", response="r0"))
        registry = ExecutableRegistry.from_paths([tmp_path], max_retained_templates=1)
        old = registry.get("x")
        _write(path, _doc("x", response="r1"))
        registry.reload()
        with pytest.raises(ExecutableEvictedError):
            registry.get_by_template(old.template_identity)
        registry.pin(old)  # the running executable re-registers itself
        assert registry.get_by_template(old.template_identity) is old

    def test_a_registry_built_adapter_pins_through_the_registry(self) -> None:
        registry = ExecutableRegistry()
        exe = registry.add_document(_doc())
        assert isinstance(exe, FsmBodyExecutable) and exe.pins is registry


class TestRebuild:
    def test_a_logged_record_rebuilds_and_a_tampered_one_fails(self) -> None:
        registry = ExecutableRegistry()
        document = _doc()
        document["variables"] = ["tone"]
        document["body"]["fsm"]["states"]["a"]["response_instructions"] = "Be {{tone}}."
        exe = registry.add_document(document)
        instance = exe.instantiate(variables={"tone": "kind"})
        rebuilt = registry.rebuild(
            exe.template_identity, instance.record, instance.instance_identity
        )
        assert rebuilt.instance_identity == instance.instance_identity
        with pytest.raises(InstanceIdentityMismatchError):
            registry.rebuild(
                exe.template_identity,
                {"options": {}, "variables": {"tone": "cold"}},
                instance.instance_identity,
            )
        with pytest.raises(InstanceIdentityMismatchError):
            registry.rebuild(exe.template_identity, instance.record, "0" * 64)

    def test_an_unretained_template_is_named_never_guessed(self) -> None:
        registry = ExecutableRegistry()
        with pytest.raises(ExecutableEvictedError, match="f" * 64):
            registry.rebuild("f" * 64, {"options": {}, "variables": {}}, "0" * 64)

    def test_a_code_backed_executable_has_no_instance_to_rebuild(self) -> None:
        registry = ExecutableRegistry()
        code = _CodeBacked("c")
        registry.register(code)  # type: ignore[arg-type]
        with pytest.raises(ExecutableError, match="no static instance"):
            registry.rebuild(code.template_identity, {}, "0" * 64)


# ---- sources ----------------------------------------------------------------------------------------


class TestSources:
    def test_each_factory_is_called_once_and_documents_and_objects_both_load(
        self,
    ) -> None:
        sources: NameRegistry[Any] = NameRegistry("executable_sources")
        calls: list[str] = []

        def mine() -> list[Any]:
            calls.append("mine")
            return [_doc("from_doc"), _CodeBacked("from_code")]

        sources.register("mine", mine)
        registry = ExecutableRegistry(sources=sources)
        report = registry.load_sources()
        assert [e.id for e in registry.list()] == ["from_code", "from_doc"]
        assert report.failed == {} and report.collisions == ()
        registry.load_sources()
        assert calls == ["mine"]  # called once
        sources.register("later", lambda: [_doc("later")])
        registry.load_sources()
        assert calls == ["mine"] and registry.get("later")

    def test_a_failing_factory_or_item_is_reported_not_raised(self) -> None:
        sources: NameRegistry[Any] = NameRegistry("executable_sources")

        def boom() -> list[Any]:
            raise RuntimeError("plugin broke")

        sources.register("boom", boom)
        sources.register("mixed", lambda: [_doc("ok"), {"kind": "fsm"}, _doc("ok")])
        registry = ExecutableRegistry(sources=sources)
        report = registry.load_sources()
        assert "boom" in report.failed and "mixed" in report.failed
        assert len(report.collisions) == 1 and "'ok'" in report.collisions[0]
        assert [e.id for e in registry.list()] == ["ok"]

    def test_the_default_sources_namespace_has_no_l6_package_behind_it(self) -> None:
        """D4: the engine registers built-ins by name only. A fresh interpreter that imports the
        executables package loads no subpackage."""
        code = (
            "import sys, fsm_llm.engine.executables\n"
            "bad = [m for m in sys.modules if m.split('.')[:2] in ("
            "['fsm_llm', 'agents'], ['fsm_llm', 'reasoning'], ['fsm_llm', 'workflows'], "
            "['fsm_llm', 'synthesis'], ['fsm_llm', 'monitor'], ['fsm_llm', 'eval'], "
            "['fsm_llm', 'testing'])]\n"
            "print(bad)\n"
        )
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(root / "src"), "PATH": os.environ.get("PATH", "")},
            check=True,
        )
        assert out.stdout.strip() == "[]"


# ---- aliases --------------------------------------------------------------------------------------------


class TestAliases:
    def _registry(self) -> tuple[ExecutableRegistry, Any, Any]:
        registry = ExecutableRegistry()
        one = registry.add_document(_doc("react", "1", response="one"))
        two = registry.add_document(_doc("react", "2", response="two"))
        return registry, one, two

    def test_the_weights_are_validated(self) -> None:
        for bad in (
            {},
            {"1": -1},
            {"1": 0},
            {"1": True},
            {"1": float("nan")},
            {"1": float("inf")},
            {"1": "2"},
            {"": 1},
        ):
            with pytest.raises(ValueError):
                make_alias("a", bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="alias name"):
            make_alias(" ", {"1": 1})
        assert make_alias("a", {"b": 1, "a": 2}).routes == (("a", 2.0), ("b", 1.0))

    def test_an_assignment_is_reproducible(self) -> None:
        registry, _, _ = self._registry()
        registry.set_alias("react", {"1": 70, "2": 30})
        first = [
            registry.resolve("react", experiment="e1", conversation_id=f"c{n}").route
            for n in range(50)
        ]
        again = [
            registry.resolve("react", experiment="e1", conversation_id=f"c{n}").route
            for n in range(50)
        ]
        assert first == again
        other = ExecutableRegistry()
        other.add_document(_doc("react", "1"))
        other.add_document(_doc("react", "2"))
        other.set_alias("react", {"2": 30, "1": 70})  # key order is irrelevant
        assert first == [
            other.resolve("react", experiment="e1", conversation_id=f"c{n}").route
            for n in range(50)
        ]

    def test_the_draw_is_pinned_by_a_golden(self) -> None:
        """A change of the hash recipe moves every assignment: this row fails on it."""
        spec = make_alias("a", {"1": 1, "2": 1})
        assert [pick_route(spec, "exp", f"c{n}") for n in range(8)] == [
            pick_route(spec, "exp", f"c{n}") for n in range(8)
        ]
        import hashlib

        for n in range(8):
            digest = hashlib.sha256(f"exp:c{n}".encode()).hexdigest()
            expected = "1" if int(digest[:16], 16) / 2**64 < 0.5 else "2"
            assert pick_route(spec, "exp", f"c{n}") == expected

    def test_the_split_is_within_tolerance_over_a_fixed_id_range(self) -> None:
        registry, _, _ = self._registry()
        registry.set_alias("react", {"1": 70, "2": 30})
        picks = [
            registry.resolve(
                "react", experiment="exp", conversation_id=f"conv-{n}"
            ).route
            for n in range(4000)
        ]
        share = picks.count("1") / len(picks)
        assert 0.66 < share < 0.74
        # a different experiment name re-draws the same conversations
        other = [
            registry.resolve(
                "react", experiment="exp2", conversation_id=f"conv-{n}"
            ).route
            for n in range(4000)
        ]
        assert other != picks

    def test_a_pinned_identity_is_served_and_never_rolls(self) -> None:
        registry, one, two = self._registry()
        registry.set_alias("react", {"1": 1, "2": 0.000001})
        resolution = registry.resolve(
            "react", pinned_template_identity=two.template_identity
        )
        assert resolution.executable is two and resolution.route == "pinned"
        assert resolution.template_identity == two.template_identity
        # no seed is needed either: a resumed conversation never rolls
        assert (
            registry.resolve(
                "react", pinned_template_identity=one.template_identity
            ).executable
            is one
        )
        with pytest.raises(ExecutableEvictedError):
            registry.resolve("react", pinned_template_identity="e" * 64)

    def test_a_route_may_name_a_template_identity(self) -> None:
        registry, one, _ = self._registry()
        assert is_template_identity(one.template_identity)
        assert not is_template_identity("1")
        registry.set_alias("pick", {one.template_identity: 1}, target="react")
        got = registry.resolve("pick")
        assert got.executable is one and got.route == one.template_identity

    def test_a_single_route_needs_no_seed_and_several_routes_do(self) -> None:
        registry, _, two = self._registry()
        registry.set_alias("only", {"2": 1}, target="react")
        assert registry.resolve("only").executable is two
        registry.set_alias("react", {"1": 1, "2": 1})
        with pytest.raises(ValueError, match="experiment name and a conversation id"):
            registry.resolve("react")
        with pytest.raises(ValueError):
            registry.resolve("react", experiment="e")

    def test_a_zero_weight_route_is_never_picked(self) -> None:
        spec = make_alias("a", {"1": 0, "2": 1, "3": 0})
        assert {pick_route(spec, "e", f"c{n}") for n in range(200)} == {"2"}

    def test_a_plain_id_resolves_to_its_greatest_version(self) -> None:
        registry, _, two = self._registry()
        got = registry.resolve("react")
        assert got.executable is two and got.route == "2"
        with pytest.raises(ExecutableNotFoundError):
            registry.resolve("ghost")

    def test_a_route_to_a_version_that_does_not_exist_is_a_named_miss(self) -> None:
        registry, _, _ = self._registry()
        registry.set_alias("react", {"9": 1})
        with pytest.raises(ExecutableNotFoundError, match="version '9'"):
            registry.resolve("react")
