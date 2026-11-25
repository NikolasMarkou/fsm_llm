"""The name registry of core (plan ace7a8c8, roadmap 0.17, contract C10) and its namespaces.

Three parts: the primitive (``NameRegistry``: register, resolve, collisions, freeze, the leaf
lock, lazy entry-point loading on the first miss), the list of every namespace with its sources,
and the guard that no NEW private name table appears (a module-level dict that a ``register*``
function mutates) beside the five that remain and the row of the roadmap that retires each.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.core import names
from fsm_llm.core.definitions import FSMError
from fsm_llm.core.names import (
    NameCollisionError,
    NameNotFoundError,
    NameRegistry,
    NameRegistryError,
    RegistryFrozenError,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"


class _Thing:
    def __init__(self, label: str = "") -> None:
        self.label = label


class _EntryPoint:
    """A fake ``importlib.metadata.EntryPoint``: a name, a ``load()`` that records and returns."""

    def __init__(self, name: str, obj: Any = None, error: Exception | None = None):
        self.name = name
        self._obj = obj if obj is not None else _Thing(name)
        self._error = error
        self.loads = 0
        self.on_load = lambda: None

    def load(self) -> Any:
        self.loads += 1
        self.on_load()
        if self._error is not None:
            raise self._error
        return self._obj


class _SpyLock:
    """A lock that answers ``locked()``: the leaf-lock test asks it from inside a load."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> _SpyLock:
        self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class _Recorder:
    def __init__(self, registry: NameRegistry[Any] | None = None) -> None:
        self.registry = registry
        self.warnings: list[str] = []
        self.locked_during_warning: list[bool] = []

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(str(message))
        if self.registry is not None:
            self.locked_during_warning.append(self.registry._lock.locked())

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


@pytest.fixture
def entry_points(monkeypatch):
    """Install a fake ``importlib.metadata.entry_points``; returns (set_entries, calls)."""
    state: dict[str, Any] = {"entries": [], "calls": [], "error": None}

    def fake(*, group: str | None = None, **kwargs: Any) -> list[_EntryPoint]:
        state["calls"].append(group)
        if state["error"] is not None:
            raise state["error"]
        return list(state["entries"])

    monkeypatch.setattr(importlib.metadata, "entry_points", fake)
    return state


class TestRegisterAndResolve:
    def test_register_then_resolve_returns_the_object(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        thing = _Thing()
        registry.register("a", thing)
        assert registry.resolve("a") is thing
        assert registry.get("a") is thing
        assert "a" in registry
        assert registry.namespace == "demo"

    def test_a_miss(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        assert "nope" not in registry
        assert registry.get("nope") is None
        sentinel = _Thing()
        assert registry.get("nope", sentinel) is sentinel
        with pytest.raises(NameNotFoundError) as caught:
            registry.resolve("nope")
        assert caught.value.name == "nope"
        assert caught.value.namespace == "demo"
        assert "nope" in str(caught.value) and "demo" in str(caught.value)

    def test_list_is_sorted_and_a_copy(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        for name in ("zeta", "alpha", "mid"):
            registry.register(name, _Thing(name))
        listed = registry.list()
        assert listed == ["alpha", "mid", "zeta"]
        listed.append("x")
        assert registry.list() == ["alpha", "mid", "zeta"]

    def test_items_are_in_registration_order_and_a_snapshot(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        things = {name: _Thing(name) for name in ("zeta", "alpha", "mid")}
        for name, thing in things.items():
            registry.register(name, thing)
        replacement = _Thing("again")
        registry.register("zeta", replacement, override=True)
        listed = registry.items()
        assert [name for name, _ in listed] == ["zeta", "alpha", "mid"]
        assert listed[0][1] is replacement
        listed.append(("x", _Thing()))
        assert len(registry.items()) == 3

    def test_the_error_classes_share_one_root_under_fsmerror(self):
        for error in (NameCollisionError, NameNotFoundError, RegistryFrozenError):
            assert issubclass(error, NameRegistryError)
        assert issubclass(NameRegistryError, FSMError)

    def test_registries_are_independent_instances(self):
        first: NameRegistry[_Thing] = NameRegistry("one")
        second: NameRegistry[_Thing] = NameRegistry("one")
        first.register("a", _Thing())
        assert "a" not in second

    @pytest.mark.parametrize("bad", ["", "   ", "\t"])
    def test_a_blank_name_is_refused(self, bad):
        with pytest.raises(ValueError):
            NameRegistry("demo").register(bad, _Thing())

    @pytest.mark.parametrize("bad", [None, 3, b"a"])
    def test_a_non_string_name_is_refused(self, bad):
        with pytest.raises(TypeError):
            NameRegistry("demo").register(bad, _Thing())  # type: ignore[arg-type]


class TestCollisions:
    def test_a_taken_name_is_refused(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        first = _Thing()
        registry.register("a", first)
        with pytest.raises(NameCollisionError) as caught:
            registry.register("a", _Thing())
        assert caught.value.name == "a" and caught.value.namespace == "demo"
        assert registry.resolve("a") is first

    def test_override_replaces_and_frees_the_old_identity(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        old, new = _Thing("old"), _Thing("new")
        registry.register("a", old)
        registry.register("a", new, override=True)
        assert registry.resolve("a") is new
        assert registry.name_of(old) is None
        assert registry.name_of(new) == "a"
        registry.register("b", old)  # the freed object may be registered again
        assert registry.name_of(old) == "b"

    def test_overriding_a_name_with_its_own_object_is_a_no_op(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        thing = _Thing()
        registry.register("a", thing)
        registry.register("a", thing, override=True)
        assert registry.resolve("a") is thing and registry.name_of(thing) == "a"

    @pytest.mark.parametrize("override", [False, True])
    def test_one_object_under_two_names_is_refused(self, override):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        thing = _Thing()
        registry.register("a", thing)
        with pytest.raises(NameCollisionError):
            registry.register("b", thing, override=override)
        assert registry.list() == ["a"]

    def test_a_refused_override_changes_nothing(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        one, two = _Thing(), _Thing()
        registry.register("a", one)
        registry.register("b", two)
        with pytest.raises(NameCollisionError):
            registry.register("a", two, override=True)
        assert registry.resolve("a") is one and registry.resolve("b") is two
        assert registry.name_of(one) == "a" and registry.name_of(two) == "b"


class TestNameOf:
    def test_by_identity_not_equality(self):
        registry: NameRegistry[list[int]] = NameRegistry("demo")
        stored = [1, 2]
        registry.register("a", stored)
        assert registry.name_of(stored) == "a"
        assert registry.name_of([1, 2]) is None

    def test_unregistered_is_none(self):
        assert NameRegistry("demo").name_of(_Thing()) is None


class TestFreeze:
    def test_register_raises_after_freeze_and_reads_work(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        thing = _Thing()
        registry.register("a", thing)
        assert registry.frozen is False
        registry.freeze()
        assert registry.frozen is True
        with pytest.raises(RegistryFrozenError):
            registry.register("b", _Thing())
        with pytest.raises(RegistryFrozenError):
            registry.register("a", _Thing(), override=True)
        assert registry.resolve("a") is thing
        assert registry.list() == ["a"]
        registry.freeze()  # idempotent
        assert registry.frozen is True


class TestEntryPoints:
    def test_no_group_reads_no_entry_points(self, entry_points):
        registry: NameRegistry[_Thing] = NameRegistry("demo")
        assert "nope" not in registry
        assert registry.get("nope") is None
        assert entry_points["calls"] == []

    def test_two_entries_load_on_the_first_miss_only(self, entry_points):
        one, two = _EntryPoint("one"), _EntryPoint("two")
        entry_points["entries"] = [one, two]
        registry: NameRegistry[Any] = NameRegistry(
            "demo", entry_point_group="fsm_llm.demo"
        )
        registry.register("built_in", _Thing())
        assert registry.resolve("built_in") is not None  # a hit loads nothing
        assert registry.list() == ["built_in"]  # a read loads nothing
        assert entry_points["calls"] == [] and (one.loads, two.loads) == (0, 0)
        assert "one" in registry  # the first miss loads, and finds the name
        assert entry_points["calls"] == ["fsm_llm.demo"]
        assert (one.loads, two.loads) == (1, 1)
        assert registry.resolve("two") is two._obj
        assert registry.list() == ["built_in", "one", "two"]
        assert registry.name_of(one._obj) == "one"
        assert "missing" not in registry  # a second miss never reloads
        assert registry.get("missing") is None
        assert entry_points["calls"] == ["fsm_llm.demo"]
        assert (one.loads, two.loads) == (1, 1)

    def test_resolve_of_a_plugin_name_loads_it(self, entry_points):
        entry = _EntryPoint("plugin")
        entry_points["entries"] = [entry]
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        assert registry.resolve("plugin") is entry._obj

    def test_a_failing_entry_is_warned_once_and_skipped(
        self, entry_points, monkeypatch
    ):
        good = _EntryPoint("good")
        bad = _EntryPoint("bad", error=ImportError("boom"))
        entry_points["entries"] = [bad, good]
        recorder = _Recorder()
        monkeypatch.setattr(names, "logger", recorder)
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        assert "good" in registry
        assert "bad" not in registry
        assert "other" not in registry
        assert bad.loads == 1 and good.loads == 1
        assert len(recorder.warnings) == 1
        assert "bad" in recorder.warnings[0] and "boom" in recorder.warnings[0]

    def test_an_existing_name_wins_over_an_entry_point(self, entry_points, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(names, "logger", recorder)
        shadow = _EntryPoint("taken")
        entry_points["entries"] = [shadow]
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        built_in = _Thing()
        registry.register("taken", built_in)
        assert "other" not in registry
        assert registry.resolve("taken") is built_in
        assert len(recorder.warnings) == 1 and "taken" in recorder.warnings[0]

    def test_entry_points_that_cannot_be_listed_warn_once(
        self, entry_points, monkeypatch
    ):
        recorder = _Recorder()
        monkeypatch.setattr(names, "logger", recorder)
        entry_points["error"] = RuntimeError("no metadata")
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        assert "x" not in registry
        assert "y" not in registry
        assert len(recorder.warnings) == 1
        assert entry_points["calls"] == ["g"]

    def test_explicit_load_is_idempotent(self, entry_points):
        entry = _EntryPoint("plugin")
        entry_points["entries"] = [entry]
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        registry.load_entry_points()
        registry.load_entry_points()
        assert registry.list() == ["plugin"]
        assert entry.loads == 1 and entry_points["calls"] == ["g"]

    def test_freeze_loads_the_pending_entries_then_seals(self, entry_points):
        entry = _EntryPoint("plugin")
        entry_points["entries"] = [entry]
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        registry.freeze()
        assert registry.list() == ["plugin"]
        with pytest.raises(RegistryFrozenError):
            registry.register("late", _Thing())
        assert "late" not in registry and entry_points["calls"] == ["g"]

    def test_no_loader_and_no_logger_call_runs_under_the_lock(
        self, entry_points, monkeypatch
    ):
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        registry._lock = _SpyLock()  # type: ignore[assignment]
        recorder = _Recorder(registry)
        monkeypatch.setattr(names, "logger", recorder)
        seen: list[bool] = []
        good = _EntryPoint("good")
        bad = _EntryPoint("bad", error=ImportError("boom"))
        for entry in (good, bad):
            entry.on_load = lambda: seen.append(registry._lock.locked())
        entry_points["entries"] = [good, bad]
        assert "good" in registry
        assert seen == [False, False]
        assert recorder.locked_during_warning == [False]

    def test_a_loader_that_reads_the_registry_does_not_deadlock(self, entry_points):
        entry = _EntryPoint("plugin")
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        probes: list[bool] = []
        entry.on_load = lambda: probes.append("anything" in registry)
        entry_points["entries"] = [entry]
        worker = threading.Thread(target=lambda: "first" in registry, daemon=True)
        worker.start()
        worker.join(10)
        assert not worker.is_alive()
        assert probes == [False] and entry.loads == 1

    def test_a_second_thread_waits_for_the_loader_instead_of_missing(
        self, entry_points
    ):
        started, release = threading.Event(), threading.Event()
        entry = _EntryPoint("plugin")

        def slow() -> None:
            started.set()
            release.wait(10)

        entry.on_load = slow
        entry_points["entries"] = [entry]
        registry: NameRegistry[Any] = NameRegistry("demo", entry_point_group="g")
        found: list[bool] = []
        first = threading.Thread(target=lambda: registry.get("plugin"), daemon=True)
        first.start()
        assert started.wait(10)
        second = threading.Thread(
            target=lambda: found.append("plugin" in registry), daemon=True
        )
        second.start()
        release.set()
        first.join(10)
        second.join(10)
        assert found == [True] and entry.loads == 1


class TestThreads:
    def test_concurrent_registrations_all_land(self):
        registry: NameRegistry[_Thing] = NameRegistry("demo")

        def work(index: int) -> None:
            for step in range(50):
                registry.register(f"n{index}-{step}", _Thing())
                registry.list()

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert len(registry.list()) == 400


class TestSource:
    _TREE = ast.parse(Path(names.__file__).read_text(encoding="utf-8"))

    def test_importlib_is_imported_inside_a_function_only(self):
        for node in self._TREE.body:
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("importlib") for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("importlib")

    def test_imports_stay_in_core_and_the_kernel(self):
        for node in ast.walk(self._TREE):
            if isinstance(node, ast.ImportFrom) and not node.level:
                module = node.module or ""
                assert not module.startswith("fsm_llm") or module.startswith(
                    ("fsm_llm.core", "fsm_llm.logging")
                ), module

    def test_all_is_static(self):
        assert sorted(names.__all__) == sorted(
            [
                "NameCollisionError",
                "NameNotFoundError",
                "NameRegistry",
                "NameRegistryError",
                "RegistryFrozenError",
            ]
        )


# --- the namespaces and the private tables --------------------------------------------------

#: namespace -> (module path under src/fsm_llm, attribute, its sources). One row per module-level
#: ``NameRegistry``; a later step adds the row of its namespace.
NAMESPACES: dict[str, tuple[str, str, str]] = {
    "state_kinds": (
        "engine/kinds/registry.py",
        "_EXECUTORS",
        "the conversation and completion executors registered at import; "
        "register_executor (a runtime kind); a per-API executors= map sits outside it",
    ),
    "executable_kinds": (
        "engine/executables/namespaces.py",
        "EXECUTABLE_KINDS",
        "the fsm, strategy and procedure adapter factories registered at import; "
        "entry-point group fsm_llm.executable_kinds (E2.3 wires pyproject later)",
    ),
    "executable_sources": (
        "engine/executables/namespaces.py",
        "EXECUTABLE_SOURCES",
        "named source factories a package registers by an explicit call (fsm_llm.agents "
        "and fsm_llm.reasoning register theirs when imported, step 23); "
        "entry-point group fsm_llm.executable_sources",
    ),
    "handlers": (
        "engine/executables/namespaces.py",
        "HANDLERS",
        "handler FACTORIES a pattern file binds by name; fsm_llm.agents.handlers.library registers "
        "the agents.* names at import; entry-point group fsm_llm.handlers",
    ),
    "generators": (
        "engine/executables/namespaces.py",
        "GENERATORS",
        "generator functions a pattern file names instead of an FSM (empty until the prompt_chain "
        "step); entry-point group fsm_llm.generators",
    ),
    "pattern_agents": (
        "engine/executables/namespaces.py",
        "PATTERN_AGENTS",
        "the agent class create_agent builds for each built-in pattern id, filled by fsm_llm.agents "
        "at import (plan a555dcfe D-030); no entry-point group",
    ),
}

#: The private name tables that remain, each with the roadmap row that retires it (docs/2026_roadmap.md).
#: ``_EXECUTORS``, the ninth, is on the registry above. A NEW table is refused: a module-level dict
#: that a ``register*`` function mutates is not in this table, so the scan below fails.
KNOWN_PRIVATE_TABLES: dict[str, tuple[str, str]] = {
    "_WORKFLOW_PRESETS": ("monitor/instance_manager/workflow_presets.py", "E5.1"),
    "_ARTIFACT_SCHEMAS": ("agents/meta_prompts.py", "E4.6"),
}

_MUTATORS = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__"}


def _module_level_names(tree: ast.Module) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node
    return found


def _is_dict_value(value: ast.AST | None) -> bool:
    if isinstance(value, ast.Dict):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return called in {"dict", "defaultdict", "OrderedDict"}
    return False


def registering_dict_tables(source: str) -> set[str]:
    """Module-level dicts that a function named ``register*`` mutates.

    The obvious shape only: ``TABLE[key] = value``, ``del TABLE[key]``,
    ``TABLE.update(...)`` and the other mutating methods inside a ``def register...``.
    A table filled some other way is a review matter; the note in the module docstring says so.
    """
    tree = ast.parse(source)
    tables = {
        name
        for name, node in _module_level_names(tree).items()
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _is_dict_value(node.value)
    }
    hit: set[str] = set()
    for func in ast.walk(tree):
        if not (
            isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
            and func.name.startswith("register")
        ):
            continue
        for node in ast.walk(func):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif isinstance(node, ast.Delete):
                targets = list(node.targets)
            for target in targets:
                if isinstance(target, ast.Subscript) and isinstance(
                    target.value, ast.Name
                ):
                    hit.add(target.value.id)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _MUTATORS
                and isinstance(node.func.value, ast.Name)
            ):
                hit.add(node.func.value.id)
    return hit & tables


def _module_level_registries(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for name, node in _module_level_names(tree).items():
        value = getattr(node, "value", None)
        if isinstance(value, ast.Call):
            func = value.func
            called = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            )
            if called == "NameRegistry":
                found.add(name)
    return found


class TestNamespaces:
    @pytest.mark.parametrize("namespace", sorted(NAMESPACES))
    def test_each_listed_namespace_is_a_registry_of_that_name(self, namespace):
        path, attribute, sources = NAMESPACES[namespace]
        module = importlib.import_module("fsm_llm." + path[:-3].replace("/", "."))
        registry = getattr(module, attribute)
        assert isinstance(registry, NameRegistry)
        assert registry.namespace == namespace
        assert sources

    def test_every_module_level_registry_in_src_is_listed(self):
        found: dict[str, str] = {}
        for path in sorted(_SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for attribute in _module_level_registries(tree):
                found[f"{path.relative_to(_SRC).as_posix()}:{attribute}"] = attribute
        listed = {f"{path}:{attr}" for path, attr, _ in NAMESPACES.values()}
        assert set(found) == listed, (
            "a NameRegistry at module level is missing from NAMESPACES (or a row is "
            f"stale): {sorted(set(found) ^ listed)}"
        )

    def test_the_executable_namespaces_have_their_entry_point_groups(self):
        from fsm_llm.engine.executables import namespaces

        assert namespaces.EXECUTABLE_KINDS._group == "fsm_llm.executable_kinds"
        assert namespaces.EXECUTABLE_SOURCES._group == "fsm_llm.executable_sources"
        assert namespaces.HANDLERS._group == "fsm_llm.handlers"
        assert namespaces.GENERATORS._group == "fsm_llm.generators"
        # in-process the list also holds what an imported package registered (the agents package
        # adds `pattern`, plan a555dcfe step 9), so the engine-only list is asked of a fresh interpreter
        assert {"fsm", "procedure", "strategy"} <= set(
            namespaces.EXECUTABLE_KINDS.list()
        )

    def test_the_engine_alone_registers_three_kinds_and_agents_adds_pattern(self):
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(_SRC.parent)!r})\n"
            "from fsm_llm.engine.executables import namespaces\n"
            "assert namespaces.EXECUTABLE_KINDS.list() == ['fsm', 'procedure', 'strategy']\n"
            "import fsm_llm.agents\n"
            "assert namespaces.EXECUTABLE_KINDS.list() == ['fsm', 'pattern', 'procedure', 'strategy']\n"
            "assert 'fsm_llm.agents.executables' not in sys.modules\n"
        )
        done = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr

    def test_importing_the_engine_namespace_registers_no_source(self):
        # In-process the list depends on which packages other tests imported (agents and
        # reasoning register their source when imported), so ask a fresh interpreter.
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(_SRC.parent)!r})\n"
            "from fsm_llm.engine.executables import namespaces\n"
            "assert namespaces.EXECUTABLE_SOURCES.list() == []\n"
            "assert not any(m in sys.modules for m in ('fsm_llm.agents', 'fsm_llm.reasoning'))\n"
        )
        done = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr

    def test_the_handler_library_registers_when_the_agents_package_is_imported(self):
        # D-013 of plan a555dcfe: the engine namespace is empty until a package registers its
        # library; the agents library registers the agents.* names at import, and nothing else
        # (one generator since step 11, D-019)
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(_SRC.parent)!r})\n"
            "from fsm_llm.engine.executables import namespaces\n"
            "assert namespaces.HANDLERS.list() == []\n"
            "assert namespaces.GENERATORS.list() == []\n"
            "import fsm_llm.agents\n"
            "names = namespaces.HANDLERS.list()\n"
            "assert names and all(n.startswith('agents.') for n in names), names\n"
            "assert namespaces.GENERATORS.list() == ['agents.prompt_chain']\n"
        )
        done = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr

    def test_the_state_kind_registry_holds_the_two_builtin_executors(self):
        from fsm_llm.engine.kinds import registry as kinds

        assert {"conversation", "completion"} <= set(kinds._EXECUTORS.list())
        assert kinds._EXECUTORS.name_of(kinds._CONVERSATION) == "conversation"


class TestNoNewPrivateTable:
    def test_the_scan_finds_a_registering_function_that_fills_a_dict(self):
        source = (
            "_T: dict[str, int] = {}\n"
            "_U = {}\n"
            "_V = {}\n"
            "def register_it(name, value):\n    _T[name] = value\n    _U.update({name: value})\n"
            "def other(name):\n    _V[name] = 1\n"
        )
        assert registering_dict_tables(source) == {"_T", "_U"}

    def test_the_scan_ignores_a_name_registry_and_local_dicts(self):
        source = (
            "_R = NameRegistry('x')\n"
            "def register_it(name, obj):\n    local = {}\n    local[name] = obj\n    _R.register(name, obj)\n"
        )
        assert registering_dict_tables(source) == set()

    def test_no_module_of_src_fills_a_private_table_in_a_register_function(self):
        found = {
            f"{path.relative_to(_SRC).as_posix()}:{table}"
            for path in sorted(_SRC.rglob("*.py"))
            for table in registering_dict_tables(path.read_text(encoding="utf-8"))
        }
        assert found == set(), (
            "a new private name table: register it through a NameRegistry "
            f"(core/names.py) or add its retiring roadmap row to KNOWN_PRIVATE_TABLES: {found}"
        )

    @pytest.mark.parametrize("symbol", sorted(KNOWN_PRIVATE_TABLES))
    def test_each_known_table_still_exists_and_names_its_retiring_row(self, symbol):
        path, row = KNOWN_PRIVATE_TABLES[symbol]
        tree = ast.parse((_SRC / path).read_text(encoding="utf-8"))
        assert symbol in _module_level_names(tree), (
            f"{symbol} is gone from {path}: drop its row from KNOWN_PRIVATE_TABLES"
        )
        assert row.startswith("E"), row
        roadmap = (_SRC.parents[1] / "docs" / "2026_roadmap.md").read_text(
            encoding="utf-8"
        )
        assert f"| {row} |" in roadmap

    def test_the_adopted_executor_table_is_not_a_known_private_table(self):
        assert "_EXECUTORS" not in KNOWN_PRIVATE_TABLES
