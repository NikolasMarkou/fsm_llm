"""The base of a code-backed executable (plan ace7a8c8 step 23, roadmap E2.4, decision D-061).

A built-in pattern or ``reasoning.solve`` has no document: its body is Python code. The base holds
what the two families share (the header, the template identity, the input checks, the resolution
of ``requires.callables`` that raises BEFORE any model call, the cancel pre-check, the sync name
as a thin wrapper) and leaves one hook, ``execute``. Rows use a tiny subclass and no model.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.codes import Lifecycle, Reason, Status
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    ExecutableError,
    InstantiationError,
    Requires,
)
from fsm_llm.core.definitions.executable_instance import canonical_digest
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import Executable
from fsm_llm.engine.executables import code_backed as module
from fsm_llm.engine.executables.code_backed import (
    CodeBackedExecutable,
    CodeBackedInstance,
)


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


class _Probe(CodeBackedExecutable):
    """Records what ``execute`` was handed; answers ``ok``."""

    def __init__(self, **kw: Any) -> None:
        kw.setdefault("id", "probe")
        kw.setdefault("kind", "pattern")
        kw.setdefault("impl", "tests.probe.Probe")
        kw.setdefault("contract", Contract(inputs=["task"]))
        super().__init__(**kw)
        self.calls: list[tuple[dict[str, Any], dict[str, Any], RunContext]] = []

    async def execute(
        self,
        inputs: Any,
        objects: Any,
        context: RunContext,
        identity: Identity,
    ) -> RunResult:
        self.calls.append((dict(inputs), dict(objects), context))
        return RunResult(status=Status.OK, answer="done", identity=identity)


def _callables(**named: Any) -> NameRegistry[Any]:
    registry: NameRegistry[Any] = NameRegistry("callables")
    for name, value in named.items():
        registry.register(name, value)
    return registry


class TestTheHeader:
    def test_the_header_is_what_the_constructor_was_given(self) -> None:
        exe = _Probe(
            version="3",
            requires=Requires(tools="some", callables=["tools"]),
            capabilities=[Capability.TOOLS],
        )
        assert (exe.id, exe.kind, exe.version) == ("probe", "pattern", "3")
        assert exe.lifecycle is Lifecycle.BUILTIN
        assert exe.requires.callables == ["tools"]
        assert exe.capabilities == [Capability.TOOLS]
        assert isinstance(exe, Executable)

    def test_the_template_identity_is_the_documented_recipe(self) -> None:
        exe = _Probe(requires=Requires(callables=["tools"]))
        expected = canonical_digest(
            {
                "kind": "pattern",
                "id": "probe",
                "version": "1",
                "impl": "tests.probe.Probe",
                "contract": _Probe().contract.model_dump(mode="json"),
                "fsm_llm": fsm_llm.__version__,
                "requires": Requires(callables=["tools"]).model_dump(mode="json"),
            }
        )
        assert exe.template_identity == expected

    @pytest.mark.parametrize(
        "change",
        [
            {"id": "other"},
            {"version": "2"},
            {"impl": "tests.probe.Other"},
            {"contract": Contract(inputs=["more"])},
            {"requires": Requires(callables=["hitl"])},
        ],
    )
    def test_each_claimed_field_moves_the_template_identity(
        self, change: dict[str, Any]
    ) -> None:
        assert _Probe(**change).template_identity != _Probe().template_identity

    def test_the_template_identity_is_stable_across_instances(self) -> None:
        assert _Probe().template_identity == _Probe().template_identity


class TestInstantiate:
    def test_an_instance_has_no_static_identity_and_an_empty_record(self) -> None:
        instance = _Probe().instantiate()
        assert isinstance(instance, CodeBackedInstance)
        assert instance.instance_identity is None
        assert dict(instance.record) == {"options": {}, "variables": {}}

    @pytest.mark.parametrize("kw", [{"options": {"a": 1}}, {"variables": {"v": "x"}}])
    def test_options_and_variables_are_refused(self, kw: dict[str, Any]) -> None:
        with pytest.raises(InstantiationError):
            _Probe().instantiate(**kw)

    def test_the_executable_keeps_no_reference_to_an_instance(self) -> None:
        exe = _Probe()
        before = set(vars(exe))
        exe.instantiate()
        assert set(vars(exe)) == before


class TestRun:
    def test_run_is_the_sync_name_of_arun(self) -> None:
        exe = _Probe()
        result = exe.run({"task": "t"})
        assert (result.status, result.answer) == (Status.OK, "done")
        assert result.identity.template_identity == exe.template_identity
        assert result.identity.instance_identity is None
        assert exe.calls[0][0] == {"task": "t"}

    def test_the_sync_names_are_thin_wrappers(self) -> None:
        for owner in (CodeBackedExecutable, CodeBackedInstance):
            source = textwrap.dedent(inspect.getsource(owner.run))
            body = ast.parse(source).body[0]
            assert isinstance(body, ast.FunctionDef)
            calls = {
                n.func.id
                for n in ast.walk(body)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert "run_sync" in calls

    def test_an_undeclared_input_is_a_value_error_before_execute(self) -> None:
        exe = _Probe()
        with pytest.raises(ValueError, match="declares no input"):
            exe.run({"task": "t", "extra": 1})
        assert exe.calls == []

    def test_an_internal_prefixed_input_is_refused(self) -> None:
        exe = _Probe(contract=Contract(inputs=["task", "_grant"]))
        with pytest.raises(ValueError, match="reserved prefix"):
            exe.run({"task": "t", "_grant": True})
        assert exe.calls == []

    def test_a_missing_declared_input_is_a_failed_result(self) -> None:
        exe = _Probe()
        result = exe.run({})
        assert (result.status, result.reason) == (Status.FAILED, Reason.INPUTS_MISSING)
        assert result.detail["missing"] == ["task"]
        assert exe.calls == []

    def test_an_optional_input_is_accepted_and_never_required(self) -> None:
        exe = _Probe(optional_inputs=("note",))
        assert exe.run({"task": "t"}).success
        assert exe.run({"task": "t", "note": "n"}).success
        assert exe.calls[1][0] == {"task": "t", "note": "n"}

    def test_the_cancel_token_is_checked_before_execute(self) -> None:
        exe = _Probe()
        context = RunContext()
        context.cancel.cancel()
        result = exe.run({"task": "t"}, context=context)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert result.identity.template_identity == exe.template_identity
        assert exe.calls == []

    def test_a_missing_context_is_a_bare_context(self) -> None:
        exe = _Probe()
        exe.run({"task": "t"})
        assert isinstance(exe.calls[0][2], RunContext)


class TestTheCallables:
    def test_required_names_are_resolved_into_objects(self) -> None:
        exe = _Probe(requires=Requires(callables=["tools", "hitl"]))
        context = RunContext(callables=_callables(tools="T", hitl="H", unrelated="U"))
        exe.run({"task": "t"}, context=context)
        assert exe.calls[0][1] == {"tools": "T", "hitl": "H"}

    def test_an_unresolved_name_raises_before_execute_and_names_it(self) -> None:
        exe = _Probe(requires=Requires(callables=["tools", "hitl"]))
        context = RunContext(callables=_callables(tools="T"))
        with pytest.raises(ExecutableError, match="hitl"):
            exe.run({"task": "t"}, context=context)
        assert exe.calls == []

    def test_no_callables_registry_at_all_raises_the_same_error(self) -> None:
        exe = _Probe(requires=Requires(callables=["tools"]))
        with pytest.raises(ExecutableError, match="tools"):
            exe.run({"task": "t"})
        assert exe.calls == []

    def test_an_optional_name_is_read_when_present_and_skipped_when_absent(
        self,
    ) -> None:
        exe = _Probe(optional_callables=("hitl", "config"))
        exe.run({"task": "t"}, context=RunContext(callables=_callables(hitl="H")))
        exe.run({"task": "t"})
        assert exe.calls[0][1] == {"hitl": "H"}
        assert exe.calls[1][1] == {}

    def test_a_required_name_wins_over_nothing_and_is_not_an_optional_twice(
        self,
    ) -> None:
        exe = _Probe(
            requires=Requires(callables=["tools"]), optional_callables=("tools",)
        )
        exe.run({"task": "t"}, context=RunContext(callables=_callables(tools="T")))
        assert exe.calls[0][1] == {"tools": "T"}

    def test_launchable_is_false_exactly_when_a_callable_is_required(self) -> None:
        from fsm_llm.engine.executables import ExecutableRegistry

        registry = ExecutableRegistry()
        registry.register(_Probe(id="free"))
        registry.register(_Probe(id="needs", requires=Requires(callables=["tools"])))
        assert [e.id for e in registry.list(launchable=True)] == ["free"]
        assert [e.id for e in registry.list(launchable=False)] == ["needs"]


class TestTheModule:
    def test_it_imports_no_package_above_the_engine(self) -> None:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = [
            n.module or ""
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and not n.level
        ]
        for name in imported:
            assert not name.startswith(
                (
                    "fsm_llm.agents",
                    "fsm_llm.reasoning",
                    "fsm_llm.workflows",
                    "fsm_llm.monitor",
                    "fsm_llm.synthesis",
                    "fsm_llm.eval",
                )
            ), name

    def test_the_package_exports_it(self) -> None:
        from fsm_llm.engine import executables

        assert executables.CodeBackedExecutable is CodeBackedExecutable
        assert "CodeBackedExecutable" in executables.__all__
        assert "CodeBackedInstance" in executables.__all__
