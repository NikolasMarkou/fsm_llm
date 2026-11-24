"""Delegated resolution (plan baa37f40, step K7.1, D-049, roadmap E3.2 second half).

A code-backed ``pattern`` has no FSM body to push, so it resolves to a
``DelegatedTarget`` (not a ``ResolvedTarget``) through ``ExecutableRegistry.resolve_delegated``,
and only for a caller that holds a ``RunLauncher``. Every refusal is the one answer ``None`` that a
caller maps to ``failed/unknown_target``. Frame resolution (``resolve_target``) is unchanged.
Nothing here starts a thread.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm.agents.executables import PatternExecutable
from fsm_llm.core.codes import Lifecycle, Reason, Status
from fsm_llm.core.definitions.executable import Capability, Contract, Requires
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import CancelToken, RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable
from fsm_llm.engine.kinds import targets as targets_module
from fsm_llm.engine.kinds.targets import (
    TARGET_MODE_DELEGATED,
    TARGET_MODE_FRAME,
    DelegatedTarget,
    ResolvedTarget,
    RunLauncher,
    TargetResolver,
)
from tests.test_fsm_llm.test_executable_adapters import _doc


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


class _Stub(CodeBackedExecutable):
    """A code-backed executable of a chosen kind, contract and requirements."""

    def __init__(
        self,
        kind: str = "pattern",
        *,
        id: str = "probe.run",
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        callables: list[str] | None = None,
        capabilities: list[Capability] | None = None,
        optional_inputs: tuple[str, ...] = (),
        refused: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            id=id,
            kind=kind,
            impl="tests.probe.Stub",
            contract=Contract(inputs=inputs or ["task"], outputs=outputs or []),
            requires=Requires(callables=callables or []),
            capabilities=capabilities or [],
            optional_inputs=optional_inputs,
        )
        self.refused_budgets = refused

    async def execute(self, inputs, objects, context, identity) -> RunResult:
        return RunResult(status=Status.OK, answer="done", identity=identity)


class _Launcher:
    """The six methods of a launcher, doing nothing: a second implementer for the Protocol."""

    def resolve_run(
        self, target_id, *, input_keys=(), max_steps=None, conversation_id=None
    ):
        return None

    def claim(self, handle, *, conversation_id):
        return True

    def start(
        self,
        handle,
        target,
        inputs,
        *,
        conversation_id,
        max_seconds=None,
        max_steps=None,
        announce=None,
    ):
        return False

    def poll(self, handle, *, conversation_id):
        return None

    def cancel(self, handle, *, conversation_id=None):
        return False

    def cancel_for(self, conversation_id):
        return 0


def _resolve(registry, target="probe.run", *, launcher=None, context=None, **kw):
    return registry.resolve_delegated(
        target,
        launcher=launcher if launcher is not None else _Launcher(),
        context=context if context is not None else RunContext(),
        **kw,
    )


def _registry(*executables) -> ExecutableRegistry:
    registry = ExecutableRegistry()
    for executable in executables:
        registry.register(executable)
    return registry


# ---- the reason, the token, the seam ---------------------------------------------------------


class TestRunLost:
    def test_run_lost_is_the_nineteenth_reason_and_outside_the_pinned_tuple(
        self,
    ) -> None:
        values = [m.value for m in Reason]
        assert len(values) == 19
        assert values[18] == "run_lost"
        assert Reason.RUN_LOST.value == "run_lost"
        assert Reason.from_string("run_lost") is Reason.RUN_LOST
        from fsm_llm.synthesis.constants import REASONS

        assert "run_lost" not in REASONS


class TestDeriveCancel:
    def test_a_given_token_is_the_childs_own_and_the_parents_is_untouched(self) -> None:
        parent = RunContext(max_steps=10, max_seconds=60)
        own = CancelToken()
        child = parent.derive(parent="conv-1", cancel=own)
        assert child.cancel is own
        assert child.cancel is not parent.cancel
        own.cancel()
        assert child.cancel.cancelled and not parent.cancel.cancelled
        parent.cancel.cancel()
        assert parent.cancel.cancelled

    def test_no_token_keeps_the_parents_as_before(self) -> None:
        parent = RunContext()
        assert parent.derive(parent="p").cancel is parent.cancel
        assert parent.derive(parent="p", cancel=None).cancel is parent.cancel

    def test_narrowing_is_unchanged_beside_a_token(self) -> None:
        parent = RunContext(max_steps=10, max_seconds=60)
        child = parent.derive(
            parent="p", max_steps=4, max_seconds=5.5, cancel=CancelToken()
        )
        assert (child.max_steps, child.max_seconds) == (4, 5.5)
        with pytest.raises(ValueError, match="max_seconds"):
            parent.derive(parent="p", max_seconds=61, cancel=CancelToken())

    def test_a_token_of_the_wrong_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="cancel"):
            RunContext().derive(parent="p", cancel=object())  # type: ignore[arg-type]

    def test_a_grandchild_may_take_another_token_and_no_field_was_added(self) -> None:
        root = RunContext()
        a, b = CancelToken(), CancelToken()
        child = root.derive(parent="a", cancel=a)
        assert child.derive(parent="b", cancel=b).cancel is b
        assert child.derive(parent="b").cancel is a
        assert {f.name for f in dataclasses.fields(RunContext)} == {
            "llm_interface",
            "llm_factory",
            "event_log",
            "max_steps",
            "max_seconds",
            "cancel",
            "callables",
            "parent",
            "registry",
        }


class TestTheSeam:
    def test_frame_resolution_is_unchanged(self) -> None:
        assert TARGET_MODE_FRAME == "frame"
        assert TARGET_MODE_DELEGATED == "delegated"
        fields = [f.name for f in dataclasses.fields(ResolvedTarget)]
        assert fields == [
            "definition",
            "template_identity",
            "instance_identity",
            "mode",
            "returns",
        ]
        assert list(inspect.signature(TargetResolver.resolve_target).parameters) == [
            "self",
            "target_id",
            "conversation_id",
        ]

    def test_a_frame_target_still_refuses_the_delegated_mode(self) -> None:
        registry = ExecutableRegistry()
        registry.add_document(_doc("fsm"))
        target = registry.resolve_target("demo")
        assert target is not None
        with pytest.raises(ValueError, match="mode"):
            dataclasses.replace(target, mode=TARGET_MODE_DELEGATED)

    def test_the_registry_frame_resolver_still_answers_nothing_for_code_backed(
        self,
    ) -> None:
        registry = _registry(_Stub("pattern"))
        assert registry.resolve_target("probe.run") is None
        registry = _registry(_Stub("workflow"))
        assert registry.resolve_target("probe.run") is None

    def test_the_delegated_value_is_frozen_and_has_the_one_mode(self) -> None:
        stub = _Stub(outputs=["a"])
        target = DelegatedTarget("probe.run", "pattern", stub, "t", None, ("a",))
        assert target.mode == TARGET_MODE_DELEGATED
        assert target.outputs == ("a",)
        with pytest.raises(dataclasses.FrozenInstanceError):
            target.kind = "workflow"  # type: ignore[misc]
        with pytest.raises(TypeError):  # the mode is not a parameter: there is one
            DelegatedTarget(  # type: ignore[call-arg]
                "probe.run", "pattern", stub, "t", None, ("a",), mode="frame"
            )

    def test_the_launcher_protocol_has_six_methods_and_is_runtime_checkable(
        self,
    ) -> None:
        assert isinstance(_Launcher(), RunLauncher)

        class _NoPoll(_Launcher):
            poll = None  # type: ignore[assignment]

        assert not isinstance(_NoPoll(), RunLauncher)
        assert not isinstance(object(), RunLauncher)
        assert not isinstance(ExecutableRegistry(), RunLauncher)
        assert {
            "resolve_run",
            "claim",
            "start",
            "poll",
            "cancel",
            "cancel_for",
        } <= set(dir(RunLauncher))

    def test_the_seam_module_imports_no_executables_and_no_run_context(self) -> None:
        text = Path(targets_module.__file__).read_text(encoding="utf-8")
        assert "engine.executables" not in text.replace("``engine.executables``", "")
        assert "from fsm_llm.engine.context" not in text


# ---- the registry resolves a delegated target -------------------------------------------------


class TestResolvesDelegated:
    def test_a_workflow_kind_resolves_to_nothing(self) -> None:
        """D-053, D-055: delegated mode is pattern targets only (E5.2 and E5.3 carry workflows)."""
        assert _resolve(_registry(_Stub("workflow"))) is None

    @pytest.mark.parametrize("kind", ["pattern"])
    def test_a_code_backed_kind_resolves_with_its_identities(self, kind) -> None:
        stub = _Stub(kind, outputs=["x", "y"])
        registry = _registry(stub)
        target = _resolve(registry)
        assert isinstance(target, DelegatedTarget)
        assert target.mode == "delegated"
        assert (target.target_id, target.kind) == ("probe.run", kind)
        assert target.executable is stub
        assert target.template_identity == stub.template_identity
        assert (
            target.instance_identity is None
        )  # a code-backed body has no static instance
        assert target.outputs == ("x", "y")

    def test_the_greatest_version_is_served(self) -> None:
        old = _Stub()
        new = _Stub()
        new.version = "2"
        registry = _registry(old)
        registry.register(new)
        target = _resolve(registry)
        assert target is not None and target.executable is new

    @pytest.mark.parametrize("kind", ["fsm", "strategy", "procedure"])
    def test_a_body_kind_is_not_a_delegated_target(self, kind) -> None:
        registry = ExecutableRegistry()
        registry.add_document(_doc(kind))
        assert _resolve(registry, "demo") is None
        assert registry.resolve_target("demo") is not None  # it stays a frame target

    def test_an_unknown_id_is_nothing(self) -> None:
        assert _resolve(ExecutableRegistry(), "no.such") is None

    def test_a_deprecated_executable_is_nothing(self) -> None:
        stub = _Stub()
        stub.lifecycle = Lifecycle.DEPRECATED
        assert _resolve(_registry(stub)) is None

    def test_the_id_is_never_loaded_as_a_module(self) -> None:
        assert _resolve(ExecutableRegistry(), "os.path") is None

    def test_the_registry_keeps_no_instance(self) -> None:
        registry = _registry(_Stub())
        before = set(vars(registry))
        _resolve(registry)
        assert set(vars(registry)) == before


class TestRefusals:
    """Each refusal is ``None``, and its control resolves."""

    def test_no_launcher_injected(self) -> None:
        registry = _registry(_Stub())
        assert _resolve(registry) is not None
        assert registry.resolve_delegated("probe.run", launcher=None) is None
        assert (
            registry.resolve_delegated(
                "probe.run", launcher=object(), context=RunContext()
            )
            is None
        )  # an object that is not a RunLauncher

    def test_a_conversational_target(self) -> None:
        registry = _registry(
            _Stub(id="probe.chat", capabilities=[Capability.CONVERSATIONAL])
        )
        assert _resolve(registry, "probe.chat") is None
        plain = _registry(_Stub(id="probe.chat", capabilities=[Capability.TOOLS]))
        assert _resolve(plain, "probe.chat") is not None

    def test_the_meta_builder_pattern_is_conversational(self) -> None:
        registry = _registry(
            PatternExecutable("meta_builder"), PatternExecutable("debate")
        )
        assert _resolve(registry, "meta_builder") is None
        assert _resolve(registry, "debate") is not None

    def test_a_required_callable_the_host_lacks(self) -> None:
        registry = _registry(_Stub(callables=["tools"]))
        assert _resolve(registry) is None  # no callables at all
        other: NameRegistry[object] = NameRegistry("callables")
        other.register("config", object())
        assert _resolve(registry, context=RunContext(callables=other)) is None
        have: NameRegistry[object] = NameRegistry("callables")
        have.register("tools", object())
        assert _resolve(registry, context=RunContext(callables=have)) is not None

    def test_the_judge_is_the_host_context_not_the_global_flag(self) -> None:
        """``list(launchable=)`` is false for a callables-requiring executable; a host that
        supplies the callables resolves it all the same."""
        registry = _registry(PatternExecutable("react"))
        assert registry.list(launchable=True) == []
        have: NameRegistry[object] = NameRegistry("callables")
        have.register("tools", [])
        assert _resolve(registry, "react", context=RunContext(callables=have))
        assert _resolve(registry, "react") is None

    def test_an_input_key_outside_the_contract(self) -> None:
        registry = _registry(_Stub(optional_inputs=("extra",)))
        assert _resolve(registry, input_keys=["task"]) is not None
        assert _resolve(registry, input_keys=("task", "extra")) is not None
        assert _resolve(registry, input_keys=["task", "other"]) is None
        assert _resolve(registry, input_keys=["other"]) is None

    def test_max_steps_on_a_target_that_refuses_it(self) -> None:
        refusing = _registry(_Stub(refused=("max_steps",)))
        assert _resolve(refusing) is not None
        assert _resolve(refusing, max_steps=3) is None
        accepting = _registry(_Stub(refused=()))
        assert _resolve(accepting, max_steps=3) is not None

    def test_the_real_patterns_refuse_max_steps_and_take_their_task(self) -> None:
        registry = _registry(PatternExecutable("debate"))
        assert _resolve(registry, "debate", max_steps=2) is None
        assert _resolve(registry, "debate", input_keys=["task"]) is not None
        assert (
            _resolve(registry, "debate", input_keys=["task", "initial_context"])
            is not None
        )
        assert _resolve(registry, "debate", input_keys=["task", "hitl"]) is None

    def test_a_refusal_never_raises_for_a_miss(self) -> None:
        registry = _registry(_Stub(callables=["tools"], refused=("max_steps",)))
        assert _resolve(registry, input_keys=["zzz"], max_steps=1) is None
