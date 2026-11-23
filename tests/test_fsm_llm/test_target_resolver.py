"""The target resolver and ``API(targets=)`` (plan baa37f40, step 10, D-021, roadmap E3.2).

A ``call`` state names its target by id; the engine below ``engine.executables`` resolves it
through a ``TargetResolver`` it is handed (``API(targets=)``, or ``RunContext.registry`` for the
nested ``API`` of ``FsmInstance.arun``), never through a registry import and never through a
string import. Frame mode resolves the kinds ``fsm``, ``strategy`` and ``procedure``; a pattern,
a workflow, an unknown id and a body that does not instantiate resolve to nothing (``None``, the
answer a caller maps to ``failed/unknown_target``). A definition holding a ``call`` state is
refused with the named ``MissingExecutorError`` where it enters an ``API`` with no resolver.
Rows use the real core and a scripted model; nothing here runs a ``call`` state (step 11).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from fsm_llm import API, FSMManager
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import FSMDefinition, MissingExecutorError
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    InstantiationError,
)
from fsm_llm.core.results import RunResult
from fsm_llm.core.state_kinds import KINDS, get_kind
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable
from fsm_llm.engine.kinds.targets import (
    TARGET_MODE_FRAME,
    ResolvedTarget,
    TargetResolver,
    check_targets,
)
from fsm_llm.runtime import EventLog
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.test_call_state import _parent
from tests.test_fsm_llm.test_executable_adapters import _doc, _fsm


def _call_definition() -> dict[str, Any]:
    return copy.deepcopy(_parent())


def _child() -> dict[str, Any]:
    return copy.deepcopy(_fsm())


class _Pattern(CodeBackedExecutable):
    """A code-backed ``pattern`` executable: it has no FSM body to push."""

    def __init__(self, kind: str = "pattern") -> None:
        super().__init__(
            id="probe.pattern",
            kind=kind,
            impl="tests.probe.Pattern",
            contract=Contract(inputs=["task"]),
            capabilities=[Capability.CONVERSATIONAL],
        )

    async def execute(self, inputs, objects, context, identity) -> RunResult:
        return RunResult(status=Status.OK, answer="done", identity=identity)


class _Fixed:
    """A resolver that answers one definition for every id (a second implementer)."""

    def __init__(self, definition: FSMDefinition | None = None) -> None:
        self.definition = definition
        self.asked: list[tuple[str, str | None]] = []

    def resolve_target(self, target_id: str, *, conversation_id: str | None = None):
        self.asked.append((target_id, conversation_id))
        if self.definition is None:
            return None
        return ResolvedTarget(self.definition, "t" * 64, "i" * 64)


# ---- the Protocol and the value --------------------------------------------------------------


class TestTheSeam:
    def test_the_value_is_frozen_and_frame_is_its_only_mode(self):
        definition = FSMDefinition(**_child())
        target = ResolvedTarget(definition, "t", "i")
        assert target.mode == TARGET_MODE_FRAME == "frame"
        with pytest.raises(dataclasses.FrozenInstanceError):
            target.mode = "frame"  # type: ignore[misc]
        with pytest.raises(ValueError, match="mode"):
            ResolvedTarget(definition, "t", "i", mode="delegated")

    def test_the_two_resolvers_both_satisfy_the_protocol(self):
        assert isinstance(ExecutableRegistry(), TargetResolver)
        assert isinstance(_Fixed(), TargetResolver)
        assert not isinstance(object(), TargetResolver)

    def test_check_targets(self):
        registry = ExecutableRegistry()
        assert check_targets(None) is None
        assert check_targets(registry) is registry
        with pytest.raises(TypeError, match="resolve_target"):
            check_targets(object())
        with pytest.raises(TypeError, match="resolve_target"):
            check_targets({"reasoning.analytical": "x"})  # a mapping is not a resolver


# ---- the registry as a resolver --------------------------------------------------------------


class TestTheRegistryResolves:
    @pytest.mark.parametrize("kind", ["fsm", "strategy", "procedure"])
    def test_a_body_kind_resolves_to_its_definition_and_both_identities(self, kind):
        registry = ExecutableRegistry()
        executable = registry.add_document(_doc(kind))
        target = registry.resolve_target("demo")
        assert target is not None and target.mode == "frame"
        instance = executable.instantiate()
        assert target.definition == instance.definition
        assert target.template_identity == executable.template_identity
        assert target.instance_identity == instance.instance_identity

    def test_the_identities_equal_those_of_a_run(self):
        """The same id through the resolver and through a run report one pair."""
        registry = ExecutableRegistry()
        executable = registry.add_document(_doc("fsm"))
        target = registry.resolve_target("demo")
        llm = ScriptedLLM(default_response="r")
        result = executable.run(
            {"topic": "t"}, context=RunContext(llm_interface=llm, max_steps=5)
        )
        assert result.identity is not None
        assert (target.template_identity, target.instance_identity) == (
            result.identity.template_identity,
            result.identity.instance_identity,
        )

    def test_the_greatest_version_of_a_plain_id_is_served(self):
        registry = ExecutableRegistry()
        registry.add_document(_doc("fsm", version="1"))
        newer = registry.add_document(_doc("fsm", version="2"))
        target = registry.resolve_target("demo")
        assert target.template_identity == newer.template_identity

    @pytest.mark.parametrize("kind", ["pattern", "workflow"])
    def test_a_code_backed_kind_resolves_to_nothing(self, kind):
        registry = ExecutableRegistry()
        registry.register(_Pattern(kind))
        assert registry.resolve_target("probe.pattern") is None

    def test_an_unknown_id_resolves_to_nothing(self):
        assert ExecutableRegistry().resolve_target("no.such.id") is None

    def test_a_body_that_does_not_instantiate_resolves_to_nothing(self):
        """A document that loads but whose instantiation fails (a declared variable is unfilled)."""
        registry = ExecutableRegistry()
        document = _doc("fsm")
        document["body"]["fsm"]["description"] = "uses {{who}}"
        document["variables"] = ["who"]
        executable = registry.add_document(document)
        with pytest.raises(InstantiationError):
            executable.instantiate()
        assert registry.resolve_target("demo") is None

    def test_the_id_is_never_loaded_as_a_module(self):
        """Resolution reads the registry only: an id that names an importable module is a miss."""
        assert ExecutableRegistry().resolve_target("os.path") is None
        assert ExecutableRegistry().resolve_target("fsm_llm.engine.executables") is None

    def test_the_conversation_id_is_accepted(self):
        registry = ExecutableRegistry()
        registry.add_document(_doc("fsm"))
        assert registry.resolve_target("demo", conversation_id="c-1") is not None

    def test_the_registry_keeps_no_instance(self):
        registry = ExecutableRegistry()
        registry.add_document(_doc("fsm"))
        before = dict(vars(registry))
        registry.resolve_target("demo")
        assert vars(registry).keys() == before.keys()
        assert registry._retained.keys() <= {
            e.executable.template_identity for e in registry._snapshot.entries.values()
        }


# ---- API(targets=) ----------------------------------------------------------------------------


def _llm() -> ScriptedLLM:
    return ScriptedLLM(default_response="r")


class TestAnApiNeedsAResolverForACallState:
    def test_the_flag_is_data_on_the_call_spec_only(self):
        assert [s.name for s in KINDS if s.requires_targets] == ["call"]
        assert get_kind("call").requires_targets is True
        assert get_kind("synthesize").requires_targets is False

    def test_construction_without_a_resolver_names_state_kind_and_resolver(self):
        with pytest.raises(MissingExecutorError) as refused:
            API.from_definition(_call_definition(), llm_interface=_llm())
        error = refused.value
        assert isinstance(error, ValueError)
        assert (error.state_id, error.kind) == ("work", "call")
        assert "TargetResolver" in str(error) and "targets=" in str(error)

    def test_construction_with_a_resolver_succeeds_and_hands_it_down(self):
        resolver = _Fixed()
        api = API.from_definition(
            _call_definition(), llm_interface=_llm(), targets=resolver
        )
        assert api._targets is resolver
        assert api.fsm_manager._pipeline.target_resolver is resolver

    def test_a_registry_is_a_resolver(self):
        registry = ExecutableRegistry()
        api = API(_call_definition(), llm_interface=_llm(), targets=registry)
        assert api.fsm_manager._pipeline.target_resolver is registry

    def test_a_definition_with_no_call_state_needs_none(self):
        api = API.from_definition(_child(), llm_interface=_llm())
        assert api._targets is None
        assert api.fsm_manager._pipeline.target_resolver is None

    def test_a_bad_resolver_is_a_type_error(self):
        with pytest.raises(TypeError, match="resolve_target"):
            API.from_definition(_child(), llm_interface=_llm(), targets=object())

    def test_push_without_a_resolver_is_refused_and_leaves_nothing(self):
        api = API.from_definition(_child(), llm_interface=_llm())
        cid, _ = api.start_conversation()
        with pytest.raises(MissingExecutorError):
            api.push_fsm(cid, _call_definition())
        assert api.get_stack_depth(cid) == 1
        assert not api._pending_push_ids and not api._temp_fsm_definitions
        assert set(api.fsm_manager.instances) == {cid}

    def test_push_with_a_resolver_works(self):
        api = API.from_definition(_child(), llm_interface=_llm(), targets=_Fixed())
        cid, _ = api.start_conversation()
        api.push_fsm(cid, _call_definition())
        assert api.get_stack_depth(cid) == 2

    def test_the_manager_and_the_pipeline_take_the_resolver_by_keyword(self):
        resolver = _Fixed()
        manager = FSMManager(
            fsm_loader=lambda _id: FSMDefinition(**_child()),
            llm_interface=_llm(),
            targets=resolver,
        )
        assert manager._pipeline.target_resolver is resolver
        with pytest.raises(TypeError, match="resolve_target"):
            FSMManager(
                fsm_loader=lambda _id: FSMDefinition(**_child()),
                llm_interface=_llm(),
                targets=object(),
            )

    def test_fork_of_a_call_frame_without_a_resolver_is_refused(self, tmp_path):
        """The frame a fork rebuilds is judged like a push: a configuration fact, raised as itself."""
        log = EventLog(tmp_path / "log")
        with_resolver = API.from_definition(
            _child(), llm_interface=_llm(), event_log=log, targets=_Fixed()
        )
        cid, _ = with_resolver.start_conversation()
        with_resolver.converse("hello", cid)
        with_resolver.push_fsm(cid, _call_definition())
        with_resolver.converse("again", cid)
        bare = API.from_definition(_child(), llm_interface=_llm(), event_log=log)
        with pytest.raises(MissingExecutorError):
            bare.fork(cid, at_turn=2)


# ---- RunContext.registry ----------------------------------------------------------------------


class TestTheRunContextCarriesTheResolver:
    def test_the_field_is_last_and_defaults_to_none(self):
        names = [f.name for f in dataclasses.fields(RunContext)]
        assert names[-1] == "registry"
        assert RunContext().registry is None

    def test_a_resolver_is_kept_and_derive_shares_it(self):
        resolver = _Fixed()
        context = RunContext(registry=resolver, max_steps=5)
        assert context.registry is resolver
        assert context.derive(parent="p").registry is resolver

    def test_a_value_that_is_not_a_resolver_is_a_type_error(self):
        with pytest.raises(TypeError, match="resolve_target"):
            RunContext(registry=object())  # type: ignore[arg-type]

    def test_the_nested_api_of_a_run_is_given_the_resolver(self):
        """``FsmInstance.arun`` builds its own ``API``: it gets ``context.registry``."""
        resolver = _Fixed()
        seen: list[Any] = []
        registry = ExecutableRegistry()
        document = _doc("fsm", fsm=_call_definition(), inputs=(), outputs=())
        executable = registry.add_document(document)
        executable.setup = lambda api: seen.append(api._targets)
        executable.run(
            {}, context=RunContext(llm_interface=_llm(), registry=resolver, max_steps=3)
        )
        assert seen == [resolver]

    def test_a_body_with_a_call_state_and_no_resolver_in_the_context_is_refused(self):
        registry = ExecutableRegistry()
        document = _doc("fsm", fsm=_call_definition(), inputs=(), outputs=())
        executable = registry.add_document(document)
        with pytest.raises(MissingExecutorError):
            executable.run({}, context=RunContext(llm_interface=_llm(), max_steps=3))
