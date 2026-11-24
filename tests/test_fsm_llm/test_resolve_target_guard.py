"""The frame guard of ``ExecutableRegistry.resolve_target`` (plan a555dcfe, step 4, D-010).

A frame is a plain child conversation of a ``call`` state: the engine runs no handler of the
child. So an executable of kind ``pattern`` and any executable whose ``requires.handlers`` is
non-empty is never pushed as a frame: ``resolve_target`` answers ``None`` and the call ends
``failed/unknown_target``. The loader accepts an ``fsm`` envelope that declares
``requires.handlers`` for an operator origin (V7 refuses it only for a model), so the document
row below is a real hole of the parent commit. Delegated resolution of a built-in pattern is
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsm_llm
from fsm_llm import API
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.definitions.executable import Contract, Requires
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.code_backed import (
    CodeBackedExecutable,
    CodeBackedInstance,
)
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.test_call_settle import _initial_call
from tests.test_fsm_llm.test_call_state import _call, _parent
from tests.test_fsm_llm.test_executable_adapters import _doc

_TARGET = "reasoning.analytical"


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


@dataclass
class _BodyInstance:
    """What a future pattern file instantiates to: it carries a definition (a body to push)."""

    definition: FSMDefinition
    instance_identity: str = "instance:probe"


class _BodyBacked(CodeBackedExecutable):
    """A code-backed executable whose instance carries an FSM body, of a chosen kind."""

    def __init__(self, kind: str, *, handlers: tuple[str, ...] = ()) -> None:
        super().__init__(
            id=_TARGET,
            kind=kind,
            impl="tests.probe.BodyBacked",
            contract=Contract(inputs=["topic"]),
            requires=Requires(handlers=list(handlers)),
        )

    def instantiate(self, options=None, variables=None):  # type: ignore[override]
        return _BodyInstance(FSMDefinition(**kc._fsm("Probe", "c1", kc._state("c1"))))

    async def execute(self, inputs, objects, context, identity) -> RunResult:
        return RunResult(status=Status.OK, answer="done", identity=identity)


def _registry(*executables: Any) -> ExecutableRegistry:
    registry = ExecutableRegistry()
    for executable in executables:
        registry.register(executable)
    return registry


class TestFrameModeRefuses:
    def test_a_document_that_requires_handlers_is_no_frame_target(self) -> None:
        """The loader accepts it (operator origin); resolve_target did return its body.
        Kills (D3): a guard that is missing or reads only the kind."""
        registry = ExecutableRegistry()
        executable = registry.add_document(
            _doc("fsm", _TARGET, requires={"handlers": ["h1"]})
        )
        assert executable.requires.handlers == ["h1"]
        assert registry.resolve_target(_TARGET) is None

    def test_a_strategy_that_requires_handlers_is_no_frame_target(self) -> None:
        registry = ExecutableRegistry()
        registry.add_document(_doc("strategy", _TARGET, requires={"handlers": ["h1"]}))
        assert registry.resolve_target(_TARGET) is None

    def test_a_pattern_that_carries_a_body_is_no_frame_target(self) -> None:
        """A pattern file will instantiate to a definition; it still must not be a frame.
        Kills (D3): a guard that reads only `requires.handlers`."""
        registry = _registry(_BodyBacked("pattern"))
        assert registry.resolve_target(_TARGET) is None

    def test_a_pattern_with_handlers_is_no_frame_target(self) -> None:
        registry = _registry(_BodyBacked("pattern", handlers=("gate",)))
        assert registry.resolve_target(_TARGET) is None

    def test_a_handler_free_body_still_resolves(self) -> None:
        """The control: the guard refuses the two shapes and nothing else."""
        registry = ExecutableRegistry()
        registry.add_document(_doc("fsm", _TARGET))
        target = registry.resolve_target(_TARGET)
        assert target is not None and target.mode == "frame"
        registry = _registry(_BodyBacked("fsm"))
        assert registry.resolve_target(_TARGET) is not None

    def test_the_refusal_is_per_version(self) -> None:
        """The greatest version is judged, as `get` serves it: a newer version that requires
        handlers hides the older plain one for a frame."""
        registry = ExecutableRegistry()
        registry.add_document(_doc("fsm", _TARGET, version="1"))
        assert registry.resolve_target(_TARGET) is not None
        registry.add_document(
            _doc("fsm", _TARGET, version="2", requires={"handlers": ["h1"]})
        )
        assert registry.resolve_target(_TARGET) is None


class TestACallEndsFailed:
    async def test_a_call_to_a_handler_requiring_target_ends_unknown_target(
        self,
    ) -> None:
        registry = ExecutableRegistry()
        registry.add_document(
            _doc("fsm", _TARGET, requires={"handlers": ["h1"]}, inputs=("topic",))
        )
        api = API.from_definition(
            _initial_call(_parent(_call())),
            llm_interface=ScriptedLLM(default_response="r"),
            targets=registry,
        )
        cid, _ = await api.astart_conversation({"topic": "why"})
        frame = api.conversation_stacks[cid][0].conversation_id
        await api.arun_until_terminal(cid, max_steps=10)
        record = api.fsm_manager.copy_raw_context_data(frame)["call_result"]
        assert (record["status"], record["reason"]) == (
            "failed",
            Reason.UNKNOWN_TARGET.value,
        )
        assert api.get_stack_depth(cid) == 1  # nothing was pushed


class TestDelegatedModeIsUntouched:
    def test_a_code_backed_pattern_still_resolves_delegated(self) -> None:
        stub = _BodyBacked("pattern")
        registry = _registry(stub)
        launcher = _Launcher()
        target = registry.resolve_delegated(
            _TARGET, launcher=launcher, context=RunContext(), input_keys=()
        )
        assert target is not None and target.mode == "delegated"
        assert target.executable is stub

    def test_a_builtin_pattern_still_resolves_delegated(self) -> None:
        import fsm_llm.agents  # noqa: F401  (registers the lazy source)

        registry = ExecutableRegistry()
        registry.load_sources()
        executable = registry.get("debate")
        assert executable.kind == "pattern"
        assert registry.resolve_target("debate") is None  # a frame: refused
        target = registry.resolve_delegated(
            "debate", launcher=_Launcher(), context=RunContext(), input_keys=("task",)
        )
        assert target is not None and target.kind == "pattern"
        assert isinstance(executable.instantiate(), CodeBackedInstance)


class _Launcher:
    """The six methods of a launcher, doing nothing."""

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
