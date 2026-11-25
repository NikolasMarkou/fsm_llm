"""V2 of a workflow document in its structure-only form (plan 7d9c0955 step 5, D-008; assumption 6).

Core cannot import ``fsm_llm.workflows``, so the structure of a document is checked when a registry
loads it: ``workflow_adapter`` builds the definition over STUBS (the loader plus ``validate_structure()``,
no host object, no registry) and refuses a step that pauses. These rows hold that probe to what it
promises: it loads EVERY built-in step tag (a new step validator that the stub cannot satisfy fails
here and not in a user's registry), it never reads a host object, and every refusal is one capped line.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.core.definitions.executable import (
    ExecutableDocumentError,
    parse_executable,
)
from fsm_llm.core.definitions.executable_workflow import walk_steps
from fsm_llm.workflows import executable as wfexec
from fsm_llm.workflows.executable import (
    WorkflowExecutable,
    _structure_problems,
    workflow_adapter,
)
from fsm_llm.workflows.loader import builtin_step_classes
from tests.test_fsm_llm_workflows.support import (
    callables_registry,
    every_step_objects,
    every_step_workflow,
)


def _every_tag_steps() -> dict[str, Any]:
    """The 11 built-in step classes of ``every_step_workflow`` plus an ``ExecutableStep``, as a body."""
    objects = every_step_objects()
    dump = every_step_workflow(objects).serialize(callables=callables_registry(objects))
    steps = dump["steps"]
    steps["s_exec"] = {
        "type": "ExecutableStep",
        "step_id": "s_exec",
        "name": "Exec",
        "target": "other",
        "success_step_id": "s_par",
    }
    steps["s_switch"]["default_step_id"] = "s_exec"
    return steps


def _envelope(steps: dict[str, Any], initial: str | None = None) -> Any:
    return parse_executable(
        {
            "schema": "fsm-llm/executable",
            "kind": "workflow",
            "id": "probe",
            "body": {"initial_step_id": initial or next(iter(steps)), "steps": steps},
        }
    )


def _chain() -> dict[str, Any]:
    return {
        "a": {"type": "AutoStep", "step_id": "a", "name": "a", "next_step_id": "z"},
        "z": {"type": "AutoStep", "step_id": "z", "name": "z", "next_step_id": ""},
    }


class TestEveryTagLoads:
    def test_the_document_holds_every_builtin_tag(self) -> None:
        tags = {step["type"] for step, _, _ in walk_steps(_every_tag_steps())}
        assert tags == set(builtin_step_classes())

    def test_the_probe_loads_every_tag_and_refuses_only_the_two_pausing_steps(
        self,
    ) -> None:
        problems = _structure_problems(
            WorkflowExecutable(_envelope(_every_tag_steps()))
        )
        assert len(problems) == 2, problems
        assert all("cannot pause" in line for line in problems)
        assert any("'s_wait'" in line for line in problems)
        assert any("'s_timer'" in line for line in problems)

    def test_without_the_pausing_steps_the_document_has_no_problem(self) -> None:
        steps = _every_tag_steps()
        # route around the two pausing steps: nothing reaches them, so drop them
        steps["s_switch"]["cases"] = {"a": "s_exec", "b": "s_exec"}
        del steps["s_wait"], steps["s_timer"]
        assert _structure_problems(WorkflowExecutable(_envelope(steps))) == []

    def test_a_new_builtin_step_class_that_the_stub_cannot_satisfy_would_show_here(
        self,
    ) -> None:
        """The row that holds assumption 6: a tag in ``builtin_step_classes`` the document above does
        not hold is a gap in this file, so the set of tags is pinned."""
        assert set(builtin_step_classes()) == {
            "AgentStep",
            "APIStep",
            "AutoStep",
            "ConditionStep",
            "ConversationStep",
            "ExecutableStep",
            "LLMStep",
            "ParallelStep",
            "RetryStep",
            "SwitchStep",
            "TimerStep",
            "WaitEventStep",
        }


class TestRefusals:
    @pytest.mark.parametrize(
        "pausing",
        [
            {
                "type": "WaitEventStep",
                "step_id": "a",
                "name": "a",
                "config": {"event_type": "go", "success_step_id": "z"},
            },
            {
                "type": "TimerStep",
                "step_id": "a",
                "name": "a",
                "delay_seconds": 1,
                "next_step_id": "z",
            },
            {
                "type": "RetryStep",
                "step_id": "a",
                "name": "a",
                "step": {
                    "type": "WaitEventStep",
                    "step_id": "in",
                    "name": "in",
                    "config": {"event_type": "go", "success_step_id": "z"},
                },
            },
        ],
    )
    def test_each_pausing_form_is_one_capped_line_through_the_factory(
        self, pausing: dict[str, Any]
    ) -> None:
        steps = _chain()
        steps["a"] = pausing
        with pytest.raises(ExecutableDocumentError) as caught:
            workflow_adapter(_envelope(steps))
        assert len(caught.value.problems) == 1
        assert len(caught.value.problems[0]) < 300

    def test_a_long_step_id_is_capped_in_the_line(self) -> None:
        name = "w" * 120  # a long id V1 admits
        steps = {
            name: {
                "type": "TimerStep",
                "step_id": name,
                "name": "t",
                "delay_seconds": 1,
                "next_step_id": "",
            }
        }
        (line,) = _structure_problems(WorkflowExecutable(_envelope(steps)))
        assert len(line) < 200 and name not in line

    def test_an_unknown_step_type_and_a_sync_cycle_are_loader_lines(self) -> None:
        unknown = _chain()
        unknown["a"] = {"type": "Nope", "step_id": "a", "name": "a"}
        (line, *_) = _structure_problems(WorkflowExecutable(_envelope(unknown)))
        assert "unknown step type" in line
        cyclic = _chain()
        cyclic["z"]["next_step_id"] = "a"
        lines = _structure_problems(WorkflowExecutable(_envelope(cyclic)))
        assert any("cycle" in line for line in lines), lines

    def test_a_good_document_builds_an_executable(self) -> None:
        assert isinstance(workflow_adapter(_envelope(_chain())), WorkflowExecutable)


class TestTheProbeReadsNoHostObject:
    def test_the_probe_binds_stubs_and_no_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[Any, Any]] = []
        original = WorkflowExecutable.build_definition

        def spy(self: Any, callables: Any = None, registry: Any = None) -> Any:
            seen.append((callables, registry))
            return original(self, callables, registry)

        monkeypatch.setattr(WorkflowExecutable, "build_definition", spy)
        steps = _chain()
        steps["a"]["action"] = "anything"
        workflow_adapter(_envelope(steps))
        ((callables, registry),) = seen
        assert registry is None
        assert [type(obj) for _, obj in callables.items()] == [wfexec._Stub]
        assert callables.list() == ["anything"]
