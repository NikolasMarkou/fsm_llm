"""``WorkflowDefinition.serialize(callables=)``: the opt-in naming path (plan ace7a8c8 step 25).

The default ``serialize()`` is byte-identical to the parent commit: ``GOLDEN`` is the output of
the parent (captured before ``definitions.py`` was edited) for a definition holding all 11
built-in step classes. With ``callables=`` a registry of namespace ``callables`` names every
callable or object field; a value it cannot name, a nested step that is not a step and metadata
that is not JSON raise ``WorkflowDefinitionError`` and never drop silently.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.names import NameRegistry
from fsm_llm.workflows import (
    AutoStep,
    ParallelStep,
    RetryStep,
    WorkflowDefinitionError,
    WorkflowStep,
    create_workflow,
)
from tests.test_fsm_llm_workflows.support import (
    callables_registry,
    every_step_objects,
    every_step_workflow,
)

GOLDEN: dict[str, Any] = {
    "workflow_id": "wf_all",
    "name": "All steps",
    "description": "every class",
    "initial_step_id": "s_auto",
    "metadata": {"owner": {"team": "x", "tags": ["a", "b"]}},
    "steps": {
        "s_auto": {
            "step_id": "s_auto",
            "name": "Auto",
            "description": "with an action",
            "timeout_seconds": 5.0,
            "next_step_id": "s_auto2",
            "error_step_id": "done",
            "type": "AutoStep",
        },
        "s_auto2": {
            "step_id": "s_auto2",
            "name": "Auto2",
            "description": "",
            "timeout_seconds": None,
            "next_step_id": "s_cond",
            "error_step_id": None,
            "type": "AutoStep",
        },
        "s_cond": {
            "step_id": "s_cond",
            "name": "Cond",
            "description": "",
            "timeout_seconds": None,
            "true_step_id": "s_api",
            "false_step_id": "s_switch",
            "error_step_id": None,
            "type": "ConditionStep",
        },
        "s_api": {
            "step_id": "s_api",
            "name": "Api",
            "description": "",
            "timeout_seconds": None,
            "success_step_id": "s_llm",
            "failure_step_id": "s_switch",
            "input_mapping": {"q": "query"},
            "output_mapping": {"answer": "result.text"},
            "type": "APIStep",
        },
        "s_switch": {
            "step_id": "s_switch",
            "name": "Switch",
            "description": "",
            "timeout_seconds": None,
            "key": "intent",
            "cases": {"a": "s_wait", "b": "s_timer"},
            "default_step_id": "s_par",
            "type": "SwitchStep",
        },
        "s_wait": {
            "step_id": "s_wait",
            "name": "Wait",
            "description": "",
            "timeout_seconds": None,
            "config": {
                "event_type": "go",
                "success_step_id": "s_timer",
                "timeout_seconds": 30.0,
                "timeout_step_id": "s_par",
                "event_mapping": {"k": "payload.k"},
                "correlation_key": "order_id",
            },
            "type": "WaitEventStep",
        },
        "s_timer": {
            "step_id": "s_timer",
            "name": "Timer",
            "description": "",
            "timeout_seconds": None,
            "delay_seconds": 2.5,
            "next_step_id": "s_par",
            "type": "TimerStep",
        },
        "s_par": {
            "step_id": "s_par",
            "name": "Par",
            "description": "",
            "timeout_seconds": None,
            "steps": [
                {
                    "step_id": "p1",
                    "name": "P1",
                    "description": "",
                    "timeout_seconds": None,
                    "next_step_id": "x",
                    "error_step_id": None,
                    "type": "AutoStep",
                },
                {
                    "step_id": "p2",
                    "name": "P2",
                    "description": "",
                    "timeout_seconds": None,
                    "next_step_id": "x",
                    "error_step_id": None,
                    "type": "AutoStep",
                },
            ],
            "next_step_id": "s_retry",
            "error_step_id": "done",
            "type": "ParallelStep",
        },
        "s_retry": {
            "step_id": "s_retry",
            "name": "Retry",
            "description": "",
            "timeout_seconds": None,
            "max_retries": 2,
            "backoff_factor": 0.5,
            "type": "RetryStep",
            "step": {
                "step_id": "inner",
                "name": "Inner",
                "description": "",
                "timeout_seconds": None,
                "next_step_id": "s_conv",
                "error_step_id": None,
                "type": "AutoStep",
            },
        },
        "s_conv": {
            "step_id": "s_conv",
            "name": "Conv",
            "description": "",
            "timeout_seconds": None,
            "fsm_file": None,
            "fsm_definition": {"name": "f"},
            "model": "m",
            "initial_context": {"a": "b"},
            "context_mapping": {"c": "d"},
            "success_step_id": "s_conv2",
            "error_step_id": None,
            "max_turns": 20,
            "conversation_timeout": None,
            "require_completion": False,
            "use_user_input": False,
            "auto_messages": ["hi"],
            "type": "ConversationStep",
        },
        "s_conv2": {
            "step_id": "s_conv2",
            "name": "Conv2",
            "description": "",
            "timeout_seconds": None,
            "fsm_file": "flows/a.json",
            "fsm_definition": None,
            "model": None,
            "initial_context": {},
            "context_mapping": {},
            "success_step_id": "s_agent",
            "error_step_id": None,
            "max_turns": 20,
            "conversation_timeout": None,
            "require_completion": False,
            "use_user_input": False,
            "auto_messages": [],
            "type": "ConversationStep",
        },
        "s_agent": {
            "step_id": "s_agent",
            "name": "Agent",
            "description": "",
            "timeout_seconds": None,
            "task_template": "do {task}",
            "success_step_id": "s_llm",
            "context_mapping": {},
            "input_mapping": {"k": "v"},
            "error_step_id": None,
            "type": "AgentStep",
        },
        "s_llm": {
            "step_id": "s_llm",
            "name": "Llm",
            "description": "",
            "timeout_seconds": None,
            "prompt_template": "say {x}",
            "context_mapping": {"x": "y"},
            "output_mapping": {"out": ""},
            "next_step_id": "done",
            "error_step_id": None,
            "system_prompt": "You are a precise assistant. Follow the "
            "instructions in the user message and answer "
            "with exactly what they ask for.",
            "type": "LLMStep",
        },
        "done": {
            "step_id": "done",
            "name": "Done",
            "description": "",
            "timeout_seconds": None,
            "next_step_id": "",
            "error_step_id": None,
            "type": "AutoStep",
        },
    },
}


class PingStep(WorkflowStep):
    """A custom step with a callable field of its own (not one of the shared names)."""

    next_step_id: str
    hook: Callable[..., Any] | None = None

    async def execute(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError


class BlobStep(WorkflowStep):
    """A custom step whose field holds a value no JSON document can carry."""

    next_step_id: str = ""
    payload: Any = None

    async def execute(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError


def _single(step: WorkflowStep, *, metadata: dict[str, Any] | None = None) -> Any:
    workflow = create_workflow("wf", "W")
    workflow.with_initial_step(step)
    if metadata is not None:
        workflow.metadata.update(metadata)
    return workflow


def _registry(**objects: Any) -> NameRegistry[Any]:
    return callables_registry(objects)


class _Duck:
    """Not a ``WorkflowStep``, but the duck type ``RetryStep`` accepts."""

    async def execute(self, context: dict[str, Any]) -> Any:
        return None


class TestDefaultSerializeIsUnchanged:
    def test_the_default_output_equals_the_golden_of_the_parent_byte_for_byte(self):
        workflow = every_step_workflow(every_step_objects())
        out = workflow.serialize()
        assert out == GOLDEN
        # key order is part of "byte identical": dict equality ignores it
        assert json.dumps(out) == json.dumps(GOLDEN)

    def test_an_explicit_none_is_the_default_path(self):
        workflow = every_step_workflow(every_step_objects())
        assert json.dumps(workflow.serialize(callables=None)) == json.dumps(GOLDEN)

    def test_the_default_path_names_no_callable(self):
        text = json.dumps(every_step_workflow(every_step_objects()).serialize())
        for name in ("act", "cond", "agg", "llm", "agent"):
            assert f'"{name}"' not in text

    def test_the_golden_covers_every_builtin_step_class(self):
        import fsm_llm.workflows.steps as steps

        tags: set[str] = set()

        def walk(step: dict[str, Any]) -> None:
            tags.add(step["type"])
            for nested in (
                step.get("steps", []) if step["type"] == "ParallelStep" else []
            ):
                walk(nested)
            if step["type"] == "RetryStep":
                walk(step["step"])

        for step in GOLDEN["steps"].values():
            walk(step)
        assert tags == set(steps.__all__) - {"WorkflowStep"}


class TestSerializeNamesCallables:
    def test_each_callable_field_holds_its_registered_name(self):
        objects = every_step_objects()
        workflow = every_step_workflow(objects)
        out = workflow.serialize(callables=callables_registry(objects))
        steps = out["steps"]
        assert steps["s_auto"]["action"] == "act"
        assert "action" not in steps["s_auto2"]  # None stays absent
        assert steps["s_cond"]["condition"] == "cond"
        assert steps["s_api"]["api_function"] == "api"
        assert steps["s_par"]["aggregation_function"] == "agg"
        assert steps["s_par"]["steps"][0]["action"] == "child_act"
        assert "action" not in steps["s_par"]["steps"][1]
        assert steps["s_retry"]["step"]["action"] == "inner_act"
        assert steps["s_llm"]["llm_interface"] == "llm"
        assert steps["s_agent"]["agent"] == "agent"

    def test_the_output_is_json_and_adds_only_the_names(self):
        objects = every_step_objects()
        out = every_step_workflow(objects).serialize(
            callables=callables_registry(objects)
        )
        json.dumps(out)  # no callable, no object left
        stripped = json.loads(json.dumps(out))
        for step in stripped["steps"].values():
            for key in (
                "action",
                "condition",
                "api_function",
                "llm_interface",
                "agent",
            ):
                step.pop(key, None)
            step.pop("aggregation_function", None)
            nested = step.get("step")
            if nested:
                nested.pop("action", None)
            for child in (
                step.get("steps", []) if step["type"] == "ParallelStep" else []
            ):
                child.pop("action", None)
        assert stripped == GOLDEN

    def test_serializing_changes_neither_the_workflow_nor_the_registry(self):
        objects = every_step_objects()
        registry = callables_registry(objects)
        workflow = every_step_workflow(objects)
        before = workflow.model_copy(deep=False)
        workflow.serialize(callables=registry)
        assert workflow == before
        assert workflow.steps["s_auto"].action is objects["act"]
        assert registry.list() == sorted(objects)

    def test_a_workflow_without_callables_serializes_the_same_with_a_registry(self):
        from fsm_llm.workflows import switch_step, timer_step

        workflow = create_workflow("wf", "W")
        workflow.with_initial_step(
            switch_step("s", "S", key="k", cases={"a": "t"}, default_step_id="t")
        )
        workflow.with_step(timer_step("t", "T", delay_seconds=1, next_step_id=""))
        assert workflow.serialize(callables=_registry()) == workflow.serialize()

    def test_a_custom_step_callable_field_is_named_too(self):
        def hook(**kw: Any) -> None:
            return None

        step = PingStep(step_id="p", name="P", next_step_id="", hook=hook)
        out = _single(step).serialize(callables=_registry(hook=hook))
        assert out["steps"]["p"]["hook"] == "hook"
        assert out["steps"]["p"]["type"] == "PingStep"


class TestSerializeRefusesASilentLoss:
    @staticmethod
    def _refuses(
        workflow: Any, registry: Any, *needles: str
    ) -> WorkflowDefinitionError:
        with pytest.raises(WorkflowDefinitionError) as info:
            workflow.serialize(callables=registry)
        for needle in needles:
            assert needle in str(info.value)
        return info.value

    def test_an_unregistered_action_is_refused_naming_step_and_field(self):
        step = AutoStep(
            step_id="a", name="A", next_step_id="", action=lambda c: {"x": 1}
        )
        self._refuses(_single(step), _registry(), "'a'", "action")

    def test_an_unregistered_llm_interface_and_agent_are_refused(self):
        from fsm_llm.workflows import agent_step, llm_step

        class Gen:
            def generate(self, prompt: str) -> str:
                return ""

        class Run:
            def run(self, task: str) -> str:
                return ""

        self._refuses(
            _single(llm_step("l", "L", Gen(), "p", next_step_id="")),
            _registry(),
            "'l'",
            "llm_interface",
        )
        self._refuses(
            _single(agent_step("g", "G", Run(), success_step_id="")),
            _registry(),
            "'g'",
            "agent",
        )

    def test_a_look_alike_of_a_registered_callable_is_not_that_callable(self):
        def registered(c: dict[str, Any]) -> dict[str, Any]:
            return {}

        def look_alike(c: dict[str, Any]) -> dict[str, Any]:
            return {}

        step = AutoStep(step_id="a", name="A", next_step_id="", action=look_alike)
        self._refuses(_single(step), _registry(registered=registered), "action")

    def test_an_unregistered_callable_of_a_nested_step_is_refused(self):
        inner = AutoStep(step_id="i", name="I", next_step_id="", action=lambda c: {})
        retry = RetryStep(step_id="r", name="R", step=inner)
        self._refuses(_single(retry), _registry(), "'i'", "action")
        child = AutoStep(step_id="c", name="C", next_step_id="", action=lambda c: {})
        parallel = ParallelStep(step_id="p", name="P", steps=[child], next_step_id="")
        self._refuses(_single(parallel), _registry(), "'c'", "action")

    def test_an_unregistered_custom_callable_field_is_refused(self):
        step = PingStep(step_id="p", name="P", next_step_id="", hook=lambda: None)
        self._refuses(_single(step), _registry(), "'p'", "hook")

    def test_a_retry_step_that_wraps_a_non_step_is_refused_not_dropped(self):
        retry = RetryStep(step_id="r", name="R", step=_Duck())
        workflow = _single(retry)
        assert "step" not in workflow.serialize()["steps"]["r"]  # the default drops it
        self._refuses(workflow, _registry(), "'r'", "step")

    def test_a_parallel_child_that_is_not_a_step_is_refused_not_passed_through(self):
        child = AutoStep(step_id="c", name="C", next_step_id="")
        parallel = ParallelStep(step_id="p", name="P", steps=[child], next_step_id="")
        parallel.steps.append(_Duck())  # type: ignore[arg-type]
        workflow = _single(parallel)
        workflow.serialize()  # the default passes the object through, as before
        self._refuses(workflow, _registry(), "'p'", "steps")

    def test_non_json_metadata_is_refused(self):
        step = AutoStep(step_id="a", name="A", next_step_id="")
        workflow = _single(step, metadata={"bad": {1, 2}})
        workflow.serialize()  # default: untouched
        self._refuses(workflow, _registry(), "metadata")

    def test_a_non_json_step_field_is_refused_naming_step_and_field(self):
        step = BlobStep(step_id="b", name="B", payload=object())
        self._refuses(_single(step), _registry(), "'b'", "payload")

    def test_the_refusal_is_a_workflow_definition_error_with_the_workflow_id(self):
        step = AutoStep(step_id="a", name="A", next_step_id="", action=lambda c: {})
        error = self._refuses(_single(step), _registry())
        assert error.workflow_id == "wf"

    @pytest.mark.parametrize("bad", [{}, ["act"], object()])
    def test_a_registry_that_is_not_a_registry_is_refused(self, bad: Any):
        workflow = _single(AutoStep(step_id="a", name="A", next_step_id=""))
        with pytest.raises(WorkflowDefinitionError, match="callables"):
            workflow.serialize(callables=bad)

    def test_a_registry_of_another_namespace_is_refused(self):
        workflow = _single(AutoStep(step_id="a", name="A", next_step_id=""))
        with pytest.raises(WorkflowDefinitionError, match="'tools'"):
            workflow.serialize(callables=NameRegistry("tools"))


class TestObjectFormFsmDefinition:
    @staticmethod
    def _fsm() -> FSMDefinition:
        return FSMDefinition.model_validate(
            {
                "name": "f",
                "description": "d",
                "initial_state": "s",
                "persona": "p",
                "states": {
                    "s": {"id": "s", "description": "d", "purpose": "p"},
                },
            }
        )

    def test_an_fsm_definition_object_serializes_as_its_dump_in_both_paths(self):
        from fsm_llm.workflows import conversation_step

        fsm = self._fsm()
        workflow = _single(
            conversation_step("c", "C", fsm_definition=fsm, success_step_id="")
        )
        default = workflow.serialize()["steps"]["c"]["fsm_definition"]
        named = workflow.serialize(callables=_registry())["steps"]["c"][
            "fsm_definition"
        ]
        assert default == named == fsm.model_dump()
        assert isinstance(default, dict)
        json.dumps(named)
