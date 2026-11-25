"""The corpus and the snapshot of differential G1 (plan 7d9c0955 step 1). Not collected.

``snapshot()`` serializes a fixed corpus of workflow definitions (every serializable step class,
nested ``RetryStep`` and ``ParallelStep``, a terminal ``AutoStep(next_step_id="")`` with no action,
an ``ExecutableStep``, a custom step class) the two ways (``serialize()`` and
``serialize(callables=)``), loads the named form back through ``load_workflow`` and records the
bytes, whether the round trip is equal, and the two tables of named-object fields. The fixture
``tests/fixtures/workflow_serialize_golden.json`` was written by running this module against a
``git archive`` of the parent commit ``a8ad5d31`` (the BASE of the plan); the test compares the
tree to it, so a field, a type tag, a key order or a table entry that moves is a red row.

``tables()`` is the one place that names where the two tables of named-object fields live: step 2
of the plan moves them to core, and this function is the single line to retarget then.
"""

from __future__ import annotations

import json
from typing import Any

from fsm_llm.core.names import NameRegistry
from fsm_llm.workflows import (
    AutoStep,
    ParallelStep,
    RetryStep,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    create_workflow,
    executable_step,
    load_workflow,
)
from tests.test_fsm_llm_workflows.support import (
    callables_registry,
    every_step_objects,
    every_step_workflow,
)


class GoldenPingStep(WorkflowStep):
    """A custom step class: its type tag is its class name; ``action`` is a table field, so named."""

    message: str = "ping"
    next_step_id: str = ""
    action: Any = None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result(next_step_id=self.next_step_id)


class _Targets:
    """The ``registry`` an ``ExecutableStep`` names: anything with ``get``."""

    def get(self, target: str) -> Any:
        raise KeyError(target)


def _noop(context: dict[str, Any]) -> dict[str, Any]:
    return {}


def tables() -> dict[str, list[str]]:
    """The two tables of named-object fields, sorted (the seam step 2 retargets)."""
    from fsm_llm.workflows import definitions

    return {
        "named_object_fields": sorted(definitions._NAMED_OBJECT_FIELDS),
        "step_callable_exclude_fields": sorted(
            definitions._STEP_CALLABLE_EXCLUDE_FIELDS
        ),
    }


def _all_steps() -> tuple[Any, NameRegistry[Any], list[type[WorkflowStep]]]:
    objects = every_step_objects()
    return every_step_workflow(objects), callables_registry(objects), []


def _all_steps_custom() -> tuple[Any, NameRegistry[Any], list[type[WorkflowStep]]]:
    objects = every_step_objects()
    hook = _noop
    objects["hook"] = hook
    custom = GoldenPingStep(
        step_id="s_custom", name="Ping", next_step_id="s_cond", action=hook
    )
    return (
        every_step_workflow(objects, custom=custom),
        callables_registry(objects),
        [GoldenPingStep],
    )


def _terminal() -> tuple[Any, NameRegistry[Any], list[type[WorkflowStep]]]:
    wf = create_workflow("wf_terminal", "Terminal", "an empty last step")
    wf.with_initial_step(auto_step("only", "Only", next_step_id=""))
    return wf, callables_registry({}), []


def _nested() -> tuple[Any, NameRegistry[Any], list[type[WorkflowStep]]]:
    objects = {"inner": _noop, "leaf": lambda context: {"leaf": True}}
    wf = create_workflow("wf_nested", "Nested", "retry in retry, parallel of retry")
    wf.with_initial_step(
        RetryStep(
            step_id="outer",
            name="Outer",
            step=RetryStep(
                step_id="middle",
                name="Middle",
                step=AutoStep(
                    step_id="inner",
                    name="Inner",
                    next_step_id="par",
                    action=objects["inner"],
                ),
                max_retries=1,
            ),
            max_retries=3,
            backoff_factor=2.0,
        )
    )
    wf.with_step(
        ParallelStep(
            step_id="par",
            name="Par",
            steps=[
                RetryStep(
                    step_id="p_retry",
                    name="PRetry",
                    step=AutoStep(
                        step_id="p_leaf",
                        name="PLeaf",
                        next_step_id="end",
                        action=objects["leaf"],
                    ),
                ),
                AutoStep(step_id="p_plain", name="PPlain", next_step_id="end"),
            ],
            next_step_id="end",
        )
    )
    wf.with_step(auto_step("end", "End", next_step_id=""))
    return wf, callables_registry(objects), []


def _executable() -> tuple[Any, NameRegistry[Any], list[type[WorkflowStep]]]:
    targets = _Targets()
    objects = {"targets": targets, "bound": _noop}
    wf = create_workflow("wf_exec", "Exec", "runs an executable by id")
    wf.with_initial_step(
        executable_step(
            "run",
            "Run",
            "strategy.demo",
            targets,
            inputs={"question": "q"},
            returns={"answer": "answer"},
            success_step_id="end",
            error_step_id="end",
            callables=None,
            max_steps=7,
            max_seconds=3.5,
        )
    )
    wf.with_step(auto_step("end", "End", next_step_id=""))
    return wf, callables_registry(objects), []


CORPUS = {
    "all_steps": _all_steps,
    "all_steps_with_custom": _all_steps_custom,
    "terminal_auto": _terminal,
    "nested_retry_parallel": _nested,
    "executable_step": _executable,
}


def _text(value: Any) -> str:
    """The bytes of a document: insertion order kept, no key sorting."""
    return json.dumps(value, ensure_ascii=False)


def snapshot() -> dict[str, Any]:
    """The G1 snapshot of the checkout that is imported (see the module docstring)."""
    cases: dict[str, Any] = {}
    for name, build in CORPUS.items():
        workflow, registry, classes = build()
        named = workflow.serialize(callables=registry)
        round_trip_error: str | None = None
        equal = bytes_equal = False
        try:
            loaded = load_workflow(
                json.loads(_text(named)), objects=registry, step_classes=classes
            )
            equal = loaded == workflow
            bytes_equal = _text(loaded.serialize(callables=registry)) == _text(named)
        except Exception as error:  # a moved table or tag is a row, not a crash
            round_trip_error = type(error).__name__
        cases[name] = {
            "serialize": _text(workflow.serialize()),
            "serialize_named": _text(named),
            "round_trip_equal": equal,
            "round_trip_bytes_equal": bytes_equal,
            "round_trip_error": round_trip_error,
        }
    return {"cases": cases, "tables": tables()}
