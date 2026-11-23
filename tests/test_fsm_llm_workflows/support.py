"""Helpers shared by several test files of this suite. Not collected (no test_ prefix)."""

from __future__ import annotations

import asyncio
from typing import Any

from fsm_llm.workflows import auto_step, create_workflow, wait_event_step
from fsm_llm.workflows.definitions import WorkflowDefinition


def _wait_then_done(
    workflow_id: str,
    event_type: str = "go",
    **wait_kwargs: Any,
) -> WorkflowDefinition:
    """wait(event) -> done (terminal)."""
    wf = create_workflow(workflow_id, workflow_id)
    wf.with_initial_step(
        wait_event_step(
            "wait", "Wait", event_type, success_step_id="done", **wait_kwargs
        )
    )
    wf.with_step(auto_step("done", "Done", next_step_id=""))
    return wf


async def _settle(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


class _FakeLLMResponse:
    def __init__(self, message: str) -> None:
        self.message = message


class _CoreLikeLLM:
    """Shaped like fsm_llm.LLMInterface: sync generate_response only."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[Any] = []

    def generate_response(self, request: Any) -> _FakeLLMResponse:
        self.requests.append(request)
        return _FakeLLMResponse(self.reply)


def _with_async_twins(api: Any) -> Any:
    """Give a MagicMock API the async twins a ``ConversationStep`` awaits (X5.1, pin A-6).

    ``astart_conversation`` and ``aconverse`` delegate to the sync mocks, so ``start_conversation``
    and ``converse`` keep every call count, argument record and ``side_effect`` the rows assert.
    A bare ``MagicMock`` cannot be awaited; the real ``API`` has both twins.
    """

    async def astart_conversation(*args: Any, **kwargs: Any) -> Any:
        return api.start_conversation(*args, **kwargs)

    async def aconverse(*args: Any, **kwargs: Any) -> Any:
        return api.converse(*args, **kwargs)

    api.astart_conversation = astart_conversation
    api.aconverse = aconverse
    return api


def _single(step) -> Any:
    wf = create_workflow("wf", "wf")
    wf.with_initial_step(step)
    return wf


class _Gen:
    def generate(self, prompt: str) -> str:
        return "x"


# ----------------------------------------------------------------
# One definition holding every built-in step class (plan ace7a8c8 step 25)
# ----------------------------------------------------------------


def _act(context: dict[str, Any]) -> dict[str, Any]:
    return {"acted": True}


def _child_act(context: dict[str, Any]) -> dict[str, Any]:
    return {"child": True}


def _inner_act(context: dict[str, Any]) -> dict[str, Any]:
    return {"inner": True}


def _cond(context: dict[str, Any]) -> bool:
    return True


def _api(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True}


def _agg(results: list[Any]) -> dict[str, Any]:
    return {"n": len(results)}


class _Llm:
    """A step LLM: any object with ``generate(prompt)``."""

    def generate(self, prompt: str) -> str:
        return "x"


class _Agent:
    """A step agent: any object with ``run(task)``."""

    def run(self, task: str) -> str:
        return "x"


def every_step_objects() -> dict[str, Any]:
    """The named objects of ``every_step_workflow``, one fresh instance per name."""
    return {
        "act": _act,
        "child_act": _child_act,
        "inner_act": _inner_act,
        "cond": _cond,
        "api": _api,
        "agg": _agg,
        "llm": _Llm(),
        "agent": _Agent(),
    }


def callables_registry(objects: dict[str, Any]) -> Any:
    """A ``callables`` ``NameRegistry`` holding ``objects`` (name to object)."""
    from fsm_llm.core.names import NameRegistry

    registry: NameRegistry[Any] = NameRegistry("callables")
    for name, obj in objects.items():
        registry.register(name, obj)
    return registry


def every_step_workflow(
    objects: dict[str, Any], *, custom: Any = None, object_form: Any = None
) -> Any:
    """A valid workflow holding all 11 built-in step classes.

    Auto with and without ``action``, Condition, API, Switch, LLM, WaitEvent, Timer,
    Parallel with two children and an ``aggregation_function``, Retry with a nested
    step, Conversation with ``fsm_definition`` as a dict and with ``fsm_file``, Agent.
    ``custom`` (a step instance whose ``step_id`` is ``"s_custom"``) is spliced in after
    ``s_auto2``; ``object_form`` (an ``FSMDefinition``) replaces the dict of the first
    conversation step.
    """
    from fsm_llm.workflows import (
        AgentStep,
        APIStep,
        AutoStep,
        ConditionStep,
        ConversationStep,
        LLMStep,
        ParallelStep,
        RetryStep,
        SwitchStep,
        TimerStep,
        WaitEventConfig,
        WaitEventStep,
    )

    steps = [
        AutoStep(
            step_id="s_auto",
            name="Auto",
            description="with an action",
            next_step_id="s_auto2",
            action=objects["act"],
            error_step_id="done",
            timeout_seconds=5.0,
        ),
        AutoStep(
            step_id="s_auto2",
            name="Auto2",
            next_step_id="s_custom" if custom is not None else "s_cond",
        ),
        ConditionStep(
            step_id="s_cond",
            name="Cond",
            condition=objects["cond"],
            true_step_id="s_api",
            false_step_id="s_switch",
        ),
        APIStep(
            step_id="s_api",
            name="Api",
            api_function=objects["api"],
            success_step_id="s_llm",
            failure_step_id="s_switch",
            input_mapping={"q": "query"},
            output_mapping={"answer": "result.text"},
        ),
        SwitchStep(
            step_id="s_switch",
            name="Switch",
            key="intent",
            cases={"a": "s_wait", "b": "s_timer"},
            default_step_id="s_par",
        ),
        WaitEventStep(
            step_id="s_wait",
            name="Wait",
            config=WaitEventConfig(
                event_type="go",
                success_step_id="s_timer",
                timeout_seconds=30.0,
                timeout_step_id="s_par",
                event_mapping={"k": "payload.k"},
                correlation_key="order_id",
            ),
        ),
        TimerStep(
            step_id="s_timer", name="Timer", delay_seconds=2.5, next_step_id="s_par"
        ),
        ParallelStep(
            step_id="s_par",
            name="Par",
            steps=[
                AutoStep(
                    step_id="p1",
                    name="P1",
                    next_step_id="x",
                    action=objects["child_act"],
                ),
                AutoStep(step_id="p2", name="P2", next_step_id="x"),
            ],
            next_step_id="s_retry",
            error_step_id="done",
            aggregation_function=objects["agg"],
        ),
        RetryStep(
            step_id="s_retry",
            name="Retry",
            step=AutoStep(
                step_id="inner",
                name="Inner",
                next_step_id="s_conv",
                action=objects["inner_act"],
            ),
            max_retries=2,
            backoff_factor=0.5,
        ),
        ConversationStep(
            step_id="s_conv",
            name="Conv",
            fsm_definition=object_form if object_form is not None else {"name": "f"},
            model="m",
            initial_context={"a": "b"},
            context_mapping={"c": "d"},
            success_step_id="s_conv2",
            auto_messages=["hi"],
        ),
        ConversationStep(
            step_id="s_conv2",
            name="Conv2",
            fsm_file="flows/a.json",
            success_step_id="s_agent",
        ),
        AgentStep(
            step_id="s_agent",
            name="Agent",
            agent=objects["agent"],
            task_template="do {task}",
            success_step_id="s_llm",
            input_mapping={"k": "v"},
        ),
        LLMStep(
            step_id="s_llm",
            name="Llm",
            llm_interface=objects["llm"],
            prompt_template="say {x}",
            context_mapping={"x": "y"},
            output_mapping={"out": ""},
            next_step_id="done",
        ),
        AutoStep(step_id="done", name="Done", next_step_id=""),
    ]
    if custom is not None:
        steps.append(custom)
    workflow = create_workflow("wf_all", "All steps", description="every class")
    workflow.metadata["owner"] = {"team": "x", "tags": ["a", "b"]}
    for index, step in enumerate(steps):
        workflow.with_step(step, is_initial=index == 0)
    return workflow
