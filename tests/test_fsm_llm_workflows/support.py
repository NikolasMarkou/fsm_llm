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


def _single(step) -> Any:
    wf = create_workflow("wf", "wf")
    wf.with_initial_step(step)
    return wf


class _Gen:
    def generate(self, prompt: str) -> str:
        return "x"
