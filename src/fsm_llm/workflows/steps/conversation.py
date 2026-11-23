"""The step that runs an FSM conversation: ``ConversationStep``.

``API`` is read at call time (``API.from_definition``, ``API.from_file`` in ``_run_conversation``), so a test patches
``fsm_llm.workflows.steps.conversation.API``: the module that performs the lookup, never the package.

The whole conversation is ONE awaited coroutine on the loop (plan ace7a8c8 step 26, X5.1): the greeting and every
turn go through ``API.astart_conversation`` and ``aconverse``, with no executor thread. A step has no sync mode, so it
calls the async names directly, never through ``acall``. The time limit cancels the turn in flight (the core rolls it
back whole, f150ec28 D-002) and ``end_conversation`` runs in ``finally`` on every exit, a cancel included.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import Field, field_validator, model_validator

from fsm_llm import API
from fsm_llm.logging import logger

from ..constants import CONTEXT_KEY_USER_INPUT
from ..models import WorkflowStepResult
from .base import WorkflowStep
from .helpers import _error_text


class ConversationStep(WorkflowStep):
    """A step that runs an FSM conversation using the fsm_llm API.

    This enables deep integration between workflows and FSM-LLM core:
    workflows can invoke full FSM conversations (including reasoning)
    as steps in a larger workflow.

    The conversation is started, driven by ``auto_messages`` (at most
    ``max_turns`` turns; plus the ``advance`` user input when
    ``use_user_input`` is set), and its collected data is returned. It is NOT
    guaranteed to reach a terminal state: ``conversation_<step_id>_ended``
    reports whether it did, and ``require_completion=True`` turns a
    conversation that did not end into a failure.

    Exactly one of ``fsm_file`` / ``fsm_definition`` (a dict or an
    ``FSMDefinition``) must be given. ``initial_context`` maps
    ``{conversation_key: workflow_key}`` (input); ``context_mapping`` maps
    ``{workflow_key: conversation_key}`` (output). The effective time limit
    is the smaller of ``timeout_seconds`` and ``conversation_timeout``. Failures
    route to ``error_step_id``, or FAIL the instance when none is set.
    """

    fsm_file: str | None = None
    fsm_definition: Any | None = None
    model: str | None = None
    initial_context: dict[str, str] = Field(default_factory=dict)
    context_mapping: dict[str, str] = Field(default_factory=dict)
    success_step_id: str = ""
    error_step_id: str | None = None
    max_turns: int = 20
    conversation_timeout: float | None = None
    require_completion: bool = False
    use_user_input: bool = False

    @field_validator("max_turns")
    @classmethod
    def _validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be >= 1")
        return v

    @field_validator("fsm_definition")
    @classmethod
    def _validate_fsm_definition(cls, v: Any) -> Any:
        if v is None or isinstance(v, dict) or callable(getattr(v, "model_dump", None)):
            return v
        raise ValueError(
            f"fsm_definition must be a dict or an FSMDefinition, got {type(v).__name__}"
        )

    @model_validator(mode="after")
    def _validate_fsm_source(self):
        has_def = self.fsm_definition is not None
        has_file = bool(self.fsm_file)
        if has_def == has_file:
            raise ValueError(
                "ConversationStep requires either fsm_file or fsm_definition "
                "(exactly one)"
            )
        return self

    auto_messages: list[str] = Field(default_factory=list)

    def _effective_timeout(self) -> float | None:
        limits = [
            t
            for t in (self.timeout_seconds, self.conversation_timeout)
            if t is not None
        ]
        return min(limits) if limits else None

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute an FSM conversation and return collected data."""
        limit = self._effective_timeout()
        raised = False

        async def conversation() -> WorkflowStepResult:
            nonlocal raised
            try:
                return await self._run_conversation(dict(context))
            except TimeoutError:
                # raised by the conversation itself: a failure, not this step's time limit
                raised = True
                raise

        try:
            # `wait_for` cancels the awaited turn at the limit: a real cancel, rolled back by the
            # core, and `_run_conversation` still ends the conversation in its `finally`.
            return await asyncio.wait_for(conversation(), timeout=limit)
        except TimeoutError as e:
            if raised:
                return self._failed(e)
            logger.error(f"ConversationStep [{self.step_id}] timed out after {limit}s")
            return WorkflowStepResult.failure_result(
                error=f"Conversation timed out after {limit}s",
                next_step_id=self.error_step_id,
                message="Conversation timed out",
            )
        except Exception as e:
            return self._failed(e)

    def _failed(self, e: Exception) -> WorkflowStepResult:
        logger.error(f"Error in conversation step {self.step_id}: {e!s}")
        return WorkflowStepResult.failure_result(
            error=_error_text(e),
            next_step_id=self.error_step_id,
            message=f"Conversation failed: {e!s}",
        )

    async def _run_conversation(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Run the FSM conversation loop as one coroutine on the loop (no executor thread).

        Readers (``has_conversation_ended``, ``get_data``) and ``end_conversation`` are called
        inline from the task that ran the turn, so the turn gate is free for them.
        """
        # Build initial context from workflow context using mapping
        conv_context: dict[str, Any] = {}
        for conv_key, workflow_key in self.initial_context.items():
            if workflow_key in context:
                conv_context[conv_key] = context[workflow_key]

        # Create API instance
        if self.fsm_definition is not None:
            fsm = API.from_definition(self.fsm_definition, model=self.model)
        else:
            # The model validator guarantees fsm_file is set here.
            fsm = API.from_file(str(self.fsm_file), model=self.model)

        messages = list(self.auto_messages)
        user_input = context.get(CONTEXT_KEY_USER_INPUT)
        if self.use_user_input and isinstance(user_input, str) and user_input:
            messages.append(user_input)

        # Start conversation
        conv_id, response = await fsm.astart_conversation(initial_context=conv_context)
        logger.info(
            f"ConversationStep [{self.step_id}] started conversation: "
            f"{str(response or '')[:100]}"
        )
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-039
        # `last_response` / `final_answer` hold the last SPOKEN reply. Do NOT
        # store the last turn's return value as is: a turn that ends on a
        # silent state (no `response_instructions`) returns "", and that
        # would overwrite the reply a workflow maps out of the conversation.
        last_spoken = response or None

        try:
            # Drive the conversation with the messages
            turn = 0
            for message in messages:
                if fsm.has_conversation_ended(conv_id) or turn >= self.max_turns:
                    break
                response = await fsm.aconverse(
                    user_message=message, conversation_id=conv_id
                )
                if response:
                    last_spoken = response
                logger.debug(
                    f"ConversationStep [{self.step_id}] turn {turn}: "
                    f"{str(response or '')[:100]}"
                )
                turn += 1

            ended = bool(fsm.has_conversation_ended(conv_id))
            # Collect results
            collected_data = fsm.get_data(conv_id)
            # Inject the last spoken reply so context_mapping can reference
            # it; a conversation that never spoke adds neither key.
            if last_spoken is not None:
                collected_data.setdefault("last_response", last_spoken)
                collected_data.setdefault("final_answer", last_spoken)
        finally:
            fsm.end_conversation(conv_id)

        # Map collected data back to workflow context
        output_data: dict[str, Any] = {}
        for workflow_key, conv_key in self.context_mapping.items():
            if conv_key in collected_data:
                output_data[workflow_key] = collected_data[conv_key]
            else:
                logger.warning(
                    f"ConversationStep [{self.step_id}] context_mapping key "
                    f"'{conv_key}' not found in collected data. "
                    f"Available keys: {list(collected_data.keys())}"
                )
        # Also include raw collected data under a namespaced key.
        # NOTE (W-ISSUE-001): key must NOT start with "_" — the workflow engine
        # filters out underscore-prefixed keys unless whitelisted.
        output_data[f"conversation_{self.step_id}_data"] = collected_data
        output_data[f"conversation_{self.step_id}_ended"] = ended

        if self.require_completion and not ended:
            return WorkflowStepResult(
                success=False,
                data=output_data,
                error=f"Conversation did not reach a terminal state in {turn} turns",
                next_step_id=self.error_step_id,
                message="Conversation did not complete",
            )

        return WorkflowStepResult.success_result(
            data=output_data,
            next_step_id=self.success_step_id,
            message=f"Conversation completed in {turn} turns",
        )
