"""``CompletionExecutor``: the completion kind, Pass 1 is one ``LLMInterface.complete`` call.

Layer ``engine`` (plan d454fd62, step 80): split out of the old ``fsm_llm.executors``; the
class and its D-021 and D-029 anchors are moved unchanged, except that ``pipeline`` is typed
``_PipelineServices`` instead of ``Any``. Interface contract: the package docstring of
``fsm_llm.engine.kinds`` (``StateExecutor``) and ``CompletionExecutor.turn`` below.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from fsm_llm.aio import acall, run_sync
from fsm_llm.constants import COMPLETION_TOOL_CHOICE_KEYWORDS
from fsm_llm.context import record_cut_off, was_cut_off
from fsm_llm.core.definitions import (
    CompletionRequest,
    CompletionStateConfig,
    DataExtractionResponse,
    FSMError,
    FSMInstance,
    LLMResponseError,
    State,
)
from fsm_llm.llm import LLMInterface, check_tool_transcript
from fsm_llm.logging import logger
from fsm_llm.runtime import allm_call

from .base import StateExecutor
from .services import _PipelineServices


class CompletionExecutor(StateExecutor):
    """The completion kind: Pass 1 is one ``complete`` call (D-003 of plan
    944e2692). No field, bulk or classification extraction runs beside it, and
    arriving in a completion state runs no call until the next turn."""

    # Why False: a completion state's data is its one typed result and is
    # committed as is (the D-021 anchor at the call site in ``engine/turn/pass1.py``,
    # ``_execute_extraction_and_transition_pass``, holds the rule). Do NOT set it
    # to True and do NOT clean the data elsewhere: ``clean_context_keys`` would
    # drop ``text: None`` and a tool argument such as ``{"_id": 1}`` from the
    # calls the consumer runs.
    cleans_extracted = False
    kind = "completion"

    def extract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        return run_sync(
            self.aextract(pipeline, instance, user_message, conversation_id, state)
        )

    async def aextract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        # A completion state's Pass-1 work is its one completion call (D-003);
        # no field, bulk or classification extraction runs beside it.
        # An explicit error, never ``assert`` (stripped under ``python -O``): the
        # dispatch picks this executor only for a state with a ``completion``
        # config, so a state without one is a mis-registered or hand-built case.
        if state.completion is None:
            raise FSMError(
                f"CompletionExecutor.extract needs a state with a completion "
                f"config; state '{state.id}' has none"
            )
        response: DataExtractionResponse = await acall(
            self,
            "turn",
            pipeline.llm_interface,
            instance,
            state.completion,
            conversation_id,
        )
        instance.last_extraction_response = response
        return response

    @staticmethod
    def turn(
        llm_interface: LLMInterface,
        instance: FSMInstance,
        config: CompletionStateConfig,
        conversation_id: str,
    ) -> DataExtractionResponse:
        """``run_sync`` over ``aturn``, the one body."""
        return run_sync(
            CompletionExecutor.aturn(llm_interface, instance, config, conversation_id)
        )

    @staticmethod
    async def aturn(
        llm_interface: LLMInterface,
        instance: FSMInstance,
        config: CompletionStateConfig,
        conversation_id: str,
    ) -> DataExtractionResponse:
        """Pass-1 work of a completion state: one ``complete`` call.

        Contract: sends exactly ``[system(config.instructions)]`` (when set)
        plus a copy of ``context[config.messages_key]`` (absent or ``None``:
        empty), with ``config.tools`` or the response format read from
        ``context[config.response_format_key]``. Returns the result
        ``{kind, text, calls}`` under ``config.result_key``, or no data when
        that key is already set (skip-if-set: no call). Never touches
        ``Conversation.exchanges`` or the transcript.

        Raises:
            LLMResponseError: the transcript is not a paired list of chat
                messages (or holds no user message on a tool turn), the
                response format is missing or not a dict, the
                interface does not implement ``complete``, or the provider
                call failed. Nothing is committed, so the turn fails with the
                context as it was.
        """
        log = logger.bind(conversation_id=conversation_id)
        data = instance.context.data
        if data.get(config.result_key) is not None:
            log.debug(f"Completion result '{config.result_key}' already set; no call")
            return DataExtractionResponse(extracted_data={}, confidence=1.0)

        # DECISION plan-2026-10-01T093600-944e2692/D-021: the request is the
        # state's instructions plus the consumer's transcript and nothing
        # else. Do NOT add prompt-builder text, history, rendered context or a
        # neutral user turn here (native_fc's request bytes are pinned by a
        # golden test), and do NOT drop or repair an unpaired transcript: an
        # assistant tool-call message without its results is refused, never
        # sent and never silently trimmed. See D-021.
        transcript = data.get(config.messages_key)
        if transcript is None:
            transcript = []
        if not isinstance(transcript, list):
            raise LLMResponseError(
                f"Completion transcript '{config.messages_key}' is "
                f"{type(transcript).__name__}, not a list of chat messages"
            )
        check_tool_transcript(transcript)
        # DECISION plan-2026-10-01T093600-944e2692/D-029 (narrowed by D-039):
        # a tool turn whose transcript holds no user message has no task:
        # refuse it. The transcript is an internal key that session files do
        # not carry, so a session restored in the middle of a tool loop
        # arrives here empty, or holding only the tool exchange a runner
        # rebuilt after the restore; sending it made the model's answer to
        # nothing the result. Do NOT reduce this to "non-empty" (an assistant
        # tool call plus its result passed that), do NOT send it, and do NOT
        # persist the transcript to "fix" this (the session format is
        # unchanged). See D-029, D-039.
        if config.tools is not None and not any(
            message.get("role") == "user" for message in transcript
        ):
            raise LLMResponseError(
                f"Completion transcript '{config.messages_key}' has no user "
                "message: a tool turn needs the task in its transcript (a "
                "session restored in the middle of a tool loop does not carry "
                "its transcript)"
            )
        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": config.instructions}]
            if config.instructions is not None
            else []
        )
        messages.extend(copy.deepcopy(transcript))

        response_format: dict[str, Any] | None = None
        if config.response_format_key is not None:
            response_format = data.get(config.response_format_key)
            if not isinstance(response_format, dict) or not response_format:
                raise LLMResponseError(
                    f"Completion response format '{config.response_format_key}' "
                    "is missing or not a dict"
                )
        tool_choice: str | dict[str, Any] | None = config.tool_choice
        if tool_choice is not None and tool_choice not in (
            COMPLETION_TOOL_CHOICE_KEYWORDS
        ):
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        try:
            request = CompletionRequest(
                messages=messages,
                tools=config.tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
        except ValidationError as e:
            raise LLMResponseError(
                f"Completion state request is invalid: {e.error_count()} error(s), "
                f"first: {e.errors()[0]['msg']}"
            ) from e
        try:
            async with allm_call(llm_interface, "completion", request) as call:
                reply = await acall(llm_interface, "complete", request)
                await call.aresponse(reply)
        except NotImplementedError as e:
            raise LLMResponseError(
                f"{type(llm_interface).__name__} does not implement "
                "complete, which a completion state needs"
            ) from e
        log.debug(
            f"Completion turn: kind={reply.kind}, calls={[c.name for c in reply.calls]}"
        )
        if was_cut_off(reply):
            record_cut_off(instance, field=config.result_key)
        return DataExtractionResponse(
            # `finish_reason` stays out of the committed result: its shape is
            # the documented {kind, text, calls} (the provider's reason is on
            # the CompletionResponse and in the WARNING for a cut reply).
            extracted_data={
                config.result_key: reply.model_dump(
                    mode="json", exclude={"finish_reason"}
                )
            },
            confidence=1.0,
        )
