"""State executors: what a state's kind does in Pass 1 of a turn.

The turn pipeline (``pipeline.py``) is kind-agnostic: at the one Pass-1 seam it
asks ``executor_for(state)`` for the state's executor and calls it. An executor
owns the three things that differ between kinds (roadmap row 0.14, plan
d454fd62, D-004, D-015):

- ``extract``: the state's Pass-1 extraction, a ``DataExtractionResponse``.
- ``cleans_extracted``: whether the extracted data goes through
  ``clean_context_keys`` before it is committed.
- ``after_transition``: what runs after a transition INTO a state of the kind
  (the conversation kind re-extracts what the same message holds for the new
  state; the other kinds do nothing).

Three kinds, three registrations: ``conversation`` (the retained field, bulk
and classification extraction of ``MessagePipeline``), ``completion`` (one
``LLMInterface.complete`` call over a consumer-owned transcript, moved here from
the pipeline unchanged) and ``synthesize`` (passive: the conversation executor,
because a runner outside core pushes the compiled procedure and a bare
``converse`` on such a state stays BLOCKED, D-002 of plan 2cefeb61; a later
synthesize executor replaces it).

Interface contract:

- ``StateExecutor.extract(pipeline, instance, user_message, conversation_id,
  state)``: ``pipeline`` is the calling ``MessagePipeline`` (the executor may
  call the retained methods the conversation kind wraps, and reads
  ``pipeline.llm_interface``); ``state`` is the instance's current state. Must
  set ``instance.last_extraction_response`` to what it returns, take no snapshot
  of its own (Pass 1 runs outside the PRE and POST atomicity blocks of the turn
  body; the anchors in ``pipeline.py`` hold the rule) and may raise
  ``LLMResponseError``.
- ``StateExecutor.after_transition(pipeline, instance, user_message,
  conversation_id, extraction_response, previous_state, new_state)``: called
  once after a transition that ended in ``new_state`` (not for an agent-managed
  FSM); may update ``extraction_response`` and the context; returns ``None``.
- ``executor_for(state)``: the executor of ``kind_of(state).name``; a kind with
  no registered executor runs as ``conversation`` (a passive kind).
- ``register_executor(kind_name, executor)``: add or replace the executor of a
  kind (a test kind registers its own beside its ``KindSpec``).

This is an engine module: it imports core modules and never ``pipeline.py``
(the pipeline imports it; the one helper both need, the cut-off record, lives in
``context.py``). ``fsm_llm/__init__.py`` does not import it; the registry is
internal until the kind extension point is public (roadmap C10).
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from .constants import COMPLETION_TOOL_CHOICE_KEYWORDS
from .context import record_cut_off, was_cut_off
from .definitions import (
    CompletionRequest,
    CompletionStateConfig,
    DataExtractionResponse,
    FSMInstance,
    LLMResponseError,
    State,
)
from .llm import LLMInterface, check_tool_transcript
from .logging import logger
from .state_kinds import kind_of

__all__ = [
    "CompletionExecutor",
    "ConversationExecutor",
    "StateExecutor",
    "executor_for",
    "register_executor",
]


class StateExecutor:
    """What the turn pipeline calls for a state of one kind (see the module
    docstring for the contract). Subclass it and register the instance."""

    # Whether the extracted data is passed through ``clean_context_keys`` before
    # it is committed (it drops None values and internal-prefixed keys).
    cleans_extracted: bool = True

    def extract(
        self,
        pipeline: Any,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        raise NotImplementedError

    def after_transition(
        self,
        pipeline: Any,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        """Nothing runs after a transition into this kind of state."""
        return None


class ConversationExecutor(StateExecutor):
    """The conversation kind: field, bulk and classification extraction.

    Wraps the retained methods of ``MessagePipeline`` (they move out of the
    pipeline with the layout split, roadmap 0.4); also the passive executor of
    ``synthesize``.
    """

    def extract(
        self,
        pipeline: Any,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        response: DataExtractionResponse = pipeline._execute_data_extraction(
            instance, user_message, conversation_id
        )
        return response

    def after_transition(
        self,
        pipeline: Any,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        pipeline._extract_after_transition(
            instance,
            user_message,
            conversation_id,
            extraction_response,
            previous_state,
            new_state,
        )


class CompletionExecutor(StateExecutor):
    """The completion kind: Pass 1 is one ``complete`` call (D-003 of plan
    944e2692). No field, bulk or classification extraction runs beside it, and
    arriving in a completion state runs no call until the next turn."""

    # Why False: a completion state's data is its one typed result and is
    # committed as is (the D-021 anchor at the call site in ``pipeline.py``,
    # ``_execute_extraction_and_transition_pass``, holds the rule). Do NOT set it
    # to True and do NOT clean the data elsewhere: ``clean_context_keys`` would
    # drop ``text: None`` and a tool argument such as ``{"_id": 1}`` from the
    # calls the consumer runs.
    cleans_extracted = False

    def extract(
        self,
        pipeline: Any,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        # A completion state's Pass-1 work is its one completion call (D-003);
        # no field, bulk or classification extraction runs beside it.
        assert state.completion is not None
        response = self.turn(
            pipeline.llm_interface, instance, state.completion, conversation_id
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
            reply = llm_interface.complete(request)
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


_CONVERSATION = ConversationExecutor()

# Kind name -> executor. ``synthesize`` is passive: it shares the conversation
# executor (a bare ``converse`` on it stays BLOCKED, D-002 of plan 2cefeb61).
_EXECUTORS: dict[str, StateExecutor] = {
    "conversation": _CONVERSATION,
    "completion": CompletionExecutor(),
    "synthesize": _CONVERSATION,
}


def register_executor(kind_name: str, executor: StateExecutor) -> None:
    """Register (or replace) the executor of the kind named ``kind_name``."""
    _EXECUTORS[kind_name] = executor


def executor_for(state: State) -> StateExecutor:
    """The executor of ``state``'s kind (``conversation`` for a kind with none)."""
    return _EXECUTORS.get(kind_of(state).name, _CONVERSATION)
