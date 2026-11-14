"""The turn-by-turn session of the meta-builder agent (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

``start`` opens a session and never builds, ``send`` adds a user turn (``_turn``: ``converse``, then one
``advance`` when it is due; ``_build_turn`` reports the build outcome; ``_record_failed_build`` routes a
failed build step to ``build_failed``; ``_collect_fallback`` is the canned reply when the reply call
failed). ``_with_build_prompt`` (module level) closes every collect reply with ``META_BUILD_PROMPT``
(D-024, D-025). ``SessionMixin`` holds the methods.
"""

from __future__ import annotations

from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import (
    META_BUILD_CALL_FAILED,
    META_BUILD_PROMPT,
    MetaBuildOutcome,
    MetaErrorMessages,
)
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.exceptions import BuilderError, MetaBuilderError
from fsm_llm.agents.meta_prompts import build_welcome_message
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.logging import logger

from .hints import _normalize_message
from .names import _K, _S, _TYPE_VALUES
from .surface import _MetaBuilderSurface

# Variance a model's own copy of the sentence may carry and still count as
# it: any quote style, markdown emphasis, "you are", case and whitespace.
_QUOTES = str.maketrans(
    {"‘": "'", "’": "'", '"': "'", "“": "'", "”": "'", "*": None, "_": None}
)


def _prompt_text(text: str) -> str:
    """``text`` normalized for finding the build prompt in it."""
    return _normalize_message(text, _QUOTES).replace("you are", "you're")


_BUILD_PROMPT_CORE = _prompt_text(META_BUILD_PROMPT).rstrip(".!")


def _with_build_prompt(reply: str) -> str:
    """A collect reply that carries ``META_BUILD_PROMPT`` exactly once.

    A reply already containing the sentence anywhere (up to the variance
    above, so a paragraph after it does not count as a missing sentence) is
    returned unchanged; otherwise the sentence is appended as a new final
    line.
    """
    if _BUILD_PROMPT_CORE in _prompt_text(reply):
        return reply
    body = reply.rstrip()
    return f"{body}\n{META_BUILD_PROMPT}" if body else META_BUILD_PROMPT


class SessionMixin(_MetaBuilderSurface):
    """The turn-by-turn session: start, send, one user turn, the build turn and the canned fallbacks."""

    def start(self, initial_message: str = "") -> str:
        """Initialize a builder session; never builds.

        With a message, the first turn classifies it and returns the
        generated collect reply, whatever the message says ("build it"
        included); without one, the welcome text.
        """
        if self._api is not None:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_ALREADY_STARTED)
        api = self._create_api()
        # The session is set only once the conversation exists: a failed
        # start leaves the object unstarted, so ``start`` can be retried.
        self._conversation_id, _ = api.start_conversation(
            initial_context=self._turn_hints("", [], build_requested=False)
        )
        self._api = api
        if not initial_message:
            return build_welcome_message()
        return self._turn(initial_message, build_requested=False)

    def send(self, message: str) -> str:
        """Send a message in a turn-by-turn session."""
        self._session()
        if self.is_complete():
            raise MetaBuilderError("Session has already completed")

        self._turn_count += 1
        if self._turn_count > self.meta_config.max_turns:
            raise MetaBuilderError(
                f"Maximum turns ({self.meta_config.max_turns}) exceeded"
            )
        return self._turn(message, build_requested=None)

    def _session(self) -> tuple[API, str]:
        """The started session's API and conversation id."""
        if self._api is None or self._conversation_id is None:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_NOT_STARTED)
        return self._api, self._conversation_id

    def _session_data(self) -> dict[str, Any]:
        """The session's public context, ``{}`` without a session."""
        if self._api is None or self._conversation_id is None:
            return {}
        data: dict[str, Any] = self._api.get_data(self._conversation_id)
        return data

    def _turn(self, message: str, *, build_requested: bool | None) -> str:
        """One user turn: ``converse``, then one ``advance`` when it is due.

        The turn ends in ``classify`` after a type switch (the message-free
        step classifies ``latest_request`` and the collect reply follows) or
        in ``build`` (the step makes the build call).
        """
        api, conversation_id = self._session()
        requirements = [*(self._session_data().get(_K.REQUIREMENTS) or []), message]
        api.update_context(
            conversation_id,
            self._turn_hints(message, requirements, build_requested=build_requested),
        )
        try:
            reply = api.converse(message, conversation_id)
            state = api.get_current_state(conversation_id)
            if state == _S.CLASSIFY:
                step = api.advance(conversation_id)
                reply = step.response or ""
                state = step.state_after
        except LLMResponseError as e:
            # The turn was rolled back; the requirement is kept in context.
            logger.warning(f"Collect reply failed, using the canned reply: {e}")
            return self._collect_fallback(message)
        if state == _S.COLLECT:
            # DECISION plan-2026-10-01T093600-944e2692/D-025: the build prompt
            # is appended to the reply the user gets, not to core's stored
            # reply. Do NOT rewrite `Conversation.exchanges` from here (no
            # public core API does it; reaching into the instance would be a
            # driver-side history editor) and do NOT add more prompt wording
            # instead (the 4b model drops the sentence regardless, D-024).
            # History keeps what the model wrote. See decisions.md D-025.
            return _with_build_prompt(reply)
        if state != _S.BUILD:
            return reply
        return self._build_turn(api, conversation_id)

    def _build_turn(self, api: API, conversation_id: str) -> str:
        """The build step of a ``send``; the reply reports its outcome."""
        try:
            self._advance_build(api, conversation_id)
        except BuilderError as e:
            logger.error(f"Build execution failed: {e}")
            self._record_failed_build(api, conversation_id, e)
        data = api.get_data(conversation_id)
        if self.is_complete():
            presentation = data.get(_K.REVIEW_PRESENTATION) or ""
            return (
                f"Build complete!\n\n{presentation}\n\n"
                f"The artifact JSON has been generated."
            )

        # Build produced validation errors — keep session open.
        errors = data.get(_K.VALIDATION_ERRORS) or ["Build failed"]
        error_list = "\n".join(f"  - {e}" for e in errors)
        return (
            f"I couldn't complete the build yet:\n{error_list}\n\n"
            f"Please provide the missing information, then say 'build it' again."
        )

    @staticmethod
    def _record_failed_build(
        api: API, conversation_id: str, error: BuilderError
    ) -> None:
        """Route a ``build`` state whose step failed to ``build_failed``.

        The failed step left the conversation in ``build`` with no result.
        Recording the failure as the result (``kind`` ``call_failed``) makes
        the next step a no-call step that takes the ``build_failed`` edge, so
        the session continues like any failed build.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-023: the failure is
        # recorded as the build state's result so the FSM's own edge takes the
        # session to `build_failed`. Do NOT leave the conversation in `build`
        # (the next `converse` there would make a build call whatever the user
        # said) and do NOT jump states with `set_conversation_state` (a
        # transition outside the FSM's rules). See decisions.md D-023.
        cause = error.__cause__
        message = (
            f"The build call failed: {error}"
            if isinstance(cause, LLMResponseError)
            else f"The build reply could not be assembled: {error}"
        )
        api.update_context(
            conversation_id,
            {
                _K.BUILD_REPLY: {
                    "kind": META_BUILD_CALL_FAILED,
                    "text": message,
                    "calls": [],
                },
                _K.BUILD_OUTCOME: MetaBuildOutcome.INVALID,
                _K.VALIDATION_ERRORS: [message],
            },
        )
        api.advance(conversation_id)

    def _collect_fallback(self, message: str) -> str:
        """The canned collect reply used when the reply call failed."""
        value = self._session_data().get(_K.ARTIFACT_TYPE)
        artifact_type = (
            ArtifactType(value)
            if value in _TYPE_VALUES
            else self._detect_type_fallback(message)
        )
        return (
            f"Got it: added to your {artifact_type.value.upper()} spec. "
            f"You can keep adding details. {META_BUILD_PROMPT}"
        )
