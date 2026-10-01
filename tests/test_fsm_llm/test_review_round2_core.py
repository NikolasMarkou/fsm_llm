"""Review round 2, core pins (plan 07ad3f8c step 22.3, D-048 item 2).

``findings/review-iter-1-pass4.md`` concern 2: three of the request sites
that must keep "the user sent an empty message" (``""``) apart from "there
is no user message" (``None``) were unpinned. Mutating ``user_message`` to
``user_message or None`` at the Pass-1 field extraction (M3), the bulk
extraction (M4) or the streaming Pass 2 (M6) sent the imperative
``NEUTRAL_USER_TURN`` for ``converse("")`` again, and every core test
passed. These tests drive the real ``API`` and ``LiteLLMInterface`` with
``litellm.completion`` patched, so they see the provider's user turn of
every request, tagged with the ``LLMInterface`` method that sent it.
They fail under M3, M4, M6 and under the reverse mutation (``None`` to
``""``) at the same three sites.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm import constants
from fsm_llm.api import API
from fsm_llm.llm import LiteLLMInterface

_KINDS = (
    "extract_field",
    "extract_bulk_data",
    "generate_response",
    "generate_response_stream",
)


def _message(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.reasoning_content = None
    return response


def _chunks(text: str) -> list[MagicMock]:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    chunk.choices[0].delta.reasoning_content = None
    return [chunk]


class _TaggedProvider:
    """``fsm_llm.llm.completion`` recording ``(LLMInterface method, user
    turn)`` per request; the method is tagged by wrapping the interface's
    four request methods."""

    def __init__(self, llm: LiteLLMInterface) -> None:
        self.turns: list[tuple[str, str]] = []
        self._kind = "?"
        for name in _KINDS:
            setattr(llm, name, self._tagged(name, getattr(llm, name)))

    def _tagged(self, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        def call(request: Any) -> Any:
            self._kind = name
            return method(request)

        return call

    def completion(self, **kwargs: Any) -> Any:
        (user,) = [m["content"] for m in kwargs["messages"] if m["role"] == "user"]
        self.turns.append((self._kind, user))
        if kwargs.get("stream"):
            return iter(_chunks("Hello there"))
        return _message(
            json.dumps(
                {
                    "field_name": "city",
                    "value": None,
                    "confidence": 0.0,
                    "reasoning": "",
                    "extracted_data": {},
                    "message": "Hello there",
                }
            )
        )

    def kinds(self) -> set[str]:
        return {kind for kind, _ in self.turns}

    def sent(self) -> set[str]:
        return {turn for _, turn in self.turns}


def _fsm(ask: dict[str, Any]) -> dict[str, Any]:
    """``ask`` (a speaking state with the given extraction config) gated on
    ``city`` to the terminal ``done``."""
    return {
        "name": "City",
        "description": "Asks for a city",
        "initial_state": "ask",
        "states": {
            "ask": {
                "id": "ask",
                "description": "Ask for the city",
                "purpose": "Learn the city",
                "response_instructions": "Ask which city the user means.",
                **ask,
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "The city is known",
                        "conditions": [
                            {
                                "description": "city is set",
                                "requires_context_keys": ["city"],
                                "logic": {"!!": [{"var": "city"}]},
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "Done",
                "purpose": "Wrap up",
                "response_instructions": "Say goodbye.",
            },
        },
    }


_FIELD = {
    "field_extractions": [
        {
            "field_name": "city",
            "field_type": "str",
            "extraction_instructions": "The city the user names.",
        }
    ]
}
_BULK = {"extraction_instructions": "Extract the city the user names as 'city'."}

_SITES = [
    pytest.param(_FIELD, "converse", "extract_field", id="field-sync"),
    pytest.param(_FIELD, "stream", "extract_field", id="field-stream"),
    pytest.param(_BULK, "converse", "extract_bulk_data", id="bulk-sync"),
    pytest.param(_BULK, "stream", "extract_bulk_data", id="bulk-stream"),
    pytest.param(_FIELD, "stream", "generate_response_stream", id="pass2-stream"),
    pytest.param(_BULK, "stream", "generate_response_stream", id="pass2-stream-bulk"),
]


def _run(ask: dict[str, Any], mode: str, message: str | None) -> _TaggedProvider:
    """Start a conversation, then run one turn (``message``) or one step
    (``None``) in ``mode``; return the provider record of that turn only."""
    llm = LiteLLMInterface(model="gpt-4o", api_key="sk-x")
    provider = _TaggedProvider(llm)
    with (
        patch("fsm_llm.llm.completion", side_effect=provider.completion),
        patch(
            "fsm_llm.llm.get_supported_openai_params",
            return_value=["response_format"],
        ),
    ):
        api = API(_fsm(ask), llm_interface=llm)
        cid, _ = api.start_conversation()
        provider.turns.clear()
        if message is None:
            if mode == "stream":
                list(api.advance_stream(cid))
            else:
                api.advance(cid)
        elif mode == "stream":
            list(api.converse_stream(message, cid))
        else:
            api.converse(message, cid)
    return provider


class TestEmptyMessageSitesSendThePlaceholder:
    """RED under M3 (field), M4 (bulk), M6 (streaming Pass 2): the request
    carried ``None`` for ``converse("")`` and the provider got the neutral
    instruction."""

    @pytest.mark.parametrize("message", ["", "   "])
    @pytest.mark.parametrize(("ask", "mode", "kind"), _SITES)
    def test_an_empty_message_reaches_every_request_as_the_placeholder(
        self, ask: dict[str, Any], mode: str, kind: str, message: str
    ):
        provider = _run(ask, mode, message)

        assert kind in provider.kinds()
        assert provider.sent() == {constants.EMPTY_USER_MESSAGE_TURN}


class TestNoMessageSitesSendTheNeutralTurn:
    """The reverse mutation (``None`` sent as ``""``) at the same three
    sites: a message-free step would tell the model the user sent an empty
    message."""

    @pytest.mark.parametrize(("ask", "mode", "kind"), _SITES)
    def test_a_step_reaches_every_request_as_the_neutral_turn(
        self, ask: dict[str, Any], mode: str, kind: str
    ):
        provider = _run(ask, mode, None)

        assert kind in provider.kinds()
        assert provider.sent() == {constants.NEUTRAL_USER_TURN}


def test_a_real_message_is_sent_unchanged_at_every_site():
    for ask in (_FIELD, _BULK):
        for mode in ("converse", "stream"):
            provider = _run(ask, mode, "Paris, please")
            assert provider.sent() == {"Paris, please"}
