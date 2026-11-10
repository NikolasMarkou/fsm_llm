"""Shared fixtures of the synthesis suite: no network, a valid compile and a
scripted strong model."""

from typing import Any

import pytest

from fsm_llm.definitions import (
    CompletionRequest,
    CompletionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from tests.conftest import block_network


@pytest.fixture(autouse=True)
def _offline_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Refuse IPv4/IPv6 connects (``tests.conftest.block_network``)."""
    block_network(monkeypatch, request.node)


class ScriptedStrongLLM(LLMInterface):
    """A strong model that answers ``complete`` from a queue.

    Each item is a reply text, a ``CompletionResponse`` or an exception
    instance the call raises. Every request is kept in ``requests``. An
    exhausted queue raises ``IndexError`` (an unexpected call).
    """

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, CompletionResponse):
            return reply
        return CompletionResponse(kind="final", text=reply)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        raise NotImplementedError("the strong model only completes")


@pytest.fixture
def valid_fsm() -> dict[str, Any]:
    """A fresh compile that passes G1 to G5 with ``return_keys=("answer",)``."""
    return {
        "name": "Collect",
        "description": "Collect one answer",
        "initial_state": "ask",
        "states": {
            "ask": {
                "id": "ask",
                "description": "ask for the answer",
                "purpose": "get the answer",
                "response_instructions": "Ask for the answer",
                "field_extractions": [
                    {
                        "field_name": "answer",
                        "field_type": "str",
                        "extraction_instructions": "The user's answer",
                    }
                ],
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "answer given",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "has answer",
                                "requires_context_keys": ["answer"],
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "finished",
                "purpose": "wrap up",
                "response_instructions": "Thank the user",
            },
        },
    }
