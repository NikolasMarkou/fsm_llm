"""Shared fixtures of the synthesis suite: no network, a valid compile and a
scripted strong model."""

import re
from typing import Any

import pytest

import fsm_llm.llm as llm_module
from fsm_llm.definitions import (
    CompletionRequest,
    CompletionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from tests.conftest import MockLLM2Interface, block_network


@pytest.fixture(autouse=True)
def _offline_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Refuse IPv4/IPv6 connects (``tests.conftest.block_network``)."""
    block_network(monkeypatch, request.node)


@pytest.fixture
def foreign_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every ``LiteLLMInterface`` built and every provider call made.

    Each one is a foreign interface: the tests inject their own doubles, so a
    real interface appears only when a compile steers the pipeline to a model
    of its own. The constructor and ``fsm_llm.llm.completion`` raise as well,
    but a pipeline may swallow that, so the test asserts the list is empty.
    """
    seen: list[str] = []

    def built(self: Any, model: str = "?", *args: Any, **kwargs: Any) -> None:
        seen.append(f"interface:{model}")
        raise AssertionError(f"a foreign interface was built for {model!r}")

    def sent(**kwargs: Any) -> None:
        seen.append(f"completion:{kwargs.get('model')}")
        raise AssertionError("a provider call was made")

    monkeypatch.setattr(llm_module.LiteLLMInterface, "__init__", built)
    monkeypatch.setattr(llm_module, "completion", sent)
    return seen


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


_STATE_TAG = re.compile(r"<current_state>([^<]+)</current_state>")


class ScriptedRunnerLLM(MockLLM2Interface):
    """The cheap model under a pushed procedure.

    ``extraction_data`` answers ``extract_field`` (a name it lacks is not
    extracted, so a child that needs it stays); ``responses`` maps a state id,
    read from the reply prompt's ``<current_state>`` tag, to the reply text.
    ``shown(kind, state=None)`` is the text of every recorded request of that
    kind (system prompt, user message, extraction context), optionally only
    the replies written for one state.
    """

    def __init__(
        self,
        extraction_data: dict[str, Any] | None = None,
        responses: dict[str, str] | None = None,
    ) -> None:
        super().__init__(extraction_data=extraction_data)
        self.responses = responses or {}

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        reply = super().generate_response(request)
        text = self.responses.get(self._state_of(request))
        return reply if text is None else reply.model_copy(update={"message": text})

    @staticmethod
    def _state_of(request: Any) -> str:
        found = _STATE_TAG.search(request.system_prompt)
        return found.group(1) if found else ""

    def count(self, kind: str, state: str | None = None) -> int:
        """How many requests of ``kind`` were made (for ``state``, if given)."""
        return sum(
            1
            for name, r in self.call_history
            if name == kind and (state is None or self._state_of(r) == state)
        )

    def shown(self, kind: str, state: str | None = None) -> str:
        return "\n".join(
            f"{r.system_prompt}\n{r.user_message}\n{getattr(r, 'context', None)}"
            for name, r in self.call_history
            if name == kind and (state is None or self._state_of(r) == state)
        )


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
