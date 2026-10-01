from __future__ import annotations

import json
from typing import Any

import pytest

from fsm_llm.agents.definitions import MetaBuilderConfig
from fsm_llm.agents.meta_builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from fsm_llm.definitions import (
    CompletionRequest,
    CompletionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from tests.conftest import block_network

SCRIPTED_REPLY = "Noted. Say 'build it' when you're ready."


class ScriptedMetaLLM(LLMInterface):
    """A scripted ``LLMInterface`` for the meta-builder FSM's three call kinds.

    Contract (shared by every meta test file that drives the FSM):

    - ``intents``: answers to classifier calls (``call_type ==
      "classification"``), in order. An item is an intent name (confidence
      0.95), an ``(intent, confidence)`` pair, or an exception instance the
      call raises. An exhausted queue answers ``"fsm"``.
    - ``builds``: answers to every other ``complete`` call (the build), in
      order: a reply text, or an exception instance the call raises. An
      exhausted queue raises ``IndexError`` (an unexpected build call).
    - ``replies``: answers to ``generate_response`` (Pass 2, the collect
      reply), in order: a text or an exception instance. An exhausted queue
      answers ``SCRIPTED_REPLY``.

    Every request is recorded: ``requests`` (``complete``), ``replies_sent``
    (``generate_response``); ``build_requests()`` and ``classifier_requests()``
    filter ``requests``.
    """

    def __init__(
        self,
        *,
        intents: list[Any] | None = None,
        builds: list[Any] | None = None,
        replies: list[Any] | None = None,
    ) -> None:
        self.intents = list(intents or [])
        self.builds = list(builds or [])
        self.reply_script = list(replies or [])
        self.requests: list[CompletionRequest] = []
        self.replies: list[ResponseGenerationRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if request.call_type == "classification":
            intent = self.intents.pop(0) if self.intents else "fsm"
            if isinstance(intent, BaseException):
                raise intent
            confidence = 0.95
            if isinstance(intent, tuple):
                intent, confidence = intent
            return CompletionResponse(
                kind="final",
                text=json.dumps(
                    {"intent": intent, "confidence": confidence, "reasoning": "x"}
                ),
            )
        build = self.builds.pop(0)
        if isinstance(build, BaseException):
            raise build
        return CompletionResponse(kind="final", text=build)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.replies.append(request)
        reply = self.reply_script.pop(0) if self.reply_script else SCRIPTED_REPLY
        if isinstance(reply, BaseException):
            raise reply
        return ResponseGenerationResponse(message=reply, message_type="response")

    def build_requests(self) -> list[CompletionRequest]:
        return [r for r in self.requests if r.call_type != "classification"]

    def classifier_requests(self) -> list[CompletionRequest]:
        return [r for r in self.requests if r.call_type == "classification"]


@pytest.fixture(autouse=True)
def _offline_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Refuse IPv4/IPv6 connects (``tests.conftest.block_network``)."""
    block_network(monkeypatch, request.node)


def _offline_completion(*args, **kwargs):
    raise RuntimeError("offline")


@pytest.fixture
def offline_llm(monkeypatch):
    """Make every LLM call from MetaBuilderAgent fail at once, with no network.

    Every call of the agent goes through core's LLM layer, whose one send
    binding is ``fsm_llm.llm.completion`` (bound at import, so patching
    ``litellm.completion`` would not reach it). The agent then takes its
    keyword type (``_detect_type_fallback``) and canned collect replies.
    """
    monkeypatch.setattr("fsm_llm.llm.completion", _offline_completion)


@pytest.fixture
def fsm_builder() -> FSMBuilder:
    """Fresh FSM builder."""
    return FSMBuilder()


@pytest.fixture
def workflow_builder() -> WorkflowBuilder:
    """Fresh workflow builder."""
    return WorkflowBuilder()


@pytest.fixture
def agent_builder() -> AgentBuilder:
    """Fresh agent builder."""
    return AgentBuilder()


@pytest.fixture
def populated_fsm_builder() -> FSMBuilder:
    """FSM builder with some states and transitions already added."""
    b = FSMBuilder()
    b.set_overview("GreetingBot", "A simple greeting bot", persona="Friendly assistant")
    b.add_state("greeting", "Greet the user", "Welcome the user and ask their name")
    b.add_state(
        "ask_name",
        "Ask for name",
        "Extract user name",
        extraction_instructions="Extract the user's name",
        response_instructions="Greet them by name",
    )
    b.add_state("farewell", "Say goodbye", "End the conversation")
    b.add_transition("greeting", "ask_name", "User responds to greeting")
    b.add_transition(
        "ask_name",
        "farewell",
        "Name has been collected",
        conditions=[
            {"description": "Name is set", "logic": {"has_context": "user_name"}}
        ],
    )
    return b


@pytest.fixture
def meta_config() -> MetaBuilderConfig:
    """Default test config."""
    return MetaBuilderConfig(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=1000,
        max_turns=20,
    )
