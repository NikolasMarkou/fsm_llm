from __future__ import annotations

import pytest

from fsm_llm.agents.definitions import MetaBuilderConfig
from fsm_llm.agents.meta_builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from tests.conftest import block_network


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
