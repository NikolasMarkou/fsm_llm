from __future__ import annotations

import pytest

from fsm_llm_meta.builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from fsm_llm_meta.definitions import MetaAgentConfig


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
def meta_config() -> MetaAgentConfig:
    """Default test config."""
    return MetaAgentConfig(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=1000,
        max_turns=20,
    )
