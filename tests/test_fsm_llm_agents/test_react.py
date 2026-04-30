from __future__ import annotations

"""Tests for fsm_llm_agents.react module."""

import pytest

from fsm_llm.stdlib.agents.definitions import AgentConfig
from fsm_llm.stdlib.agents.exceptions import AgentError
from fsm_llm.stdlib.agents.react import ReactAgent
from fsm_llm.stdlib.agents.tools import ToolRegistry


def _search(params):
    return f"Results for: {params.get('query', '')}"


def _calculate(params):
    return eval(params.get("expression", "0"))


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_search, name="search", description="Search the web")
    registry.register_function(
        _calculate, name="calculate", description="Calculate expression"
    )
    return registry


class TestReactAgentCreation:
    """Tests for ReactAgent initialization."""

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = ReactAgent(tools=registry)
        assert agent.tools is registry
        assert agent.config is not None
        assert agent.hitl is None

    def test_create_with_config(self):
        registry = _make_registry()
        config = AgentConfig(max_iterations=5, model="gpt-4o-mini")
        agent = ReactAgent(tools=registry, config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4o-mini"

    def test_create_empty_registry_raises(self):
        registry = ToolRegistry()
        with pytest.raises(AgentError, match="empty tool registry"):
            ReactAgent(tools=registry)

    def test_create_with_hitl(self):
        from fsm_llm.stdlib.agents.hitl import HumanInTheLoop

        registry = _make_registry()
        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True,
            approval_callback=lambda req: True,
        )
        agent = ReactAgent(tools=registry, hitl=hitl)
        assert agent.hitl is hitl


class TestReactAgentIntegration:
    """Integration tests for ReactAgent.run() — require mocking LLM."""

    @pytest.mark.slow
    def test_run_requires_llm(self):
        """ReactAgent.run() needs a real or mock LLM — skip in unit tests."""
        pytest.skip("Requires LLM interface — run with real_llm marker")
