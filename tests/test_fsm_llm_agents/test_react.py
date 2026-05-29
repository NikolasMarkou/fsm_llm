from __future__ import annotations

"""Tests for fsm_llm_agents.react module."""

import pytest

from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_agents.react import ReactAgent
from fsm_llm_agents.tools import ToolRegistry


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
        from fsm_llm_agents.hitl import HumanInTheLoop

        registry = _make_registry()
        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True,
            approval_callback=lambda req: True,
        )
        agent = ReactAgent(tools=registry, hitl=hitl)
        assert agent.hitl is hitl


class TestReactAgentHitlGating:
    """Regression tests for F-01 (plan_2026-05-29_1d66f861 / D-001).

    The await_approval FSM state must be built whenever the runtime approval
    gate is active. Approval is policy-driven; a per-tool requires_approval
    attribute must NOT be required to activate gating, otherwise a policy-only
    HITL config (the documented usage) executes tools un-gated.
    """

    @staticmethod
    def _policy_only_hitl():
        from fsm_llm_agents.hitl import HumanInTheLoop

        # Tools from _make_registry() default to requires_approval=False.
        return HumanInTheLoop(
            approval_policy=lambda call, ctx: True,
            approval_callback=lambda req: True,
        )

    def test_hitl_active_true_for_policy_only(self):
        agent = ReactAgent(tools=_make_registry(), hitl=self._policy_only_hitl())
        assert agent._hitl_active is True

    def test_hitl_inactive_without_hitl(self):
        agent = ReactAgent(tools=_make_registry())
        assert agent._hitl_active is False

    def test_hitl_inactive_without_policy(self):
        from fsm_llm_agents.hitl import HumanInTheLoop

        hitl = HumanInTheLoop(approval_callback=lambda req: True)  # no policy
        agent = ReactAgent(tools=_make_registry(), hitl=hitl)
        assert agent._hitl_active is False

    def test_policy_only_builds_await_approval_state(self):
        # The FSM that run() would build for a policy-only HITL must contain the
        # await_approval gate state; otherwise the gate handler sets
        # approval_required=True with no state to intercept it and the tool
        # executes before _handle_hitl_approval can request approval.
        from fsm_llm_agents.fsm_definitions import build_react_fsm

        agent = ReactAgent(tools=_make_registry(), hitl=self._policy_only_hitl())
        fsm = build_react_fsm(
            agent.tools, include_approval_state=agent._hitl_active
        )
        assert "await_approval" in fsm["states"]


class TestReactAgentIntegration:
    """Integration tests for ReactAgent.run() — require mocking LLM."""

    @pytest.mark.slow
    def test_run_requires_llm(self):
        """ReactAgent.run() needs a real or mock LLM — skip in unit tests."""
        pytest.skip("Requires LLM interface — run with real_llm marker")
