"""Tests for additive AgentConfig fields + _create_api passthrough wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm import API, LiteLLMInterface
from fsm_llm.agents import AgentConfig, ReactAgent, ToolRegistry, tool
from fsm_llm.core.definitions import AgentError
from fsm_llm.reasoning import ReasoningEngine
from fsm_llm.testing import MockLLM2Interface


@tool
def noop(query: str) -> str:
    """No-op tool."""
    return query


def _registry():
    from fsm_llm.agents import ToolRegistry

    reg = ToolRegistry()
    reg.register(noop._tool_definition)
    return reg


class TestNewConfigFields:
    def test_defaults_preserve_behavior(self):
        c = AgentConfig()
        assert c.max_history_size is None
        assert c.enable_prompt_cache is False
        assert c.reflect_every_n is None
        assert c.auto_summarize_after is None
        assert c.verification_fn is None

    def test_accepts_values(self):
        c = AgentConfig(
            max_history_size=3,
            enable_prompt_cache=True,
            reflect_every_n=2,
            auto_summarize_after=10,
            verification_fn=lambda answer, ctx: True,
        )
        assert c.max_history_size == 3
        assert c.enable_prompt_cache is True
        assert c.reflect_every_n == 2
        assert c.auto_summarize_after == 10
        assert callable(c.verification_fn)

    @pytest.mark.parametrize(
        "field", ["max_history_size", "reflect_every_n", "auto_summarize_after"]
    )
    def test_rejects_non_positive(self, field):
        with pytest.raises(ValueError):
            AgentConfig(**{field: 0})

    def test_verification_fn_excluded_from_serialization(self):
        c = AgentConfig(verification_fn=lambda a, ctx: True)
        assert "verification_fn" not in c.model_dump()


class TestCreateApiPassthrough:
    def _capture_kwargs(self, config):
        captured = {}

        def fake_from_definition(fsm_def, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        agent = ReactAgent(tools=_registry(), config=config)
        with patch(
            "fsm_llm.agents.base.runs.API.from_definition",
            side_effect=fake_from_definition,
        ):
            agent._create_api({"name": "x", "initial_state": "s", "states": {}})
        return captured

    def test_max_history_size_forwarded(self):
        captured = self._capture_kwargs(AgentConfig(max_history_size=2))
        assert captured.get("max_history_size") == 2

    def test_caching_forwarded_when_enabled(self):
        captured = self._capture_kwargs(AgentConfig(enable_prompt_cache=True))
        assert captured.get("caching") is True

    def test_defaults_inject_nothing_extra(self):
        captured = self._capture_kwargs(AgentConfig())
        assert "max_history_size" not in captured
        assert "caching" not in captured


def _review_round2_callers_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(lambda query: f"found {query}", name="search")
    return registry


class TestPromptCacheWithAnInjectedInterface:
    def test_refused_at_construction_naming_the_flag(self):
        with pytest.raises(AgentError, match="enable_prompt_cache"):
            ReactAgent(
                _review_round2_callers_registry(),
                config=AgentConfig(model="x/y", enable_prompt_cache=True),
                llm_interface=MockLLM2Interface(),
            )

    def test_a_config_replaced_after_construction_is_refused_at_run(self):
        agent = ReactAgent(
            _review_round2_callers_registry(),
            config=AgentConfig(model="x/y"),
            llm_interface=MockLLM2Interface(),
        )
        agent.config = AgentConfig(model="x/y", enable_prompt_cache=True)
        with pytest.raises(AgentError, match="enable_prompt_cache"):
            agent._create_api({"name": "x"})

    def test_without_an_interface_caching_still_reaches_the_interface(self):
        agent = ReactAgent(
            _review_round2_callers_registry(),
            config=AgentConfig(model="gpt-4o", enable_prompt_cache=True),
        )
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        api = agent._create_api(build_react_fsm(agent.tools))
        assert api.llm_interface.kwargs.get("caching") is True


class TestInjectedInterfaceOwnsItsSettings:
    def test_agent_config_defaults_are_not_passed_beside_it(self):
        injected = LiteLLMInterface("ollama_chat/qwen3.5:4b", temperature=0.2)
        agent = ReactAgent(
            _review_round2_callers_registry(),
            config=AgentConfig(model="gpt-4o", temperature=0.9, max_tokens=50),
            llm_interface=injected,
        )
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        api = agent._create_api(build_react_fsm(agent.tools))
        assert api.llm_interface is injected
        assert injected.temperature == 0.2

    def test_a_setting_the_caller_passes_beside_it_is_still_refused(self):
        agent = ReactAgent(
            _review_round2_callers_registry(),
            config=AgentConfig(model="gpt-4o"),
            llm_interface=MockLLM2Interface(),
            seed=7,
        )
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        with pytest.raises(ValueError, match="seed"):
            agent._create_api(build_react_fsm(agent.tools))

    def test_the_reasoning_engine_builds_on_an_injected_interface(self):
        injected = MockLLM2Interface()
        engine = ReasoningEngine(model="gpt-4o", llm_interface=injected)
        assert engine.orchestrator.llm_interface is injected
        with pytest.raises(ValueError, match="model"):
            API.from_definition(engine.main_fsm, model="gpt-4o", llm_interface=injected)
