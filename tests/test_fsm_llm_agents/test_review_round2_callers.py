"""Callers that combined LLM settings with an injected interface (plan 944e2692,
step 18.5, D-037 item "API refuses every LLM setting", D-038).

Pass 8 warnings 4 and 5: ``AgentConfig.enable_prompt_cache`` with an injected
interface failed at run time with core's message about a ``caching`` kwarg
the caller never passed, and SelfConsistency's per-sample temperatures were
silently dropped with an injected interface (every sample ran at the
interface's temperature). Both tests fail on the parent 7ccd722.
"""

from __future__ import annotations

import pytest

from fsm_llm import API, LiteLLMInterface
from fsm_llm.agents import AgentConfig, ReactAgent, SelfConsistencyAgent, ToolRegistry
from fsm_llm.agents.constants import Defaults
from fsm_llm.agents.exceptions import AgentError
from fsm_llm.reasoning import ReasoningEngine
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm.test_llm_complete import _Provider, _reply


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(lambda query: f"found {query}", name="search")
    return registry


class TestPromptCacheWithAnInjectedInterface:
    def test_refused_at_construction_naming_the_flag(self):
        with pytest.raises(AgentError, match="enable_prompt_cache"):
            ReactAgent(
                _registry(),
                config=AgentConfig(model="x/y", enable_prompt_cache=True),
                llm_interface=MockLLM2Interface(),
            )

    def test_a_config_replaced_after_construction_is_refused_at_run(self):
        agent = ReactAgent(
            _registry(),
            config=AgentConfig(model="x/y"),
            llm_interface=MockLLM2Interface(),
        )
        agent.config = AgentConfig(model="x/y", enable_prompt_cache=True)
        with pytest.raises(AgentError, match="enable_prompt_cache"):
            agent._create_api({"name": "x"})

    def test_without_an_interface_caching_still_reaches_the_interface(self):
        agent = ReactAgent(
            _registry(), config=AgentConfig(model="gpt-4o", enable_prompt_cache=True)
        )
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        api = agent._create_api(build_react_fsm(agent.tools))
        assert api.llm_interface.kwargs.get("caching") is True


class TestInjectedInterfaceOwnsItsSettings:
    def test_agent_config_defaults_are_not_passed_beside_it(self):
        injected = LiteLLMInterface("ollama_chat/qwen3.5:4b", temperature=0.2)
        agent = ReactAgent(
            _registry(),
            config=AgentConfig(model="gpt-4o", temperature=0.9, max_tokens=50),
            llm_interface=injected,
        )
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        api = agent._create_api(build_react_fsm(agent.tools))
        assert api.llm_interface is injected
        assert injected.temperature == 0.2

    def test_a_setting_the_caller_passes_beside_it_is_still_refused(self):
        agent = ReactAgent(
            _registry(),
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


class TestSelfConsistencySampleTemperatures:
    """Each sample's temperature reaches the provider, with the interface
    injected or built by core."""

    @staticmethod
    def _expected(n: int) -> list[float]:
        low, high = Defaults.SAMPLE_TEMPERATURE_RANGE
        return [low + (high - low) * i / (n - 1) for i in range(n)]

    def test_through_an_injected_interface(self):
        agent = SelfConsistencyAgent(
            config=AgentConfig(model="gpt-4o"),
            num_samples=3,
            llm_interface=LiteLLMInterface("gpt-4o", temperature=0.1),
        )
        with _Provider(_reply('{"message": "Answer: 4"}')) as provider:
            result = agent.run("What is 2 + 2?")
        assert result.answer
        sent = [call["temperature"] for call in provider.calls]
        assert sent == pytest.approx(self._expected(3))

    def test_through_the_interface_core_builds(self):
        agent = SelfConsistencyAgent(config=AgentConfig(model="gpt-4o"), num_samples=3)
        with _Provider(_reply('{"message": "Answer: 4"}')) as provider:
            agent.run("What is 2 + 2?")
        sent = [call["temperature"] for call in provider.calls]
        assert sent == pytest.approx(self._expected(3))
