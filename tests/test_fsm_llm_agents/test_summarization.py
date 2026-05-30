"""Tests for observation summarization (auto_summarize_after)."""

from __future__ import annotations

import pytest

from fsm_llm_agents import (
    AgentConfig,
    ReactAgent,
    ToolRegistry,
    make_observation_summarizer,
    tool,
)
from fsm_llm_agents.constants import ContextKeys


@tool
def noop(query: str) -> str:
    """No-op."""
    return query


def _registry():
    reg = ToolRegistry()
    reg.register(noop._tool_definition)
    return reg


class TestSummarizer:
    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            make_observation_summarizer(0)

    def test_below_threshold_noop(self):
        s = make_observation_summarizer(5)
        ctx = {ContextKeys.OBSERVATIONS: ["a", "b", "c"]}
        assert s(ctx) == {}

    def test_condenses_above_threshold(self):
        s = make_observation_summarizer(4, keep_last=2)
        obs = [f"step{i}" for i in range(6)]
        out = s({ContextKeys.OBSERVATIONS: obs})
        new_obs = out[ContextKeys.OBSERVATIONS]
        # 1 summary + 2 kept
        assert len(new_obs) == 3
        assert new_obs[0].startswith("[Summary of 4 earlier steps]")
        assert new_obs[1:] == ["step4", "step5"]
        assert out[ContextKeys.OBSERVATION_COUNT] == 3

    def test_custom_summarize_fn(self):
        s = make_observation_summarizer(
            2, keep_last=1, summarize_fn=lambda entries: f"<{len(entries)}>"
        )
        out = s({ContextKeys.OBSERVATIONS: ["a", "b", "c"]})
        assert "<2>" in out[ContextKeys.OBSERVATIONS][0]

    def test_non_list_observations_noop(self):
        s = make_observation_summarizer(2)
        assert s({ContextKeys.OBSERVATIONS: "not a list"}) == {}

    def test_no_observations_key_noop(self):
        s = make_observation_summarizer(2)
        assert s({}) == {}


class TestWiring:
    def test_registered_when_config_set(self):
        """When auto_summarize_after is set, the handler is registered."""
        from unittest.mock import MagicMock

        agent = ReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model", auto_summarize_after=5),
        )
        api = MagicMock()
        agent._register_lifecycle_handlers(api, "react")
        names = [c.args[0] for c in api.create_handler.call_args_list if c.args]
        assert "AgentObservationSummarizer" in names

    def test_not_registered_by_default(self):
        from unittest.mock import MagicMock

        agent = ReactAgent(tools=_registry(), config=AgentConfig(model="mock/model"))
        api = MagicMock()
        agent._register_lifecycle_handlers(api, "react")
        names = [c.args[0] for c in api.create_handler.call_args_list if c.args]
        assert "AgentObservationSummarizer" not in names
