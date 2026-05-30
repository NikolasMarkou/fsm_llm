"""Tests for additive AgentConfig fields + _create_api passthrough wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm_agents import AgentConfig, ReactAgent, tool


@tool
def noop(query: str) -> str:
    """No-op tool."""
    return query


def _registry():
    from fsm_llm_agents import ToolRegistry

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
            "fsm_llm_agents.base.API.from_definition",
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
