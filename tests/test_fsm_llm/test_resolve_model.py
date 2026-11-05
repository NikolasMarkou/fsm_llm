"""One model resolution rule and one home for the LLM default numbers."""

from __future__ import annotations

import pytest

from fsm_llm import API
from fsm_llm.constants import DEFAULT_LLM_MODEL, ENV_LLM_MODEL
from fsm_llm.llm import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    LiteLLMEmbedder,
    LiteLLMInterface,
    resolve_model,
)

_FSM = {
    "name": "m",
    "description": "d",
    "initial_state": "a",
    "states": {
        "a": {
            "id": "a",
            "description": "start",
            "purpose": "p",
            "transitions": [{"target_state": "b", "description": "go", "priority": 1}],
        },
        "b": {"id": "b", "description": "end", "purpose": "p"},
    },
}


class TestResolveModel:
    def test_explicit_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        assert resolve_model("given") == "given"

    def test_env_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, " env-model ")
        assert resolve_model() == "env-model"
        assert resolve_model(None) == "env-model"

    @pytest.mark.parametrize("env", [None, "", "   "])
    @pytest.mark.parametrize("explicit", [None, "", "  "])
    def test_blank_is_unset(self, monkeypatch, env, explicit):
        if env is None:
            monkeypatch.delenv(ENV_LLM_MODEL, raising=False)
        else:
            monkeypatch.setenv(ENV_LLM_MODEL, env)
        assert resolve_model(explicit) == DEFAULT_LLM_MODEL


class TestApiBlankEnvModel:
    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_env_falls_back_to_the_default(self, monkeypatch, value):
        monkeypatch.setenv(ENV_LLM_MODEL, value)
        api = API.from_definition(_FSM)
        assert api.llm_interface.model == DEFAULT_LLM_MODEL

    def test_env_model_is_used(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert API.from_definition(_FSM).llm_interface.model == "env/model"

    def test_explicit_model_beats_env(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        api = API.from_definition(_FSM, model="given/model")
        assert api.llm_interface.model == "given/model"


class TestDefaultNumbersHaveOneHome:
    def test_interface_defaults_are_the_named_ones(self):
        import inspect

        params = inspect.signature(LiteLLMInterface.__init__).parameters
        assert params["max_tokens"].default == DEFAULT_MAX_TOKENS
        assert params["timeout"].default == DEFAULT_LLM_TIMEOUT_SECONDS
        embed = inspect.signature(LiteLLMEmbedder.__init__).parameters
        assert embed["timeout"].default == DEFAULT_LLM_TIMEOUT_SECONDS

    def test_api_default_max_tokens(self, monkeypatch):
        monkeypatch.delenv(ENV_LLM_MODEL, raising=False)
        api = API.from_definition(_FSM)
        assert api.llm_interface.max_tokens == DEFAULT_MAX_TOKENS

    def test_values_are_unchanged(self):
        assert DEFAULT_MAX_TOKENS == 1000
        assert DEFAULT_LLM_TIMEOUT_SECONDS == 120.0
        assert DEFAULT_EMBEDDING_MODEL == "ollama/qwen3-embedding:0.6b"
