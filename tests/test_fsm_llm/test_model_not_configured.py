"""No silent default model: ``resolve_model`` raises ``ModelNotConfiguredError``.

Plan d454fd62 D-005 (roadmap row 0.8). The root autouse fixture in
``tests/conftest.py`` sets ``LLM_MODEL`` for every test, so each test that needs
the unset state removes it explicitly.
"""

from __future__ import annotations

import os
import pickle
from unittest.mock import Mock

import pytest

import fsm_llm
from fsm_llm import API, FSMError, ModelNotConfiguredError
from fsm_llm.constants import ENV_LLM_MODEL
from fsm_llm.llm import resolve_model
from fsm_llm.testing import MockLLM2Interface

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


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLM_MODEL`` removed for this test (the autouse fixture set it)."""
    monkeypatch.delenv(ENV_LLM_MODEL, raising=False)


class TestTheErrorClass:
    def test_it_is_an_fsm_error_and_a_value_error(self):
        assert issubclass(ModelNotConfiguredError, FSMError)
        assert issubclass(ModelNotConfiguredError, ValueError)

    def test_it_is_exported_from_the_package(self):
        assert "ModelNotConfiguredError" in fsm_llm.__all__
        assert fsm_llm.ModelNotConfiguredError is ModelNotConfiguredError

    def test_the_default_message_is_one_line_naming_the_three_ways_to_set_a_model(self):
        message = str(ModelNotConfiguredError())
        assert "\n" not in message
        assert "model=" in message
        assert ENV_LLM_MODEL in message
        assert "load_env" in message

    def test_the_message_names_no_concrete_model(self):
        message = str(ModelNotConfiguredError())
        for name in ("bonsai", "qwen", "gpt-", "gemma", "ollama", "claude"):
            assert name not in message

    def test_an_explicit_message_and_details_are_kept(self):
        error = ModelNotConfiguredError("custom", details={"where": "x"})
        assert str(error) == "custom"
        assert error.details == {"where": "x"}

    def test_it_survives_pickling(self):
        again = pickle.loads(pickle.dumps(ModelNotConfiguredError()))
        assert isinstance(again, ModelNotConfiguredError)
        assert str(again) == str(ModelNotConfiguredError())


class TestResolveModelRaises:
    def test_nothing_set_raises(self, unset):
        with pytest.raises(ModelNotConfiguredError) as excinfo:
            resolve_model()
        message = str(excinfo.value)
        assert "model=" in message
        assert ENV_LLM_MODEL in message
        assert "load_env" in message

    @pytest.mark.parametrize("env", ["", "   "])
    @pytest.mark.parametrize("explicit", [None, "", "  "])
    def test_blank_counts_as_unset(self, monkeypatch, env, explicit):
        monkeypatch.setenv(ENV_LLM_MODEL, env)
        with pytest.raises(ModelNotConfiguredError):
            resolve_model(explicit)

    def test_the_error_is_catchable_as_a_value_error(self, unset):
        with pytest.raises(ValueError, match="LLM_MODEL"):
            resolve_model()

    def test_explicit_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        assert resolve_model(" given ") == "given"

    def test_env_is_used_without_an_argument(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, " env-model ")
        assert resolve_model() == "env-model"

    def test_explicit_works_with_nothing_in_the_environment(self, unset):
        assert resolve_model("given/model") == "given/model"


class TestTheRootFixture:
    def test_every_test_starts_with_a_model_in_the_environment(self):
        assert os.environ[ENV_LLM_MODEL] == "test/model"
        assert resolve_model() == "test/model"

    def test_a_test_may_remove_it_and_the_next_test_still_has_it(self, unset):
        assert ENV_LLM_MODEL not in os.environ


class TestApiRaisesBeforeAnyNetwork:
    def test_unset_raises_before_the_definition_is_read(self, unset, monkeypatch):
        interface = Mock()
        monkeypatch.setattr("fsm_llm.api.LiteLLMInterface", interface)
        # A broken definition would raise a validation error if the FSM were
        # processed first: the model message comes first.
        with pytest.raises(ModelNotConfiguredError):
            API.from_definition({"not": "a definition"})
        with pytest.raises(ModelNotConfiguredError):
            API.from_definition(_FSM)
        interface.assert_not_called()

    def test_a_blank_env_raises_too(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "  ")
        with pytest.raises(ModelNotConfiguredError):
            API.from_definition(_FSM)

    def test_an_injected_interface_needs_no_model(self, unset):
        api = API.from_definition(_FSM, llm_interface=MockLLM2Interface())
        assert isinstance(api.llm_interface, MockLLM2Interface)

    def test_an_explicit_model_needs_no_environment(self, unset):
        api = API.from_definition(_FSM, model="given/model")
        assert api.llm_interface.model == "given/model"
