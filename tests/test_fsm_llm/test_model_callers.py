"""Callers of the model resolution after ``resolve_model`` stopped defaulting.

Step 14 of plan d454fd62 (D-005): ``Classifier``, ``HierarchicalClassifier`` and
``AgentConfig()`` resolve the model only through ``resolve_model`` and raise
``ModelNotConfiguredError`` when nothing is set; an agent, an agent builder
and ``create_agent`` that hold an injected ``llm_interface`` need no model;
the meta artifact builder's ``_config_problems`` (documented "never raises")
reports a missing model as a problem instead of raising.
"""

from __future__ import annotations

import pytest

from fsm_llm import Classifier, HierarchicalClassifier, ModelNotConfiguredError
from fsm_llm.agents import (
    AgentConfig,
    ConfiguredAgentBuilder,
    ReactAgent,
    ToolRegistry,
    create_agent,
    default_llm_judge,
    tool,
)
from fsm_llm.agents.meta_builders import AgentArtifactBuilder
from fsm_llm.constants import ENV_LLM_MODEL
from fsm_llm.definitions import (
    ClassificationSchema,
    HierarchicalSchema,
    IntentDefinition,
)
from fsm_llm.llm import CompletionRequest, CompletionResponse
from tests.conftest import MockLLM2Interface


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_LLM_MODEL, raising=False)


def _schema(*names: str) -> ClassificationSchema:
    return ClassificationSchema(
        intents=[
            IntentDefinition(name=n, description=f"the {n} intent") for n in names
        ],
        fallback_intent=names[0],
    )


def _hierarchical() -> HierarchicalSchema:
    return HierarchicalSchema(
        domain_schema=_schema("sales", "support"),
        intent_schemas={"support": _schema("buy", "browse")},
    )


@tool
def _noop(query: str) -> str:
    """No-op tool."""
    return query


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_noop._tool_definition)
    return registry


class _Double(MockLLM2Interface):
    """A test double that, like a real interface, names no model, and
    implements ``complete`` so that a ``Classifier`` accepts it."""

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(text="{}")


class TestClassifierResolvesThroughResolveModel:
    def test_unset_raises(self, unset):
        with pytest.raises(ModelNotConfiguredError):
            Classifier(_schema("a", "b"))

    def test_hierarchical_unset_raises(self, unset):
        with pytest.raises(ModelNotConfiguredError):
            HierarchicalClassifier(_hierarchical())

    def test_env_is_used(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert Classifier(_schema("a", "b")).model == "env/model"

    def test_explicit_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env/model")
        assert Classifier(_schema("a", "b"), model="arg/model").model == "arg/model"

    def test_an_injected_interface_needs_no_model(self, unset):
        classifier = Classifier(_schema("a", "b"), llm_interface=_Double())
        assert classifier.model is None

    def test_hierarchical_injected_interface_needs_no_model(self, unset):
        HierarchicalClassifier(_hierarchical(), llm_interface=_Double())


class TestAgentConfigRaisesWhenUnset:
    def test_unset_raises(self, unset):
        with pytest.raises(ModelNotConfiguredError):
            AgentConfig()

    def test_explicit_model_needs_no_environment(self, unset):
        assert AgentConfig(model="m/x").model == "m/x"

    def test_the_judge_without_a_model_raises(self, unset):
        with pytest.raises(ModelNotConfiguredError):
            default_llm_judge()


class TestInjectedInterfaceNeedsNoAgentModel:
    def test_create_agent_with_a_double_and_no_model(self, unset):
        agent = create_agent("react", [_noop], llm_interface=_Double())
        assert isinstance(agent, ReactAgent)
        assert agent.config.model

    def test_create_agent_with_system_prompt_and_a_double(self, unset):
        agent = create_agent(
            "react", [_noop], system_prompt="Be brief.", llm_interface=_Double()
        )
        assert agent.config.instructions == "Be brief."

    def test_the_agent_constructor_with_a_double(self, unset):
        assert ReactAgent(tools=_registry(), llm_interface=_Double()).config.model

    def test_the_interface_model_is_the_config_model(self, unset):
        llm = _Double()
        llm.model = "iface/model"
        assert (
            ReactAgent(tools=_registry(), llm_interface=llm).config.model
            == "iface/model"
        )

    def test_the_builder_with_a_double_and_a_config_option(self, unset):
        agent = (
            ConfiguredAgentBuilder()
            .set_pattern("react")
            .add_tool(_noop)
            .set_max_iterations(3)
            .set_option("llm_interface", _Double())
            .build()
        )
        assert agent.config.max_iterations == 3

    def test_without_an_interface_the_agent_still_raises(self, unset):
        with pytest.raises(ModelNotConfiguredError):
            create_agent("react", [_noop])


class TestMetaConfigProblemsNeverRaises:
    def test_a_config_without_a_model_is_a_problem(self, unset):
        problems = AgentArtifactBuilder._config_problems({"max_iterations": 3})
        assert "model" in problems
        assert "LLM_MODEL" in problems["model"]

    def test_a_valid_config_has_no_problem(self, unset):
        assert AgentArtifactBuilder._config_problems({"model": "m/x"}) == {}

    def test_another_invalid_key_is_still_reported_beside_the_model(self, unset):
        problems = AgentArtifactBuilder._config_problems({"max_iterations": 0})
        assert "max_iterations" in problems


class TestOtherEntryPoints:
    def test_the_meta_builder_agent_with_a_double_needs_no_model(self, unset):
        from fsm_llm.agents import MetaBuilderAgent

        assert MetaBuilderAgent(llm_interface=_Double()).meta_config.model

    def test_the_judge_with_a_custom_backend_needs_no_model(self, unset):
        judge = default_llm_judge(complete_fn=lambda model, prompt: "{}")
        assert callable(judge)
