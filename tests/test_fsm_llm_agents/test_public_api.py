"""Step 24 (API-01 / API-03 / PAT-13): the public agents entry points.

``create_agent`` takes the pattern first, ``AgentConfig`` rejects unknown
fields and carries ``instructions``, and the default model honours
``LLM_MODEL``.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from fsm_llm.agents import (
    AgentConfig,
    AgentError,
    DebateAgent,
    NativeFunctionCallingReactAgent,
    ReactAgent,
    ToolRegistry,
    create_agent,
    default_llm_judge,
)
from fsm_llm.agents.base import with_instructions
from fsm_llm.agents.definitions import MetaBuilderConfig
from fsm_llm.agents.sop import SOPDefinition, SOPRegistry
from fsm_llm.constants import DEFAULT_LLM_MODEL
from tests.conftest import PromptGroundedLLM


def _search(query: str) -> str:
    """Search the web."""
    return "results"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_search, name="search", description="Search")
    return registry


class TestCreateAgentPatternFirst:
    def test_positional_pattern_builds_that_pattern(self):
        assert isinstance(create_agent("debate"), DebateAgent)

    def test_positional_pattern_and_tools(self):
        agent = create_agent("react", [_search])

        assert isinstance(agent, ReactAgent)
        assert agent.tools.list_tools()[0].name == "_search"

    @pytest.mark.parametrize(
        ("name", "cls"),
        [("debate ", DebateAgent), (" React", ReactAgent), ("DEBATE", DebateAgent)],
    )
    def test_pattern_names_are_stripped_and_lowercased(self, name, cls):
        # Fix 24.1 (review api #4): "debate " and " React" name a pattern.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent = (
                create_agent(name, [_search])
                if cls is ReactAgent
                else (create_agent(name))
            )

        assert type(agent) is cls
        assert agent.config.instructions is None

    @pytest.mark.parametrize("name", ["debat", "Reactt", "x" * 32])
    def test_short_unknown_name_raises_listing_patterns(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"Available:.*'debate'"):
                create_agent(name)

    def test_system_prompt_keyword_sets_instructions(self):
        agent = create_agent("debate", system_prompt="Be terse.")

        assert agent.config.instructions == "Be terse."

    def test_system_prompt_keeps_config_fields_without_mutating_it(self):
        config = AgentConfig(max_iterations=4, model="mock/m")
        agent = create_agent("debate", config=config, system_prompt="Be terse.")

        assert (agent.config.max_iterations, agent.config.model) == (4, "mock/m")
        assert agent.config.instructions == "Be terse."
        assert config.instructions is None

    def test_conflicting_instructions_raise(self):
        with pytest.raises(ValueError, match="once"):
            create_agent(
                "debate",
                config=AgentConfig(instructions="A"),
                system_prompt="B",
            )

    @pytest.mark.parametrize("pattern", ["swarm", "meta_builder"])
    def test_system_prompt_on_pattern_without_prompts_raises(self, pattern):
        with pytest.raises(ValueError, match="does not use system_prompt"):
            create_agent(pattern, system_prompt="Be terse.")

    def test_native_fc_takes_instructions_as_system_policy(self):
        agent = create_agent("native_fc", [_search], system_prompt="Rule 1.")

        assert isinstance(agent, NativeFunctionCallingReactAgent)
        assert agent.system_policy == "Rule 1."
        assert agent._system_message().endswith("\n\nRule 1.")


class TestNativeFcPolicyPrecedence:
    def test_explicit_system_policy_wins(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(instructions="from config"),
            system_policy="explicit",
        )

        assert agent.system_policy == "explicit"

    def test_no_policy_and_no_instructions_leaves_policy_unset(self):
        agent = NativeFunctionCallingReactAgent(tools=_registry())

        assert agent.system_policy is None


class TestAgentConfigStrict:
    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="bogus"):
            AgentConfig(bogus=1)

    def test_meta_builder_config_is_strict_too(self):
        with pytest.raises(ValidationError, match="bogus"):
            MetaBuilderConfig(bogus=1)
        assert MetaBuilderConfig(max_turns=3).max_turns == 3

    def test_sop_typo_override_is_rejected_at_registration(self):
        sop = SOPDefinition(name="typo", config_overrides={"max_iteration": 3})

        with pytest.raises(ValueError, match=r"(?s)typo.*max_iteration"):
            SOPRegistry().register(sop)

    def test_instructions_are_length_capped(self):
        with pytest.raises(ValidationError, match="instructions"):
            AgentConfig(instructions="x" * 2001)


class TestModelResolution:
    def test_llm_model_env_is_the_default(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "foo")

        assert AgentConfig().model == "foo"
        assert MetaBuilderConfig().model == "foo"

    def test_explicit_model_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "foo")

        assert AgentConfig(model="bar").model == "bar"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_unset_or_blank_env_uses_the_core_default(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("LLM_MODEL", raising=False)
        else:
            monkeypatch.setenv("LLM_MODEL", value)

        assert AgentConfig().model == DEFAULT_LLM_MODEL

    def test_default_llm_judge_resolves_model_the_same_way(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "judge-env")
        seen: list[str] = []

        def complete(model: str, prompt: str) -> str:
            seen.append(model)
            return '{"passed": true, "score": 0.9, "feedback": "ok"}'

        default_llm_judge(complete_fn=complete)("out", {"task": "t"})
        default_llm_judge(model="explicit", complete_fn=complete)("out", {})

        assert seen == ["judge-env", "explicit"]


class TestPromptSlotOverflow:
    """Fix 24.1 (review api #1): instructions plus a tool catalogue that
    overflow core's instruction limit fail with an actionable AgentError, not
    core's bare FSM-load ValueError."""

    @staticmethod
    def _big_registry(count: int) -> ToolRegistry:
        registry = ToolRegistry()
        for i in range(count):
            registry.register_function(
                _search,
                name=f"tool_{i}",
                description=f"Searches the knowledge base for topic {i}. " * 6,
            )
        return registry

    def test_overflow_raises_agent_error_naming_the_budget(self):
        agent = ReactAgent(
            tools=self._big_registry(8),
            config=AgentConfig(instructions="Be careful. " * 100),
            llm_interface=PromptGroundedLLM(),
        )

        with pytest.raises(AgentError) as info:
            agent.run("hello")

        message = str(info.value)
        assert "AgentConfig.instructions (1199 characters)" in message
        assert "(8 tools)" in message
        assert "5000-character" in message
        assert "states.think" in message
        assert isinstance(info.value.__cause__, ValueError)

    def test_other_load_errors_are_not_rewritten(self):
        agent = ReactAgent(tools=_registry(), llm_interface=None)

        with pytest.raises(ValueError, match="Invalid FSM definition"):
            agent._create_api({"name": "x"})


class TestWithInstructions:
    _FSM: ClassVar[dict] = {
        "persona": "p",
        "states": {
            "think": {
                "extraction_instructions": "",
                "response_instructions": "",
                "field_extractions": [
                    {"field_name": "a", "extraction_instructions": "get a"}
                ],
            },
            "conclude": {"response_instructions": "answer"},
        },
    }

    def test_none_or_blank_returns_the_same_dict(self):
        assert with_instructions(self._FSM, None) is self._FSM
        assert with_instructions(self._FSM, "  ") is self._FSM

    def test_fills_only_non_empty_slots_and_copies(self):
        out = with_instructions(self._FSM, "RULE")
        think = out["states"]["think"]

        assert think["extraction_instructions"] == ""
        assert think["response_instructions"] == ""
        assert think["field_extractions"][0]["extraction_instructions"] == (
            "Agent instructions: RULE\n\nget a"
        )
        assert out["states"]["conclude"]["response_instructions"].startswith(
            "Agent instructions: RULE"
        )
        assert out["persona"] == "p"
        assert (
            self._FSM["states"]["think"]["field_extractions"][0][
                "extraction_instructions"
            ]
            == "get a"
        )


class TestStaticAll:
    """Step 25 (API-05): ``fsm_llm.agents.__all__`` is one static list."""

    _INIT = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fsm_llm"
        / "agents"
        / "__init__.py"
    )

    def _all_bindings(self) -> list[ast.stmt]:
        tree = ast.parse(self._INIT.read_text())
        bindings: list[ast.stmt] = []
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                bindings.append(node)
        return bindings

    def test_all_is_a_single_list_of_string_literals(self):
        bindings = self._all_bindings()
        assert len(bindings) == 1
        (binding,) = bindings
        assert isinstance(binding, ast.Assign)
        assert isinstance(binding.value, ast.List)
        assert all(
            isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            for elt in binding.value.elts
        )

    def test_all_names_resolve_once(self):
        import fsm_llm.agents as agents

        assert len(agents.__all__) == len(set(agents.__all__))
        for name in agents.__all__:
            assert hasattr(agents, name), name

    def test_reasoning_react_and_stop_reason_exported(self):
        import fsm_llm.agents as agents
        from fsm_llm.agents.constants import StopReason

        assert "ReasoningReactAgent" in agents.__all__
        assert agents.StopReason is StopReason
        assert "StopReason" in agents.__all__
        assert type(create_agent("reasoning_react", [_search])).__name__ == (
            "ReasoningReactAgent"
        )
