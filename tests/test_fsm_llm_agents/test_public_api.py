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


class TestCoreOwnsTypedFieldsAndKeyClearing:
    """Step 21 of plan 944e2692: the typed field builder and the key-clearing
    handler live in core; agents import them and keep no copy or alias."""

    @pytest.mark.parametrize(
        "name",
        ["_typed_field_extraction", "_EXTRACTION_ENVELOPE_KEYS", "TypedFieldType"],
    )
    def test_agents_definitions_are_gone(self, name):
        import fsm_llm.agents as agents
        import fsm_llm.agents.fsm_definitions as fsm_definitions

        assert not hasattr(fsm_definitions, name)
        assert not hasattr(agents, name)

    def test_core_names_exported(self):
        import fsm_llm
        from fsm_llm import definitions, handlers

        assert fsm_llm.typed_field_extraction is definitions.typed_field_extraction
        assert fsm_llm.clear_keys_on_entry is handlers.clear_keys_on_entry
        assert {"typed_field_extraction", "clear_keys_on_entry"} <= set(fsm_llm.__all__)

    def test_agents_build_with_the_core_builder(self):
        import fsm_llm
        import fsm_llm.agents as agents
        import fsm_llm.agents.fsm_definitions as fsm_definitions

        assert fsm_definitions.typed_field_extraction is fsm_llm.typed_field_extraction
        assert "typed_field_extraction" not in agents.__all__
        assert "clear_keys_on_entry" not in agents.__all__

    def test_no_agents_module_defines_a_key_clearing_copy(self):
        # make_fresh_keys_handler and the meta-builder build their plain clear
        # with core clear_keys_delta (D-051); no agents module re-implements it.
        src = Path(__file__).resolve().parents[2] / "src" / "fsm_llm" / "agents"
        for path in ("handlers.py", "meta_builder.py"):
            assert "clear_keys_delta(" in (src / path).read_text(), path


class TestConfiguredAgentBuilder:
    """ConfiguredAgentBuilder: records, then calls create_agent unchanged."""

    @staticmethod
    def _builder():
        from fsm_llm.agents import ConfiguredAgentBuilder

        return ConfiguredAgentBuilder()

    def test_mutators_return_same_builder(self):
        b = self._builder()
        for call in (
            lambda: b.set_pattern("react"),
            lambda: b.add_tool(_search),
            lambda: b.set_config(AgentConfig()),
            lambda: b.set_config_option("max_iterations", 3),
            lambda: b.set_model("m"),
            lambda: b.set_temperature(0.1),
            lambda: b.set_max_tokens(10),
            lambda: b.set_max_iterations(4),
            lambda: b.set_timeout_seconds(5.0),
            lambda: b.set_system_prompt("hi"),
            lambda: b.set_option("seed", 1),
        ):
            assert call() is b

    def test_default_build_matches_create_agent(self):
        from fsm_llm.definitions import BuildError  # noqa: F401

        built = self._builder().add_tool(_search).build()
        direct = create_agent("react", [_search])
        assert type(built) is type(direct)
        assert built.config.model_dump() == direct.config.model_dump()
        assert built.tools.tool_names == direct.tools.tool_names

    def test_typed_setters_write_config(self):
        agent = (
            self._builder()
            .set_pattern("debate")
            .set_model("m")
            .set_temperature(0.3)
            .set_max_tokens(77)
            .set_max_iterations(4)
            .set_timeout_seconds(9.0)
            .build()
        )
        c = agent.config
        assert (c.model, c.temperature, c.max_tokens) == ("m", 0.3, 77)
        assert (c.max_iterations, c.timeout_seconds) == (4, 9.0)

    def test_set_option_model_is_not_filtered(self):
        from fsm_llm.definitions import BuildError

        with pytest.raises(BuildError) as info:
            self._builder().set_option("model", "x").build()
        assert isinstance(info.value.__cause__, TypeError)

    def test_set_option_seed_lands_in_api_kwargs(self):
        agent = self._builder().set_pattern("debate").set_option("seed", 3).build()
        assert agent._api_kwargs["seed"] == 3

    def test_system_prompt_conflict_is_wrapped(self):
        from fsm_llm.definitions import BuildError

        b = (
            self._builder()
            .set_pattern("debate")
            .set_config(AgentConfig(instructions="a"))
            .set_system_prompt("b")
        )
        with pytest.raises(BuildError) as info:
            b.build()
        assert isinstance(info.value.__cause__, ValueError)
        assert "pass the instructions once" in str(info.value)

    def test_system_prompt_applied(self):
        agent = self._builder().set_pattern("debate").set_system_prompt("Cite.").build()
        assert agent.config.instructions == "Cite."

    def test_config_isolation_and_independent_builds(self):
        cfg = AgentConfig(max_iterations=5)
        b = self._builder().set_pattern("debate").set_config(cfg)
        first = b.build()
        cfg.max_iterations = 99
        assert first.config.max_iterations == 5
        second = b.build()
        assert second.config is not first.config
        assert second.config.max_iterations == 99

    def test_callables_and_objects_in_config_stay_shared(self):
        import threading

        class _Verifier:
            def __init__(self):
                self.lock = threading.Lock()

            def check(self, answer, ctx):
                return True

        v = _Verifier()
        transition = object()
        cfg = AgentConfig(verification_fn=v.check, transition_config=transition)
        b = self._builder().set_pattern("debate").set_config(cfg)
        agent = b.build()
        assert agent.config.verification_fn.__self__ is v
        assert agent.config.transition_config is transition
        assert agent.config is not cfg
        cfg.max_iterations = 77
        assert agent.config.max_iterations != 77
        second = b.build()
        assert second.config is not agent.config
        assert second.config.verification_fn.__self__ is v

    @pytest.mark.parametrize(
        ("name", "value", "method"),
        [
            ("config", AgentConfig(), "set_config"),
            ("pattern", "react", "set_pattern"),
            ("tools", [_search], "add_tool"),
            ("system_prompt", "x", "set_system_prompt"),
        ],
    )
    def test_set_option_naming_a_create_agent_parameter_is_refused(
        self, name, value, method
    ):
        from fsm_llm.definitions import BuildError

        with pytest.raises(BuildError) as info:
            self._builder().set_option(name, value).build()
        assert method in str(info.value)
        assert any(repr(name) in e for e in info.value.errors)

    def test_set_option_beside_typed_setter_is_refused_by_name(self):
        from fsm_llm.definitions import BuildError

        b = self._builder().set_pattern("debate").set_option("pattern", "react")
        with pytest.raises(BuildError) as info:
            b.build()
        assert "set_pattern" in str(info.value)

    def test_second_build_independent(self):
        b = self._builder().add_tool(_search).set_option("seed", 1)
        a1, a2 = b.build(), b.build()
        assert a1 is not a2
        assert a1.tools is not a2.tools
        assert a1._api_kwargs is not a2._api_kwargs

    def test_add_tool_registers_tool_function(self):
        from fsm_llm.agents import tool

        @tool
        def lookup(q: str) -> str:
            """Look up."""
            return q

        agent = self._builder().add_tool(lookup).build()
        assert agent.tools.tool_names == ["lookup"]

    def test_add_tool_with_registry_errors(self):
        from fsm_llm.definitions import BuildError

        b = self._builder().add_tool(_search).set_tool_registry(_registry())
        with pytest.raises(BuildError):
            b.build()

    def test_set_tool_registry_shared_by_reference(self):
        reg = _registry()
        agent = self._builder().set_tool_registry(reg).build()
        assert agent.tools is reg

    def test_bad_config_option_wraps_validation_error(self):
        from fsm_llm.definitions import BuildError

        with pytest.raises(BuildError) as info:
            self._builder().set_config_option("bogus", 1).build()
        assert isinstance(info.value.__cause__, ValidationError)

    def test_toolless_pattern_with_tool_errors(self):
        from fsm_llm.definitions import BuildError

        with pytest.raises(BuildError) as info:
            self._builder().set_pattern("debate").add_tool(_search).build()
        assert isinstance(info.value.__cause__, TypeError)

    def test_toolless_pattern_builds(self):
        agent = self._builder().set_pattern("self_consistency").set_model("m").build()
        assert agent.config.model == "m"
        assert type(agent) is type(create_agent("self_consistency"))
