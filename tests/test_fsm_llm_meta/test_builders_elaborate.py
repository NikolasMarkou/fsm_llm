from __future__ import annotations

"""Elaborate tests for artifact builders: edge cases, type validation,
config validation, WorkflowArtifactBuilder ClassVar, and false-positive fixes."""

import copy
import pickle

import pytest

from fsm_llm.agents.constants import MetaDefaults
from fsm_llm.agents.exceptions import BuilderError
from fsm_llm.agents.meta_builders import (
    AgentArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from fsm_llm.agents.meta_tools import create_fsm_tools
from fsm_llm.definitions import FSMDefinition

# ---- FSMArtifactBuilder Edge Cases -------------------------------------------


class TestFSMBuilderUpdateStateTypeChecking:
    """Bug fix: update_state should reject non-string values."""

    def test_none_value_warns(self, populated_fsm_builder: FSMArtifactBuilder):
        warnings = populated_fsm_builder.update_state(
            "greeting", description=None
        ).take_warnings()
        assert any("None" in w for w in warnings)
        # Original value should be preserved
        assert populated_fsm_builder.states["greeting"]["description"] != ""

    def test_int_value_converts_with_warning(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        warnings = populated_fsm_builder.update_state(
            "greeting", purpose=42
        ).take_warnings()
        assert any("string" in w.lower() for w in warnings)
        assert populated_fsm_builder.states["greeting"]["purpose"] == "42"

    def test_dict_value_converts_with_warning(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        warnings = populated_fsm_builder.update_state(
            "greeting", description={"bad": "value"}
        ).take_warnings()
        assert any("string" in w.lower() for w in warnings)


class TestFSMBuilderToDict:
    """Fix false positive: verify to_dict produces correct field values."""

    def test_to_dict_field_values(self, populated_fsm_builder: FSMArtifactBuilder):
        d = populated_fsm_builder.to_dict()
        assert d["name"] == "GreetingBot"
        assert d["description"] == "A simple greeting bot"
        assert d["persona"] == "Friendly assistant"
        assert d["initial_state"] == "greeting"
        assert d["version"] == "4.1"
        assert set(d["states"].keys()) == {"greeting", "ask_name", "farewell"}

    def test_to_dict_produces_valid_fsm_with_values(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        d = populated_fsm_builder.to_dict()
        definition = FSMDefinition(**d)
        assert definition.name == "GreetingBot"
        assert len(definition.states) == 3
        assert definition.initial_state == "greeting"

    def test_to_dict_transitions_preserved(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        d = populated_fsm_builder.to_dict()
        greeting_transitions = d["states"]["greeting"]["transitions"]
        assert len(greeting_transitions) == 1
        assert greeting_transitions[0]["target_state"] == "ask_name"


class TestFSMBuilderEdgeCases:
    def test_self_transition(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("s1", "State", "Purpose")
        fsm_builder.add_transition("s1", "s1", "Loop")
        assert len(fsm_builder.states["s1"]["transitions"]) == 1

    def test_multiple_transitions_from_same_source(
        self, fsm_builder: FSMArtifactBuilder
    ):
        fsm_builder.add_state("s1", "State 1", "P1")
        fsm_builder.add_state("s2", "State 2", "P2")
        fsm_builder.add_state("s3", "State 3", "P3")
        fsm_builder.add_transition("s1", "s2", "To s2")
        fsm_builder.add_transition("s1", "s3", "To s3")
        assert len(fsm_builder.states["s1"]["transitions"]) == 2

    def test_remove_only_state_leaves_empty(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("s1", "State", "Purpose")
        fsm_builder.remove_state("s1")
        assert len(fsm_builder.states) == 0
        assert fsm_builder.initial_state is None

    def test_add_transition_uses_defaults_constant(self):
        b = FSMArtifactBuilder()
        b.add_state("s1", "S1", "P1")
        b.add_state("s2", "S2", "P2")
        b.add_transition("s1", "s2", "Go")
        t = b.states["s1"]["transitions"][0]
        assert t["priority"] == MetaDefaults.DEFAULT_PRIORITY


# ---- WorkflowArtifactBuilder ClassVar and Validation -------------------------


class TestWorkflowBuilderClassVar:
    def test_valid_step_types_is_classvar(self):
        assert hasattr(WorkflowArtifactBuilder, "VALID_STEP_TYPES")
        assert isinstance(WorkflowArtifactBuilder.VALID_STEP_TYPES, set)
        assert "auto_transition" in WorkflowArtifactBuilder.VALID_STEP_TYPES
        assert len(WorkflowArtifactBuilder.VALID_STEP_TYPES) == 8

    def test_all_eight_types_present(self):
        expected = {
            "auto_transition",
            "api_call",
            "condition",
            "llm_processing",
            "wait_for_event",
            "timer",
            "parallel",
            "conversation",
        }
        assert WorkflowArtifactBuilder.VALID_STEP_TYPES == expected


class TestWorkflowBuilderEdgeCases:
    def test_remove_step_cleans_transitions(
        self, workflow_builder: WorkflowArtifactBuilder
    ):
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        workflow_builder.add_step("s2", "auto_transition", "Step 2")
        workflow_builder.set_step_transition("s1", "s2")
        workflow_builder.remove_step("s2")
        assert len(workflow_builder.steps["s1"]["transitions"]) == 0

    def test_set_initial_step_explicit(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        workflow_builder.add_step("s2", "auto_transition", "Step 2")
        workflow_builder.set_initial_step("s2")
        assert workflow_builder.initial_step_id == "s2"

    def test_validate_complete_checks_transition_targets(
        self, workflow_builder: WorkflowArtifactBuilder
    ):
        workflow_builder.set_overview("wf1", "Workflow", "Desc")
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        # Manually add invalid transition
        workflow_builder.steps["s1"]["transitions"].append({"target": "nonexistent"})
        errors = workflow_builder.validate_complete()
        assert any("nonexistent" in e for e in errors)


# ---- AgentArtifactBuilder Config Validation ----------------------------------


class TestAgentBuilderConfigValidation:
    def test_set_config_rejects_wrong_type_for_max_iterations(
        self, agent_builder: AgentArtifactBuilder
    ):
        warnings = agent_builder.set_config(max_iterations="ten").take_warnings()
        assert any("max_iterations" in w for w in warnings)
        # Should not have changed
        assert (
            agent_builder.config["max_iterations"] == MetaDefaults.AGENT_MAX_ITERATIONS
        )

    def test_set_config_rejects_string_for_temperature(
        self, agent_builder: AgentArtifactBuilder
    ):
        warnings = agent_builder.set_config(temperature="warm").take_warnings()
        assert any("temperature" in w for w in warnings)
        assert agent_builder.config["temperature"] == MetaDefaults.AGENT_TEMPERATURE

    def test_set_config_accepts_int_for_temperature(
        self, agent_builder: AgentArtifactBuilder
    ):
        warnings = agent_builder.set_config(temperature=0).take_warnings()
        assert warnings == []
        assert agent_builder.config["temperature"] == 0

    def test_set_config_accepts_valid_model(self, agent_builder: AgentArtifactBuilder):
        warnings = agent_builder.set_config(model="gpt-4o").take_warnings()
        assert warnings == []
        assert agent_builder.config["model"] == "gpt-4o"

    def test_set_config_rejects_unknown_field(
        self, agent_builder: AgentArtifactBuilder
    ):
        warnings = agent_builder.set_config(unknown_field="value").take_warnings()
        assert any("unknown" in w.lower() for w in warnings)

    def test_defaults_use_constants(self):
        b = AgentArtifactBuilder()
        assert b.config["model"] == MetaDefaults.AGENT_MODEL
        assert b.config["max_iterations"] == MetaDefaults.AGENT_MAX_ITERATIONS
        assert b.config["timeout_seconds"] == MetaDefaults.AGENT_TIMEOUT_SECONDS
        assert b.config["temperature"] == MetaDefaults.AGENT_TEMPERATURE
        assert b.config["max_tokens"] == MetaDefaults.AGENT_MAX_TOKENS


class TestAgentConfigValidation:
    """F-AM-06: the artifact config is validated through ``AgentConfig``."""

    @pytest.mark.parametrize(
        "key,value",
        [
            ("max_iterations", -1),
            ("temperature", 9),
            ("timeout_seconds", -3),
            ("max_tokens", True),
            ("model", ""),
        ],
    )
    def test_set_config_skips_each_bad_value_with_a_warning(
        self, agent_builder: AgentArtifactBuilder, key: str, value
    ):
        before = dict(agent_builder.config)
        warnings = agent_builder.set_config(**{key: value}).take_warnings()
        assert any(key in w for w in warnings)
        assert agent_builder.config == before

    def test_set_config_keeps_the_good_keys_of_a_mixed_call(
        self, agent_builder: AgentArtifactBuilder
    ):
        warnings = agent_builder.set_config(
            max_iterations=-1, temperature=0.2, model="  "
        ).take_warnings()
        assert len(warnings) == 2
        assert agent_builder.config["temperature"] == 0.2
        assert agent_builder.config["max_iterations"] == (
            MetaDefaults.AGENT_MAX_ITERATIONS
        )

    def test_validate_complete_rejects_a_poked_invalid_config(
        self, agent_builder: AgentArtifactBuilder
    ):
        agent_builder.set_agent_type("react").set_overview("A", "d")
        agent_builder.add_tool("t", "d")
        assert agent_builder.validate_complete() == []
        agent_builder.config.update(max_iterations=-1, temperature=9, model=" ")
        errors = agent_builder.validate_complete()
        for key in ("max_iterations", "temperature", "model"):
            assert any(f"'{key}'" in e for e in errors)

    def test_validate_complete_rejects_a_poked_bool_and_unknown_key(
        self, agent_builder: AgentArtifactBuilder
    ):
        agent_builder.set_agent_type("react").set_overview("A", "d")
        agent_builder.add_tool("t", "d")
        agent_builder.config["max_tokens"] = True
        agent_builder.config["bogus"] = 1
        errors = agent_builder.validate_complete()
        assert any("'max_tokens'" in e for e in errors)
        assert any("'bogus'" in e for e in errors)

    def test_in_range_values_are_unchanged(self, agent_builder: AgentArtifactBuilder):
        warnings = agent_builder.set_config(
            model="gpt-4o",
            max_iterations=25,
            timeout_seconds=30,
            temperature=0,
            max_tokens=2048,
        ).take_warnings()
        assert warnings == []
        assert agent_builder.config == {
            "model": "gpt-4o",
            "max_iterations": 25,
            "timeout_seconds": 30,
            "temperature": 0,
            "max_tokens": 2048,
        }
        assert AgentArtifactBuilder().set_config(temperature=2).take_warnings() == []

    def test_artifact_config_accepts_every_agentconfig_key(self):
        """Step 9 interaction: the artifact builder is judged by AgentConfig only.

        ``MetaBuilderConfig`` refuses unread fields such as ``enable_prompt_cache``
        and ``max_history_size``; they are valid data about the generated agent
        and must not be rejected here. The builder's own allow-list still decides
        which keys ``set_config`` takes; ``validate_complete`` accepts a config
        dict carrying them.
        """
        from fsm_llm.agents.definitions import AgentConfig, MetaBuilderConfig

        with pytest.raises(ValueError):
            MetaBuilderConfig(enable_prompt_cache=True)
        assert AgentConfig(enable_prompt_cache=True, max_history_size=7)

        b = AgentArtifactBuilder()
        b.set_agent_type("react").set_overview("A", "d")
        b.add_tool("t", "d")
        b.config.update(enable_prompt_cache=True, max_history_size=7)
        assert b.validate_complete() == []
        b.config["max_history_size"] = 0
        assert any("'max_history_size'" in e for e in b.validate_complete())


class TestAgentBuilderSetType:
    def test_all_valid_types_accepted(self, agent_builder: AgentArtifactBuilder):
        for agent_type in AgentArtifactBuilder.VALID_AGENT_TYPES:
            b = AgentArtifactBuilder()
            warnings = b.set_agent_type(agent_type).take_warnings()
            assert warnings == []
            assert b.agent_type == agent_type

    def test_invalid_type_raises(self, agent_builder: AgentArtifactBuilder):
        with pytest.raises(BuilderError, match="Unknown agent type"):
            agent_builder.set_agent_type("invalid")
        # Should NOT have set the type
        assert agent_builder.agent_type is None

    def test_type_is_normalized(self, agent_builder: AgentArtifactBuilder):
        agent_builder.set_agent_type("  REACT  ")
        assert agent_builder.agent_type == "react"


# ---- Exception Classes -----------------------------------------------


class TestExceptionAttributes:
    def test_builder_error_action(self):
        e = BuilderError("test error", action="add_state")
        assert e.action == "add_state"
        assert str(e) == "test error"

    def test_builder_error_no_action(self):
        e = BuilderError("test error")
        assert e.action is None

    def test_meta_validation_error_errors(self):
        from fsm_llm.agents.exceptions import MetaValidationError

        e = MetaValidationError("validation failed", errors=["err1", "err2"])
        assert e.errors == ["err1", "err2"]

    def test_meta_validation_error_default_errors(self):
        from fsm_llm.agents.exceptions import MetaValidationError

        e = MetaValidationError("validation failed")
        assert e.errors == []

    def test_output_error_path(self):
        from fsm_llm.agents.exceptions import OutputError

        e = OutputError("write failed", path="/tmp/test.json")
        assert e.path == "/tmp/test.json"

    def test_output_error_no_path(self):
        from fsm_llm.agents.exceptions import OutputError

        e = OutputError("write failed")
        assert e.path is None

    def test_exception_hierarchy(self):
        from fsm_llm.agents.exceptions import (
            BuilderError,
            MetaBuilderError,
            MetaValidationError,
            OutputError,
        )

        assert issubclass(MetaBuilderError, Exception)
        assert issubclass(BuilderError, MetaBuilderError)
        assert issubclass(MetaValidationError, MetaBuilderError)
        assert issubclass(OutputError, MetaBuilderError)


# ---- Definitions Validators ------------------------------------------


class TestMetaBuilderConfigValidators:
    """MetaBuilderConfig inherits from AgentConfig. AgentConfig now validates
    temperature (0.0-2.0) and max_tokens (>= 1). These tests verify enforcement."""

    def test_temperature_below_zero_rejected(self):
        import pytest

        from fsm_llm.agents.definitions import MetaBuilderConfig

        with pytest.raises(Exception, match="temperature"):
            MetaBuilderConfig(temperature=-0.1)

    def test_temperature_above_two_rejected(self):
        import pytest

        from fsm_llm.agents.definitions import MetaBuilderConfig

        with pytest.raises(Exception, match="temperature"):
            MetaBuilderConfig(temperature=2.1)

    def test_temperature_boundary_zero(self):
        from fsm_llm.agents.definitions import MetaBuilderConfig

        config = MetaBuilderConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_boundary_two(self):
        from fsm_llm.agents.definitions import MetaBuilderConfig

        config = MetaBuilderConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_max_tokens_zero_rejected(self):
        import pytest

        from fsm_llm.agents.definitions import MetaBuilderConfig

        with pytest.raises(Exception, match="max_tokens"):
            MetaBuilderConfig(max_tokens=0)

    def test_max_tokens_negative_rejected(self):
        import pytest

        from fsm_llm.agents.definitions import MetaBuilderConfig

        with pytest.raises(Exception, match="max_tokens"):
            MetaBuilderConfig(max_tokens=-1)

    def test_max_tokens_valid(self):
        from fsm_llm.agents.definitions import MetaBuilderConfig

        config = MetaBuilderConfig(max_tokens=1)
        assert config.max_tokens == 1


# ---- Summary Detail Levels Content Assertions ------------------------


class TestSummaryContentAssertions:
    """Replace fragile length-based test with content assertions."""

    def test_minimal_omits_extraction_instructions(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        summary = populated_fsm_builder.get_summary("minimal")
        assert "extraction:" not in summary
        assert "response:" not in summary

    def test_standard_omits_extraction_instructions(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        summary = populated_fsm_builder.get_summary("standard")
        assert "extraction:" not in summary

    def test_full_includes_extraction_instructions(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        summary = populated_fsm_builder.get_summary("full")
        assert "extraction:" in summary

    def test_minimal_has_state_count(self, populated_fsm_builder: FSMArtifactBuilder):
        summary = populated_fsm_builder.get_summary("minimal")
        assert "States (3)" in summary

    def test_standard_has_transition_targets(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        summary = populated_fsm_builder.get_summary("standard")
        assert "-> ask_name" in summary

    def test_full_has_persona(self, populated_fsm_builder: FSMArtifactBuilder):
        summary = populated_fsm_builder.get_summary("full")
        assert "Persona:" in summary
        assert "Friendly assistant" in summary


class _NoSuperFSMBuilder(FSMArtifactBuilder):
    """A subclass that skips ``super().__init__()``."""

    def __init__(self) -> None:
        self.name = None
        self.description = None
        self.persona = None
        self.initial_state = None
        self.states = {}


class TestLazyWarnings:
    """``_warnings`` exists for a builder subclass that skips ``__init__``."""

    def test_subclass_without_super_init_survives(self):
        b = _NoSuperFSMBuilder()
        assert b.take_warnings() == []
        assert _NoSuperFSMBuilder().take_warnings() == []
        b.set_overview("Bot", "A bot")
        b.add_state("a", "desc", "purpose")
        b.update_state("a", description=None)
        assert any("None" in w for w in b.take_warnings())
        registry = create_fsm_tools(b)
        for name in ("validate", "get_summary"):
            out = registry.get(name).execute_fn()
            assert isinstance(out, str)

    def test_instances_do_not_share_warnings(self):
        a, b = FSMArtifactBuilder(), FSMArtifactBuilder()
        a._warnings.append("x")
        assert b.take_warnings() == []

    def test_copy_and_pickle_keep_warnings(self):
        b = FSMArtifactBuilder()
        b._warnings.append("kept")
        assert copy.deepcopy(b).take_warnings() == ["kept"]
        assert pickle.loads(pickle.dumps(b)).take_warnings() == ["kept"]
