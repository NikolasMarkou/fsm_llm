from __future__ import annotations

"""Tests for fsm_llm_agents.fsm_definitions module."""

from fsm_llm.definitions import FSMDefinition, State
from fsm_llm.ollama import build_ollama_response_format
from fsm_llm.pipeline import MessagePipeline
from fsm_llm_agents.fsm_definitions import (
    build_orchestrator_fsm,
    build_plan_execute_fsm,
    build_react_fsm,
    build_reflexion_fsm,
    build_rewoo_fsm,
)
from fsm_llm_agents.tools import ToolRegistry


def _dummy_tool(params):
    return "result"


def _make_registry(*tool_names: str) -> ToolRegistry:
    """Create a registry with dummy tools."""
    registry = ToolRegistry()
    for name in tool_names:
        registry.register_function(_dummy_tool, name=name, description=f"Tool: {name}")
    return registry


class TestBuildReactFsm:
    """Tests for build_react_fsm function."""

    def test_basic_fsm_structure(self):
        registry = _make_registry("search", "calculate")
        fsm = build_react_fsm(registry)

        assert fsm["name"] == "react_agent"
        assert fsm["initial_state"] == "think"
        assert "think" in fsm["states"]
        assert "act" in fsm["states"]
        assert "conclude" in fsm["states"]

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        registry = _make_registry("search")
        fsm = build_react_fsm(registry)
        # Should not raise
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "react_agent"

    def test_think_state_has_tool_info(self):
        registry = _make_registry("search", "calc")
        fsm = build_react_fsm(registry)
        think = fsm["states"]["think"]

        assert "extraction_instructions" in think
        assert "search" in think["extraction_instructions"]
        assert "calc" in think["extraction_instructions"]

    def test_think_transitions(self):
        registry = _make_registry("search")
        fsm = build_react_fsm(registry)
        transitions = fsm["states"]["think"]["transitions"]

        targets = {t["target_state"] for t in transitions}
        assert "conclude" in targets
        assert "act" in targets

    def test_conclude_is_terminal(self):
        registry = _make_registry("search")
        fsm = build_react_fsm(registry)
        assert fsm["states"]["conclude"]["transitions"] == []

    def test_act_transitions(self):
        registry = _make_registry("search")
        fsm = build_react_fsm(registry)
        act_transitions = fsm["states"]["act"]["transitions"]
        targets = {t["target_state"] for t in act_transitions}
        assert "think" in targets
        assert "conclude" in targets
        # conclude escape has highest priority (lowest number)
        conclude_t = next(t for t in act_transitions if t["target_state"] == "conclude")
        think_t = next(t for t in act_transitions if t["target_state"] == "think")
        assert conclude_t["priority"] < think_t["priority"]

    def test_with_approval_state(self):
        registry = _make_registry("search")
        fsm = build_react_fsm(registry, include_approval_state=True)

        assert "await_approval" in fsm["states"]

        # Think should have transition to await_approval
        think_targets = {
            t["target_state"] for t in fsm["states"]["think"]["transitions"]
        }
        assert "await_approval" in think_targets

        # Await approval should transition to act or think
        approval_targets = {
            t["target_state"] for t in fsm["states"]["await_approval"]["transitions"]
        }
        assert "act" in approval_targets
        assert "think" in approval_targets

    def test_without_approval_state(self):
        registry = _make_registry("search")
        fsm = build_react_fsm(registry, include_approval_state=False)
        assert "await_approval" not in fsm["states"]

    def test_custom_task_description(self):
        registry = _make_registry("search")
        fsm = build_react_fsm(registry, task_description="Find weather")
        assert fsm["description"] == "Find weather"

    def test_persona_mentions_tools(self):
        registry = _make_registry("search")
        fsm = build_react_fsm(registry)
        assert "tool" in fsm["persona"].lower()


class TestPlanningStatesRequiredContextKeys:
    """Step 2a: the three planner planning states carry required_context_keys
    for their plan field, routing extraction through the typed per-field path
    (mirroring adapt.decompose.subtasks)."""

    def test_plan_execute_plan_state_requires_plan_steps(self):
        registry = _make_registry("search")
        fsm = build_plan_execute_fsm(registry)
        plan_state = fsm["states"]["plan"]
        assert "plan_steps" in plan_state["required_context_keys"]
        # Still a valid FSMDefinition.
        assert FSMDefinition(**fsm).name == "plan_execute_agent"

    def test_orchestrate_state_requires_subtasks(self):
        fsm = build_orchestrator_fsm()
        orchestrate_state = fsm["states"]["orchestrate"]
        assert "subtasks" in orchestrate_state["required_context_keys"]
        FSMDefinition(**fsm)

    def test_rewoo_plan_all_state_requires_plan_blueprint(self):
        registry = _make_registry("search")
        fsm = build_rewoo_fsm(registry)
        plan_all_state = fsm["states"]["plan_all"]
        assert "plan_blueprint" in plan_all_state["required_context_keys"]
        FSMDefinition(**fsm)


def _think_configs(fsm: dict) -> dict:
    """Per-field configs the pipeline derives for the think state, by name."""
    state = State(**fsm["states"]["think"])
    return {
        c.field_name: c for c in MessagePipeline._build_field_configs_from_state(state)
    }


def _value_types(cfg) -> set[str]:
    rf = build_ollama_response_format("field_extraction", cfg.field_type)
    return set(rf["json_schema"]["schema"]["properties"]["value"]["type"])


class TestThinkStateToolSelectionTypes:
    """RA-01b (D-024): under the auto-minted ``any`` grammar (no ``object``) the
    9b model returned null for tool_name/tool_input and no tool ever ran (live
    s15 A/B, 0/3). The think state declares both keys explicitly: tool_name as
    ``str``, tool_input as ``dict`` (its grammar admits an object)."""

    def test_react_declares_typed_tool_selection_fields(self):
        cfgs = _think_configs(build_react_fsm(_make_registry("search")))
        assert cfgs["tool_name"].field_type == "str"
        assert cfgs["tool_input"].field_type == "dict"
        assert cfgs["should_terminate"].field_type == "any"
        assert "object" in _value_types(cfgs["tool_input"])
        assert "object" not in _value_types(cfgs["tool_name"])

    def test_reflexion_declares_typed_tool_selection_fields(self):
        cfgs = _think_configs(build_reflexion_fsm(_make_registry("search")))
        assert cfgs["tool_name"].field_type == "str"
        assert cfgs["tool_input"].field_type == "dict"
        assert "object" in _value_types(cfgs["tool_input"])

    def test_explicit_configs_carry_the_think_instructions(self):
        fsm = build_react_fsm(_make_registry("search"))
        think = fsm["states"]["think"]
        cfgs = _think_configs(fsm)
        for key in ("tool_name", "tool_input"):
            assert think["extraction_instructions"] in cfgs[key].extraction_instructions
            assert key in cfgs[key].extraction_instructions

    def test_classification_owned_tool_name_is_not_redeclared(self):
        # use_classification: tool_name belongs to the classifier (D-006); an
        # explicit config would make the plain extractor fill it again.
        fsm = build_react_fsm(_make_registry("search"), use_classification=True)
        names = [fc["field_name"] for fc in fsm["states"]["think"]["field_extractions"]]
        assert names == ["tool_input"]
        assert "tool_name" not in _think_configs(fsm)

    def test_fsms_stay_valid_definitions(self):
        FSMDefinition(**build_react_fsm(_make_registry("search")))
        FSMDefinition(
            **build_react_fsm(_make_registry("search"), use_classification=True)
        )
        FSMDefinition(**build_reflexion_fsm(_make_registry("search")))
