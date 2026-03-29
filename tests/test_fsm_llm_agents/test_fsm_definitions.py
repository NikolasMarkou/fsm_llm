from __future__ import annotations

"""Tests for fsm_llm_agents.fsm_definitions module."""

from fsm_llm.definitions import FSMDefinition
from fsm_llm_agents.fsm_definitions import build_react_fsm
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
        conclude_t = [t for t in act_transitions if t["target_state"] == "conclude"][0]
        think_t = [t for t in act_transitions if t["target_state"] == "think"][0]
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
