from __future__ import annotations

"""Tests for meta-agent builder tools."""

import pytest

from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.meta_builders import (
    AgentArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from fsm_llm.agents.meta_tools import (
    create_agent_tools,
    create_builder_tools,
    create_fsm_tools,
    create_workflow_tools,
)


class TestFSMTools:
    def test_creates_registry(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        names = [t.name for t in registry.list_tools()]
        assert "set_overview" in names
        assert "add_state" in names
        assert "add_transition" in names
        assert "remove_state" in names
        assert "remove_transition" in names
        assert "set_initial_state" in names
        assert "validate" in names
        assert "get_summary" in names

    def test_set_overview(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(
            _make_call("set_overview", name="Bot", description="A bot", persona="Nice")
        )
        assert result.success
        assert fsm_builder.name == "Bot"
        assert fsm_builder.description == "A bot"
        assert fsm_builder.persona == "Nice"

    def test_add_state(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(
            _make_call("add_state", state_id="s1", description="S1", purpose="P1")
        )
        assert result.success
        assert "s1" in fsm_builder.states
        assert fsm_builder.initial_state == "s1"  # Auto-set

    def test_add_multiple_states(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        for sid in ["a", "b", "c"]:
            result = registry.execute(
                _make_call(
                    "add_state",
                    state_id=sid,
                    description=f"State {sid}",
                    purpose=f"P-{sid}",
                )
            )
            assert result.success
        assert len(fsm_builder.states) == 3
        assert fsm_builder.initial_state == "a"

    def test_add_transition(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="a", description="A", purpose="PA")
        )
        registry.execute(
            _make_call("add_state", state_id="b", description="B", purpose="PB")
        )
        result = registry.execute(
            _make_call(
                "add_transition", from_state="a", target_state="b", description="A to B"
            )
        )
        assert result.success
        assert len(fsm_builder.states["a"]["transitions"]) == 1

    def test_add_transition_missing_state_returns_error(
        self, fsm_builder: FSMArtifactBuilder
    ):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(
            _make_call(
                "add_transition", from_state="x", target_state="y", description="bad"
            )
        )
        assert result.success  # Tool doesn't crash
        assert "Error" in result.result

    def test_remove_state(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="s1", description="S", purpose="P")
        )
        result = registry.execute(_make_call("remove_state", state_id="s1"))
        assert result.success
        assert "s1" not in fsm_builder.states

    def test_validate_empty_returns_errors(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(_make_call("validate"))
        assert result.success
        assert "ERRORS" in result.result

    def test_validate_complete_returns_valid(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        registry = create_fsm_tools(populated_fsm_builder)
        result = registry.execute(_make_call("validate"))
        assert result.success
        assert "ERRORS" not in result.result

    def test_get_summary(self, populated_fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(populated_fsm_builder)
        result = registry.execute(_make_call("get_summary"))
        assert result.success
        assert "GreetingBot" in result.result

    def test_update_state(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="s1", description="Old", purpose="P")
        )
        result = registry.execute(
            _make_call("update_state", state_id="s1", description="New")
        )
        assert result.success
        assert fsm_builder.states["s1"]["description"] == "New"

    def test_set_initial_state(self, fsm_builder: FSMArtifactBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="a", description="A", purpose="PA")
        )
        registry.execute(
            _make_call("add_state", state_id="b", description="B", purpose="PB")
        )
        result = registry.execute(_make_call("set_initial_state", state_id="b"))
        assert result.success
        assert fsm_builder.initial_state == "b"


class TestWorkflowTools:
    def test_creates_registry(self, workflow_builder: WorkflowArtifactBuilder):
        registry = create_workflow_tools(workflow_builder)
        names = [t.name for t in registry.list_tools()]
        assert "set_overview" in names
        assert "add_step" in names
        assert "set_step_transition" in names
        assert "validate" in names

    def test_set_overview(self, workflow_builder: WorkflowArtifactBuilder):
        registry = create_workflow_tools(workflow_builder)
        result = registry.execute(
            _make_call(
                "set_overview", workflow_id="wf1", name="Flow", description="A flow"
            )
        )
        assert result.success
        assert workflow_builder.name == "Flow"

    def test_add_step(self, workflow_builder: WorkflowArtifactBuilder):
        registry = create_workflow_tools(workflow_builder)
        result = registry.execute(
            _make_call(
                "add_step", step_id="start", step_type="auto_transition", name="Start"
            )
        )
        assert result.success
        assert "start" in workflow_builder.steps

    def test_set_step_transition(self, workflow_builder: WorkflowArtifactBuilder):
        registry = create_workflow_tools(workflow_builder)
        registry.execute(
            _make_call("add_step", step_id="a", step_type="auto_transition", name="A")
        )
        registry.execute(
            _make_call("add_step", step_id="b", step_type="auto_transition", name="B")
        )
        result = registry.execute(
            _make_call("set_step_transition", from_step="a", to_step="b")
        )
        assert result.success


class TestAgentTools:
    def test_creates_registry(self, agent_builder: AgentArtifactBuilder):
        registry = create_agent_tools(agent_builder)
        names = [t.name for t in registry.list_tools()]
        assert "set_overview" in names
        assert "set_agent_type" in names
        assert "add_tool" in names
        assert "validate" in names

    def test_set_overview(self, agent_builder: AgentArtifactBuilder):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(
            _make_call("set_overview", name="MyAgent", description="An agent")
        )
        assert result.success
        assert agent_builder.name == "MyAgent"

    def test_set_agent_type(self, agent_builder: AgentArtifactBuilder):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(_make_call("set_agent_type", agent_type="react"))
        assert result.success
        assert agent_builder.agent_type == "react"

    def test_set_agent_type_invalid_returns_error(
        self, agent_builder: AgentArtifactBuilder
    ):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(_make_call("set_agent_type", agent_type="invalid"))
        assert result.success
        assert "Error" in result.result

    def test_add_tool(self, agent_builder: AgentArtifactBuilder):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(
            _make_call("add_tool", name="search", description="Search the web")
        )
        assert result.success
        assert len(agent_builder.tools) == 1

    def test_remove_tool(self, agent_builder: AgentArtifactBuilder):
        registry = create_agent_tools(agent_builder)
        registry.execute(_make_call("add_tool", name="search", description="Search"))
        result = registry.execute(_make_call("remove_tool", name="search"))
        assert result.success
        assert len(agent_builder.tools) == 0


class TestCreateBuilderTools:
    def test_dispatch_fsm(self, fsm_builder: FSMArtifactBuilder):
        registry = create_builder_tools(fsm_builder, ArtifactType.FSM)
        assert any(t.name == "add_state" for t in registry.list_tools())

    def test_dispatch_workflow(self, workflow_builder: WorkflowArtifactBuilder):
        registry = create_builder_tools(workflow_builder, ArtifactType.WORKFLOW)
        assert any(t.name == "add_step" for t in registry.list_tools())

    def test_dispatch_agent(self, agent_builder: AgentArtifactBuilder):
        registry = create_builder_tools(agent_builder, ArtifactType.AGENT)
        assert any(t.name == "set_agent_type" for t in registry.list_tools())

    def test_type_mismatch_raises(self, fsm_builder: FSMArtifactBuilder):
        with pytest.raises(TypeError):
            create_builder_tools(fsm_builder, ArtifactType.WORKFLOW)


class TestToolReplyText:
    """Golden reply text of every meta tool (byte-identical to base 0f03810).

    Each script runs on a fresh builder, so a later reply depends on the
    earlier calls (overwrite warnings, auto-set initial state, removals).
    """

    def test_fsm_tool_replies(self) -> None:
        reg = create_fsm_tools(FSMArtifactBuilder())
        script = [
            (
                "add_transition",
                {"from_state": "a", "target_state": "b", "description": "early"},
                "Error: Source state 'a' not found",
            ),
            (
                "update_state",
                {"state_id": "a", "description": "x"},
                "Error: State 'a' not found",
            ),
            ("set_initial_state", {"state_id": "a"}, "Error: State 'a' not found"),
            (
                "validate",
                {},
                "ERRORS: FSM name is required; FSM description is required; At least one state is required; Initial state is required",
            ),
            (
                "get_summary",
                {},
                "=== FSM Builder Status ===\nName: (not set)\nDescription: (not set)\nInitial state: (not set)\nStates (0):\n\nStill missing:\n  - FSM name\n  - FSM description\n  - At least one state",
            ),
            (
                "set_overview",
                {"name": "Bot", "description": "A bot", "persona": "Nice"},
                "Overview set: name='Bot'",
            ),
            (
                "set_overview",
                {"name": "Bot2", "description": "B"},
                "Overview set: name='Bot2'",
            ),
            (
                "add_state",
                {"state_id": "a", "description": "A", "purpose": "PA"},
                "Added state 'a' (warnings: Auto-set initial state to 'a' (first state added))",
            ),
            (
                "add_state",
                {
                    "state_id": "a",
                    "description": "A again",
                    "purpose": "PA2",
                    "extraction_instructions": "e",
                    "response_instructions": "r",
                },
                "Added state 'a' (warnings: State 'a' already exists; overwriting)",
            ),
            (
                "add_state",
                {"state_id": "b", "description": "B", "purpose": "PB"},
                "Added state 'b'",
            ),
            (
                "add_state",
                {"state_id": "", "description": "B", "purpose": "PB"},
                "Error: State ID cannot be empty",
            ),
            (
                "update_state",
                {"state_id": "a", "description": "A upd", "purpose": "P upd"},
                "Updated state 'a'",
            ),
            (
                "update_state",
                {"state_id": "zz", "description": "no"},
                "Error: State 'zz' not found",
            ),
            (
                "add_transition",
                {"from_state": "a", "target_state": "b", "description": "A to B"},
                "Added transition 'a' -> 'b'",
            ),
            (
                "add_transition",
                {
                    "from_state": "a",
                    "target_state": "b",
                    "description": "dup",
                    "priority": 5,
                },
                "Added transition 'a' -> 'b'",
            ),
            (
                "add_transition",
                {
                    "from_state": "a",
                    "target_state": "missing",
                    "description": "to missing",
                },
                "Error: Target state 'missing' not found",
            ),
            (
                "add_transition",
                {"from_state": "nope", "target_state": "b", "description": "bad from"},
                "Error: Source state 'nope' not found",
            ),
            (
                "add_transition",
                {
                    "from_state": "a",
                    "target_state": "b",
                    "description": "bad prio",
                    "priority": 5000,
                },
                "Added transition 'a' -> 'b'",
            ),
            (
                "validate",
                {},
                "ERRORS: states -> a -> transitions -> 2 -> priority: Input should be less than or equal to 1000",
            ),
            (
                "get_summary",
                {},
                "=== FSM Builder Status ===\nName: Bot2\nDescription: B\nPersona: Nice\nInitial state: a\nStates (2):\n  - a [INITIAL]: A upd\n    extraction: e...\n    response: r...\n    -> b: A to B\n    -> b: dup\n    -> b: bad prio\n  - b [TERMINAL]: B",
            ),
            ("set_initial_state", {"state_id": "b"}, "Initial state set to 'b'"),
            (
                "set_initial_state",
                {"state_id": "nope"},
                "Error: State 'nope' not found",
            ),
            (
                "remove_transition",
                {"from_state": "a", "target_state": "b"},
                "Removed transition",
            ),
            (
                "remove_transition",
                {"from_state": "a", "target_state": "b"},
                "Transition not found",
            ),
            (
                "remove_transition",
                {"from_state": "nope", "target_state": "b"},
                "Transition not found",
            ),
            ("remove_state", {"state_id": "b"}, "Removed state 'b'"),
            ("remove_state", {"state_id": "b"}, "State 'b' not found"),
            ("validate", {}, "ERRORS: Initial state is required"),
            (
                "get_summary",
                {},
                "=== FSM Builder Status ===\nName: Bot2\nDescription: B\nPersona: Nice\nInitial state: (not set)\nStates (1):\n  - a [TERMINAL]: A upd\n    extraction: e...\n    response: r...\n\nStill missing:\n  - Initial state",
            ),
        ]
        assert _replay(reg, script) == [exp for _, _, exp in script]

    def test_workflow_tool_replies(self) -> None:
        reg = create_workflow_tools(WorkflowArtifactBuilder())
        script = [
            (
                "set_step_transition",
                {"from_step": "s1", "to_step": "s2"},
                "Error: Source step 's1' not found",
            ),
            ("set_initial_step", {"step_id": "s1"}, "Error: Step 's1' not found"),
            (
                "validate",
                {},
                "ERRORS: Workflow ID is required; Workflow name is required; At least one step is required; Initial step is required",
            ),
            (
                "get_summary",
                {},
                "=== Workflow Builder Status ===\nID: (not set)\nName: (not set)\nDescription: (not set)\nInitial step: (not set)\nSteps (0):\n\nStill missing:\n  - Workflow ID\n  - Workflow name\n  - At least one step",
            ),
            (
                "set_overview",
                {"workflow_id": "wf1", "name": "WF", "description": "d"},
                "Overview set: name='WF'",
            ),
            (
                "add_step",
                {"step_id": "s1", "step_type": "auto_transition", "name": "S1"},
                "Added step 's1' (auto_transition) (warnings: Auto-set initial step to 's1')",
            ),
            (
                "add_step",
                {
                    "step_id": "s1",
                    "step_type": "condition",
                    "name": "S1 again",
                    "description": "dd",
                },
                "Added step 's1' (condition) (warnings: Step 's1' already exists; overwriting)",
            ),
            (
                "add_step",
                {"step_id": "s2", "step_type": "timer", "name": "S2"},
                "Added step 's2' (timer)",
            ),
            (
                "add_step",
                {"step_id": "s3", "step_type": "bogus", "name": "S3"},
                "Added step 's3' (bogus) (warnings: Unknown step type 'bogus'. Valid: api_call, auto_transition, condition, conversation, llm_processing, parallel, timer, wait_for_event)",
            ),
            (
                "add_step",
                {"step_id": "", "step_type": "timer", "name": "S3"},
                "Error: Step ID cannot be empty",
            ),
            (
                "set_step_transition",
                {"from_step": "s1", "to_step": "s2"},
                "Connected 's1' -> 's2'",
            ),
            (
                "set_step_transition",
                {"from_step": "s1", "to_step": "s2", "condition": "x > 1"},
                "Connected 's1' -> 's2'",
            ),
            (
                "set_step_transition",
                {"from_step": "s1", "to_step": "missing"},
                "Error: Target step 'missing' not found",
            ),
            (
                "set_step_transition",
                {"from_step": "nope", "to_step": "s2"},
                "Error: Source step 'nope' not found",
            ),
            ("validate", {}, "ERRORS: Step 's3' has unknown step type 'bogus'"),
            (
                "get_summary",
                {},
                "=== Workflow Builder Status ===\nID: wf1\nName: WF\nDescription: d\nInitial step: s1\nSteps (3):\n  - s1 [INITIAL] (condition): S1 again\n    -> s2\n    -> s2 [if: x > 1]\n  - s2 [TERMINAL] (timer): S2\n  - s3 [TERMINAL] (bogus): S3",
            ),
            ("set_initial_step", {"step_id": "s2"}, "Initial step set to 's2'"),
            ("set_initial_step", {"step_id": "nope"}, "Error: Step 'nope' not found"),
            ("remove_step", {"step_id": "s2"}, "Removed step 's2'"),
            ("remove_step", {"step_id": "s2"}, "Step 's2' not found"),
            (
                "validate",
                {},
                "ERRORS: Initial step is required; Step 's3' has unknown step type 'bogus'",
            ),
            (
                "get_summary",
                {},
                "=== Workflow Builder Status ===\nID: wf1\nName: WF\nDescription: d\nInitial step: (not set)\nSteps (2):\n  - s1 [TERMINAL] (condition): S1 again\n  - s3 [TERMINAL] (bogus): S3\n\nStill missing:\n  - Initial step",
            ),
        ]
        assert _replay(reg, script) == [exp for _, _, exp in script]

    def test_agent_tool_replies(self) -> None:
        reg = create_agent_tools(AgentArtifactBuilder())
        script = [
            (
                "validate",
                {},
                "ERRORS: Agent type is required; Agent name is required; At least one tool is required",
            ),
            (
                "get_summary",
                {},
                "=== Agent Builder Status ===\nName: (not set)\nDescription: (not set)\nAgent type: (not set)\nConfig: model=gpt-4o-mini, max_iter=10, temp=0.5\nTools (0):\n\nStill missing:\n  - Agent type\n  - Agent name\n  - At least one tool",
            ),
            (
                "set_overview",
                {"name": "Ag", "description": "An agent"},
                "Overview set: name='Ag'",
            ),
            ("set_agent_type", {"agent_type": "react"}, "Agent type set to 'react'"),
            (
                "set_agent_type",
                {"agent_type": "bogus"},
                "Error: Unknown agent type 'bogus'. Valid: adapt, debate, evaluator_optimizer, maker_checker, orchestrator, plan_execute, prompt_chain, react, reflexion, rewoo, self_consistency",
            ),
            (
                "add_tool",
                {"name": "search", "description": "Search"},
                "Added tool 'search'",
            ),
            (
                "add_tool",
                {"name": "search", "description": "Search again"},
                "Added tool 'search' (warnings: Tool 'search' already exists; overwriting)",
            ),
            (
                "add_tool",
                {"name": "", "description": "x"},
                "Error: Tool name cannot be empty",
            ),
            ("remove_tool", {"name": "search"}, "Removed tool 'search'"),
            ("remove_tool", {"name": "search"}, "Tool 'search' not found"),
            ("add_tool", {"name": "calc", "description": "Calc"}, "Added tool 'calc'"),
            ("set_config", {}, "No config fields to update"),
            (
                "set_config",
                {
                    "model": "gpt-x",
                    "max_iterations": 7,
                    "timeout_seconds": 12.5,
                    "temperature": 0.3,
                    "max_tokens": 99,
                },
                "Config updated",
            ),
            ("set_config", {"max_iterations": 100000}, "Config updated"),
            ("validate", {}, "Valid: no errors or warnings"),
            (
                "get_summary",
                {},
                "=== Agent Builder Status ===\nName: Ag\nDescription: An agent\nAgent type: react\nConfig: model=gpt-x, max_iter=100000, temp=0.3\nTools (1):\n  - calc: Calc",
            ),
        ]
        assert _replay(reg, script) == [exp for _, _, exp in script]

    @pytest.mark.parametrize(
        ("artifact_type", "factory", "tool_names", "validate_reply"),
        [
            (
                ArtifactType.FSM,
                FSMArtifactBuilder,
                [
                    "add_state",
                    "add_transition",
                    "get_summary",
                    "remove_state",
                    "remove_transition",
                    "set_initial_state",
                    "set_overview",
                    "update_state",
                    "validate",
                ],
                "ERRORS: FSM name is required; FSM description is required; At least one state is required; Initial state is required",
            ),
            (
                ArtifactType.WORKFLOW,
                WorkflowArtifactBuilder,
                [
                    "add_step",
                    "get_summary",
                    "remove_step",
                    "set_initial_step",
                    "set_overview",
                    "set_step_transition",
                    "validate",
                ],
                "ERRORS: Workflow ID is required; Workflow name is required; At least one step is required; Initial step is required",
            ),
            (
                ArtifactType.AGENT,
                AgentArtifactBuilder,
                [
                    "add_tool",
                    "get_summary",
                    "remove_tool",
                    "set_agent_type",
                    "set_config",
                    "set_overview",
                    "validate",
                ],
                "ERRORS: Agent type is required; Agent name is required; At least one tool is required",
            ),
        ],
    )
    def test_dispatch_tool_set_and_validate_reply(
        self, artifact_type, factory, tool_names, validate_reply
    ) -> None:
        reg = create_builder_tools(factory(), artifact_type)
        assert sorted(t.name for t in reg.list_tools()) == tool_names
        assert reg.execute(_make_call("validate")).result == validate_reply


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_call(tool_name: str, **kwargs):
    """Create a ToolCall for testing."""
    from fsm_llm.agents.definitions import ToolCall

    return ToolCall(tool_name=tool_name, parameters=kwargs)


def _replay(registry, script) -> list[str]:
    """Run (tool, kwargs, _) steps in order and return each tool's reply text."""
    replies = []
    for tool_name, kwargs, _ in script:
        result = registry.execute(_make_call(tool_name, **kwargs))
        assert result.success, result.error
        replies.append(result.result)
    return replies
