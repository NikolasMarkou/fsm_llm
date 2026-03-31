"""
Integration tests for the meta builder.

Tests the actual spec-to-builder pipeline, workflow ordering fix,
run() return values, MonitorBuilder, and reachability validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fsm_llm_agents.definitions import ArtifactType, MetaBuilderConfig
from fsm_llm_agents.meta_builder import MetaBuilderAgent
from fsm_llm_agents.meta_builders import (
    FSMBuilder,
    MonitorBuilder,
    WorkflowBuilder,
)
from fsm_llm_agents.meta_fsm import build_meta_builder_fsm
from fsm_llm_agents.meta_prompts import build_welcome_message
from fsm_llm_agents.meta_tools import create_builder_tools, create_monitor_tools

# ---------------------------------------------------------------------------
# BUG-1: Workflow step ordering
# ---------------------------------------------------------------------------


class TestWorkflowStepOrdering:
    """Verify that workflow specs with forward-referencing transitions work."""

    def test_apply_workflow_spec_two_pass(self):
        """Steps added first, transitions second — no BuilderError."""
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._builder = WorkflowBuilder()
        agent._requirements = {"artifact_name": "test", "artifact_description": "test"}
        agent._artifact_type = ArtifactType.WORKFLOW
        agent._build_errors = []
        agent._api_kwargs = {}

        spec = {
            "workflow_id": "test_wf",
            "name": "Test Workflow",
            "description": "A test workflow",
            "initial_step_id": "step_1",
            "steps": [
                {
                    "id": "step_1",
                    "step_type": "auto_transition",
                    "name": "Step 1",
                    "next_step": "step_2",
                },
                {
                    "id": "step_2",
                    "step_type": "llm_processing",
                    "name": "Step 2",
                    "next_step": "step_3",
                },
                {
                    "id": "step_3",
                    "step_type": "auto_transition",
                    "name": "Step 3",
                },
            ],
        }
        agent._apply_workflow_spec(agent._builder, spec)

        builder = agent._builder
        assert len(builder.steps) == 3
        assert "step_1" in builder.steps
        assert "step_2" in builder.steps
        assert "step_3" in builder.steps
        # Transitions should be set correctly
        assert len(builder.steps["step_1"]["transitions"]) == 1
        assert builder.steps["step_1"]["transitions"][0]["target"] == "step_2"
        assert len(builder.steps["step_2"]["transitions"]) == 1
        assert builder.steps["step_2"]["transitions"][0]["target"] == "step_3"
        assert len(builder.steps["step_3"]["transitions"]) == 0

    def test_workflow_spec_with_step_id_key(self):
        """Workflow steps using 'step_id' instead of 'id' key."""
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._builder = WorkflowBuilder()
        agent._requirements = {"artifact_name": "test", "artifact_description": "test"}
        agent._artifact_type = ArtifactType.WORKFLOW
        agent._build_errors = []
        agent._api_kwargs = {}

        spec = {
            "workflow_id": "wf",
            "name": "Test",
            "description": "Test",
            "initial_step_id": "a",
            "steps": [
                {"step_id": "a", "step_type": "auto_transition", "name": "A", "next_step": "b"},
                {"step_id": "b", "step_type": "auto_transition", "name": "B"},
            ],
        }
        agent._apply_workflow_spec(agent._builder, spec)
        assert len(agent._builder.steps) == 2
        assert len(agent._builder.steps["a"]["transitions"]) == 1

    def test_assemble_workflow_ordering_by_construction(self):
        """Hybrid assembly creates sequential transitions — no ordering bug."""
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._builder = None
        agent._artifact_type = None
        agent._requirements = {}
        agent._build_errors = []
        agent._api_kwargs = {}
        agent.meta_config = MetaBuilderConfig()
        agent._call_llm_json = lambda *a, **kw: []

        context = {
            "artifact_type": "workflow",
            "artifact_name": "Pipeline",
            "artifact_description": "Data pipeline",
            "component_names": ["upload", "process", "store", "notify"],
        }
        agent._assemble_artifact(context)

        builder = agent._builder
        assert len(builder.steps) == 4
        # Sequential transitions created by construction
        assert len(builder.steps["upload"]["transitions"]) == 1
        assert builder.steps["upload"]["transitions"][0]["target"] == "process"
        assert len(builder.steps["process"]["transitions"]) == 1
        assert builder.steps["process"]["transitions"][0]["target"] == "store"
        assert len(builder.steps["store"]["transitions"]) == 1
        assert builder.steps["store"]["transitions"][0]["target"] == "notify"
        assert len(builder.steps["notify"]["transitions"]) == 0

    def test_workflow_invalid_next_step_is_skipped(self):
        """Transition to non-existent step is silently skipped."""
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._builder = WorkflowBuilder()
        agent._requirements = {}
        agent._artifact_type = ArtifactType.WORKFLOW
        agent._build_errors = []
        agent._api_kwargs = {}

        spec = {
            "workflow_id": "wf",
            "name": "Test",
            "description": "Test",
            "steps": [
                {"id": "a", "step_type": "auto_transition", "name": "A", "next_step": "missing"},
            ],
        }
        agent._apply_workflow_spec(agent._builder, spec)
        assert len(agent._builder.steps) == 1
        assert len(agent._builder.steps["a"]["transitions"]) == 0


# ---------------------------------------------------------------------------
# BUG-2: run() populates final_context
# ---------------------------------------------------------------------------


class TestRunPopulatesFinalContext:
    """Verify that _build_result populates final_context."""

    def test_build_result_has_final_context(self):
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._artifact_type = ArtifactType.FSM
        agent._turn_count = 1
        agent._builder = FSMBuilder()
        agent._builder.set_overview(name="Test", description="A test FSM")
        agent._builder.add_state("start", "Start", "Start state")
        agent._builder.initial_state = "start"

        agent._build_result()

        assert agent._result is not None
        fc = agent._result.final_context
        assert fc is not None
        assert "artifact_json" in fc
        assert isinstance(fc["artifact_json"], dict)
        assert fc["artifact_json"]["name"] == "Test"
        assert "artifact_type" in fc
        assert fc["artifact_type"] == "fsm"

    def test_build_result_no_builder(self):
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._artifact_type = ArtifactType.FSM
        agent._turn_count = 0
        agent._builder = None

        agent._build_result()

        assert agent._result is not None
        assert agent._result.success is False
        assert agent._result.final_context == {}


# ---------------------------------------------------------------------------
# BUG-3: MonitorBuilder
# ---------------------------------------------------------------------------


class TestMonitorBuilder:
    def test_artifact_type(self):
        b = MonitorBuilder()
        assert b.artifact_type == ArtifactType.MONITOR

    def test_add_panel(self):
        b = MonitorBuilder()
        b.set_overview("Dashboard", "Test dashboard")
        b.add_panel("p1", "CPU Usage", panel_type="gauge", metric="cpu_percent")
        assert "p1" in b.panels
        assert b.panels["p1"]["title"] == "CPU Usage"
        assert b.panels["p1"]["panel_type"] == "gauge"

    def test_add_alert(self):
        b = MonitorBuilder()
        b.add_alert("a1", metric="cpu_percent", condition=">", threshold=90.0)
        assert "a1" in b.alerts
        assert b.alerts["a1"]["threshold"] == 90.0

    def test_remove_panel(self):
        b = MonitorBuilder()
        b.add_panel("p1", "Test", metric="m")
        assert b.remove_panel("p1") is True
        assert "p1" not in b.panels
        assert b.remove_panel("p1") is False

    def test_remove_alert(self):
        b = MonitorBuilder()
        b.add_alert("a1", metric="m")
        assert b.remove_alert("a1") is True
        assert b.remove_alert("a1") is False

    def test_validate_complete_empty(self):
        b = MonitorBuilder()
        errors = b.validate_complete()
        assert any("name" in e.lower() for e in errors)
        assert any("panel" in e.lower() for e in errors)

    def test_validate_complete_valid(self):
        b = MonitorBuilder()
        b.set_overview("Dashboard", "Test")
        b.add_panel("p1", "CPU", metric="cpu_percent")
        assert b.validate_complete() == []

    def test_validate_panel_needs_metric(self):
        b = MonitorBuilder()
        b.set_overview("Dashboard", "Test")
        b.add_panel("p1", "CPU")
        errors = b.validate_complete()
        assert any("metric" in e.lower() for e in errors)

    def test_to_dict(self):
        b = MonitorBuilder()
        b.set_overview("Dashboard", "Test")
        b.add_panel("p1", "CPU", metric="cpu_percent")
        b.add_alert("a1", metric="cpu_percent", condition=">", threshold=90.0)
        d = b.to_dict()
        assert d["name"] == "Dashboard"
        assert "p1" in d["panels"]
        assert "a1" in d["alerts"]
        assert d["config"]["refresh_interval_seconds"] == 30

    def test_set_config(self):
        b = MonitorBuilder()
        b.set_config(refresh_interval_seconds=10)
        assert b.config["refresh_interval_seconds"] == 10

    def test_set_config_unknown_field(self):
        b = MonitorBuilder()
        warnings = b.set_config(unknown_field="x")
        assert len(warnings) == 1

    def test_update_panel(self):
        b = MonitorBuilder()
        b.add_panel("p1", "Old Title", metric="m")
        b.update_panel("p1", title="New Title")
        assert b.panels["p1"]["title"] == "New Title"

    def test_progress(self):
        b = MonitorBuilder()
        p = b.get_progress()
        assert p.percentage == 0.0
        b.set_overview("Dashboard", "Test")
        b.add_panel("p1", "CPU", metric="m")
        p = b.get_progress()
        assert p.percentage == 100.0

    def test_summary_levels(self):
        b = MonitorBuilder()
        b.set_overview("Dashboard", "Test dashboard")
        b.add_panel("p1", "CPU", panel_type="gauge", metric="cpu_percent")
        for level in ["minimal", "standard", "full"]:
            summary = b.get_summary(detail_level=level)
            assert "Dashboard" in summary


class TestMonitorInArtifactType:
    def test_monitor_enum_value(self):
        assert ArtifactType.MONITOR.value == "monitor"

    def test_monitor_from_string(self):
        assert ArtifactType("monitor") == ArtifactType.MONITOR


class TestMonitorTools:
    def test_creates_registry(self):
        b = MonitorBuilder()
        registry = create_monitor_tools(b)
        tool_names = {t.name for t in registry.list_tools()}
        assert "add_panel" in tool_names
        assert "remove_panel" in tool_names
        assert "add_alert" in tool_names
        assert "validate" in tool_names

    def test_dispatch_monitor(self):
        b = MonitorBuilder()
        registry = create_builder_tools(b, ArtifactType.MONITOR)
        assert registry is not None


# ---------------------------------------------------------------------------
# DESIGN-3: Type alias matching (longest first)
# ---------------------------------------------------------------------------


class TestTypeAliasMatching:
    def test_longest_match_first(self):
        aliases = MetaBuilderAgent._build_type_aliases()
        keys = list(aliases.keys())
        # "finite state machine" should come before "state machine"
        assert keys.index("finite state machine") < keys.index("state machine")
        # "data pipeline" should come before "pipeline"
        assert keys.index("data pipeline") < keys.index("pipeline")

    def test_monitor_aliases_present(self):
        aliases = MetaBuilderAgent._build_type_aliases()
        assert aliases.get("dashboard") == "monitor"
        assert aliases.get("monitoring") == "monitor"
        assert aliases.get("telemetry") == "monitor"


# ---------------------------------------------------------------------------
# DESIGN-4: Reachability validation
# ---------------------------------------------------------------------------


class TestReachabilityValidation:
    def test_unreachable_state_detected(self):
        b = FSMBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("mid", "Mid", "Mid")
        b.add_state("orphan", "Orphan", "Orphan")
        b.add_transition("start", "mid", "go")
        b.initial_state = "start"

        errors = b.validate_complete()
        assert any("orphan" in e.lower() for e in errors)

    def test_all_reachable_no_errors(self):
        b = FSMBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("end", "End", "End")
        b.add_transition("start", "end", "finish")
        b.initial_state = "start"

        errors = b.validate_complete()
        unreachable_errors = [e for e in errors if "unreachable" in e.lower()]
        assert len(unreachable_errors) == 0

    def test_transition_to_nonexistent_state(self):
        b = FSMBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("end", "End", "End")
        b.add_transition("start", "end", "finish")
        b.initial_state = "start"
        # Manually inject a bad transition
        b.states["start"]["transitions"].append(
            {"target_state": "ghost", "description": "bad", "priority": 100, "conditions": []}
        )
        errors = b.validate_complete()
        assert any("ghost" in e for e in errors)


# ---------------------------------------------------------------------------
# FSM spec with dict format
# ---------------------------------------------------------------------------


class TestFSMSpecDictFormat:
    """Test that _apply_fsm_spec handles both list and dict format states."""

    def test_dict_format_states(self):
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._builder = FSMBuilder()
        agent._requirements = {"artifact_name": "Test", "artifact_description": "Test"}
        agent._artifact_type = ArtifactType.FSM
        agent._build_errors = []
        agent._api_kwargs = {}

        spec = {
            "name": "Test",
            "description": "Test",
            "initial_state": "greeting",
            "states": {
                "greeting": {
                    "id": "greeting",
                    "description": "Welcome",
                    "purpose": "Greet user",
                    "transitions": [
                        {"target_state": "end", "description": "Finish"}
                    ],
                },
                "end": {
                    "id": "end",
                    "description": "Goodbye",
                    "purpose": "End",
                    "transitions": [],
                },
            },
        }
        agent._apply_fsm_spec(agent._builder, spec)

        assert len(agent._builder.states) == 2
        assert "greeting" in agent._builder.states
        assert "end" in agent._builder.states
        assert agent._builder.initial_state == "greeting"
        assert len(agent._builder.states["greeting"]["transitions"]) == 1

    def test_list_format_states(self):
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._builder = FSMBuilder()
        agent._requirements = {"artifact_name": "Test", "artifact_description": "Test"}
        agent._artifact_type = ArtifactType.FSM
        agent._build_errors = []
        agent._api_kwargs = {}

        spec = {
            "name": "Test",
            "initial_state": "s1",
            "states": [
                {"id": "s1", "description": "State 1", "purpose": "P1"},
                {"id": "s2", "description": "State 2", "purpose": "P2"},
            ],
            "transitions": [
                {"source": "s1", "target": "s2", "description": "Go"},
            ],
        }
        agent._apply_fsm_spec(agent._builder, spec)

        assert len(agent._builder.states) == 2
        assert len(agent._builder.states["s1"]["transitions"]) == 1


# ---------------------------------------------------------------------------
# Monitor spec apply
# ---------------------------------------------------------------------------


class TestMonitorSpecApply:
    def test_apply_monitor_spec(self):
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent._builder = MonitorBuilder()
        agent._requirements = {"artifact_name": "Test", "artifact_description": "Test"}
        agent._artifact_type = ArtifactType.MONITOR
        agent._build_errors = []
        agent._api_kwargs = {}

        spec = {
            "name": "API Dashboard",
            "description": "Monitor API health",
            "panels": [
                {
                    "id": "p1",
                    "title": "Response Time",
                    "panel_type": "chart",
                    "metric": "response_time_ms",
                },
                {
                    "id": "p2",
                    "title": "Error Rate",
                    "panel_type": "gauge",
                    "metric": "error_rate",
                },
            ],
            "alerts": [
                {
                    "id": "a1",
                    "metric": "error_rate",
                    "condition": ">",
                    "threshold": 5.0,
                    "description": "Error rate too high",
                },
            ],
            "config": {"refresh_interval_seconds": 15},
        }
        agent._apply_monitor_spec(agent._builder, spec)

        assert len(agent._builder.panels) == 2
        assert len(agent._builder.alerts) == 1
        assert agent._builder.config["refresh_interval_seconds"] == 15


# ---------------------------------------------------------------------------
# FSM definition includes monitor
# ---------------------------------------------------------------------------


class TestMetaFSMDefinition:
    def test_fsm_includes_monitor_intent(self):
        fsm = build_meta_builder_fsm()
        classify = fsm["states"]["classify"]
        classifications = classify["classification_extractions"]
        intents = classifications[0]["intents"]
        intent_names = [i["name"] for i in intents]
        assert "monitor" in intent_names

    def test_fsm_transition_includes_monitor(self):
        fsm = build_meta_builder_fsm()
        classify = fsm["states"]["classify"]
        for t in classify["transitions"]:
            if t["target_state"] == "collect":
                logic = t["conditions"][0]["logic"]
                assert "monitor" in logic["in"][1]

    def test_has_four_states(self):
        fsm = build_meta_builder_fsm()
        assert len(fsm["states"]) == 4
        assert "classify" in fsm["states"]
        assert "collect" in fsm["states"]
        assert "confirm" in fsm["states"]
        assert "output" in fsm["states"]

    def test_collect_has_field_extractions(self):
        fsm = build_meta_builder_fsm()
        collect = fsm["states"]["collect"]
        assert "field_extractions" in collect
        field_names = [fe["field_name"] for fe in collect["field_extractions"]]
        assert "component_names" in field_names

    def test_classify_has_field_extractions(self):
        fsm = build_meta_builder_fsm()
        classify = fsm["states"]["classify"]
        assert "field_extractions" in classify
        field_names = [fe["field_name"] for fe in classify["field_extractions"]]
        assert "artifact_name" in field_names
        assert "artifact_description" in field_names


# ---------------------------------------------------------------------------
# Welcome message mentions monitors
# ---------------------------------------------------------------------------


class TestWelcomeMessage:
    def test_mentions_monitor(self):
        msg = build_welcome_message()
        assert "Monitor" in msg or "monitor" in msg


# ---------------------------------------------------------------------------
# LLM retry logic
# ---------------------------------------------------------------------------


class TestLLMRetryLogic:
    def test_retries_on_empty_content(self):
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent.meta_config = MetaBuilderConfig()
        agent._api_kwargs = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        call_count = 0

        def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return mock_response
            good_response = MagicMock()
            good_response.choices = [MagicMock()]
            good_response.choices[0].message.content = '{"name": "test"}'
            return good_response

        with patch("fsm_llm_agents.meta_builder.litellm.completion", side_effect=mock_completion):
            result = agent._call_llm_json("system", "user")

        assert result == {"name": "test"}
        assert call_count == 2

    def test_gives_up_after_max_retries(self):
        agent = MetaBuilderAgent.__new__(MetaBuilderAgent)
        agent.meta_config = MetaBuilderConfig()
        agent._api_kwargs = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not json"

        with patch("fsm_llm_agents.meta_builder.litellm.completion", return_value=mock_response):
            result = agent._call_llm_json("system", "user")

        assert result == {}
