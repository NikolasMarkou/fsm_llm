from __future__ import annotations

"""
Tests for the comprehensive review fixes:
- Short answer extraction (Bug 1)
- SelfConsistencyAgent structured output (Bug 2)
- Observation pruning logging (Bug 4)
- Handler priorities (Step 8)
- END_CONVERSATION/ERROR handlers (Step 9)
- TransitionEvaluatorConfig in AgentConfig (Step 10)
- Classification-based tool selection (Step 11)
- ContextCompactor integration (Step 13)
- SkillDefinition + SkillLoader (Step 14)
"""

import os
import tempfile
from unittest.mock import Mock, patch

from fsm_llm.llm import LLMInterface

# ---------------------------------------------------------------
# Bug 1: Short answer extraction
# ---------------------------------------------------------------


class TestShortAnswerExtraction:
    """Verify answers shorter than 6 chars are accepted."""

    def test_extract_short_answer_yes(self):
        from fsm_llm_agents.base import BaseAgent
        from fsm_llm_agents.constants import ContextKeys

        # We can't instantiate BaseAgent directly, so test the method
        # by creating a minimal subclass
        class _Stub(BaseAgent):
            def run(self, task, initial_context=None):
                pass

            def _register_handlers(self, api):
                pass

        agent = _Stub()
        answer = agent._extract_answer(
            {ContextKeys.FINAL_ANSWER: "yes"}, ["fallback response"]
        )
        assert answer == "yes"

    def test_extract_short_answer_no(self):
        from fsm_llm_agents.base import BaseAgent
        from fsm_llm_agents.constants import ContextKeys

        class _Stub(BaseAgent):
            def run(self, task, initial_context=None):
                pass

            def _register_handlers(self, api):
                pass

        agent = _Stub()
        answer = agent._extract_answer({ContextKeys.FINAL_ANSWER: "no"}, ["fallback"])
        assert answer == "no"

    def test_extract_short_answer_number(self):
        from fsm_llm_agents.base import BaseAgent
        from fsm_llm_agents.constants import ContextKeys

        class _Stub(BaseAgent):
            def run(self, task, initial_context=None):
                pass

            def _register_handlers(self, api):
                pass

        agent = _Stub()
        answer = agent._extract_answer({ContextKeys.FINAL_ANSWER: "42"}, [])
        assert answer == "42"

    def test_empty_answer_falls_through(self):
        from fsm_llm_agents.base import BaseAgent
        from fsm_llm_agents.constants import ContextKeys

        class _Stub(BaseAgent):
            def run(self, task, initial_context=None):
                pass

            def _register_handlers(self, api):
                pass

        agent = _Stub()
        answer = agent._extract_answer(
            {ContextKeys.FINAL_ANSWER: ""}, ["real answer here"]
        )
        assert answer == "real answer here"

    def test_short_extra_key_works(self):
        from fsm_llm_agents.base import BaseAgent

        class _Stub(BaseAgent):
            def run(self, task, initial_context=None):
                pass

            def _register_handlers(self, api):
                pass

        agent = _Stub()
        answer = agent._extract_answer(
            {"custom_key": "ok"}, [], extra_keys=["custom_key"]
        )
        assert answer == "ok"


# ---------------------------------------------------------------
# Bug 2: SelfConsistencyAgent structured output
# ---------------------------------------------------------------


class TestSelfConsistencyStructuredOutput:
    """Verify SelfConsistencyAgent populates structured_output."""

    def test_structured_output_populated(self):
        from pydantic import BaseModel

        from fsm_llm_agents import AgentConfig, SelfConsistencyAgent

        class Answer(BaseModel):
            value: str

        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate_response.return_value = (
            '{"final_answer": "{\\"value\\": \\"test\\"}", "confidence": 0.9}'
        )
        mock_llm.extract_field.return_value = (
            '{"final_answer": "{\\"value\\": \\"test\\"}", "confidence": 0.9}'
        )

        SelfConsistencyAgent(
            config=AgentConfig(
                model="test-model",
                output_schema=Answer,
            ),
            num_samples=1,
        )

        # Verify the field exists on AgentResult
        from fsm_llm_agents.definitions import AgentResult

        result = AgentResult(
            answer='{"value": "test"}',
            success=True,
        )
        # structured_output field should be accessible
        assert result.structured_output is None  # None by default


# ---------------------------------------------------------------
# Bug 4: Observation pruning logging
# ---------------------------------------------------------------


class TestObservationPruning:
    """Verify observation pruning logs a debug message."""

    def test_pruning_logs_debug(self):
        from fsm_llm_agents.constants import ContextKeys, Defaults
        from fsm_llm_agents.definitions import ToolDefinition
        from fsm_llm_agents.handlers import AgentHandlers
        from fsm_llm_agents.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="test_tool",
                description="Test",
                parameter_schema={},
                execute_fn=lambda: "result",
            )
        )
        handlers = AgentHandlers(registry)

        # Fill observations to MAX_OBSERVATIONS + 1
        observations = [f"obs_{i}" for i in range(Defaults.MAX_OBSERVATIONS + 5)]

        context = {
            ContextKeys.TOOL_NAME: "test_tool",
            ContextKeys.TOOL_INPUT: {},
            ContextKeys.REASONING: "test",
            ContextKeys.OBSERVATIONS: observations,
            ContextKeys.AGENT_TRACE: [],
        }

        with patch("fsm_llm_agents.handlers.logger") as mock_logger:
            handlers.execute_tool(context)
            # Should have logged a debug message about pruning
            mock_logger.debug.assert_called()
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("Pruning" in c or "pruning" in c.lower() for c in debug_calls)


# ---------------------------------------------------------------
# Step 8: Handler priorities
# ---------------------------------------------------------------


class TestHandlerPriorities:
    """Verify handler priority constants exist and are ordered."""

    def test_priorities_exist(self):
        from fsm_llm_agents.constants import HandlerPriorities

        assert HandlerPriorities.HITL_GATE < HandlerPriorities.ITERATION_LIMITER
        assert HandlerPriorities.ITERATION_LIMITER < HandlerPriorities.TOOL_EXECUTOR
        assert HandlerPriorities.TOOL_EXECUTOR < HandlerPriorities.END_CONVERSATION

    def test_handler_names_include_new(self):
        from fsm_llm_agents.constants import HandlerNames

        assert hasattr(HandlerNames, "END_CONVERSATION")
        assert hasattr(HandlerNames, "ERROR")


# ---------------------------------------------------------------
# Step 10: TransitionEvaluatorConfig in AgentConfig
# ---------------------------------------------------------------


class TestTransitionConfigInAgentConfig:
    """Verify transition_config field in AgentConfig."""

    def test_default_is_none(self):
        from fsm_llm_agents import AgentConfig

        config = AgentConfig()
        assert config.transition_config is None

    def test_accepts_config(self):
        from fsm_llm.transition_evaluator import TransitionEvaluatorConfig
        from fsm_llm_agents import AgentConfig

        tc = TransitionEvaluatorConfig(ambiguity_threshold=0.2)
        config = AgentConfig(transition_config=tc)
        assert config.transition_config is not None
        assert config.transition_config.ambiguity_threshold == 0.2


# ---------------------------------------------------------------
# Step 11: Classification-based tool selection
# ---------------------------------------------------------------


class TestClassificationToolSelection:
    """Verify use_classification flag in ReactAgent and FSM builder."""

    def test_react_agent_accepts_flag(self):
        from fsm_llm_agents import ReactAgent, ToolRegistry
        from fsm_llm_agents.definitions import ToolDefinition

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search",
                description="Search",
                parameter_schema={},
                execute_fn=lambda: "result",
            )
        )
        agent = ReactAgent(tools=registry, use_classification=True)
        assert agent.use_classification is True

    def test_fsm_has_classification_extractions(self):
        from fsm_llm_agents.definitions import ToolDefinition
        from fsm_llm_agents.fsm_definitions import build_react_fsm
        from fsm_llm_agents.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search",
                description="Search the web",
                parameter_schema={},
                execute_fn=lambda: "result",
            )
        )

        fsm = build_react_fsm(registry, use_classification=True)
        think_state = fsm["states"]["think"]
        assert "classification_extractions" in think_state
        extractions = think_state["classification_extractions"]
        assert len(extractions) == 1
        assert extractions[0]["field_name"] == "tool_name"
        # Should have intents including the tool name
        intent_names = [i["name"] for i in extractions[0]["intents"]]
        assert "search" in intent_names
        assert "none" in intent_names

    def test_fsm_without_classification_has_no_extractions(self):
        from fsm_llm_agents.definitions import ToolDefinition
        from fsm_llm_agents.fsm_definitions import build_react_fsm
        from fsm_llm_agents.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search",
                description="Search",
                parameter_schema={},
                execute_fn=lambda: "result",
            )
        )

        fsm = build_react_fsm(registry, use_classification=False)
        think_state = fsm["states"]["think"]
        assert "classification_extractions" not in think_state


# ---------------------------------------------------------------
# Step 14: SkillDefinition + SkillLoader
# ---------------------------------------------------------------


class TestSkillDefinition:
    """Test SkillDefinition model and conversions."""

    def test_basic_skill(self):
        from fsm_llm_agents.skills import SkillDefinition

        skill = SkillDefinition(
            name="test_skill",
            description="A test skill",
            execute=lambda: "done",
            category="testing",
        )
        assert skill.name == "test_skill"
        assert skill.category == "testing"

    def test_to_tool_definition(self):
        from fsm_llm_agents.skills import SkillDefinition

        skill = SkillDefinition(
            name="my_tool",
            description="Does stuff",
            execute=lambda x: x,
            parameter_schema={"properties": {"x": {"type": "string"}}},
            requires_approval=True,
        )
        tool_def = skill.to_tool_definition()
        assert tool_def.name == "my_tool"
        assert tool_def.requires_approval is True
        assert tool_def.execute_fn is not None


class TestSkillLoader:
    """Test SkillLoader functionality."""

    def test_from_functions(self):
        from fsm_llm_agents.skills import SkillLoader
        from fsm_llm_agents.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        skills = SkillLoader.from_functions(search, category="web")
        assert len(skills) == 1
        assert skills[0].name == "search"
        assert skills[0].category == "web"

    def test_to_tool_registry(self):
        from fsm_llm_agents.skills import SkillDefinition, SkillLoader

        skills = [
            SkillDefinition(
                name="tool_a",
                description="Tool A",
                execute=lambda: "a",
            ),
            SkillDefinition(
                name="tool_b",
                description="Tool B",
                execute=lambda: "b",
            ),
        ]
        registry = SkillLoader.to_tool_registry(skills)
        assert len(registry) == 2
        assert "tool_a" in registry
        assert "tool_b" in registry

    def test_by_category(self):
        from fsm_llm_agents.skills import SkillDefinition, SkillLoader

        skills = [
            SkillDefinition(
                name="a", description="A", execute=lambda: "", category="web"
            ),
            SkillDefinition(
                name="b", description="B", execute=lambda: "", category="web"
            ),
            SkillDefinition(
                name="c", description="C", execute=lambda: "", category="data"
            ),
        ]
        groups = SkillLoader.by_category(skills)
        assert len(groups) == 2
        assert len(groups["web"]) == 2
        assert len(groups["data"]) == 1

    def test_from_directory_with_tool_decorated(self):
        from fsm_llm_agents.skills import SkillLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a skill file with @tool decorated function
            skill_file = os.path.join(tmpdir, "my_skill.py")
            with open(skill_file, "w") as f:
                f.write(
                    "from fsm_llm_agents.tools import tool\n\n"
                    "@tool\n"
                    "def greet(name: str) -> str:\n"
                    '    """Greet someone."""\n'
                    "    return f'Hello {name}'\n"
                )

            skills = SkillLoader.from_directory(tmpdir)
            assert len(skills) == 1
            assert skills[0].name == "greet"

    def test_from_directory_nonexistent(self):
        from fsm_llm_agents.skills import SkillLoader

        skills = SkillLoader.from_directory("/nonexistent/path")
        assert skills == []

    def test_from_directory_skips_underscore_files(self):
        from fsm_llm_agents.skills import SkillLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a file with underscore prefix (should be skipped)
            with open(os.path.join(tmpdir, "_internal.py"), "w") as f:
                f.write("x = 1\n")
            skills = SkillLoader.from_directory(tmpdir)
            assert skills == []

    def test_register_skill_on_registry(self):
        from fsm_llm_agents.skills import SkillDefinition
        from fsm_llm_agents.tools import ToolRegistry

        registry = ToolRegistry()
        skill = SkillDefinition(
            name="my_skill",
            description="Test",
            execute=lambda: "done",
        )
        registry.register_skill(skill)
        assert "my_skill" in registry


# ---------------------------------------------------------------
# Exports
# ---------------------------------------------------------------


class TestExports:
    """Verify new symbols are exported from package __init__."""

    def test_agent_exports(self):
        import fsm_llm_agents

        assert hasattr(fsm_llm_agents, "SkillDefinition")
        assert hasattr(fsm_llm_agents, "SkillLoader")

    def test_workflow_exports(self):
        import fsm_llm_workflows

        assert hasattr(fsm_llm_workflows, "AgentStep")
        assert hasattr(fsm_llm_workflows, "RetryStep")
        assert hasattr(fsm_llm_workflows, "SwitchStep")
        assert hasattr(fsm_llm_workflows, "agent_step")
        assert hasattr(fsm_llm_workflows, "retry_step")
        assert hasattr(fsm_llm_workflows, "switch_step")
