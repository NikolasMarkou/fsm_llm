from __future__ import annotations

"""
Tests for new workflow step types: AgentStep, RetryStep, SwitchStep.
Also tests for workflow engine improvements (instance cleanup, event warnings).
"""

from unittest.mock import MagicMock

import pytest

from fsm_llm.workflows.engine import WorkflowEngine
from fsm_llm.workflows.models import WorkflowStepResult
from fsm_llm.workflows.steps import (
    AgentStep,
    AutoTransitionStep,
    RetryStep,
    SwitchStep,
)

# ---------------------------------------------------------------
# SwitchStep
# ---------------------------------------------------------------


class TestSwitchStep:
    """Test n-way branching on context keys."""

    @pytest.fixture
    def switch(self):
        return SwitchStep(
            step_id="route",
            name="Route",
            key="intent",
            cases={"buy": "checkout", "browse": "catalog", "help": "support"},
            default_state="fallback",
        )

    async def test_routes_to_matching_case(self, switch):
        result = await switch.execute({"intent": "buy"})
        assert result.success is True
        assert result.next_state == "checkout"

    async def test_routes_to_default(self, switch):
        result = await switch.execute({"intent": "unknown"})
        assert result.success is True
        assert result.next_state == "fallback"

    async def test_missing_key_uses_default(self, switch):
        result = await switch.execute({})
        assert result.success is True
        assert result.next_state == "fallback"

    async def test_no_default_fails(self):
        switch = SwitchStep(
            step_id="route",
            name="Route",
            key="intent",
            cases={"buy": "checkout"},
        )
        result = await switch.execute({"intent": "unknown"})
        assert result.success is False

    async def test_numeric_value_converted_to_string(self):
        switch = SwitchStep(
            step_id="route",
            name="Route",
            key="level",
            cases={"1": "low", "2": "high"},
            default_state="unknown",
        )
        result = await switch.execute({"level": 1})
        assert result.next_state == "low"


# ---------------------------------------------------------------
# RetryStep
# ---------------------------------------------------------------


class TestRetryStep:
    """Test retry logic with backoff."""

    async def test_succeeds_on_first_try(self):
        inner = AutoTransitionStep(
            step_id="inner",
            name="Inner",
            next_state="done",
        )
        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=inner,
            max_retries=3,
        )
        result = await step.execute({})
        assert result.success is True
        assert result.next_state == "done"

    async def test_retries_on_failure(self):
        call_count = {"n": 0}

        class _FailThenSucceed:
            step_id = "inner"
            name = "Inner"

            async def execute(self, context):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    return WorkflowStepResult.failure_result(
                        error="failed", next_state="err"
                    )
                return WorkflowStepResult.success_result(
                    data={"ok": True}, next_state="done"
                )

        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=_FailThenSucceed(),
            max_retries=3,
            backoff_factor=0.01,  # Fast for testing
        )
        result = await step.execute({})
        assert result.success is True
        assert call_count["n"] == 3

    async def test_exhausts_retries(self):
        class _AlwaysFails:
            step_id = "inner"

            async def execute(self, context):
                return WorkflowStepResult.failure_result(
                    error="always fails", next_state="err"
                )

        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=_AlwaysFails(),
            max_retries=2,
            backoff_factor=0.01,
        )
        result = await step.execute({})
        assert result.success is False

    async def test_retries_on_non_workflow_step_error(self):
        """A non-conforming inner step raising a bare exception (e.g.
        ValueError) must still be retried, not propagated immediately.
        Regression test for D-010 (steps.py RetryStep.execute except clause
        widened from WorkflowStepError to Exception)."""
        call_count = {"n": 0}

        class _RaisesValueErrorThenSucceeds:
            step_id = "inner"
            name = "Inner"

            async def execute(self, context):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise ValueError("boom")
                return WorkflowStepResult.success_result(
                    data={"ok": True}, next_state="done"
                )

        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=_RaisesValueErrorThenSucceeds(),
            max_retries=3,
            backoff_factor=0.01,
        )
        result = await step.execute({})
        assert result.success is True
        assert call_count["n"] == 3

    async def test_exhausts_retries_reraises_non_workflow_step_error(self):
        """On exhaustion, the original non-WorkflowStepError exception type
        propagates unchanged."""

        class _AlwaysRaisesValueError:
            step_id = "inner"

            async def execute(self, context):
                raise ValueError("always fails")

        step = RetryStep(
            step_id="retry",
            name="Retry",
            step=_AlwaysRaisesValueError(),
            max_retries=2,
            backoff_factor=0.01,
        )
        with pytest.raises(ValueError, match="always fails"):
            await step.execute({})


# ---------------------------------------------------------------
# AgentStep
# ---------------------------------------------------------------


class TestAgentStep:
    """Test running agents as workflow steps."""

    async def test_basic_execution(self):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "The answer is 42"
        mock_result.success = True
        mock_result.final_context = {"key": "value"}
        mock_agent.run.return_value = mock_result

        step = AgentStep(
            step_id="research",
            name="Research",
            agent=mock_agent,
            task_template="Research {topic}",
            success_state="analyze",
            context_mapping={"findings": "key"},
        )

        result = await step.execute({"topic": "AI"})

        assert result.success is True
        assert result.next_state == "analyze"
        mock_agent.run.assert_called_once_with("Research AI")
        assert result.data["findings"] == "value"
        assert result.data["agent_answer"] == "The answer is 42"

    async def test_context_mapping_answer_key(self):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Summary text"
        mock_result.success = True
        mock_result.final_context = {"task": "Summarize X"}  # non-empty so mapping runs
        mock_agent.run.return_value = mock_result

        step = AgentStep(
            step_id="summarize",
            name="Summarize",
            agent=mock_agent,
            success_state="next",
            context_mapping={"summary": "answer"},
        )

        result = await step.execute({"task": "Summarize X"})

        assert result.data["summary"] == "Summary text"

    async def test_agent_failure(self):
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Agent crashed")

        step = AgentStep(
            step_id="fail",
            name="Fail",
            agent=mock_agent,
            success_state="next",
            error_state="error",
        )

        result = await step.execute({"task": "Do something"})

        assert result.success is False
        assert result.next_state == "error"

    async def test_missing_template_key(self):
        mock_agent = MagicMock()

        step = AgentStep(
            step_id="fail",
            name="Fail",
            agent=mock_agent,
            task_template="Research {missing_key}",
            success_state="next",
            error_state="error",
        )

        result = await step.execute({})

        assert result.success is False


class TestStepsOverTheRealCore:
    """plan-2026-09-30T062855-07ad3f8c step 14: ``AgentStep`` runs a real
    ``ReactAgent`` (driven by core's bounded run, no user turn), and
    ``ConversationStep.auto_messages`` stays a scripted conversation on
    ``converse``."""

    @pytest.fixture
    def converse_calls(self, monkeypatch):
        from fsm_llm import API

        calls: list[str] = []
        real = API.converse

        def _converse(api, user_message, conversation_id):
            calls.append(user_message)
            return real(api, user_message, conversation_id)

        monkeypatch.setattr(API, "converse", _converse)
        return calls

    async def test_agent_step_runs_a_react_agent_to_success(self, converse_calls):
        pytest.importorskip("fsm_llm.agents")
        from fsm_llm.agents import AgentConfig, ReactAgent, ToolRegistry
        from tests.conftest import PromptGroundedLLM

        answer = "The capital of France is Paris."
        queries: list[str] = []

        def lookup(query: str) -> str:
            queries.append(query)
            return answer

        registry = ToolRegistry()
        registry.register_function(lookup, name="lookup", description="Look up a fact")
        llm = PromptGroundedLLM(
            facts={
                "tool_name": ("lookup", "capital"),
                "tool_input": ({"query": "capital of France"}, "capital"),
                "should_terminate": (True, "is Paris"),
            },
            default_response=answer,
        )
        step = AgentStep(
            step_id="research",
            name="Research",
            agent=ReactAgent(
                tools=registry,
                config=AgentConfig(max_iterations=6),
                llm_interface=llm,
            ),
            task_template="What is the capital of {country}?",
            success_state="report",
            context_mapping={"finding": "answer", "calls": "observation_count"},
        )

        result = await step.execute({"country": "France"})

        assert (result.success, result.next_state) == (True, "report")
        assert queries == ["capital of France"]
        assert result.data["agent_answer"] == answer
        assert result.data["agent_research_success"] is True
        assert result.data["finding"] == answer
        assert result.data["calls"] == 1
        assert converse_calls == []
        assert {request.user_message for _, request in llm.requests} == {None}

    async def test_conversation_step_sends_its_auto_messages_as_user_turns(
        self, converse_calls, monkeypatch
    ):
        from fsm_llm import API
        from fsm_llm.workflows.steps import ConversationStep
        from tests.conftest import MockLLM2Interface

        llm = MockLLM2Interface(extraction_data={"name": "Alice"}, response_text="Hi!")
        real_from_definition = API.from_definition
        monkeypatch.setattr(
            API,
            "from_definition",
            lambda definition, **kwargs: real_from_definition(
                definition, llm_interface=llm
            ),
        )
        name_gate = {
            "description": "name was given",
            "requires_context_keys": ["name"],
            "logic": {"!!": [{"var": "name"}]},
        }
        step = ConversationStep(
            step_id="intake",
            name="Intake",
            fsm_definition={
                "name": "NameCapture",
                "description": "Ask for a name, then greet",
                "initial_state": "ask",
                "states": {
                    "ask": {
                        "id": "ask",
                        "description": "Ask for the name",
                        "purpose": "Learn the user's name",
                        "response_instructions": "Ask for the user's name",
                        "required_context_keys": ["name"],
                        "transitions": [
                            {
                                "target_state": "greet",
                                "description": "The name is known",
                                "conditions": [name_gate],
                            }
                        ],
                    },
                    "greet": {
                        "id": "greet",
                        "description": "Greet by name",
                        "purpose": "Greet the user",
                        "response_instructions": "Greet the user by name",
                    },
                },
            },
            success_state="done",
            auto_messages=["My name is Alice", "never sent: the FSM has ended"],
            context_mapping={"user_name": "name"},
            require_completion=True,
        )

        result = await step.execute({})

        assert (result.success, result.next_state) == (True, "done")
        assert converse_calls == ["My name is Alice"]
        assert result.data["user_name"] == "Alice"
        assert result.data["conversation_intake_ended"] is True
        assert "My name is Alice" in {req.user_message for _, req in llm.call_history}


def _name_capture_fsm(*, ask_speaks: bool) -> dict:
    """Two-state FSM whose terminal ``record`` state is silent (empty
    ``response_instructions``); ``ask`` speaks only when ``ask_speaks``."""
    ask: dict = {
        "id": "ask",
        "description": "Ask for the name",
        "purpose": "Learn the user's name",
        "response_instructions": "Ask for the user's name" if ask_speaks else "",
        "required_context_keys": ["name"],
        "transitions": [
            {
                "target_state": "record",
                "description": "The name is known",
                "conditions": [
                    {
                        "description": "name was given",
                        "requires_context_keys": ["name"],
                        "logic": {"!!": [{"var": "name"}]},
                    }
                ],
            }
        ],
    }
    return {
        "name": "SilentEnd",
        "description": "Ask for a name, then record it without a reply",
        "initial_state": "ask",
        "states": {
            "ask": ask,
            "record": {
                "id": "record",
                "description": "Record the name",
                "purpose": "Store the name; nothing is said",
                "response_instructions": "",
            },
        },
    }


class TestConversationStepLastSpokenReply:
    """``last_response`` / ``final_answer`` hold the last SPOKEN reply: a turn
    that ends on a silent state returns ``""`` and must not overwrite it."""

    @pytest.fixture
    def llm(self, monkeypatch):
        from fsm_llm import API
        from tests.conftest import MockLLM2Interface

        llm = MockLLM2Interface(
            extraction_data={"name": "Alice"}, response_text="What is your name?"
        )
        real_from_definition = API.from_definition
        monkeypatch.setattr(
            API,
            "from_definition",
            lambda definition, **kwargs: real_from_definition(
                definition, llm_interface=llm
            ),
        )
        return llm

    def _step(self, *, ask_speaks: bool):
        from fsm_llm.workflows.steps import ConversationStep

        return ConversationStep(
            step_id="intake",
            name="Intake",
            fsm_definition=_name_capture_fsm(ask_speaks=ask_speaks),
            success_state="done",
            auto_messages=["My name is Alice"],
            context_mapping={"user_name": "name", "answer": "final_answer"},
            require_completion=True,
        )

    async def test_silent_last_turn_keeps_the_last_spoken_reply(self, llm):
        result = await self._step(ask_speaks=True).execute({})

        assert (result.success, result.next_state) == (True, "done")
        collected = result.data["conversation_intake_data"]
        assert collected["name"] == "Alice"
        assert collected["final_answer"] == "What is your name?"
        assert collected["last_response"] == "What is your name?"
        assert result.data["answer"] == "What is your name?"
        # Only the opening reply was generated; the silent turn made no reply call.
        assert [kind for kind, _ in llm.call_history].count("generate_response") == 1

    async def test_conversation_that_never_spoke_adds_no_reply_keys(self, llm):
        result = await self._step(ask_speaks=False).execute({})

        assert (result.success, result.next_state) == (True, "done")
        collected = result.data["conversation_intake_data"]
        assert collected["name"] == "Alice"
        assert "final_answer" not in collected
        assert "last_response" not in collected
        assert "answer" not in result.data


class TestRemovedWorkflowLegacy:
    """Absence pins: each name existed on the commit before its removal."""

    def test_engine_rejects_handler_system(self):
        with pytest.raises(TypeError, match="handler_system"):
            WorkflowEngine(handler_system=object())
        assert not hasattr(WorkflowEngine(), "handler_system")

    @pytest.mark.parametrize("first", [object(), None, 5])
    def test_engine_takes_no_positional_argument(self, first):
        # Plan 07ad3f8c step 22.2. RED on the parent: the old positional
        # handler_system bound to max_concurrent_workflows and failed only
        # inside start_workflow ("'>=' not supported").
        args = (first,)
        with pytest.raises(TypeError):
            WorkflowEngine(*args)
        engine = WorkflowEngine(max_concurrent_workflows=3, max_completed_instances=1)
        assert engine.max_concurrent_workflows == 3

    @pytest.mark.parametrize(
        "module",
        [
            "fsm_llm.workflows",
            "fsm_llm.workflows.constants",
            "fsm_llm.workflows.engine",
        ],
    )
    def test_max_step_depth_is_gone(self, module):
        import importlib

        mod = importlib.import_module(module)
        assert not hasattr(mod, "MAX_STEP_DEPTH")
        assert "MAX_STEP_DEPTH" not in getattr(mod, "__all__", ())
        assert mod.MAX_STEPS_PER_RUN == 1000

    @pytest.mark.parametrize(
        "name",
        [
            "_KEY_WAITING_INFO",
            "_KEY_TIMER_INFO",
            "_KEY_WORKFLOW_INFO",
            "_KEY_TIMEOUT",
            "_KEY_TIMER_EXPIRED",
            "_KEY_LAST_EVENT",
            "_KEY_USER_INPUT",
            "_KEY_CANCELLATION_REASON",
            "_STEP_INTERNAL_WHITELIST",
        ],
    )
    def test_engine_private_alias_is_gone(self, name):
        from fsm_llm.workflows import constants, engine

        assert not hasattr(engine, name)
        assert hasattr(constants, name.lstrip("_"))

    @pytest.mark.parametrize(
        "method",
        [
            "_execute_workflow_step",
            "_handle_successful_step",
            "_handle_failed_step",
            "_transition_to_state",
        ],
    )
    def test_engine_methods_take_no_depth(self, method):
        import inspect

        assert (
            "_depth"
            not in inspect.signature(getattr(WorkflowEngine, method)).parameters
        )


# ---------------------------------------------------------------
# DSL functions
# ---------------------------------------------------------------


class TestNewDSLFunctions:
    """Test DSL factory functions for new step types."""

    def test_agent_step_factory(self):
        from fsm_llm.workflows.dsl import agent_step

        mock_agent = MagicMock()
        step = agent_step(
            "research",
            "Research",
            mock_agent,
            task_template="Research {topic}",
            success_state="analyze",
        )
        assert isinstance(step, AgentStep)
        assert step.step_id == "research"

    def test_retry_step_factory(self):
        from fsm_llm.workflows.dsl import retry_step

        inner = AutoTransitionStep(step_id="inner", name="Inner", next_state="done")
        step = retry_step("retry", "Retry", inner, max_retries=5)
        assert isinstance(step, RetryStep)
        assert step.max_retries == 5

    def test_switch_step_factory(self):
        from fsm_llm.workflows.dsl import switch_step

        step = switch_step(
            "route",
            "Route",
            "intent",
            cases={"buy": "checkout"},
            default_state="fallback",
        )
        assert isinstance(step, SwitchStep)
        assert step.key == "intent"


# ---------------------------------------------------------------
# Workflow Engine: instance cleanup
# ---------------------------------------------------------------


class TestWorkflowInstanceCleanup:
    """Test instance removal and auto-purge."""

    def test_remove_terminal_instance(self):
        from fsm_llm.workflows.engine import WorkflowEngine
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine()
        instance = WorkflowInstance(
            instance_id="test-1",
            workflow_id="wf-1",
            current_step_id="done",
            status=WorkflowStatus.COMPLETED,
        )
        engine.workflow_instances["test-1"] = instance

        assert engine.remove_instance("test-1") is True
        assert "test-1" not in engine.workflow_instances

    def test_cannot_remove_active_instance(self):
        from fsm_llm.workflows.engine import WorkflowEngine
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine()
        instance = WorkflowInstance(
            instance_id="test-1",
            workflow_id="wf-1",
            current_step_id="running",
            status=WorkflowStatus.RUNNING,
        )
        engine.workflow_instances["test-1"] = instance

        assert engine.remove_instance("test-1") is False
        assert "test-1" in engine.workflow_instances

    def test_remove_nonexistent_instance(self):
        from fsm_llm.workflows.engine import WorkflowEngine

        engine = WorkflowEngine()
        assert engine.remove_instance("nonexistent") is False

    def test_max_completed_instances_purge(self):
        from datetime import datetime, timezone

        from fsm_llm.workflows.engine import WorkflowEngine
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine(max_completed_instances=2)

        # Add 4 completed instances with different timestamps
        for i in range(4):
            instance = WorkflowInstance(
                instance_id=f"test-{i}",
                workflow_id="wf-1",
                current_step_id="done",
                status=WorkflowStatus.COMPLETED,
                completed_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
            )
            engine.workflow_instances[f"test-{i}"] = instance

        engine._purge_oldest_terminal_instances()

        # Should keep only 2 newest
        assert len(engine.workflow_instances) == 2
        assert "test-2" in engine.workflow_instances
        assert "test-3" in engine.workflow_instances
        assert "test-0" not in engine.workflow_instances
        assert "test-1" not in engine.workflow_instances

    def test_no_purge_when_limit_none(self):
        from fsm_llm.workflows.engine import WorkflowEngine
        from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

        engine = WorkflowEngine(max_completed_instances=None)

        for i in range(10):
            engine.workflow_instances[f"test-{i}"] = WorkflowInstance(
                instance_id=f"test-{i}",
                workflow_id="wf-1",
                current_step_id="done",
                status=WorkflowStatus.COMPLETED,
            )

        engine._purge_oldest_terminal_instances()
        assert len(engine.workflow_instances) == 10
