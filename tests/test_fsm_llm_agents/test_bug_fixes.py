from __future__ import annotations

"""
Regression tests for three agent bug fixes:

Bug 1: ReflexionAgent.evaluation_fn was dead code — get_data() filtered
       internal keys so `_current_state` was always None. Fixed by moving
       evaluation_fn into a CONTEXT_UPDATE handler on the EVALUATE state.

Bug 2: ReasoningReactAgent reasoning interception timing — the run-loop
       checked tool_name after the POST_TRANSITION handler already executed
       the placeholder. Fixed by replacing the standard execute_tool handler
       with a custom one that intercepts "reason" and calls ReasoningEngine.

Bug 3: ADaPTAgent subtask execution timing — DECOMPOSE->COMBINE transition
       fired inside converse(), making COMBINE terminal before run-loop
       could check subtasks. Fixed by adding a PRE_TRANSITION handler that
       executes subtasks when leaving DECOMPOSE state.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.definitions import AgentConfig, EvaluationResult
from fsm_llm_agents.tools import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _search(params: dict) -> str:
    return f"Results for: {params.get('query', '')}"


def _make_registry() -> ToolRegistry:
    """Create a registry with a dummy search tool."""
    registry = ToolRegistry()
    registry.register_function(_search, name="search", description="Search the web")
    return registry


class SequenceMockLLM(LLMInterface):
    """Mock LLM that replays pre-programmed extraction sequences.

    Each call to ``extract_field`` looks up the requested field in the current
    extraction sequence entry.  ``generate_response`` always returns a
    canned string.
    """

    def __init__(
        self,
        extraction_sequence: list[dict[str, Any]],
        response_text: str = "Agent response.",
    ) -> None:
        self._extractions = list(extraction_sequence)
        self._call_index = 0
        self._response_text = response_text
        self._extracted_this_cycle = False
        self.model = "mock-model"

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        """Per-field extraction using the current extraction sequence entry.

        The pipeline calls extract_field once per required_context_key.
        We look up the requested field_name in the current extraction dict.
        The index advances once per converse cycle: on the first
        extract_field call after a generate_response call.
        """
        if not self._extracted_this_cycle:
            # First extraction call of a new cycle — don't advance on the
            # very first cycle (index 0), but advance for subsequent ones.
            self._extracted_this_cycle = True

        if self._call_index < len(self._extractions):
            data = self._extractions[self._call_index]
        else:
            data = {}

        value = data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="mock field extraction",
            is_valid=value is not None,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        # Advance the extraction sequence index once per converse cycle.
        # Only advance if extraction actually happened this cycle.
        if self._extracted_this_cycle:
            self._call_index += 1
            self._extracted_this_cycle = False
        return ResponseGenerationResponse(
            message=self._response_text,
            message_type="response",
            reasoning="mock response",
        )


# ---------------------------------------------------------------------------
# Bug 1 — ReflexionAgent.evaluation_fn was dead code
# ---------------------------------------------------------------------------


class TestReflexionEvaluationFnActuallyCalled:
    """
    Verify that a user-provided evaluation_fn is actually invoked during
    the Reflexion evaluate state.

    Previously the fn was dead code because the run-loop tried to detect
    the EVALUATE state via ``get_data()["_current_state"]`` which always
    returned None (internal keys are stripped). The fix registers the fn
    as a CONTEXT_UPDATE handler on the EVALUATE state.
    """

    def test_evaluation_fn_is_called(self):
        """evaluation_fn must be called at least once during agent execution."""
        from fsm_llm_agents.reflexion import ReflexionAgent

        registry = _make_registry()

        # Track whether evaluation_fn fires
        eval_called = {"count": 0}

        def mock_eval_fn(context: dict[str, Any]) -> EvaluationResult:
            eval_called["count"] += 1
            return EvaluationResult(passed=True, score=1.0, feedback="good")

        # Build extraction sequence for the Reflexion FSM.
        # Only states with required_context_keys trigger extract_field
        # calls. States without required_context_keys (like act) are
        # skipped in the sequence.
        # Flow: think -> act -> evaluate -> conclude
        #
        # NOTE: The evaluate state extraction must NOT include
        # evaluation_passed because the transition evaluator re-merges
        # raw extraction data on top of handler context updates. The
        # evaluation_fn handler sets evaluation_passed in context; if
        # the extraction also has it, the extraction value wins.
        extraction_sequence = [
            # 1. think state: select tool (has required_context_keys)
            {
                "tool_name": "search",
                "tool_input": {"query": "test"},
                "reasoning": "searching",
            },
            # 2. evaluate state: LLM extraction (has required_context_keys)
            {"evaluation_score": 0.5, "evaluation_feedback": "partial"},
            # 3. conclude state: final answer (has required_context_keys)
            {"final_answer": "The answer is 42."},
        ]

        mock_llm = SequenceMockLLM(extraction_sequence)
        config = AgentConfig(max_iterations=10, timeout_seconds=30.0)

        agent = ReflexionAgent(
            tools=registry,
            config=config,
            evaluation_fn=mock_eval_fn,
            llm_interface=mock_llm,
        )

        result = agent.run("What is 6 times 7?")

        # The critical assertion: evaluation_fn must have been called
        assert eval_called["count"] >= 1, (
            "evaluation_fn was never called — the bug is back "
            "(evaluation_fn was dead code)"
        )
        assert result.success is True

    def test_evaluation_fn_result_drives_transition(self):
        """When evaluation_fn returns passed=True, the agent should conclude
        (not reflect). When it returns passed=False, the agent should reflect."""
        from fsm_llm_agents.reflexion import ReflexionAgent

        registry = _make_registry()

        eval_results: list[bool] = []

        def failing_eval_fn(context: dict[str, Any]) -> EvaluationResult:
            """First call fails, second call passes."""
            call_num = len(eval_results)
            if call_num == 0:
                eval_results.append(False)
                return EvaluationResult(
                    passed=False, score=0.2, feedback="insufficient"
                )
            eval_results.append(True)
            return EvaluationResult(passed=True, score=0.9, feedback="good")

        # Only states with required_context_keys trigger extract_field.
        # Flow: think -> act -> evaluate (fail) -> reflect -> think
        #       -> act -> evaluate (pass) -> conclude
        #
        # NOTE: evaluate state extraction must NOT include evaluation_passed
        # (see note in test_evaluation_fn_is_called for rationale).
        extraction_sequence = [
            # 1. think: select tool (has required_context_keys)
            {
                "tool_name": "search",
                "tool_input": {"query": "test"},
                "reasoning": "searching",
            },
            # 2. evaluate: LLM extraction (has required_context_keys)
            {"evaluation_score": 0.3},
            # 3. reflect: produce reflection (has required_context_keys)
            {
                "reflection": "I should try a different approach",
                "lessons": ["be more specific"],
            },
            # 4. think again: select tool (has required_context_keys)
            {
                "tool_name": "search",
                "tool_input": {"query": "refined test"},
                "reasoning": "retrying",
            },
            # 5. evaluate again: LLM extraction (has required_context_keys)
            {"evaluation_score": 0.9},
            # 6. conclude: final answer (has required_context_keys)
            {"final_answer": "The refined answer is 42."},
        ]

        mock_llm = SequenceMockLLM(extraction_sequence)
        config = AgentConfig(max_iterations=15, timeout_seconds=30.0)

        agent = ReflexionAgent(
            tools=registry,
            config=config,
            evaluation_fn=failing_eval_fn,
            llm_interface=mock_llm,
        )

        result = agent.run("What is the meaning of life?")

        # evaluation_fn was called exactly twice (fail then pass)
        assert len(eval_results) == 2
        assert eval_results[0] is False
        assert eval_results[1] is True
        assert result.success is True


# ---------------------------------------------------------------------------
# Bug 2 — ReasoningReactAgent reasoning interception timing
# ---------------------------------------------------------------------------


class TestReasoningReactAgentInterception:
    """
    Verify that when the LLM selects the "reason" tool, the
    ReasoningEngine is invoked instead of the placeholder function.

    Previously the run-loop checked tool_name after the POST_TRANSITION
    handler had already executed the placeholder. The fix replaces the
    standard execute_tool handler with a custom one that intercepts
    "reason" before executing and calls ReasoningEngine.solve_problem().
    """

    @pytest.fixture(autouse=True)
    def _check_reasoning_installed(self):
        """Skip if fsm_llm_reasoning is not installed."""
        try:
            import fsm_llm_reasoning  # noqa: F401
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

    def test_reasoning_engine_invoked_for_reason_tool(self):
        """When the LLM picks tool_name='reason', ReasoningEngine.solve_problem
        must be called — not the placeholder function."""
        from fsm_llm_agents.reasoning_react import ReasoningReactAgent

        registry = _make_registry()

        # Only states with required_context_keys trigger extract_field.
        # Flow: think -> act -> think -> conclude
        # Act has no required_context_keys, so it's skipped.
        extraction_sequence = [
            # 1. think: select the "reason" tool (has required_context_keys)
            {
                "tool_name": "reason",
                "tool_input": {"problem": "Is 97 prime?"},
                "reasoning": "I need structured reasoning",
            },
            # 2. think: terminate after reasoning (has required_context_keys)
            {"tool_name": "none", "should_terminate": True, "reasoning": "done"},
            # 3. conclude: final answer (has required_context_keys)
            {"final_answer": "Yes, 97 is a prime number."},
        ]

        mock_llm = SequenceMockLLM(extraction_sequence)
        config = AgentConfig(max_iterations=10, timeout_seconds=30.0)

        # Patch ReasoningEngine to track if solve_problem is called
        with patch(
            "fsm_llm_agents.reasoning_react.ReasoningEngine"
        ) as MockReasoningEngine:
            mock_engine_instance = MagicMock()
            mock_engine_instance.solve_problem.return_value = (
                "97 is prime because it has no divisors other than 1 and itself.",
                {
                    "reasoning_trace": {
                        "reasoning_types_used": ["analytical"],
                        "final_confidence": 0.95,
                    }
                },
            )
            MockReasoningEngine.return_value = mock_engine_instance

            agent = ReasoningReactAgent(
                tools=registry,
                config=config,
                llm_interface=mock_llm,
            )

            result = agent.run("Is 97 prime?")

            # The critical assertion: solve_problem was called, not the placeholder
            mock_engine_instance.solve_problem.assert_called_once()
            call_args = mock_engine_instance.solve_problem.call_args
            assert "97" in call_args[0][0] or "97" in str(call_args)

            assert result.success is True

    def test_non_reason_tool_delegates_normally(self):
        """When the LLM picks a regular tool (not 'reason'), the standard
        execute_tool handler should run it normally."""
        from fsm_llm_agents.reasoning_react import ReasoningReactAgent

        # Only states with required_context_keys trigger extract_field.
        # Flow: think -> act -> think -> conclude
        # Act has no required_context_keys, so it's skipped.
        extraction_sequence = [
            # 1. think: select regular "search" tool (has required_context_keys)
            {
                "tool_name": "search",
                "tool_input": {"query": "hello world"},
                "reasoning": "need to search first",
            },
            # 2. think: terminate (has required_context_keys)
            {"tool_name": "none", "should_terminate": True, "reasoning": "done"},
            # 3. conclude: final answer (has required_context_keys)
            {"final_answer": "Hello world results."},
        ]

        registry = _make_registry()
        mock_llm = SequenceMockLLM(extraction_sequence)
        config = AgentConfig(max_iterations=10, timeout_seconds=30.0)

        with patch(
            "fsm_llm_agents.reasoning_react.ReasoningEngine"
        ) as MockReasoningEngine:
            mock_engine_instance = MagicMock()
            MockReasoningEngine.return_value = mock_engine_instance

            agent = ReasoningReactAgent(
                tools=registry,
                config=config,
                llm_interface=mock_llm,
            )

            result = agent.run("Search hello world")

            # ReasoningEngine.solve_problem should NOT have been called
            mock_engine_instance.solve_problem.assert_not_called()
            assert result.success is True


# ---------------------------------------------------------------------------
# Bug 3 — ADaPTAgent subtask execution timing
# ---------------------------------------------------------------------------


class TestADaPTSubtaskExecution:
    """
    Verify that _execute_subtasks is actually called when the assessment
    says failure and decomposition produces subtasks.

    Previously the DECOMPOSE->COMBINE transition fired inside converse(),
    making COMBINE terminal before the run-loop could check for subtasks.
    The fix adds a PRE_TRANSITION handler that executes subtasks when
    leaving the DECOMPOSE state.
    """

    def test_execute_subtasks_is_called(self):
        """When attempt fails and subtasks are produced, _execute_subtasks
        must be called during the DECOMPOSE->COMBINE transition."""
        from fsm_llm_agents.adapt import ADaPTAgent

        # Extraction sequence for ADaPT FSM:
        #   attempt -> extract attempt_result
        #   assess  -> extract attempt_succeeded=False
        #   decompose -> extract subtasks
        #   combine -> extract final_answer
        extraction_sequence = [
            # 1. attempt: produce an attempt result
            {"attempt_result": "Partial answer, need more detail."},
            # 2. assess: attempt failed
            {"attempt_succeeded": False, "assessment": "needs decomposition"},
            # 3. decompose: produce subtasks
            {
                "subtasks": ["Define neural networks", "Explain backpropagation"],
                "operator": "AND",
            },
            # 4. combine: final answer
            {"final_answer": "Neural networks learn through backpropagation."},
        ]

        mock_llm = SequenceMockLLM(extraction_sequence)
        config = AgentConfig(max_iterations=15, timeout_seconds=30.0)

        agent = ADaPTAgent(
            config=config,
            max_depth=2,
            llm_interface=mock_llm,
        )

        # Patch _execute_subtasks to track if it's called and return mock results
        execute_called = {"count": 0, "args": None}

        def mock_execute_subtasks(subtasks, operator, depth, initial_context):
            execute_called["count"] += 1
            execute_called["args"] = {
                "subtasks": subtasks,
                "operator": operator,
                "depth": depth,
            }
            # Return mock subtask results
            return [
                {
                    "subtask": str(s),
                    "answer": f"Answer for: {s}",
                    "success": True,
                    "depth": depth,
                }
                for s in subtasks
            ]

        agent._execute_subtasks = mock_execute_subtasks

        # Patch Classifier to avoid real LLM calls for ambiguous transitions.
        # The mock picks the first intent from the schema (like old decide_transition).
        def _make_mock_classifier(schema, **kwargs):
            classifier_instance = MagicMock()
            first_intent = schema.intents[0].name if schema.intents else "unknown"

            def _classify(msg):
                result = MagicMock()
                result.intent = first_intent
                result.confidence = 0.9
                result.reasoning = "mock"
                result.entities = {}
                return result

            classifier_instance.classify.side_effect = _classify
            return classifier_instance

        with patch("fsm_llm.pipeline.Classifier", side_effect=_make_mock_classifier):
            result = agent.run("Explain how neural networks learn")

        # The critical assertion: _execute_subtasks was called
        assert execute_called["count"] >= 1, (
            "_execute_subtasks was never called — the bug is back "
            "(subtask execution timing is wrong)"
        )

        # Verify it was called with the correct subtasks
        assert execute_called["args"] is not None
        assert len(execute_called["args"]["subtasks"]) == 2
        assert execute_called["args"]["operator"] == "AND"
        assert result.success is True

    def test_subtasks_not_executed_when_attempt_succeeds(self):
        """When the attempt succeeds, _execute_subtasks should NOT be called."""
        from fsm_llm_agents.adapt import ADaPTAgent

        extraction_sequence = [
            # 1. attempt: produce attempt result
            {"attempt_result": "Complete answer here."},
            # 2. assess: attempt succeeded
            {"attempt_succeeded": True, "assessment": "looks good"},
            # 3. combine: final answer (direct from success path)
            {"final_answer": "Complete answer here."},
        ]

        mock_llm = SequenceMockLLM(extraction_sequence)
        config = AgentConfig(max_iterations=15, timeout_seconds=30.0)

        agent = ADaPTAgent(
            config=config,
            max_depth=2,
            llm_interface=mock_llm,
        )

        execute_called = {"count": 0}

        def mock_execute_subtasks(subtasks, operator, depth, initial_context):
            execute_called["count"] += 1
            return []

        agent._execute_subtasks = mock_execute_subtasks

        # Patch Classifier to avoid real LLM calls for ambiguous transitions.
        def _make_mock_classifier(schema, **kwargs):
            classifier_instance = MagicMock()
            first_intent = schema.intents[0].name if schema.intents else "unknown"

            def _classify(msg):
                result = MagicMock()
                result.intent = first_intent
                result.confidence = 0.9
                result.reasoning = "mock"
                result.entities = {}
                return result

            classifier_instance.classify.side_effect = _classify
            return classifier_instance

        with patch("fsm_llm.pipeline.Classifier", side_effect=_make_mock_classifier):
            result = agent.run("Simple question")

        # _execute_subtasks should NOT have been called
        assert execute_called["count"] == 0
        assert result.success is True
