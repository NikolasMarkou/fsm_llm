"""Scripted end-to-end solves of the reasoning engine (plan 944e2692 step 23).

A scripted ``LLMInterface`` answers every per-field, bulk and Pass-2 call from
a fixed table and records each request, so a whole solve runs offline: the
orchestrator, the classifier FSM run from a handler, a pushed strategy FSM,
its pop and the validation retry loop.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm import RunBudgetExceededError
from fsm_llm.definitions import (
    BulkExtractionRequest,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.reasoning import ReasoningEngine, ReasoningExecutionError
from fsm_llm.reasoning.constants import (
    HYBRID_EVALUATION_STATE,
    ORCHESTRATOR_HANDLER_ONLY_KEYS,
    ContextKeys,
    Defaults,
    OrchestratorStates,
    ReasoningType,
)

_PROBLEM = "Is the argument valid? All men are mortal; Socrates is a man."

# A solve that validates on the first attempt: deductive strategy, a solution
# long enough and sharing content words with the problem, non-empty insights.
_VALID_SCRIPT: dict[str, Any] = {
    "problem_type": "logic puzzle",
    "problem_components": ["premise one", "premise two"],
    "problem_domain": "logic",
    "domain_indicators": ["syllogism"],
    "problem_structure": "two premises",
    "structural_elements": ["major premise", "minor premise"],
    "reasoning_requirements": "deduction",
    "key_challenges": "validity",
    "recommended_reasoning_type": "deductive",
    "strategy_justification": "a syllogism",
    "alternative_approaches": ["analytical"],
    "reasoning_strategy": "deductive",
    "strategy_rationale": "premises given",
    "premises": ["All men are mortal", "Socrates is a man"],
    "assumptions": ["terms are used consistently"],
    "logical_steps": ["apply the syllogism"],
    "intermediate_conclusions": ["Socrates belongs to the mortal class"],
    "conclusion": "Socrates is mortal",
    "logical_validity": True,
    "proposed_solution": "The argument is valid: Socrates is a man, so Socrates is mortal.",
    "key_insights": ["a valid syllogism"],
    "validation_result": True,
    "solution_confidence": 0.9,
    "final_solution": "The argument is valid: Socrates is mortal.",
}

# The same solve whose proposed solution always fails validation (too short,
# so `sufficient_detail` fails on a non-arithmetic problem).
_FAILING_SCRIPT: dict[str, Any] = {
    **_VALID_SCRIPT,
    "proposed_solution": "x",
    "validation_result": False,
    "solution_confidence": 0.1,
}


class _ScriptedLLM(LLMInterface):
    """Answers each call from ``script``; records every request.

    A field missing from the script is answered with ``None`` (nothing
    extracted). Bulk extraction extracts nothing; Pass 2 replies ``"ok"``.
    """

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.requests: list[Any] = []

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.requests.append(request)
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning=""
        )

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.requests.append(request)
        value = self.script.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=0.9 if value is not None else 0.0,
        )

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.requests.append(request)
        return DataExtractionResponse(extracted_data={}, confidence=0.9)

    def field_requests(self, field_name: str) -> list[FieldExtractionRequest]:
        return [
            r
            for r in self.requests
            if isinstance(r, FieldExtractionRequest) and r.field_name == field_name
        ]


def _engine(llm: _ScriptedLLM) -> ReasoningEngine:
    return ReasoningEngine(model="mock", llm_interface=llm)


def _record_histories(
    engine: ReasoningEngine, monkeypatch: pytest.MonkeyPatch
) -> dict[str, list[dict[str, str]]]:
    """Record each conversation's history just before the engine ends it."""
    histories: dict[str, list[dict[str, str]]] = {}
    for api in (engine.orchestrator, engine.classifier):
        original = api.end_conversation

        def spy(conversation_id: str, _api=api, _original=original) -> None:
            histories.setdefault(
                conversation_id, _api.get_conversation_history(conversation_id)
            )
            _original(conversation_id)

        monkeypatch.setattr(api, "end_conversation", spy)
    return histories


def _visited(trace_info: dict[str, Any]) -> list[str]:
    return [step["to"] for step in trace_info["reasoning_trace"]["steps"]]


class TestScriptedSolve:
    """A full solve with push and pop of the deductive strategy FSM."""

    def test_returns_the_documented_shape(self):
        solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM
        )

        # `final_solution` (final_answer state) is the first solution key read.
        assert solution == _VALID_SCRIPT["final_solution"]
        assert set(trace_info) == {
            "reasoning_trace",
            "summary",
            "final_context",
            "all_responses",
        }
        assert isinstance(trace_info["summary"], str)
        assert all(isinstance(r, str) for r in trace_info["all_responses"])
        assert trace_info["reasoning_trace"]["reasoning_types_used"] == ["deductive"]

    def test_strategy_fsm_is_pushed_driven_and_merged_back(self):
        _solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM
        )
        context = trace_info["final_context"]

        assert context[ContextKeys.REASONING_TYPE_SELECTED] == "deductive"
        assert context["deductive_reasoning_completed"] is True
        assert context[ContextKeys.DEDUCTIVE_CONCLUSION] == "Socrates is mortal"
        assert not context.get(ContextKeys.REASONING_PUSH_PENDING)
        assert _visited(trace_info)[-1] == OrchestratorStates.FINAL_ANSWER

    def test_no_user_exchange_and_no_continue_message(self, monkeypatch):
        """Every call is message-free; no conversation history holds a user turn."""
        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = _engine(llm)
        histories = _record_histories(engine, monkeypatch)

        engine.solve_problem(_PROBLEM)

        assert llm.requests
        assert all(request.user_message is None for request in llm.requests)
        assert len(histories) == 2  # orchestrator and classifier
        for history in histories.values():
            assert history
            assert all("user" not in exchange for exchange in history)
        texts = [r.system_prompt for r in llm.requests]
        texts += [m for h in histories.values() for e in h for m in e.values()]
        assert not any("Continue reasoning" in text for text in texts)

    def test_problem_reaches_the_model_through_context(self):
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _engine(llm).solve_problem(_PROBLEM)

        first = llm.field_requests(ContextKeys.PROBLEM_TYPE)[0]
        assert "All men are mortal" in first.system_prompt

    def test_no_prompt_carries_a_pushed_fsm_dict(self):
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _engine(llm).solve_problem(_PROBLEM)

        for request in llm.requests:
            assert "reasoning_fsm_to_push" not in request.system_prompt
            assert '"initial_state"' not in request.system_prompt
            assert '"transitions"' not in request.system_prompt


class TestRetryLoop:
    """Validation that always fails runs the retry loop to its limit."""

    def test_failing_validation_reaches_the_retry_limit(self):
        llm = _ScriptedLLM(_FAILING_SCRIPT)
        solution, trace_info = _engine(llm).solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        assert context[ContextKeys.RETRY_COUNT] == Defaults.MAX_RETRIES == 3
        assert context[ContextKeys.MAX_RETRIES_REACHED] is True
        assert context[ContextKeys.VALIDATION_RESULT] is False
        assert _visited(trace_info)[-1] == OrchestratorStates.FINAL_ANSWER
        assert solution == _FAILING_SCRIPT["final_solution"]

    def test_each_attempt_re_runs_the_strategy_and_the_synthesis(self):
        """The execute_reasoning entry clear lets every retry extract anew."""
        llm = _ScriptedLLM(_FAILING_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)

        # Each failed validation counts one retry; the third reaches the limit.
        attempts = Defaults.MAX_RETRIES
        assert len(llm.field_requests(ContextKeys.PROPOSED_SOLUTION)) == attempts
        assert len(llm.field_requests(ContextKeys.CONCLUSION)) == attempts
        entries = _visited(trace_info).count(OrchestratorStates.EXECUTE_REASONING)
        assert entries == attempts


_STUCK_FSM: dict[str, Any] = {
    "name": "stuck_reasoning",
    "description": "A strategy FSM whose gate never opens",
    "initial_state": "gather",
    "persona": "A careful reasoner",
    "states": {
        "gather": {
            "id": "gather",
            "description": "Gather the premises",
            "purpose": "Collect premises",
            "extraction_instructions": "Extract the premises.",
            "response_instructions": "Summarise the premises.",
            "required_context_keys": ["premises"],
            "transitions": [{"target_state": "wait", "description": "Premises in"}],
        },
        "wait": {
            "id": "wait",
            "description": "Wait for a key the model never gives",
            "purpose": "Wait for the gate",
            "extraction_instructions": "Extract the gate key.",
            "response_instructions": "Report progress on the gate WAIT-MARKER.",
            "required_context_keys": ["never_set"],
            "transitions": [
                {
                    "target_state": "done",
                    "description": "Gate opened",
                    "conditions": [
                        {
                            "description": "The gate key is present",
                            "requires_context_keys": ["never_set"],
                        }
                    ],
                }
            ],
        },
        "done": {
            "id": "done",
            "description": "Terminal",
            "purpose": "Finish",
            "response_instructions": "Give the conclusion.",
        },
    },
}


class TestForcedPop:
    """A strategy FSM that never ends is popped after MAX_SUB_FSM_ITERATIONS."""

    def test_sub_fsm_is_force_popped_and_the_solve_finishes(self):
        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = _engine(llm)
        engine.reasoning_fsms[ReasoningType.DEDUCTIVE] = _STUCK_FSM

        _solution, trace_info = engine.solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        # Every sub step replies from `wait` (the first step moves `gather`
        # to `wait`), so the Pass-2 calls carrying its marker count the steps.
        wait_replies = [
            r
            for r in llm.requests
            if isinstance(r, ResponseGenerationRequest)
            and "WAIT-MARKER" in r.system_prompt
        ]
        assert len(wait_replies) == Defaults.MAX_SUB_FSM_ITERATIONS
        assert context["deductive_reasoning_completed"] is True
        assert _visited(trace_info)[-1] == OrchestratorStates.FINAL_ANSWER


class _ClaimingLLM(_ScriptedLLM):
    """A model that claims, in every bulk reply, the keys it may not write."""

    def __init__(self, script: dict[str, Any], claims: dict[str, Any]) -> None:
        super().__init__(script)
        self.claims = claims

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.requests.append(request)
        return DataExtractionResponse(extracted_data=dict(self.claims), confidence=0.9)


def _replies_from(llm: _ScriptedLLM, state_id: str) -> int:
    """Pass-2 calls made from ``state_id`` (one per step that ends there)."""
    tag = f"<current_state>{state_id}</current_state>"
    return sum(
        1
        for r in llm.requests
        if isinstance(r, ResponseGenerationRequest) and tag in r.system_prompt
    )


# The model never synthesizes a solution but claims it is valid.
_CLAIMED_VALID_SCRIPT: dict[str, Any] = {
    key: value
    for key, value in _VALID_SCRIPT.items()
    if key != ContextKeys.PROPOSED_SOLUTION
}


class TestHandlerOnlyGate:
    """A model reply cannot write the validation verdict or the counters (D-054)."""

    def test_a_bulk_claim_of_validity_does_not_open_the_gate(self):
        llm = _ClaimingLLM(
            _CLAIMED_VALID_SCRIPT,
            claims={
                ContextKeys.VALIDATION_RESULT: True,
                ContextKeys.SOLUTION_CONFIDENCE: 1.0,
            },
        )
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        # No solution was ever proposed: every attempt fails validation.
        assert context[ContextKeys.VALIDATION_RESULT] is False
        assert context[ContextKeys.RETRY_COUNT] == Defaults.MAX_RETRIES
        assert context[ContextKeys.MAX_RETRIES_REACHED] is True
        assert context[ContextKeys.SOLUTION_CONFIDENCE] < 1.0
        entries = _visited(trace_info).count(OrchestratorStates.EXECUTE_REASONING)
        assert entries == Defaults.MAX_RETRIES

    def test_no_field_extraction_is_minted_for_a_handler_only_key(self):
        # validate_refine is entered with no verdict on every attempt.
        llm = _ScriptedLLM(_CLAIMED_VALID_SCRIPT)
        _engine(llm).solve_problem(_PROBLEM)

        for key in ORCHESTRATOR_HANDLER_ONLY_KEYS:
            assert llm.field_requests(key) == [], key

    def test_a_merged_calculator_solution_gets_a_handler_verdict(self):
        """The calculator's solution, merged back on pop, is judged by the
        validator handler: no model verdict is needed to finish."""
        script = {
            "problem_type": "arithmetic",
            "problem_components": ["2", "+", "3"],
            "problem_domain": "math",
            "recommended_reasoning_type": "simple_calculator",
            "reasoning_strategy": "simple_calculator",
            "strategy_rationale": "plain addition",
            "operand1": 2,
            "operand2": 3,
            "operator": "+",
            "calculation_result": 5,
            "key_insights": ["addition"],
            "final_solution": "5",
        }
        llm = _ScriptedLLM(script)
        solution, trace_info = _engine(llm).solve_problem("What is 2 + 3?")
        context = trace_info["final_context"]

        assert solution == "5"
        assert context[ContextKeys.PROPOSED_SOLUTION] == 5
        assert context[ContextKeys.VALIDATION_RESULT] is True
        assert context[ContextKeys.RETRY_COUNT] == 0
        assert _visited(trace_info)[-1] == OrchestratorStates.FINAL_ANSWER


_HYBRID_SCRIPT: dict[str, Any] = {
    **_VALID_SCRIPT,
    "problem_aspects": ["validity"],
    "reasoning_map": {"validity": "deductive"},
    "analytical_breakdown": "two premises",
    "component_relationships": "premise chain",
    "logical_conclusions": ["Socrates is mortal"],
    "reasoning_chain": ["major", "minor", "conclusion"],
    "creative_insights": ["none needed"],
    "novel_approaches": ["none"],
    "evaluation_results": "sound",
    "integrated_solution": "Socrates is mortal",
    "reasoning_synthesis_notes": "deduction suffices",
    "final_hybrid_solution": "Socrates is mortal",
    "reasoning_synthesis": "deduction",
}


class TestHybridLoopCounter:
    """The hybrid back edge runs at most MAX_HYBRID_LOOPS times (D-054)."""

    def test_the_back_edge_runs_at_most_twice(self):
        # The model always asks for refinement and keeps resetting the count.
        llm = _ClaimingLLM(
            _HYBRID_SCRIPT,
            claims={
                ContextKeys.NEEDS_REFINEMENT: True,
                ContextKeys.HYBRID_LOOP_COUNT: 0,
            },
        )
        _solution, trace_info = _engine(llm).solve_problem(
            _PROBLEM,
            {ContextKeys.PREFERRED_REASONING_TYPE: ReasoningType.HYBRID.value},
        )
        context = trace_info["final_context"]

        # One pass plus MAX_HYBRID_LOOPS refinements, then on to the terminal.
        passes = 1 + Defaults.MAX_HYBRID_LOOPS
        assert _replies_from(llm, "identify_components") == passes
        assert _replies_from(llm, HYBRID_EVALUATION_STATE) == passes
        assert _replies_from(llm, "finalize_hybrid") == 1
        assert context["hybrid_reasoning_completed"] is True
        assert context[ContextKeys.FINAL_HYBRID_SOLUTION] == "Socrates is mortal"


# problem_components never arrives: problem_analysis never leaves.
_NEVER_ENDING_SCRIPT: dict[str, Any] = {
    key: value
    for key, value in _VALID_SCRIPT.items()
    if key != ContextKeys.PROBLEM_COMPONENTS
}


class TestSpentBudget:
    """A solve that does not finish raises with its partial context (D-014)."""

    def test_raises_with_the_partial_context(self):
        llm = _ScriptedLLM(_NEVER_ENDING_SCRIPT)
        engine = _engine(llm)

        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM)

        error = info.value
        assert isinstance(error.__cause__, RunBudgetExceededError)
        assert error.__cause__.steps_done == Defaults.MAX_SOLVE_STEPS
        details = error.details
        assert set(details) == {
            "conversation_id",
            "responses_so_far",
            "partial_context",
        }
        partial = details["partial_context"]
        assert partial[ContextKeys.PROBLEM_STATEMENT] == _PROBLEM
        assert partial[ContextKeys.PROBLEM_TYPE] == "logic puzzle"
        assert ContextKeys.PROBLEM_COMPONENTS not in partial
        # The conversation was ended after the context was read.
        assert engine.orchestrator.list_active_conversations() == []
