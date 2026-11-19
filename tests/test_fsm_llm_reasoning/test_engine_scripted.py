"""Scripted end-to-end solves of the reasoning engine (plan 944e2692 step 23).

A scripted ``LLMInterface`` answers every per-field, bulk and Pass-2 call from
a fixed table and records each request, so a whole solve runs offline: the
orchestrator, the classifier FSM run from a handler, a pushed strategy FSM,
its pop and the validation retry loop.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from fsm_llm import RunBudgetExceededError
from fsm_llm.core.definitions import (
    BulkExtractionRequest,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.reasoning import (
    ReasoningClassificationError,
    ReasoningEngine,
    ReasoningExecutionError,
)
from fsm_llm.reasoning.constants import (
    HYBRID_CLEARED_KEYS,
    HYBRID_EVALUATION_STATE,
    ORCHESTRATOR_HANDLER_ONLY_KEYS,
    ContextKeys,
    Defaults,
    OrchestratorStates,
    ReasoningType,
)
from fsm_llm.reasoning.handlers import ReasoningHandlers

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
    """A strategy FSM that never ends is popped after MAX_SUB_FSM_STEPS."""

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
        assert len(wait_replies) == Defaults.MAX_SUB_FSM_STEPS
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
            "calculation_expression": "2 + 3",
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


# The keys the five looped hybrid states extract (required_context_keys).
_HYBRID_BODY_KEYS = (
    ContextKeys.PROBLEM_ASPECTS,
    ContextKeys.REASONING_MAP,
    ContextKeys.ANALYTICAL_BREAKDOWN,
    ContextKeys.COMPONENT_RELATIONSHIPS,
    ContextKeys.LOGICAL_CONCLUSIONS,
    ContextKeys.REASONING_CHAIN,
    ContextKeys.CREATIVE_INSIGHTS,
    ContextKeys.NOVEL_APPROACHES,
    ContextKeys.EVALUATION_RESULTS,
)


class TestHybridRefinementReExtracts:
    """The hybrid back edge re-runs the body extractions (F-RS-01)."""

    def _solve(self):
        llm = _ClaimingLLM(_HYBRID_SCRIPT, claims={ContextKeys.NEEDS_REFINEMENT: True})
        _solution, trace_info = _engine(llm).solve_problem(
            _PROBLEM,
            {ContextKeys.PREFERRED_REASONING_TYPE: ReasoningType.HYBRID.value},
        )
        return llm, trace_info["final_context"]

    def test_each_body_key_is_requested_once_per_pass(self):
        llm, _context = self._solve()

        passes = 1 + Defaults.MAX_HYBRID_LOOPS
        for key in _HYBRID_BODY_KEYS:
            assert len(llm.field_requests(key)) == passes, key

    def test_the_loop_still_ends_at_the_cap(self):
        llm, context = self._solve()

        passes = 1 + Defaults.MAX_HYBRID_LOOPS
        assert context["hybrid_reasoning_completed"] is True
        assert _replies_from(llm, HYBRID_EVALUATION_STATE) == passes
        assert _replies_from(llm, "finalize_hybrid") == 1

    def test_the_cleared_set_is_the_body_plus_the_refinement_flag(self):
        assert set(HYBRID_CLEARED_KEYS) == {
            *_HYBRID_BODY_KEYS,
            ContextKeys.NEEDS_REFINEMENT,
        }
        # Counter stays handler-only (D-054); later keys are not looped.
        assert ContextKeys.HYBRID_LOOP_COUNT not in HYBRID_CLEARED_KEYS
        assert ContextKeys.INTEGRATED_SOLUTION not in HYBRID_CLEARED_KEYS
        assert ContextKeys.FINAL_HYBRID_SOLUTION not in HYBRID_CLEARED_KEYS


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


def _causes(error: BaseException) -> list[BaseException]:
    """``error`` and every exception it was raised from, outermost first."""
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None:
        chain.append(current)
        current = current.__cause__
    return chain


class _FinalReplyFailsLLM(_ScriptedLLM):
    """The final_answer reply fails (as a prompt over core's cap does)."""

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        tag = f"<current_state>{OrchestratorStates.FINAL_ANSWER}</current_state>"
        if tag in request.system_prompt:
            self.requests.append(request)
            raise LLMResponseError("system_prompt over the 30000-character cap")
        return super().generate_response(request)


class TestFailedSolveKeepsWhatItHad:
    """Review pass 15 W1 (D-058): any failed solve, not only a spent budget,
    reports the context it reached, so an answer validated before a failing
    last step is not lost."""

    def test_a_failing_final_step_carries_the_partial_context(self):
        llm = _FinalReplyFailsLLM(_VALID_SCRIPT)
        engine = _engine(llm)

        with pytest.raises(ReasoningExecutionError, match="execution failed") as info:
            engine.solve_problem(_PROBLEM)

        error = info.value
        assert any(isinstance(e, LLMResponseError) for e in _causes(error))
        assert set(error.details) == {
            "conversation_id",
            "responses_so_far",
            "partial_context",
        }
        partial = error.details["partial_context"]
        assert (
            partial[ContextKeys.PROPOSED_SOLUTION]
            == _VALID_SCRIPT[ContextKeys.PROPOSED_SOLUTION]
        )
        assert partial[ContextKeys.VALIDATION_RESULT] is True
        assert engine.orchestrator.list_active_conversations() == []


class TestHandlerFailuresStopTheSolve:
    """Review pass 15 W2 (D-058): the engine's critical handlers raise through
    the step instead of being logged and skipped by core's "continue" mode."""

    def test_a_spent_classifier_budget_fails_the_solve(self, monkeypatch):
        monkeypatch.setattr(Defaults, "MAX_CLASSIFICATION_STEPS", 1)
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM)

        chain = _causes(info.value)
        classification = [
            e for e in chain if isinstance(e, ReasoningClassificationError)
        ]
        assert classification, chain
        assert isinstance(classification[0].__cause__, RunBudgetExceededError)
        assert engine.orchestrator.list_active_conversations() == []
        assert engine.classifier.list_active_conversations() == []

    def test_no_strategy_definitions_fails_the_solve(self):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        del engine.reasoning_fsms[ReasoningType.DEDUCTIVE]
        del engine.reasoning_fsms[ReasoningType.ANALYTICAL]

        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM)

        assert any(
            isinstance(e, ReasoningExecutionError)
            and "No reasoning FSM definitions available" in str(e)
            for e in _causes(info.value)[1:]
        )
        assert engine.orchestrator.list_active_conversations() == []

    def test_a_validator_crash_fails_the_solve_at_once(self, monkeypatch):
        def _always_crash(context: dict[str, Any]) -> dict[str, Any]:
            raise _HandlerCrash("validate_solution crashed")

        monkeypatch.setattr(
            ReasoningHandlers, "validate_solution", staticmethod(_always_crash)
        )
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM)

        chain = _causes(info.value)
        assert any(isinstance(e, _HandlerCrash) for e in chain), chain
        # Not the step budget hiding the real error after 170 free steps.
        assert not any(isinstance(e, RunBudgetExceededError) for e in chain)
        assert engine.orchestrator.list_active_conversations() == []


class TestFailingPop:
    """Review pass 15 W3 (D-053(3)): a pop that fails stops the solve; it is
    not swallowed into a "successful" solve on the strategy's context."""

    def test_a_failing_pop_raises_and_leaves_nothing_live(self, monkeypatch):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        def _explode(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("pop exploded")

        monkeypatch.setattr(engine.orchestrator, "pop_fsm", _explode)

        with pytest.raises(ReasoningExecutionError, match="pop exploded") as info:
            engine.solve_problem(_PROBLEM)

        assert isinstance(info.value.__cause__, RuntimeError)
        assert engine.orchestrator.list_active_conversations() == []


class TestNumericZeroAnswer:
    """Review pass 15 W4: a calculator result of 0 is an answer."""

    def test_a_zero_result_validates_on_the_first_attempt(self):
        script = {
            "problem_type": "arithmetic",
            "problem_components": ["5", "-", "5"],
            "problem_domain": "math",
            "recommended_reasoning_type": "simple_calculator",
            "reasoning_strategy": "simple_calculator",
            "strategy_rationale": "plain subtraction",
            "calculation_expression": "5 - 5",
            "key_insights": ["subtraction"],
            "final_solution": "0",
        }
        solution, trace_info = _engine(_ScriptedLLM(script)).solve_problem(
            "What is 5 - 5?"
        )
        context = trace_info["final_context"]

        assert solution == "0"
        assert context[ContextKeys.PROPOSED_SOLUTION] == 0
        assert context[ContextKeys.VALIDATION_RESULT] is True
        assert context[ContextKeys.RETRY_COUNT] == 0


_CALC = ContextKeys.CALCULATION_RESULT
_CALC_SCRIPT: dict[str, Any] = {
    "problem_type": "arithmetic",
    "problem_components": ["15%", "of", "240"],
    "problem_domain": "math",
    "recommended_reasoning_type": "simple_calculator",
    "reasoning_strategy": "simple_calculator",
    "strategy_rationale": "a percentage",
    "calculation_expression": "0.15 * 240",
    "key_insights": ["percentage"],
    "final_solution": "36",
}


class TestCalculatorEvaluatesTheExpression:
    """The calculator strategy asks the model for one expression; a handler evaluates it."""

    def test_the_handler_computes_what_the_model_only_wrote_down(self):
        claims = {_CALC: 999, ContextKeys.CALCULATION_ERROR: "x"}  # a bulk claim
        llm = _ClaimingLLM(_CALC_SCRIPT, claims)
        _solution, info = _engine(llm).solve_problem("What is 15% of 240?")
        context = info["final_context"]

        assert context[_CALC] == context[ContextKeys.PROPOSED_SOLUTION] == 36
        assert type(context[_CALC]) is int
        assert context[ContextKeys.VALIDATION_RESULT] is True
        assert len(llm.field_requests(ContextKeys.CALCULATION_EXPRESSION)) == 1
        for key in (_CALC, ContextKeys.CALCULATION_ERROR, "operand1", "operator"):
            assert llm.field_requests(key) == [], key
        # no Pass 2 reply from either state, and no request at all from perform_calculation
        assert _replies_from(llm, "extract_elements") == 0
        assert not any(
            "<current_state>perform_calculation<" in getattr(r, "system_prompt", "")
            for r in llm.requests
        )

    @pytest.mark.parametrize("written", [5, 7.5])
    def test_a_number_written_as_the_expression_is_still_an_expression(self, written):
        """The field is a ``str`` field: a JSON number reads as its text, it is not refused."""
        script = {**_CALC_SCRIPT, "calculation_expression": written}
        _solution, info = _engine(_ScriptedLLM(script)).solve_problem("What is 5?")

        assert info["final_context"][_CALC] == written
        assert info["final_context"][ContextKeys.RETRY_COUNT] == 0

    def test_a_long_result_reaches_the_answer_digit_for_digit(self):
        """The model's own ``final_solution`` (a cut number) is never what is returned."""
        script = {
            **_CALC_SCRIPT,
            "calculation_expression": "2 ** 100",
            "final_solution": "126765060022822",
        }
        llm = _ScriptedLLM(script)
        solution, info = _engine(llm).solve_problem("What is 2^100?")

        assert solution == info["final_context"][ContextKeys.FINAL_SOLUTION]
        assert solution == "1267650600228229401496703205376"
        assert llm.field_requests(ContextKeys.FINAL_SOLUTION) == []

    @pytest.mark.parametrize(
        ("expression", "written", "reason"),
        [
            ("10 / 0", "Division by zero is undefined.", "division by zero"),
            ("2 ^ 10", "1000", "not an arithmetic expression"),
        ],
    )
    def test_a_refused_expression_never_ends_with_the_models_own_answer(
        self, expression, written, reason
    ):
        """D-052: the model's text beside a refusal is no solution; the bounded
        retry runs and the solve ends with the fixed message."""
        script = {
            **_CALC_SCRIPT,
            "calculation_expression": expression,
            "proposed_solution": written,
            "final_solution": written,
        }
        llm = _ScriptedLLM(script)
        solution, info = _engine(llm).solve_problem("What is 2^10?")
        context = info["final_context"]

        assert solution == f"The calculation could not be completed: {reason}"
        assert context[ContextKeys.CALCULATION_ERROR_DETAILS] == reason
        assert context.get(_CALC) is None
        assert context[ContextKeys.RETRY_COUNT] == Defaults.MAX_RETRIES
        assert context[ContextKeys.MAX_RETRIES_REACHED] is True
        assert llm.field_requests(ContextKeys.FINAL_SOLUTION) == []

    def test_a_bulk_claim_of_the_result_is_not_the_answer_of_a_refused_solve(self):
        script = {**_CALC_SCRIPT, "calculation_expression": "10 / 0"}
        llm = _ClaimingLLM(script, {_CALC: 999})
        solution, info = _engine(llm).solve_problem("10 / 0?")

        assert "999" not in solution and "division by zero" in solution
        assert info["final_context"].get(_CALC) is None

    def test_a_refusal_then_a_success_keeps_no_stale_error(self):
        class Seq(_ScriptedLLM):
            asked = 0

            def extract_field(self, request):
                if request.field_name == ContextKeys.CALCULATION_EXPRESSION:
                    self.asked += 1
                    expression = "10 / 0" if self.asked == 1 else "6 * 7"
                    self.script = {**self.script, request.field_name: expression}
                return super().extract_field(request)

        script = {k: v for k, v in _CALC_SCRIPT.items() if k != "final_solution"}
        solution, info = _engine(Seq(script)).solve_problem("What is 6 times 7?")
        context = info["final_context"]

        assert solution == "42" and context[_CALC] == 42
        assert context[ContextKeys.RETRY_COUNT] == 1
        assert context.get(ContextKeys.CALCULATION_ERROR_DETAILS) is None

    def test_a_callers_claim_does_not_survive_into_the_answer(self):
        context = {_CALC: 999, ContextKeys.CALCULATION_ERROR_DETAILS: "stale"}
        script = {k: v for k, v in _CALC_SCRIPT.items() if k != "final_solution"}
        solution, info = _engine(_ScriptedLLM(script)).solve_problem(
            "What is 15% of 240?", initial_context=context
        )

        assert solution == "36"
        assert info["final_context"].get(ContextKeys.CALCULATION_ERROR_DETAILS) is None

    @pytest.mark.parametrize("missing", ["10 / 0", None])
    def test_an_error_nobody_explains_retries_at_most_the_limit(self, missing):
        script = {**_CALC_SCRIPT, "calculation_expression": missing}
        llm = _ScriptedLLM(script)
        _solution, info = _engine(llm).solve_problem("10 / 0?")

        assert info["final_context"][ContextKeys.RETRY_COUNT] == Defaults.MAX_RETRIES
        assert len(llm.requests) < 120  # an edge gated on the key cost 428 calls

    def test_a_handler_failure_stops_the_solve(self, monkeypatch):
        def explode(context):
            raise RuntimeError("calculator bug")

        monkeypatch.setattr(
            ReasoningHandlers, "execute_calculation", staticmethod(explode)
        )
        with pytest.raises(ReasoningExecutionError, match="calculator bug"):
            _engine(_ScriptedLLM(_CALC_SCRIPT)).solve_problem("What is 15% of 240?")


class TestCallerCannotDriveThePush:
    """Review pass 15 note 9: the push hook's driver keys are dropped from a
    caller's initial_context."""

    def test_a_caller_push_flag_alone_does_not_break_the_solve(self):
        solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM, {ContextKeys.REASONING_PUSH_PENDING: True}
        )

        assert solution == _VALID_SCRIPT["final_solution"]
        assert trace_info["final_context"][ContextKeys.VALIDATION_RESULT] is True

    def test_a_caller_cannot_push_a_strategy_before_the_analysis(self):
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(
            _PROBLEM,
            {
                ContextKeys.REASONING_PUSH_PENDING: True,
                ContextKeys.REASONING_TYPE_SELECTED: ReasoningType.CREATIVE.value,
            },
        )
        context = trace_info["final_context"]

        # Only the strategy the orchestrator chose ran.
        assert context[ContextKeys.REASONING_TYPE_SELECTED] == "deductive"
        assert "creative_reasoning_completed" not in context
        assert context["deductive_reasoning_completed"] is True


class TestMalformedClassifierValues:
    """Review pass 16 C1 (D-060): a badly shaped classifier value from the
    model is normalised, not a failed solve, while the classifier handler
    stays critical for real failures."""

    @pytest.mark.parametrize(
        ("overrides", "classified", "alternatives"),
        [
            # Seen live (e1f63a9 P2_t1): a str for an (any) list field.
            (
                {ContextKeys.ALTERNATIVE_APPROACHES: '["cost first", "ratio"]'},
                "deductive",
                ['["cost first", "ratio"]'],
            ),
            (
                {ContextKeys.ALTERNATIVE_APPROACHES: [{"name": "x"}, "inductive"]},
                "deductive",
                ["inductive"],
            ),
            ({ContextKeys.ALTERNATIVE_APPROACHES: 3}, "deductive", []),
            (
                {ContextKeys.RECOMMENDED_REASONING_TYPE: ["deductive", "x"]},
                "analytical",
                ["analytical"],
            ),
            (
                {ContextKeys.RECOMMENDED_REASONING_TYPE: ""},
                "analytical",
                ["analytical"],
            ),
            (
                {ContextKeys.STRATEGY_JUSTIFICATION: ["a", "b"]},
                "deductive",
                ["analytical"],
            ),
            (
                {ContextKeys.PROBLEM_DOMAIN: {"primary": "logic"}},
                "deductive",
                ["analytical"],
            ),
        ],
        ids=[
            "alternatives-str",
            "alternatives-dicts",
            "alternatives-number",
            "recommended-list",
            "recommended-empty",
            "justification-list",
            "domain-dict",
        ],
    )
    def test_a_malformed_value_still_solves(self, overrides, classified, alternatives):
        llm = _ScriptedLLM({**_VALID_SCRIPT, **overrides})
        solution, trace_info = _engine(llm).solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        assert solution == _VALID_SCRIPT["final_solution"]
        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == classified
        assert context[ContextKeys.REASONING_TYPE_SELECTED] == classified
        assert context[ContextKeys.ALTERNATIVE_APPROACHES] == alternatives
        for key in (
            ContextKeys.CLASSIFICATION_JUSTIFICATION,
            ContextKeys.PROBLEM_DOMAIN,
        ):
            assert isinstance(context[key], str)

    def test_well_shaped_values_pass_unchanged(self):
        _solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM
        )
        context = trace_info["final_context"]

        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "deductive"
        assert context[ContextKeys.CLASSIFICATION_JUSTIFICATION] == "a syllogism"
        assert context[ContextKeys.PROBLEM_DOMAIN] == "logic"
        assert context[ContextKeys.ALTERNATIVE_APPROACHES] == ["analytical"]


class _HandlerCrash(RuntimeError):
    """Raised by a patched engine handler on its first call."""


def _crash_on_first_call(
    monkeypatch: pytest.MonkeyPatch, owner: type, name: str
) -> list[Any]:
    """Patch ``owner.name`` (before the engine registers it) to raise
    ``_HandlerCrash`` on its first call and delegate afterwards; returns the
    list of recorded calls."""
    original = inspect.getattr_static(owner, name)
    is_static = isinstance(original, staticmethod)
    function = original.__func__ if is_static else original
    calls: list[Any] = []

    def patched(*args: Any) -> Any:
        calls.append(args)
        if len(calls) == 1:
            raise _HandlerCrash(f"{name} crashed")
        return function(*args)

    monkeypatch.setattr(owner, name, staticmethod(patched) if is_static else patched)
    return calls


class TestEachCriticalRegistration:
    """Review pass 16 W2: each `.critical()` registration is pinned on its
    own. The first call of one handler crashes; later calls work, so a
    non-critical registration would log the crash, skip it and finish (the
    retry limiter's own validate_solution call cannot mask the validator)."""

    @pytest.mark.parametrize(
        ("owner", "name", "initial_context"),
        [
            (ReasoningEngine, "_classify_problem", None),
            (ReasoningEngine, "_prepare_reasoning_execution", None),
            (ReasoningHandlers, "validate_solution", None),
            (ReasoningEngine, "_check_retry_limit", None),
            (
                ReasoningHandlers,
                "count_hybrid_loop",
                {ContextKeys.PREFERRED_REASONING_TYPE: ReasoningType.HYBRID.value},
            ),
        ],
        ids=["classifier", "executor", "validator", "retry-limiter", "hybrid-counter"],
    )
    def test_a_crash_stops_the_solve(self, monkeypatch, owner, name, initial_context):
        calls = _crash_on_first_call(monkeypatch, owner, name)
        engine = _engine(_ScriptedLLM(_HYBRID_SCRIPT))

        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM, initial_context)

        assert len(calls) == 1
        assert any(isinstance(e, _HandlerCrash) for e in _causes(info.value))
        assert engine.orchestrator.list_active_conversations() == []


class TestPartialContextReadFailure:
    """Review pass 16 note 5: a get_data that raises ValueError (a stack torn
    down mid-run) while a failed solve reads its partial context still gives
    ReasoningExecutionError and ends the conversation."""

    def test_value_error_on_the_read_is_reported_and_the_solve_ended(self, monkeypatch):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        api = engine.orchestrator
        original_get_data = api.get_data
        popped: list[bool] = []

        def failing_pop(*args: Any, **kwargs: Any) -> str:
            popped.append(True)
            raise RuntimeError("pop exploded")

        def get_data(conversation_id: str) -> dict[str, Any]:
            if popped:
                raise ValueError(f"Unknown conversation ID: {conversation_id}")
            return original_get_data(conversation_id)

        monkeypatch.setattr(api, "pop_fsm", failing_pop)
        monkeypatch.setattr(api, "get_data", get_data)

        with pytest.raises(ReasoningExecutionError, match="pop exploded") as info:
            engine.solve_problem(_PROBLEM)

        assert info.value.details["partial_context"] is None
        assert isinstance(info.value.__cause__, RuntimeError)
        assert api.list_active_conversations() == []


class TestCallerCannotCarryAClassification:
    """Review pass 16 note 7: a caller-set classified_problem_type is dropped.
    With no problem_type in the caller's context the classifier then runs; a
    carried problem_type still skips it (pass 17 W3, Known open)."""

    def test_a_carried_classification_is_dropped(self):
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(
            _PROBLEM,
            {ContextKeys.CLASSIFIED_PROBLEM_TYPE: ReasoningType.CREATIVE.value},
        )
        context = trace_info["final_context"]

        assert llm.field_requests(ContextKeys.RECOMMENDED_REASONING_TYPE)
        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "deductive"
        assert context[ContextKeys.REASONING_TYPE_SELECTED] == "deductive"
        assert "creative_reasoning_completed" not in context


_NON_STR_VALUES = [["logic"], 7, {"kind": "logic"}]
_NON_STR_IDS = ["list", "number", "dict"]


class TestNonStrProblemType:
    """Review pass 17 W1 (D-061): a non-str problem_type (caller context or
    model value) reads as "" in the validator and the retry limiter, so the
    solve finishes instead of failing on `.lower()`."""

    @pytest.mark.parametrize("problem_type", _NON_STR_VALUES, ids=_NON_STR_IDS)
    def test_a_caller_non_str_problem_type_solves(self, problem_type):
        solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM, {ContextKeys.PROBLEM_TYPE: problem_type}
        )
        context = trace_info["final_context"]

        assert solution == _VALID_SCRIPT["final_solution"]
        assert context[ContextKeys.VALIDATION_RESULT] is True

    @pytest.mark.parametrize("problem_type", _NON_STR_VALUES, ids=_NON_STR_IDS)
    def test_the_validator_reads_it_as_empty(self, problem_type):
        context = {
            ContextKeys.PROPOSED_SOLUTION: "x",
            ContextKeys.PROBLEM_TYPE: problem_type,
        }

        result = ReasoningHandlers.validate_solution(context)

        assert result[ContextKeys.VALIDATION_RESULT] is False

    @pytest.mark.parametrize("problem_type", _NON_STR_VALUES, ids=_NON_STR_IDS)
    def test_the_retry_limiter_gives_a_verdict(self, problem_type):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        result = engine._check_retry_limit({ContextKeys.PROBLEM_TYPE: problem_type})

        assert result[ContextKeys.VALIDATION_RESULT] is False
        assert result[ContextKeys.MAX_RETRIES_REACHED] is False


class TestNonStrReasoningStrategy:
    """Review pass 17 W2 (D-061): with problem_type preset the classifier does
    not run, so the model's reasoning_strategy decides; a non-str value is
    treated as absent (analytical), not a crash in map_reasoning_type."""

    @pytest.mark.parametrize("strategy", _NON_STR_VALUES, ids=_NON_STR_IDS)
    def test_a_non_str_strategy_runs_analytical(self, strategy):
        llm = _ScriptedLLM({**_VALID_SCRIPT, ContextKeys.REASONING_STRATEGY: strategy})
        _solution, trace_info = _engine(llm).solve_problem(
            _PROBLEM, {ContextKeys.PROBLEM_TYPE: "logic puzzle"}
        )
        context = trace_info["final_context"]

        assert not llm.field_requests(ContextKeys.RECOMMENDED_REASONING_TYPE)
        assert (
            context[ContextKeys.REASONING_TYPE_SELECTED]
            == ReasoningType.ANALYTICAL.value
        )


class _InterruptingLLM(_ScriptedLLM):
    """Raises ``exc`` when asked for ``field`` (an interrupt inside a pushed FSM).

    ``frames_at_raise`` records how many instances the orchestrator manager held
    at that moment (the orchestrator and the pushed strategy frame: 2).
    """

    def __init__(self, script: dict[str, Any], field: str, exc: BaseException) -> None:
        super().__init__(script)
        self.field = field
        self.exc = exc
        self.frames_at_raise: int | None = None
        self.api: Any = None

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        if request.field_name == self.field:
            self.frames_at_raise = len(self.api.fsm_manager.instances)
            raise self.exc
        return super().extract_field(request)


_INTERRUPTS = [KeyboardInterrupt, SystemExit]


class TestAnInterruptEndsTheSolve:
    """F1 (reasoning audit 1.2): a KeyboardInterrupt or SystemExit is the
    caller's signal and reaches it unwrapped, but the orchestrator conversation
    (and a pushed strategy frame) is ended, not left in the manager."""

    @pytest.mark.parametrize("interrupt", _INTERRUPTS)
    def test_an_interrupt_in_the_run_leaves_no_conversation(
        self, monkeypatch, interrupt
    ):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        def interrupted(*args: Any, **kwargs: Any) -> Any:
            raise interrupt()

        monkeypatch.setattr(engine.orchestrator, "run_until_terminal", interrupted)

        with pytest.raises(interrupt):
            engine.solve_problem(_PROBLEM)

        assert engine.orchestrator.list_active_conversations() == []
        assert dict(engine.orchestrator.fsm_manager.instances) == {}
        assert not engine._solve_lock.locked()

    @pytest.mark.parametrize("interrupt", _INTERRUPTS)
    def test_an_interrupt_inside_a_pushed_strategy_frame_ends_both(self, interrupt):
        llm = _InterruptingLLM(_VALID_SCRIPT, "conclusion", interrupt())
        engine = _engine(llm)
        llm.api = engine.orchestrator

        with pytest.raises(interrupt):
            engine.solve_problem(_PROBLEM)

        assert llm.frames_at_raise == 2  # the strategy frame was pushed and live
        assert engine.orchestrator.list_active_conversations() == []
        assert dict(engine.orchestrator.fsm_manager.instances) == {}

    def test_an_ordinary_error_is_still_wrapped(self, monkeypatch):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        monkeypatch.setattr(
            engine.orchestrator,
            "run_until_terminal",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(ReasoningExecutionError, match="boom"):
            engine.solve_problem(_PROBLEM)

        assert engine.orchestrator.list_active_conversations() == []

    def test_the_event_log_writer_is_released_after_an_interrupt(
        self, monkeypatch, tmp_path
    ):
        from fsm_llm.runtime import EventLog

        log = EventLog(tmp_path / "log")
        engine = ReasoningEngine(
            model="mock", llm_interface=_ScriptedLLM(_VALID_SCRIPT), event_log=log
        )
        seen: list[str] = []

        def interrupted(conversation_id: str, **kwargs: Any) -> Any:
            seen.append(conversation_id)
            assert log.has_writer(conversation_id)
            raise KeyboardInterrupt

        monkeypatch.setattr(engine.orchestrator, "run_until_terminal", interrupted)

        with pytest.raises(KeyboardInterrupt):
            engine.solve_problem(_PROBLEM)
        engine.close()

        assert seen and not log.has_writer(seen[0])
        log.close()


class TestEngineClose:
    """F1: ``close()`` ends what is open on both APIs and is idempotent; the
    engine is a context manager."""

    def test_close_ends_every_open_conversation_on_both_apis(self):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        orchestrator_id, _ = engine.orchestrator.start_conversation({})
        classifier_id, _ = engine.classifier.start_conversation({})

        engine.close()

        assert engine.orchestrator.list_active_conversations() == []
        assert engine.classifier.list_active_conversations() == []
        assert orchestrator_id and classifier_id

    def test_close_twice_is_harmless(self, tmp_path):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        engine.close()
        engine.close()

        # APIs that own their event log (a path) close it once, then again
        owning = ReasoningEngine(
            model="mock",
            llm_interface=_ScriptedLLM(_VALID_SCRIPT),
            event_log=tmp_path / "owned",
        )
        owning.orchestrator.start_conversation({})
        owning.close()
        owning.close()
        assert owning.orchestrator.list_active_conversations() == []

    def test_the_context_manager_closes_on_exit_and_on_error(self):
        with _engine(_ScriptedLLM(_VALID_SCRIPT)) as engine:
            engine.orchestrator.start_conversation({})
            assert engine.orchestrator.list_active_conversations()
        assert engine.orchestrator.list_active_conversations() == []

        with pytest.raises(RuntimeError, match="body failed"):
            with _engine(_ScriptedLLM(_VALID_SCRIPT)) as failing:
                failing.classifier.start_conversation({})
                raise RuntimeError("body failed")
        assert failing.classifier.list_active_conversations() == []

    def test_enter_returns_the_engine(self):
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        with engine as entered:
            assert entered is engine


class _InsightsLateLLM(_ScriptedLLM):
    """Answers ``key_insights`` with null on its FIRST ask and a value after."""

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        if request.field_name == ContextKeys.KEY_INSIGHTS:
            asked = len(self.field_requests(ContextKeys.KEY_INSIGHTS))
            self.requests.append(request)
            value = None if asked == 0 else self.script[ContextKeys.KEY_INSIGHTS]
            return FieldExtractionResponse(
                field_name=request.field_name,
                value=value,
                confidence=0.9 if value is not None else 0.0,
            )
        return super().extract_field(request)


class TestAnEarlyVerdictWaitsForTheInsights:
    """F2 (reasoning audit 1.1): the validator fires when the proposed solution
    is committed. When key_insights arrive in a LATER commit, that early verdict
    is a stale False; it must not cost a whole strategy pass and a retry slot.
    The attempt is judged at validate_refine entry instead (D-054)."""

    def test_late_insights_do_not_cost_a_retry(self):
        llm = _InsightsLateLLM(_VALID_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        assert len(llm.field_requests(ContextKeys.KEY_INSIGHTS)) >= 2  # was late
        assert _visited(trace_info).count(OrchestratorStates.EXECUTE_REASONING) == 1
        assert context[ContextKeys.RETRY_COUNT] == 0
        assert context[ContextKeys.VALIDATION_RESULT] is True

    def test_insights_that_never_arrive_still_count_one_failed_attempt(self):
        script = {**_VALID_SCRIPT, ContextKeys.KEY_INSIGHTS: None}
        llm = _ScriptedLLM(script)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        assert context[ContextKeys.RETRY_COUNT] == Defaults.MAX_RETRIES
        assert context[ContextKeys.MAX_RETRIES_REACHED] is True

    def test_the_wrapper_defers_only_on_missing_insights(self):
        ready = {
            ContextKeys.PROPOSED_SOLUTION: _VALID_SCRIPT["proposed_solution"],
            ContextKeys.PROBLEM_STATEMENT: _PROBLEM,
            ContextKeys.KEY_INSIGHTS: ["x"],
        }
        late = {k: v for k, v in ready.items() if k != ContextKeys.KEY_INSIGHTS}

        assert ReasoningHandlers.validate_when_insights_ready(
            ready
        ) == ReasoningHandlers.validate_solution(ready)
        assert ReasoningHandlers.validate_when_insights_ready(late) == {}
        # a simple problem needs no insights, so its verdict is given at once
        simple = {**late, ContextKeys.REASONING_STRATEGY: "simple_calculator"}
        assert (
            ReasoningHandlers.validate_when_insights_ready(simple)[
                ContextKeys.VALIDATION_RESULT
            ]
            is True
        )


class _StatedModelLLM(_ScriptedLLM):
    """An injected interface that states its own model (as LiteLLMInterface does)."""

    model = "model-b"


class TestReportedModel:
    """F3 (reasoning audit 2.1): ``engine.model`` reports the model the injected
    interface really uses, never refuses a differing ``model=`` (D-038)."""

    def test_the_interface_model_wins_over_a_differing_argument(self):
        engine = ReasoningEngine(model="model-a", llm_interface=_StatedModelLLM({}))

        assert engine.model == "model-b"
        assert engine.orchestrator.llm_interface.model == "model-b"

    def test_an_interface_that_states_no_model_keeps_the_argument(self):
        engine = ReasoningEngine(model="model-a", llm_interface=_ScriptedLLM({}))

        assert engine.model == "model-a"

    def test_neither_gives_the_label(self):
        engine = ReasoningEngine(llm_interface=_ScriptedLLM({}))

        assert isinstance(engine.model, str) and engine.model


class TestHybridLoopCountIsHandlerOnlyOnTheOrchestrator:
    """F3 (reasoning audit section 3): a model bulk claim of ``hybrid_loop_count``
    must not land in the orchestrator frame and the caller's final_context."""

    def test_the_key_is_listed(self):
        assert ContextKeys.HYBRID_LOOP_COUNT in ORCHESTRATOR_HANDLER_ONLY_KEYS

    def test_a_bulk_claim_is_left_out_of_the_final_context(self):
        llm = _ClaimingLLM(_HYBRID_SCRIPT, claims={ContextKeys.HYBRID_LOOP_COUNT: 99})
        _solution, trace_info = _engine(llm).solve_problem(
            _PROBLEM, {ContextKeys.PREFERRED_REASONING_TYPE: "hybrid"}
        )

        assert ContextKeys.HYBRID_LOOP_COUNT not in trace_info["final_context"]
