from __future__ import annotations

"""Tests for fsm_llm.agents.adapt module."""

from typing import Any

import pytest

from fsm_llm.agents.adapt import ADaPTAgent
from fsm_llm.agents.constants import (
    ADaPTStates,
    ContextKeys,
    Defaults,
    HandlerNames,
)
from fsm_llm.agents.definitions import AgentConfig, DecompositionResult
from fsm_llm.agents.exceptions import (
    AgentError,
    AgentTimeoutError,
    BudgetExhaustedError,
)
from fsm_llm.agents.fsm_definitions import build_adapt_fsm
from fsm_llm.agents.tools import ToolRegistry
from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface


def _dummy_tool(params):
    return "result"


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_dummy_tool, name="search", description="Search the web")
    return registry


class TestADaPTCreation:
    """Tests for ADaPTAgent initialization."""

    def test_create_with_defaults(self):
        agent = ADaPTAgent()
        assert agent.max_depth == Defaults.MAX_DECOMPOSITION_DEPTH
        assert agent.config is not None
        assert agent.tools is None

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = ADaPTAgent(tools=registry)
        assert agent.tools is registry

    def test_create_without_tools(self):
        agent = ADaPTAgent()
        assert agent.tools is None

    def test_create_with_max_depth(self):
        agent = ADaPTAgent(max_depth=5)
        assert agent.max_depth == 5

    def test_create_with_config_override(self):
        config = AgentConfig(max_iterations=15, model="gpt-4o-mini")
        agent = ADaPTAgent(config=config)
        assert agent.config.max_iterations == 15
        assert agent.config.model == "gpt-4o-mini"

    def test_has_run_method(self):
        agent = ADaPTAgent()
        assert callable(getattr(agent, "run", None))

    def test_run_accepts_depth_parameter(self):
        """run() accepts _depth for recursive tracking."""
        import inspect

        sig = inspect.signature(ADaPTAgent.run)
        assert "_depth" in sig.parameters

    def test_create_with_tools_and_max_depth(self):
        registry = _make_registry()
        agent = ADaPTAgent(tools=registry, max_depth=10)
        assert agent.tools is registry
        assert agent.max_depth == 10


class TestADaPTFSM:
    """Tests for build_adapt_fsm function."""

    def test_basic_fsm_structure(self):
        fsm = build_adapt_fsm()
        assert fsm["name"] == "adapt_agent"
        assert fsm["initial_state"] == "attempt"
        assert len(fsm["states"]) == 4

    def test_fsm_has_all_four_states(self):
        fsm = build_adapt_fsm()
        expected = {"attempt", "assess", "decompose", "combine"}
        assert set(fsm["states"].keys()) == expected

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_adapt_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "adapt_agent"

    def test_attempt_transitions_to_assess(self):
        fsm = build_adapt_fsm()
        targets = {t["target_state"] for t in fsm["states"]["attempt"]["transitions"]}
        assert "assess" in targets

    def test_assess_transitions_to_combine_and_decompose(self):
        fsm = build_adapt_fsm()
        targets = {t["target_state"] for t in fsm["states"]["assess"]["transitions"]}
        assert "combine" in targets
        assert "decompose" in targets

    def test_decompose_transitions_to_combine(self):
        fsm = build_adapt_fsm()
        targets = {t["target_state"] for t in fsm["states"]["decompose"]["transitions"]}
        assert "combine" in targets

    def test_combine_is_terminal(self):
        fsm = build_adapt_fsm()
        assert fsm["states"]["combine"]["transitions"] == []

    def test_custom_task_description(self):
        fsm = build_adapt_fsm(task_description="Explain quantum computing")
        assert fsm["description"] == "Explain quantum computing"

    def test_default_task_description(self):
        fsm = build_adapt_fsm()
        assert fsm["description"] == "ADaPT agent with recursive decomposition"

    def test_persona_mentions_adaptive(self):
        fsm = build_adapt_fsm()
        assert "adaptive" in fsm["persona"].lower()

    def test_fsm_with_registry(self):
        """build_adapt_fsm accepts an optional ToolRegistry."""
        registry = _make_registry()
        fsm = build_adapt_fsm(registry=registry)
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "adapt_agent"

    def test_assess_success_priority_higher_than_decompose(self):
        """Lower priority number = higher confidence in TransitionEvaluator."""
        fsm = build_adapt_fsm()
        assess_transitions = fsm["states"]["assess"]["transitions"]
        combine_success_priority = None
        decompose_priority = None
        for t in assess_transitions:
            if t["target_state"] == "combine" and combine_success_priority is None:
                combine_success_priority = t["priority"]
            elif t["target_state"] == "decompose":
                decompose_priority = t["priority"]
        assert combine_success_priority is not None
        assert decompose_priority is not None
        assert combine_success_priority < decompose_priority

    def test_max_depth_embedded_in_fsm(self):
        """The max_depth parameter should be referenced in the FSM conditions."""
        fsm = build_adapt_fsm(max_depth=7)
        # The assess->decompose transition should reference max_depth in its logic
        assess_transitions = fsm["states"]["assess"]["transitions"]
        decompose_transition = None
        for t in assess_transitions:
            if t["target_state"] == "decompose":
                decompose_transition = t
                break
        assert decompose_transition is not None
        # The condition logic should include the depth check with value 7
        conditions = decompose_transition.get("conditions", [])
        assert len(conditions) > 0


class TestDecompositionResultModel:
    """Tests for DecompositionResult Pydantic model."""

    def test_basic_creation(self):
        result = DecompositionResult()
        assert result.subtasks == []
        assert result.operator == "AND"
        assert result.depth == 0

    def test_creation_with_subtasks(self):
        result = DecompositionResult(
            subtasks=["Find data", "Analyze data", "Report findings"],
            operator="AND",
            depth=1,
        )
        assert len(result.subtasks) == 3
        assert result.operator == "AND"
        assert result.depth == 1

    def test_or_operator(self):
        result = DecompositionResult(
            subtasks=["Try method A", "Try method B"],
            operator="OR",
        )
        assert result.operator == "OR"

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match=r"AND.*OR"):
            DecompositionResult(operator="XOR")

    def test_operator_lowercase_normalized(self):
        result = DecompositionResult(operator="and")
        assert result.operator == "AND"
        result2 = DecompositionResult(operator="or")
        assert result2.operator == "OR"

    def test_depth_field(self):
        result = DecompositionResult(depth=3)
        assert result.depth == 3

    def test_serialization(self):
        result = DecompositionResult(
            subtasks=["a", "b"],
            operator="OR",
            depth=2,
        )
        data = result.model_dump(mode="json")
        assert data["subtasks"] == ["a", "b"]
        assert data["operator"] == "OR"
        assert data["depth"] == 2


class TestADaPTConstants:
    """Tests for ADaPT-related constants."""

    def test_adapt_states_attempt(self):
        assert ADaPTStates.ATTEMPT == "attempt"

    def test_adapt_states_assess(self):
        assert ADaPTStates.ASSESS == "assess"

    def test_adapt_states_decompose(self):
        assert ADaPTStates.DECOMPOSE == "decompose"

    def test_adapt_states_combine(self):
        assert ADaPTStates.COMBINE == "combine"

    def test_context_keys_attempt_result(self):
        assert ContextKeys.ATTEMPT_RESULT == "attempt_result"

    def test_context_keys_attempt_succeeded(self):
        assert ContextKeys.ATTEMPT_SUCCEEDED == "attempt_succeeded"

    def test_context_keys_subtask_results(self):
        assert ContextKeys.SUBTASK_RESULTS == "subtask_results"

    def test_context_keys_current_depth(self):
        assert ContextKeys.CURRENT_DEPTH == "current_depth"

    def test_defaults_max_decomposition_depth(self):
        assert Defaults.MAX_DECOMPOSITION_DEPTH == 3

    def test_handler_name_adapt_assessor(self):
        assert HandlerNames.ADAPT_ASSESSOR == "ADaPTAssessor"


class TestADaPTJSONLeakFix:
    """Regression tests for the adapt JSON-leak bug (plan_2026-05-31_03830272/D-001):
    (A) success no longer hard-coded True; (B) raw extraction-envelope responses
    must not surface as the final answer."""

    # --- Part B: _is_extraction_envelope -------------------------------------
    def test_envelope_detected_raw(self):
        assert ADaPTAgent._is_extraction_envelope(
            '{"extracted_data": {}, "confidence": 0.9, "reasoning": "x"}'
        )

    def test_envelope_detected_fenced(self):
        assert ADaPTAgent._is_extraction_envelope(
            '```json\n{"extracted_data": {"a": 1}}\n```'
        )

    def test_envelope_not_detected_prose(self):
        assert not ADaPTAgent._is_extraction_envelope("Paris is the capital of France.")

    def test_envelope_not_detected_prose_mentions_key(self):
        # Prose that merely mentions the word must NOT be dropped.
        assert not ADaPTAgent._is_extraction_envelope(
            "The field extracted_data was empty in my analysis."
        )

    def test_envelope_not_detected_other_json(self):
        assert not ADaPTAgent._is_extraction_envelope('{"answer": "42"}')

    # --- Part B: _extract_answer skips the envelope --------------------------
    def test_extract_answer_skips_envelope_returns_default(self):
        agent = ADaPTAgent()
        answer = agent._extract_answer(
            {}, ['{"extracted_data": {}, "reasoning": "Continue. has no info"}']
        )
        assert answer == "ADaPT agent could not determine an answer."

    def test_extract_answer_keeps_prose_response(self):
        agent = ADaPTAgent()
        answer = agent._extract_answer({}, ["Paris is the capital of France."])
        assert answer == "Paris is the capital of France."

    def test_extract_answer_prefers_final_answer(self):
        agent = ADaPTAgent()
        answer = agent._extract_answer(
            {ContextKeys.FINAL_ANSWER: "The capital names are equal length."},
            ['{"extracted_data": {}}'],
        )
        assert answer == "The capital names are equal length."

    # --- Part A: success guard (same call adapt.run() now makes) --------------
    def test_degenerate_completion_is_not_success(self):
        from fsm_llm.agents.definitions import AgentTrace

        # No final_answer, no tool calls → degenerate (was hard-coded True).
        assert (
            ADaPTAgent._completion_is_real(
                {ContextKeys.ATTEMPT_RESULT: "Tokyo"},
                AgentTrace(tool_calls=[], total_iterations=6),
                None,
            )
            is False
        )

    def test_real_completion_is_success(self):
        from fsm_llm.agents.definitions import AgentTrace

        assert ADaPTAgent._completion_is_real(
            {ContextKeys.FINAL_ANSWER: "Paris vs Tokyo: equal length."},
            AgentTrace(tool_calls=[], total_iterations=2),
            None,
        )

    def test_succeeded_attempt_result_is_success(self):
        # ADaPT's legitimate "attempt succeeded" path: attempt_result is the
        # answer (no separate final_answer). run() passes ATTEMPT_RESULT as an
        # answer key ONLY when attempt_succeeded is true — distinguishing it from
        # the leak case above (failed attempt → attempt_result is partial).
        from fsm_llm.agents.definitions import AgentTrace

        assert ADaPTAgent._completion_is_real(
            {ContextKeys.ATTEMPT_RESULT: "Complete answer here."},
            AgentTrace(tool_calls=[], total_iterations=2),
            [ContextKeys.ATTEMPT_RESULT],
        )


# -------------------------------------------------------------------------
# D-002 sibling sweep: assess and decompose must not BLOCK when their gating
# key never extracts (no PRE_TRANSITION limiter runs on a BLOCKED turn).
# -------------------------------------------------------------------------


class _SilentKeysLLM(LLMInterface):
    """Mock LLM that never extracts ``silent`` keys; ``values`` overrides the
    plain-string value every other field gets."""

    def __init__(self, silent: set[str], values: dict[str, Any] | None = None):
        self.model = "mock-model"
        self.silent = silent
        self.values = values or {}

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        silent = request.field_name in self.silent
        value = None if silent else self.values.get(request.field_name, "some text")
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=0.0 if silent else 0.9,
            reasoning="mock",
            is_valid=not silent,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="final answer", message_type="response", reasoning="mock"
        )


class TestADaPTLoopStatesNeverBlock:
    """Plan plan-2026-09-24T045559-3e4eb3e5 step 2.1 (review concern 4)."""

    MAX_ITERATIONS = 4

    @pytest.mark.parametrize("state", ["assess", "decompose"])
    def test_state_has_unconditional_fallback_edge(self, state):
        transitions = build_adapt_fsm()["states"][state]["transitions"]
        fallbacks = [t for t in transitions if not t.get("conditions")]
        assert len(fallbacks) == 1
        assert fallbacks[0]["target_state"] == ADaPTStates.COMBINE
        assert fallbacks[0]["priority"] > max(
            t["priority"] for t in transitions if t.get("conditions")
        )

    def _run(self, llm: LLMInterface) -> tuple[Any, list[int]]:
        agent = ADaPTAgent(
            config=AgentConfig(max_iterations=self.MAX_ITERATIONS),
            llm_interface=llm,
        )
        loop_counts: list[int] = []
        real_loop = agent._run_conversation_loop

        def _recording_loop(*args: Any, **kwargs: Any):
            responses, final_context, iteration = real_loop(*args, **kwargs)
            loop_counts.append(iteration)
            return responses, final_context, iteration

        agent._run_conversation_loop = _recording_loop  # type: ignore[method-assign]
        return agent.run("Solve 2+2"), loop_counts

    def test_missing_attempt_succeeded_reaches_combine(self):
        # Pre-fix: assess only had should_terminate / attempt_succeeded edges,
        # so a missing value BLOCKED assess until BudgetExhaustedError.
        llm = _SilentKeysLLM(
            {ContextKeys.ATTEMPT_SUCCEEDED, ContextKeys.SHOULD_TERMINATE}
        )
        result, loop_counts = self._run(llm)
        assert loop_counts, "conversation loop never completed"
        assert loop_counts[0] <= self.MAX_ITERATIONS
        assert result.answer

    def test_missing_subtasks_reaches_combine(self):
        # Pre-fix: decompose only had should_terminate / has_context subtasks
        # edges, so a missing subtasks list BLOCKED decompose.
        llm = _SilentKeysLLM(
            {ContextKeys.SUBTASKS, ContextKeys.SHOULD_TERMINATE},
            {ContextKeys.ATTEMPT_SUCCEEDED: False},
        )
        result, loop_counts = self._run(llm)
        assert loop_counts, "conversation loop never completed"
        assert loop_counts[0] <= self.MAX_ITERATIONS
        assert result.answer


# -------------------------------------------------------------------------
# PAT-04 (plan-2026-09-29T103145-06a5ec0a step 12): budget errors propagate,
# fan-out is capped, a failed decomposition is not a success.
# -------------------------------------------------------------------------


class _DecomposeScriptLLM(LLMInterface):
    """Scripted ADaPT model: the root attempt fails and decomposes into
    ``subtasks``; each subtask run answers from ``sub_success``.

    Extraction reads ``task`` from the request context: a task that is one of
    ``subtasks`` is a subtask run (depth 1), anything else the root run. The
    typed ADaPT field prompts no longer show ``current_depth`` (fix 21.1 of
    plan 06a5ec0a narrowed them to the task and the attempt). ``sub_tasks_seen`` records every subtask run
    in order (one entry per subtask ``attempt`` extraction).
    """

    ROOT_ATTEMPT = "ROOT FAILED ATTEMPT that must never be the answer"

    def __init__(
        self,
        subtasks: list[str],
        operator: str = "AND",
        sub_success: dict[str, bool] | None = None,
        root_final: str | None = None,
    ) -> None:
        self.model = "mock-model"
        self.subtasks = subtasks
        self.operator = operator
        self.sub_success = sub_success or {}
        self.root_final = root_final
        self.sub_tasks_seen: list[str] = []

    def _value(self, field: str, depth: int, task: str) -> Any:
        ok = self.sub_success.get(task, True)
        if field == ContextKeys.ATTEMPT_RESULT:
            if depth == 0:
                return self.ROOT_ATTEMPT
            self.sub_tasks_seen.append(task)
            return f"answer for {task}" if ok else f"partial attempt for {task}"
        if field == ContextKeys.ATTEMPT_SUCCEEDED:
            return False if depth == 0 else ok
        if field == ContextKeys.SUBTASKS:
            return list(self.subtasks) if depth == 0 else None
        if field == ContextKeys.FINAL_ANSWER:
            if depth == 0:
                return self.root_final
            return f"answer for {task}" if ok else None
        return None

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        context = request.context or {}
        task = str(context.get(ContextKeys.TASK, ""))
        value = self._value(request.field_name, 1 if task in self.subtasks else 0, task)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=0.9 if value is not None else 0.0,
            reasoning="script",
            is_valid=value is not None,
        )

    def extract_bulk_data(self, request: Any) -> Any:
        from fsm_llm.definitions import DataExtractionResponse

        data = (
            {"operator": self.operator} if '"operator"' in request.system_prompt else {}
        )
        return DataExtractionResponse(extracted_data=data)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="step done", message_type="response", reasoning="script"
        )


def _adapt(llm: LLMInterface, max_depth: int = 1) -> ADaPTAgent:
    return ADaPTAgent(
        config=AgentConfig(max_iterations=10, timeout_seconds=60.0),
        max_depth=max_depth,
        llm_interface=llm,
    )


class TestADaPTBudgetErrorsPropagate:
    """A subtask's budget/timeout error ends the whole run (D-014 holder)."""

    @pytest.mark.parametrize(
        "error",
        [
            BudgetExhaustedError("iterations", 1),
            AgentTimeoutError(1.0),
        ],
        ids=["budget", "timeout"],
    )
    def test_subtask_budget_error_propagates_from_run(self, error):
        llm = _DecomposeScriptLLM(["sub one", "sub two", "sub three"])
        agent = _adapt(llm)

        def _before_step(api: object, conv_id: str, iteration: int) -> None:
            # A subtask run exhausts its budget once its attempt was extracted.
            if llm.sub_tasks_seen:
                raise error

        agent._on_loop_iteration = _before_step  # type: ignore[method-assign]
        with pytest.raises(type(error)):
            agent.run("root task")
        # No further subtask starts after the budget error.
        assert llm.sub_tasks_seen == ["sub one"]

    def test_holder_contract(self):
        from fsm_llm.agents.handlers import RunEndingErrorHolder

        holder = RunEndingErrorHolder()
        holder.raise_if_set()  # empty: no-op
        assert holder.capture(AgentError("ordinary")) is False
        assert not holder.is_set
        first = BudgetExhaustedError("iterations", 3)
        assert holder.capture(first) is True
        assert holder.capture(BudgetExhaustedError("time", 1)) is True
        with pytest.raises(BudgetExhaustedError) as info:
            holder.raise_if_set()
        assert info.value is first


class TestADaPTFanOutCap:
    def test_fifty_subtasks_run_at_most_the_cap(self, caplog):
        llm = _DecomposeScriptLLM([f"sub {i}" for i in range(50)])
        result = _adapt(llm).run("root task")
        assert len(llm.sub_tasks_seen) == Defaults.ADAPT_MAX_SUBTASKS
        assert len(result.final_context[ContextKeys.SUBTASK_RESULTS]) == (
            Defaults.ADAPT_MAX_SUBTASKS
        )


class TestADaPTDecomposedOutcome:
    def test_all_subtasks_failed_is_not_success(self):
        llm = _DecomposeScriptLLM(
            ["sub a", "sub b"],
            sub_success={"sub a": False, "sub b": False},
            root_final="Combined text that papers over two failed subtasks.",
        )
        result = _adapt(llm).run("root task")
        assert [e["success"] for e in result.final_context["subtask_results"]] == [
            False,
            False,
        ]
        assert result.success is False

    def test_and_with_one_failed_subtask_is_not_success(self):
        llm = _DecomposeScriptLLM(
            ["sub a", "sub b"], operator="AND", sub_success={"sub a": False}
        )
        assert _adapt(llm).run("root task").success is False

    def test_lowercase_or_stops_at_first_success(self):
        llm = _DecomposeScriptLLM(
            ["sub a", "sub b", "sub c"], operator="or", sub_success={"sub a": False}
        )
        result = _adapt(llm).run("root task")
        assert llm.sub_tasks_seen == ["sub a", "sub b"]
        assert result.success is True

    def test_answer_comes_from_subtasks_not_failed_attempt(self):
        llm = _DecomposeScriptLLM(["sub a", "sub b"])
        result = _adapt(llm).run("root task")
        assert result.success is True
        assert _DecomposeScriptLLM.ROOT_ATTEMPT not in result.answer
        assert "answer for sub a" in result.answer
        assert "answer for sub b" in result.answer

    def test_decompose_is_not_a_tool_call(self):
        llm = _DecomposeScriptLLM(["sub a"], sub_success={"sub a": False})
        result = _adapt(llm).run("root task")
        assert all(c.tool_name != "decompose" for c in result.trace.tool_calls)
        assert result.success is False
