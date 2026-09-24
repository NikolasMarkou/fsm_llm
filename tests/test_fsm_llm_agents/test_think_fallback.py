"""CR-01 / F-LIVE-02: a ReAct-family ``think`` state must never BLOCK.

Plan: plan-2026-09-24T045559-3e4eb3e5 / D-002.

Root cause: ``think -> act`` was gated on a valid tool selection
(``tool_name in [...tools, NO_TOOL]``, or ``has_context: tool_calls`` for
ParallelReact). A null or unknown selection left ``think`` BLOCKED, and on a
BLOCKED turn neither the ``act`` entry handler (no-tool feedback, stall
detector) nor any PRE_TRANSITION handler (iteration limiter) runs, so the run
burned the whole 3x loop ceiling and raised ``BudgetExhaustedError``. The fix
makes ``think -> act`` the unconditional lowest-priority fallback, so the
existing no-tool handling runs and the agent concludes.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.parallel_react import ParallelReactAgent
from fsm_llm_agents.react import ReactAgent
from fsm_llm_agents.reflexion import ReflexionAgent
from fsm_llm_agents.tools import ToolRegistry

MAX_ITERATIONS = 6
_MISSING = object()


def _add_numbers(params: dict[str, Any]) -> str:
    return str(int(params.get("a", 0)) + int(params.get("b", 0)))


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        _add_numbers,
        name="add_numbers",
        description="Add two numbers",
        parameter_schema={
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}
        },
    )
    return registry


class _NoSelectionLLM(LLMInterface):
    """Mock LLM that never makes a usable tool selection.

    ``field_map`` gives the value returned for each extracted field; a field
    mapped to ``_MISSING`` (or absent) extracts as invalid, as a weak model's
    unparseable output would.
    """

    def __init__(self, field_map: dict[str, Any]) -> None:
        self.field_map = field_map

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        value = self.field_map.get(request.field_name, _MISSING)
        valid = value is not _MISSING and value is not None
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None if value is _MISSING else value,
            confidence=0.9 if valid else 0.0,
            reasoning="mock",
            is_valid=valid,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="final answer", message_type="response", reasoning="mock"
        )


_REACT_NULL = {"tool_name": None, "tool_input": {}, "should_terminate": False}
_REACT_UNKNOWN = {
    "tool_name": "totally_unknown_tool",
    "tool_input": {},
    "should_terminate": False,
}
_PARALLEL_MISSING = {"should_terminate": False}
_PARALLEL_EMPTY = {"tool_calls": [], "should_terminate": False}


def _build_reasoning_react(registry: ToolRegistry, config: AgentConfig, llm):
    pytest.importorskip("fsm_llm_reasoning")
    from fsm_llm_agents.reasoning_react import ReasoningReactAgent

    return ReasoningReactAgent(tools=registry, config=config, llm_interface=llm)


def _build(agent_cls):
    def factory(registry: ToolRegistry, config: AgentConfig, llm):
        return agent_cls(tools=registry, config=config, llm_interface=llm)

    return factory


# Loop-turn bounds. The hard ceiling is FSM_BUDGET_MULTIPLIER x max_iterations
# (18 here). A think/act cycle is 2 turns, so the 2-state patterns must conclude
# within max_iterations + 2. Reflexion's cycle is 4 turns (think, act, evaluate,
# reflect), so after the iteration limiter fires it can need one more turn.
_TWO_STATE_BOUND = MAX_ITERATIONS + 2
_REFLEXION_BOUND = MAX_ITERATIONS + 3
_HARD_CEILING = 3 * MAX_ITERATIONS

_CASES = [
    pytest.param(_build(ReactAgent), _REACT_NULL, _TWO_STATE_BOUND, id="react-null"),
    pytest.param(
        _build(ReactAgent), _REACT_UNKNOWN, _TWO_STATE_BOUND, id="react-unknown"
    ),
    pytest.param(
        _build(ReflexionAgent), _REACT_NULL, _REFLEXION_BOUND, id="reflexion-null"
    ),
    pytest.param(
        _build(ReflexionAgent),
        _REACT_UNKNOWN,
        _REFLEXION_BOUND,
        id="reflexion-unknown",
    ),
    pytest.param(
        _build(ParallelReactAgent),
        _PARALLEL_MISSING,
        _TWO_STATE_BOUND,
        id="parallel-missing",
    ),
    pytest.param(
        _build_reasoning_react, _REACT_NULL, _TWO_STATE_BOUND, id="reasoning_react-null"
    ),
    pytest.param(
        _build_reasoning_react,
        _REACT_UNKNOWN,
        _TWO_STATE_BOUND,
        id="reasoning_react-unknown",
    ),
]


def _run_and_count(factory, field_map) -> tuple[Any, int]:
    """Run the agent; return (result, conversation-loop iteration count)."""
    config = AgentConfig(max_iterations=MAX_ITERATIONS)
    agent = factory(_make_registry(), config, _NoSelectionLLM(field_map))

    loop_counts: list[int] = []
    real_loop = agent._run_conversation_loop

    def _recording_loop(*args: Any, **kwargs: Any):
        responses, final_context, iteration = real_loop(*args, **kwargs)
        loop_counts.append(iteration)
        return responses, final_context, iteration

    agent._run_conversation_loop = _recording_loop  # type: ignore[method-assign]

    result = agent.run("What is 2 + 2?")
    assert loop_counts, "conversation loop never completed"
    return result, loop_counts[0]


class TestThinkNeverBlocks:
    @pytest.mark.parametrize(("factory", "field_map", "max_loops"), _CASES)
    def test_no_usable_selection_concludes_within_budget(
        self, factory, field_map, max_loops
    ):
        # Pre-fix every case raised BudgetExhaustedError after 3 x max_iterations
        # BLOCKED think turns.
        result, loops = _run_and_count(factory, field_map)

        assert result.answer
        assert max_loops < _HARD_CEILING
        assert loops <= max_loops, (
            f"took {loops} loop iterations, expected at most {max_loops}"
        )

    def test_parallel_empty_batch_concludes_within_budget(self):
        # Regression pin, not a RED case: ``has_context`` is pure key
        # existence, so ``tool_calls=[]`` already reached ``act`` before the
        # fix. It must keep concluding through the empty-batch branch.
        result, loops = _run_and_count(_build(ParallelReactAgent), _PARALLEL_EMPTY)

        assert result.answer
        assert loops <= _TWO_STATE_BOUND


# D-002 review concern 3: an unknown tool name is not evidence. It must take
# the no-tool path (corrective feedback, stall counter, no observation), so a
# hallucinated call cannot satisfy the think->conclude evidence guard.
_UNKNOWN_TERMINATE = {
    "tool_name": "made_up",
    "tool_input": {},
    "should_terminate": True,
}

_UNKNOWN_TERMINATE_CASES = [
    pytest.param(_build(ReactAgent), _TWO_STATE_BOUND, id="react"),
    pytest.param(_build(ReflexionAgent), _REFLEXION_BOUND, id="reflexion"),
    pytest.param(_build_reasoning_react, _TWO_STATE_BOUND, id="reasoning_react"),
]


class TestUnknownToolIsNotEvidence:
    @pytest.mark.parametrize(("factory", "max_loops"), _UNKNOWN_TERMINATE_CASES)
    def test_unknown_tool_with_terminate_does_not_succeed(self, factory, max_loops):
        result, loops = _run_and_count(factory, _UNKNOWN_TERMINATE)

        assert result.success is False
        assert not result.final_context.get("observation_count")
        assert not any(
            "made_up" in str(o) for o in result.final_context.get("observations") or []
        )
        assert loops <= max_loops

    def test_unknown_tool_feedback_names_it_and_counts_as_stall(self):
        from fsm_llm_agents.handlers import AgentHandlers

        handlers = AgentHandlers(_make_registry())
        handlers._current_iteration = 2
        ctx = {"tool_name": "made_up", "tool_input": {}, "should_terminate": True}

        first = handlers.execute_tool(ctx)
        assert "made_up" in first["tool_result"]
        assert "add_numbers" in first["tool_result"]
        assert "observations" not in first
        assert "observation_count" not in first
        handlers.execute_tool(ctx)
        third = handlers.execute_tool(ctx)
        assert third["should_terminate"] is True
        assert third["max_iterations_reached"] is True
