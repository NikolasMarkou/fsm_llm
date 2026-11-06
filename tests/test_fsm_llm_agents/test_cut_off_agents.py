"""A final reply cut by ``max_tokens`` ends an agent run with ``truncated``.

Plan 0068b055, step 12.28, D-025. Core records a provider cut-off of the turn in
one internal context key (``definitions.CONTEXT_KEY_CUT_OFF``); the agents' one
run loop (``BaseAgent._run_conversation_loop``) reads it for the LAST turn and
records ``StopReason.TRUNCATED``, so every pattern that runs on that loop reports
``success=False, stop_reason="truncated"`` with the (cut) answer still shipped.
REWOO also stops on a cut ``plan_blueprint`` instead of running an empty plan.
Live evidence: findings/live-runs-2.md Defect 2, live-runs-3.md Defects 1 and 3.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.agents import ReactAgent, REWOOAgent
from fsm_llm.agents.constants import ContextKeys, ErrorMessages, StopReason
from fsm_llm.definitions import (
    CONTEXT_KEY_CUT_OFF,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from tests.conftest import PromptGroundedLLM
from tests.test_fsm_llm_agents.test_advance_driver import (
    _ANSWER,
    _FACTS,
    _TASK,
    _registry,
)
from tests.test_fsm_llm_agents.test_grounded_patterns import (
    _lookup_step,
    _rewoo_registry,
)

# What core ships for a reply cut inside a JSON envelope (llm.py, D-024).
_CUT_OFF_MESSAGE = (
    "I'm sorry, my answer was cut off before it was finished. "
    "Please try again or ask for a shorter answer."
)


class _CutReplyLLM(PromptGroundedLLM):
    """Grounded fake whose reply in ``cut_states`` comes back cut off."""

    def __init__(self, *args: Any, cut_states: tuple[str, ...] = (), **kw: Any):
        super().__init__(*args, **kw)
        self.cut_states = cut_states

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        out = super().generate_response(request)
        state = next(
            (
                s
                for s in self.cut_states
                if f"<current_state>{s}<" in request.system_prompt
            ),
            None,
        )
        if state is None:
            return out
        return out.model_copy(update={"message": _CUT_OFF_MESSAGE, "truncated": True})


class _CutFirstToolNameLLM(PromptGroundedLLM):
    """The first ``tool_name`` extraction comes back cut off; later ones are whole."""

    cut_calls = 0

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        if request.field_name == ContextKeys.TOOL_NAME and not self.cut_calls:
            self.cut_calls += 1
            self.requests.append(("extract_field", request))
            return FieldExtractionResponse(
                field_name=request.field_name,
                value=None,
                confidence=0.0,
                is_valid=False,
                truncated=True,
            )
        return super().extract_field(request)


def _react(llm: PromptGroundedLLM, runs: list[str]) -> ReactAgent:
    return ReactAgent(tools=_registry(runs), llm_interface=llm)


class TestReactFinalReplyCut:
    def test_a_normal_run_is_unchanged(self):
        runs: list[str] = []
        llm = _CutReplyLLM(facts=_FACTS, default_response=_ANSWER)

        result = _react(llm, runs).run(_TASK)

        assert runs == ["capital of France"]
        assert (result.success, result.stop_reason) == (True, StopReason.ANSWERED)
        assert result.answer == _ANSWER

    def test_a_cut_final_reply_is_success_false_truncated(self):
        runs: list[str] = []
        llm = _CutReplyLLM(
            facts=_FACTS, default_response=_ANSWER, cut_states=("conclude",)
        )

        result = _react(llm, runs).run(_TASK)

        assert runs == [
            "capital of France"
        ]  # the tool ran: it is the answer that is cut
        assert result.success is False
        assert result.stop_reason == StopReason.TRUNCATED == "truncated"
        assert result.answer == _CUT_OFF_MESSAGE  # the text still ships

    def test_an_earlier_cut_that_a_later_turn_recovers_is_not_flagged(self):
        """Only the last turn counts: the record is cleared at every turn."""
        runs: list[str] = []
        llm = _CutFirstToolNameLLM(facts=_FACTS, default_response=_ANSWER)

        result = _react(llm, runs).run(_TASK)

        assert llm.cut_calls == 1  # the first think turn lost its tool_name to a cut
        assert runs == ["capital of France"]
        assert (result.success, result.stop_reason) == (True, StopReason.ANSWERED)

    def test_the_key_is_not_in_the_result_context(self):
        llm = _CutReplyLLM(
            facts=_FACTS, default_response=_ANSWER, cut_states=("conclude",)
        )

        result = _react(llm, []).run(_TASK)

        assert CONTEXT_KEY_CUT_OFF not in result.final_context

    def test_a_forced_stop_keeps_its_own_reason(self):
        """max_iterations forced the stop: the cut does not rename it."""
        runs: list[str] = []
        lookup_forever = {
            "tool_name": ("lookup", "capital"),
            "tool_input": ({"query": "capital of France"}, "capital"),
        }
        llm = _CutReplyLLM(
            facts=lookup_forever, default_response=_ANSWER, cut_states=("conclude",)
        )
        from fsm_llm.agents import AgentConfig

        agent = ReactAgent(
            tools=_registry(runs),
            config=AgentConfig(max_iterations=3),
            llm_interface=llm,
        )

        result = agent.run(_TASK)

        assert result.success is False
        assert result.stop_reason == StopReason.MAX_ITERATIONS

    def test_truncated_is_a_forced_reason_with_a_warning(self):
        assert StopReason.TRUNCATED in StopReason.FORCED


class _CutPlanLLM(PromptGroundedLLM):
    """The ``plan_blueprint`` extraction comes back cut off (no usable value)."""

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        if request.field_name != ContextKeys.PLAN_BLUEPRINT:
            return super().extract_field(request)
        self.requests.append(("extract_field", request))
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None,
            confidence=0.0,
            reasoning="cut",
            is_valid=False,
            validation_error="Extraction produced no usable value",
            truncated=True,
        )


_REWOO_TASK = "What is the capital of France?"


class TestRewooCutPlan:
    def test_a_whole_plan_runs_as_before(self):
        calls: list[str] = []
        llm = PromptGroundedLLM(
            facts={
                "plan_blueprint": ([_lookup_step(1, "capital")], "capital of France")
            }
        )
        agent = REWOOAgent(tools=_rewoo_registry(calls), llm_interface=llm)

        result = agent.run(_REWOO_TASK)

        assert calls == ["capital"]
        assert (result.success, result.stop_reason) == (True, StopReason.EVIDENCE)
        assert ContextKeys.PLAN_CUT_OFF not in result.final_context

    def test_a_cut_plan_runs_no_tool_and_says_why(self):
        calls: list[str] = []
        llm = _CutPlanLLM(default_response="I have planned the tool executions.")
        agent = REWOOAgent(tools=_rewoo_registry(calls), llm_interface=llm)

        result = agent.run(_REWOO_TASK)

        assert calls == []
        assert result.success is False
        assert result.stop_reason == StopReason.TRUNCATED
        assert result.answer == ErrorMessages.REWOO_PLAN_CUT_OFF
        assert "max_tokens" in result.answer
        assert result.final_context[ContextKeys.PLAN_CUT_OFF] is True

    def test_a_plan_that_is_just_missing_is_still_the_old_no_result(self):
        """RED guard: no cut, no plan: the old empty-plan outcome is unchanged."""
        calls: list[str] = []
        llm = PromptGroundedLLM(default_response="I have planned the tool executions.")
        agent = REWOOAgent(tools=_rewoo_registry(calls), llm_interface=llm)

        result = agent.run(_REWOO_TASK)

        assert calls == []
        assert (result.success, result.stop_reason) == (False, StopReason.NO_RESULT)
        assert result.answer == "I have planned the tool executions."

    def test_a_caller_cannot_seed_the_marker(self):
        calls: list[str] = []
        llm = PromptGroundedLLM(
            facts={
                "plan_blueprint": ([_lookup_step(1, "capital")], "capital of France")
            }
        )
        agent = REWOOAgent(tools=_rewoo_registry(calls), llm_interface=llm)

        result = agent.run(
            _REWOO_TASK, initial_context={ContextKeys.PLAN_CUT_OFF: True}
        )

        assert calls == ["capital"]
        assert (result.success, result.stop_reason) == (True, StopReason.EVIDENCE)
