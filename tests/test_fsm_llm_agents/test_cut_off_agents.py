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
    _CHAIN_TASK,
    _HAIKU,
    _SILENT_ROOT_ATTEMPT,
    _SILENT_TASK,
    _chain_derived,
    _lookup_step,
    _passing_evaluation,
    _rewoo_registry,
    _stage,
    _TurnAwareLLM,
)

# What core ships for a reply cut inside a JSON envelope (llm.py, D-024).
_CUT_OFF_MESSAGE = (
    "I'm sorry, my answer was cut off before it was finished. "
    "Please try again or ask for a shorter answer."
)


def _cut_in_states(
    out: ResponseGenerationResponse,
    request: ResponseGenerationRequest,
    cut_states: tuple[str, ...],
) -> ResponseGenerationResponse:
    """``out`` as a cut-off reply when the request is for one of ``cut_states``."""
    if not any(f"<current_state>{s}<" in request.system_prompt for s in cut_states):
        return out
    return out.model_copy(update={"message": _CUT_OFF_MESSAGE, "truncated": True})


class _CutReplyLLM(PromptGroundedLLM):
    """Grounded fake whose reply in ``cut_states`` comes back cut off."""

    def __init__(self, *args: Any, cut_states: tuple[str, ...] = (), **kw: Any):
        super().__init__(*args, **kw)
        self.cut_states = cut_states

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return _cut_in_states(
            super().generate_response(request), request, self.cut_states
        )


class _CutTurnAwareLLM(_TurnAwareLLM):
    """``_TurnAwareLLM`` whose reply in ``cut_states`` comes back cut off."""

    def __init__(self, derived: dict, *, cut_states: tuple[str, ...], **kw: Any):
        super().__init__(derived, **kw)
        self.cut_states = cut_states

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return _cut_in_states(
            super().generate_response(request), request, self.cut_states
        )


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


class TestTheAnswerSourceDecidesTruncation:
    """Step 12.33 (D-027): a cut reply that is not the answer is not a truncated
    run. EvalOpt, MakerChecker, ADaPT and PromptChain read their answer from a
    context key; only the unused closing reply was cut (review pass 3, WARNING 1)."""

    def _evalopt(self, llm: PromptGroundedLLM):
        from fsm_llm.agents import AgentConfig, EvaluatorOptimizerAgent

        return EvaluatorOptimizerAgent(
            evaluation_fn=_passing_evaluation,
            max_refinements=1,
            config=AgentConfig(max_iterations=10),
            llm_interface=llm,
        )

    def _maker_checker(self, llm: PromptGroundedLLM):
        from fsm_llm.agents import AgentConfig, MakerCheckerAgent

        return MakerCheckerAgent(
            maker_instructions="Write a haiku.",
            checker_instructions="Review the haiku.",
            max_revisions=1,
            config=AgentConfig(max_iterations=10),
            llm_interface=llm,
        )

    def test_evalopt_whole_answer_with_a_cut_closing_reply_is_answered(self):
        llm = _CutReplyLLM(
            facts={"generated_output": (_HAIKU, "haiku")}, cut_states=("output",)
        )

        result = self._evalopt(llm).run("Write a haiku about rain")

        assert llm.calls("generate_response")  # the closing reply was asked for
        assert result.answer == _HAIKU
        assert (result.success, result.stop_reason) == (True, StopReason.ANSWERED)

    def test_maker_checker_whole_answer_with_a_cut_closing_reply_is_answered(self):
        llm = _CutReplyLLM(
            facts={
                "draft_output": (_HAIKU, "haiku"),
                "quality_score": (0.95, "haiku"),
            },
            cut_states=("output",),
        )

        result = self._maker_checker(llm).run("Write a haiku about rain")

        assert result.answer == _HAIKU
        assert (result.success, result.stop_reason) == (True, StopReason.ANSWERED)

    def test_adapt_whole_attempt_with_a_cut_combine_reply_is_answered(self):
        from fsm_llm.agents import ADaPTAgent, AgentConfig

        llm = _CutReplyLLM(
            facts={
                ContextKeys.ATTEMPT_RESULT: (_SILENT_ROOT_ATTEMPT, _SILENT_TASK),
                ContextKeys.ATTEMPT_SUCCEEDED: (True, _SILENT_ROOT_ATTEMPT),
            },
            cut_states=("combine",),
        )

        result = ADaPTAgent(
            config=AgentConfig(max_iterations=10), max_depth=1, llm_interface=llm
        ).run(_SILENT_TASK)

        assert result.answer == _SILENT_ROOT_ATTEMPT
        assert (result.success, result.stop_reason) == (True, StopReason.ANSWERED)

    def test_adapt_without_an_attempt_answers_from_the_cut_reply_and_flags_it(self):
        """The answer IS the reply here: the cut is the answer's."""
        from fsm_llm.agents import ADaPTAgent, AgentConfig

        llm = _CutReplyLLM(default_response="whole", cut_states=("combine",))

        result = ADaPTAgent(
            config=AgentConfig(max_iterations=10), max_depth=1, llm_interface=llm
        ).run(_SILENT_TASK)

        assert result.answer == _CUT_OFF_MESSAGE
        assert (result.success, result.stop_reason) == (False, StopReason.TRUNCATED)

    def test_prompt_chain_whole_result_with_a_cut_output_reply_is_answered(self):
        from fsm_llm.agents import ChainStep, PromptChainAgent

        chain = [
            ChainStep(
                step_id=f"s{i}",
                name=f"Stage {i}",
                extraction_instructions=f"Extract stage {i} text.",
                response_instructions=f"Present stage {i}.",
            )
            for i in range(2)
        ]
        llm = _CutTurnAwareLLM(
            _chain_derived(),
            cut_states=("output",),
            responses={"output": "Final note."},
        )

        result = PromptChainAgent(chain=chain, llm_interface=llm).run(_CHAIN_TASK)

        assert result.answer == _stage(1)
        assert (result.success, result.stop_reason) == (True, StopReason.ANSWERED)

    def test_react_cut_final_reply_is_still_truncated(self):
        """RED guard: the answer IS the reply, so the cut still counts."""
        llm = _CutReplyLLM(
            facts=_FACTS, default_response=_ANSWER, cut_states=("conclude",)
        )

        result = _react(llm, []).run(_TASK)

        assert (result.success, result.stop_reason) == (False, StopReason.TRUNCATED)
        assert result.answer == _CUT_OFF_MESSAGE


class TestAnswerWasCutOffSeam:
    """``BaseAgent._answer_was_cut_off`` and the judged last-round path."""

    @staticmethod
    def _agent(**kw: Any):
        from fsm_llm.agents import MakerCheckerAgent

        return MakerCheckerAgent(
            maker_instructions="m",
            checker_instructions="c",
            llm_interface=PromptGroundedLLM(),
            **kw,
        )

    def test_no_record_is_not_cut(self):
        agent = self._agent()
        assert (
            agent._answer_was_cut_off({"draft_output": "x"}, ["draft_output"]) is False
        )

    def test_a_malformed_record_is_not_cut(self):
        agent = self._agent()
        for bad in ("yes", 1, [], {}, None):
            assert agent._answer_was_cut_off({CONTEXT_KEY_CUT_OFF: bad}, None) is False

    def test_reply_answer_counts_a_cut_reply_or_a_cut_field(self):
        agent = self._agent()
        assert agent._answer_was_cut_off({CONTEXT_KEY_CUT_OFF: {"reply": True}}, None)
        assert agent._answer_was_cut_off(
            {CONTEXT_KEY_CUT_OFF: {"fields": ["x"]}}, ["draft_output"]
        )

    def test_key_answer_ignores_a_cut_reply_and_other_fields(self):
        agent = self._agent()
        ctx = {"draft_output": "whole"}
        assert not agent._answer_was_cut_off(
            {**ctx, CONTEXT_KEY_CUT_OFF: {"reply": True}}, ["draft_output"]
        )
        assert not agent._answer_was_cut_off(
            {**ctx, CONTEXT_KEY_CUT_OFF: {"fields": ["quality_score"]}},
            ["draft_output"],
        )

    def test_key_answer_counts_its_own_cut_field(self):
        agent = self._agent()
        assert agent._answer_was_cut_off(
            {"draft_output": "half", CONTEXT_KEY_CUT_OFF: {"fields": ["draft_output"]}},
            ["draft_output"],
        )

    def test_an_empty_answer_key_falls_back_to_the_reply(self):
        agent = self._agent()
        assert agent._answer_was_cut_off(
            {"draft_output": "", CONTEXT_KEY_CUT_OFF: {"reply": True}},
            ["draft_output"],
        )

    def test_a_judged_pattern_on_the_limiter_round_still_reports_a_cut(self):
        """Review pass 3 note 4: the bare limiter flag hid the cut."""
        from fsm_llm.agents.base import BaseAgent
        from fsm_llm.agents.definitions import AgentTrace

        agent = self._agent()
        ctx = {
            ContextKeys.MAX_ITERATIONS_REACHED: True,
            CONTEXT_KEY_CUT_OFF: {"reply": True},
        }
        cut = agent._answer_was_cut_off(ctx, None)
        trace = AgentTrace(tool_calls=[], total_iterations=1)

        assert BaseAgent._run_outcome(
            ctx, trace, None, judged=True, answer_cut_off=cut
        ) == (
            False,
            StopReason.TRUNCATED,
        )

    def test_a_recorded_forced_reason_outranks_a_cut(self):
        from fsm_llm.agents.base import BaseAgent
        from fsm_llm.agents.definitions import AgentTrace

        ctx = {ContextKeys.FORCED_STOP_REASON: StopReason.FORCED_PASS}
        trace = AgentTrace(tool_calls=[], total_iterations=1)

        assert BaseAgent._run_outcome(
            ctx, trace, ["k"], judged=True, answer_cut_off=True
        ) == (False, StopReason.FORCED_PASS)
