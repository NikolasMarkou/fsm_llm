"""Agent patterns and shared scaffolding (plan 5980fbb7 step 11).

Each class pins one finding: F-AP-01 (the model's answer outranks context
hints in structured output), F-AP-02 (ADaPT fills ``structured_output`` and
registers the lifecycle handlers), F-AP-03/F-XD-01 (the LLM judge reads a
malformed score as 0), F-AP-04 (``aggregation_fn`` failures are agent errors),
F-AP-05 (one empty-registry guard), F-XD-05 (the base hook asks for HITL
approval), F-AP-10 (no END_CONVERSATION marker), F-AP-11 (Debate needs no
``_extract_answer`` override), F-AP-15 (the summariser never re-wraps its own
summary and an unreachable threshold is refused).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from fsm_llm.agents import (
    ADaPTAgent,
    AgentConfig,
    PromptChainAgent,
    ReactAgent,
    SelfConsistencyAgent,
)
from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.composition import default_llm_judge
from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.agents.definitions import ChainStep
from fsm_llm.agents.exceptions import AgentError
from fsm_llm.agents.summarization import make_observation_summarizer
from fsm_llm.agents.tools import ToolRegistry
from tests.conftest import PromptGroundedLLM


class _Out(BaseModel):
    title: str = "unset"
    answer: str = ""


class TestStructuredOutputPrefersTheModelsAnswer:
    """F-AP-01: a context key named like a schema field never replaces the answer."""

    def _agent(self) -> PromptChainAgent:
        return PromptChainAgent(
            chain=[
                ChainStep(
                    step_id="s",
                    name="s",
                    extraction_instructions="x",
                    response_instructions="y",
                )
            ],
            config=AgentConfig(output_schema=_Out),
            llm_interface=PromptGroundedLLM(),
        )

    def test_the_answer_json_outranks_a_caller_hint(self):
        answer = json.dumps({"title": "Model title", "answer": "x"})
        out = self._agent()._try_parse_structured_output(
            answer, {"title": "CALLER HINT"}
        )
        assert out == _Out(title="Model title", answer="x")

    def test_context_keys_still_fill_in_when_the_answer_is_prose(self):
        out = self._agent()._try_parse_structured_output(
            "prose, no json", {"title": "T", "answer": "A"}
        )
        assert out == _Out(title="T", answer="A")


class TestADaPTStructuredOutputAndLifecycle:
    """F-AP-02: ADaPT honours ``output_schema`` and registers lifecycle handlers."""

    def _run(self, monkeypatch, **extra):
        calls: list[str] = []
        original = BaseAgent._register_lifecycle_handlers

        def spy(self, api, agent_type):
            calls.append(agent_type)
            return original(self, api, agent_type)

        monkeypatch.setattr(BaseAgent, "_register_lifecycle_handlers", spy)
        llm = PromptGroundedLLM(
            facts={
                "attempt_result": ('{"answer": "x"}', "Plan the trip"),
                "attempt_succeeded": (True, '{"answer": "x"}'),
            }
        )
        agent = ADaPTAgent(
            config=AgentConfig(max_iterations=10, output_schema=_Out),
            max_depth=1,
            llm_interface=llm,
            **extra,
        )
        return agent.run("Plan the trip"), calls

    def test_structured_output_is_filled(self, monkeypatch):
        result, _ = self._run(monkeypatch)
        assert result.answer == '{"answer": "x"}'
        assert result.structured_output == _Out(answer="x")

    def test_lifecycle_handlers_are_registered(self, monkeypatch):
        _, calls = self._run(monkeypatch)
        assert calls == ["adapt"]


class TestJudgeReadsAMalformedScoreAsZero:
    """F-AP-03 / F-XD-01: NaN, inf and a boolean never pass the judge."""

    @pytest.mark.parametrize(
        "reply",
        [
            '{"score": NaN, "feedback": "f"}',
            '{"score": "nan", "feedback": "f"}',
            '{"score": "inf", "feedback": "f"}',
            '{"score": Infinity, "feedback": "f"}',
            '{"score": true, "feedback": "f"}',
        ],
    )
    def test_malformed_score_fails(self, reply):
        judge = default_llm_judge(model="m", complete_fn=lambda _m, _p: reply)
        result = judge("out", {"task": "t"})
        assert (result.passed, result.score) == (False, 0.0)

    @pytest.mark.parametrize(
        ("reply", "passed", "score"),
        [
            ('{"score": 0.9}', True, 0.9),
            ('{"score": 5}', True, 1.0),
            ('{"score": -5}', False, 0.0),
            ('{"score": "0.8"}', True, 0.8),
            ('{"feedback": "no score"}', False, 0.0),
        ],
    )
    def test_ordinary_scores_are_unchanged(self, reply, passed, score):
        judge = default_llm_judge(model="m", complete_fn=lambda _m, _p: reply)
        result = judge("out", None)
        assert (result.passed, result.score) == (passed, score)


class TestAggregationFailureIsAnAgentError:
    """F-AP-04: a failing or non-str ``aggregation_fn`` ends as ``AgentError``."""

    def _agent(self, fn) -> SelfConsistencyAgent:
        return SelfConsistencyAgent(
            num_samples=2,
            aggregation_fn=fn,
            config=AgentConfig(max_iterations=5),
            llm_interface=PromptGroundedLLM(default_response="Paris"),
        )

    def test_a_raising_function(self):
        with pytest.raises(AgentError) as info:
            self._agent(lambda _s: 1 / 0).run("Capital of France?")
        assert isinstance(info.value.__cause__, ZeroDivisionError)

    def test_a_non_str_result(self):
        with pytest.raises(AgentError):
            self._agent(lambda _s: 4).run("Capital of France?")

    def test_none_is_no_result(self):
        result = self._agent(lambda _s: None).run("Capital of France?")
        assert (result.answer, result.success, result.stop_reason) == (
            "",
            False,
            "no_result",
        )


class TestOneEmptyRegistryGuard:
    """F-AP-05: the six tool patterns refuse an empty registry the same way."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "react",
            "reflexion",
            "rewoo",
            "parallel_react",
            "native_fc",
        ],
    )
    def test_create_agent_refuses_an_empty_registry(self, pattern):
        from fsm_llm.agents import create_agent

        with pytest.raises(AgentError, match="empty tool registry"):
            create_agent(pattern, ToolRegistry())


class TestBaseHookAsksForHitlApproval:
    """F-XD-05: the base ``_on_loop_iteration`` is the HITL driver."""

    def test_the_hook_runs_the_driver(self):
        agent = SelfConsistencyAgent(llm_interface=PromptGroundedLLM())
        agent._handle_hitl_approval = MagicMock()  # type: ignore[method-assign]
        agent._on_loop_iteration("api", "conv", 1)  # type: ignore[arg-type]
        agent._handle_hitl_approval.assert_called_once_with("api", "conv")

    def test_the_three_hitl_patterns_inherit_it(self):
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent
        from fsm_llm.agents.reflexion import ReflexionAgent

        for cls in (ReactAgent, ReflexionAgent, ReasoningReactAgent):
            assert "_on_loop_iteration" not in vars(cls)


class TestNoEndConversationMarker:
    """F-AP-10: the lifecycle handlers write no key nothing reads."""

    def test_no_end_conversation_handler(self):
        agent = SelfConsistencyAgent(llm_interface=PromptGroundedLLM())
        api = MagicMock()
        agent._register_lifecycle_handlers(api, "x")
        names = [c.args[0] for c in api.create_handler.call_args_list if c.args]
        assert "AgentEndConversation" not in names
        assert "AgentContextCompactor" in names


class TestDebateUsesTheBaseAnswerLookup:
    """F-AP-11: the success key is separate from the answer keys."""

    def test_debate_has_no_extract_answer_override(self):
        from fsm_llm.agents.debate import DebateAgent

        assert "_extract_answer" not in vars(DebateAgent)

    def test_the_conclude_reply_is_the_answer(self):
        from fsm_llm.agents.debate import DebateAgent

        agent = DebateAgent(llm_interface=PromptGroundedLLM())
        ctx = {ContextKeys.PROPOSITION: "a proposition"}
        assert agent._extract_answer(ctx, ["the verdict"]) == "the verdict"


class TestSummariserDoesNotRewrapItsSummary:
    """F-AP-15: output within the threshold is left alone."""

    @pytest.mark.parametrize("threshold", [1, 2, 3, 4])
    def test_a_second_pass_is_a_no_op(self, threshold):
        summarise = make_observation_summarizer(threshold)
        observations = [f"step{i}" for i in range(threshold + 3)]
        first = summarise({ContextKeys.OBSERVATIONS: observations})
        assert len(first[ContextKeys.OBSERVATIONS]) <= threshold
        assert (
            summarise({ContextKeys.OBSERVATIONS: first[ContextKeys.OBSERVATIONS]}) == {}
        )

    def test_keep_last_must_be_below_the_threshold(self):
        with pytest.raises(ValueError, match="keep_last"):
            make_observation_summarizer(3, keep_last=3)

    def test_a_threshold_the_list_can_never_reach_is_refused(self):
        AgentConfig(auto_summarize_after=Defaults.MAX_OBSERVATIONS - 1)
        with pytest.raises(ValueError, match="auto_summarize_after"):
            AgentConfig(auto_summarize_after=Defaults.MAX_OBSERVATIONS)
