"""Review round 2, agents fixes (plan 07ad3f8c step 22.3, D-048).

``findings/review-iter-1-pass4.md``:

- concern 1 (D-049): the ``refused_actions`` record was false in the reverse
  order: a call approved and run, then re-asked identically and denied, was
  recorded as "was not performed", and the conclude prompt told the model to
  deny an action that ran. A denial of a call that already ran in this run
  adds no record; a denial of a call that did not run still does, in every
  order.
- concern 6 (D-050): a run whose conversation was ended from outside (a hook,
  another thread, a monitor) failed in its own cleanup with ``AgentError
  "... Conversation ... not found"``. It now returns a result, reported as a
  forced stop: ``(False, "ended")``.

The HITL tests drive the real ``API`` through ``_HitlProbe`` (typed gated
``transfer``, ungated ``check_balance``) from ``test_advance_driver.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from fsm_llm import API
from fsm_llm.agents import AgentConfig, ReactAgent, ReflexionAgent
from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.constants import StopReason
from fsm_llm.agents.definitions import EvaluationResult
from fsm_llm.definitions import ConversationBusyError
from tests.conftest import PromptGroundedLLM
from tests.test_fsm_llm_agents.test_advance_driver import (
    _CALL,
    _HITL_TASK,
    _OTHER_CALL,
    _TASK,
    _conclude_prompt,
    _HitlLLM,
    _HitlProbe,
    _reasoning_react,
    _registry,
)

_REFUSED_KEY = "refused_actions"
_REFUSED_TEXT = "was refused by the human approver"
# Literal on purpose: the record of the gated _OTHER_CALL.
_OTHER_RECORD = (
    "transfer({'account': 'X-99', 'amount': 9999}): was refused by the human "
    "approver and was not performed."
)


def _reflexion_fails_once(**kwargs: Any) -> ReflexionAgent:
    """A Reflexion agent whose first evaluation fails, so the loop returns
    to ``think`` after the first tool ran (where the model re-asks)."""
    verdicts = iter([False])

    def evaluate(context: dict[str, Any]) -> EvaluationResult:
        passed = next(verdicts, True)
        return EvaluationResult(passed=passed, score=float(passed), feedback="ok")

    return ReflexionAgent(evaluation_fn=evaluate, **kwargs)


_BUILDERS = [
    pytest.param(ReactAgent, id="react"),
    pytest.param(_reflexion_fails_once, id="reflexion"),
    pytest.param(_reasoning_react, id="reasoning_react"),
]


def _approver(*answers: bool) -> tuple[list[dict[str, Any]], Any]:
    """An approver giving ``answers`` in order (then denying); records asks."""
    asks: list[dict[str, Any]] = []

    def decide(request: Any) -> bool:
        asks.append(dict(request.parameters))
        return answers[len(asks) - 1] if len(asks) <= len(answers) else False

    return asks, decide


class _RepeatLLM(_HitlLLM):
    """Re-selects ``transfer(_CALL)`` even after it ran (the live over-call
    pattern, 3-4 asks per run); concludes once a denial follows a transfer
    that ran."""

    def _grounded(self, name: str, text: str) -> object | None:
        done = "Result: Transferred" in text and "denied the call transfer" in text
        if name == "tool_name":
            return None if done else "transfer"
        if name == "tool_input":
            return None if done else dict(_CALL)
        if name == "should_terminate":
            return True if done else None
        return None


class _ThenOtherLLM(_HitlLLM):
    """``transfer(_CALL)``; once it ran, ``transfer(_OTHER_CALL)``; concludes
    on that call's denial."""

    def _grounded(self, name: str, text: str) -> object | None:
        done = "denied the call transfer({'account': 'X-99'" in text
        call = _OTHER_CALL if "Result: Transferred" in text else _CALL
        if name == "tool_name":
            return None if done else "transfer"
        if name == "tool_input":
            return None if done else dict(call)
        if name == "should_terminate":
            return True if done else None
        return None


class TestRefusalRecordInEveryOrder:
    """D-049. RED on the parent 086e0aa: approve then deny the identical
    re-ask left ``transfer(...): was refused by the human approver and was
    not performed.`` in ``refused_actions`` and in the conclude prompt,
    although the transfer ran (``review_r2/probe_ad.py``)."""

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_approve_then_deny_of_the_same_call_leaves_no_record(
        self, monkeypatch: pytest.MonkeyPatch, build: Any
    ):
        asks, decide = _approver(True, False)
        probe = _HitlProbe(
            monkeypatch, build, decide=decide, llm=_RepeatLLM(default_response="Done.")
        )

        result = probe.agent.run(_HITL_TASK)

        assert asks == [_CALL, _CALL]
        assert probe.kinds("transfer") == [("transfer", "A-17", 250)]
        (final,) = probe.final
        assert _REFUSED_KEY not in final
        assert _REFUSED_KEY not in result.final_context
        prompt = _conclude_prompt(probe.llm)
        assert _REFUSED_TEXT not in prompt
        assert "Transferred 250 from A-17" in prompt
        assert (result.success, result.stop_reason) == (True, "answered")

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_deny_approve_deny_of_the_same_call_leaves_no_record(
        self, monkeypatch: pytest.MonkeyPatch, build: Any
    ):
        asks, decide = _approver(False, True, False)
        probe = _HitlProbe(
            monkeypatch, build, decide=decide, llm=_RepeatLLM(default_response="Done.")
        )

        probe.agent.run(_HITL_TASK)

        assert asks == [_CALL, _CALL, _CALL]
        assert probe.kinds("transfer") == [("transfer", "A-17", 250)]
        (final,) = probe.final
        assert _REFUSED_KEY not in final
        assert _REFUSED_TEXT not in _conclude_prompt(probe.llm)

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_a_different_call_denied_after_one_ran_is_still_recorded(
        self, monkeypatch: pytest.MonkeyPatch, build: Any
    ):
        asks, decide = _approver(True, False)
        probe = _HitlProbe(
            monkeypatch,
            build,
            decide=decide,
            llm=_ThenOtherLLM(default_response="Done."),
        )

        probe.agent.run(_HITL_TASK)

        assert asks == [_CALL, _OTHER_CALL]
        assert probe.kinds("transfer") == [("transfer", "A-17", 250)]
        (final,) = probe.final
        assert final[_REFUSED_KEY] == [_OTHER_RECORD]
        assert _OTHER_RECORD in _conclude_prompt(probe.llm)


# ---------------------------------------------------------------------------
# A conversation ended from outside the run
# ---------------------------------------------------------------------------

# Selects the lookup tool forever: only the iteration budget or an outside
# end stops the run.
_LOOKUP_FOREVER: dict[str, tuple[object, str]] = {
    "tool_name": ("lookup", "capital"),
    "tool_input": ({"query": "capital of France"}, "capital"),
}


class _EndsItsConversation(ReactAgent):
    """Ends its own conversation from the hook before step ``end_at`` (as a
    monitor "end conversation" click or another thread would)."""

    end_at = 3

    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        super()._on_loop_iteration(api, conv_id, iteration)
        if iteration == self.end_at:
            api.end_conversation(conv_id)


def _ending_agent(runs: list[str]) -> _EndsItsConversation:
    return _EndsItsConversation(
        tools=_registry(runs),
        config=AgentConfig(max_iterations=6),
        llm_interface=PromptGroundedLLM(facts=_LOOKUP_FOREVER),
    )


class TestRunEndedFromOutside:
    """D-050. RED on the parent 086e0aa: ``run()`` raised ``AgentError
    "React execution failed: Conversation ... not found"`` from its own
    ``end_conversation`` in the ``finally`` (``review_r2/probe_end.py``), and
    without that error the run reported ``(True, "answered")`` on the one
    tool call made before the end."""

    def test_run_returns_a_forced_stop(self):
        runs: list[str] = []

        result = _ending_agent(runs).run(_TASK)

        assert runs == ["capital of France"]  # step 2 ran the tool; step 3 never ran
        assert (result.success, result.stop_reason) == (False, StopReason.ENDED)
        assert StopReason.ENDED in StopReason.FORCED

    def test_stream_ends_without_an_error(self):
        runs: list[str] = []

        chunks = list(_ending_agent(runs).run_stream(_TASK))

        assert runs == ["capital of France"]
        assert all("[" not in chunk for chunk in chunks)

    def test_a_run_that_concludes_is_not_reported_as_ended(self):
        runs: list[str] = []
        agent = _ending_agent(runs)
        agent.end_at = 99  # never reached: the budget forces the stop

        result = agent.run(_TASK)

        assert result.stop_reason == StopReason.MAX_ITERATIONS


class TestEndRunConversation:
    """The cleanup tolerates only a conversation that is already gone."""

    def test_an_end_refused_while_the_conversation_is_active_is_raised(self):
        def refuse(conv_id: str) -> None:
            raise ConversationBusyError("a turn is running")

        api = SimpleNamespace(
            end_conversation=refuse, list_active_conversations=lambda: ["c-1"]
        )

        with pytest.raises(ConversationBusyError):
            BaseAgent._end_run_conversation(cast(API, api), "c-1")

    def test_an_error_that_is_not_an_fsm_error_is_raised(self):
        def broken(conv_id: str) -> None:
            raise RuntimeError("boom")

        api = SimpleNamespace(
            end_conversation=broken, list_active_conversations=lambda: []
        )

        with pytest.raises(RuntimeError):
            BaseAgent._end_run_conversation(cast(API, api), "c-1")
