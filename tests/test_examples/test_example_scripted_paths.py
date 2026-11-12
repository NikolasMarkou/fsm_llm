"""Drive shipped example FSMs offline with the public `fsm_llm.testing` module.

This is the consumer test of row 0.9: it uses only names a user project can
import (`ScriptedUser`, `ScriptedLLM`, `PromptGroundedLLM`, `PathRecorder` and
the four assertions), reads the example FSM files as they are (no example is
edited, no example `run.py` is executed) and reaches no model, no network and
no clock. A conversation here is a path through a real example definition, so
a change to an example's transitions fails with the first state that differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from fsm_llm import API
from fsm_llm.testing import (
    PathRecorder,
    PromptGroundedLLM,
    ScriptedLLM,
    ScriptedUser,
    assert_context,
    assert_path,
    assert_result,
    assert_state,
    final_reply,
    tool_call_reply,
)

pytestmark = pytest.mark.examples

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _api(relative: str, llm, **kwargs) -> API:
    return API.from_file(str(_EXAMPLES / relative), llm_interface=llm, **kwargs)


class TestSimpleGreeting:
    """`basic/simple_greeting`: a scripted user, a scripted model, every assertion."""

    _REPLIES: ClassVar[dict] = {
        "greeting": "Hi! How are you?",
        "conversation": "Glad to hear it.",
        "farewell": "Goodbye!",
    }

    def _run(self):
        llm = ScriptedLLM(
            extraction_data={"mood": "happy", "intent": "chat"},
            responses=self._REPLIES,
        )
        api = _api("basic/simple_greeting/fsm.json", llm)
        return llm, ScriptedUser(["hello", "doing great"]).run(api)

    def test_the_conversation_walks_the_whole_example(self):
        llm, run = self._run()
        assert_path(run, ["greeting", "conversation", "farewell"])
        assert_state(run, "farewell")
        assert run.replies == ("Glad to hear it.", "Goodbye!")
        assert run.ended is True
        llm.assert_exhausted()

    def test_the_extracted_value_is_in_the_context(self):
        _, run = self._run()
        assert_context(run, mood="happy")

    def test_a_subsequence_path_ignores_the_states_in_between(self):
        _, run = self._run()
        assert_path(run, ["greeting", "farewell"], exact=False)

    def test_a_wrong_path_names_the_first_difference(self):
        _, run = self._run()
        with pytest.raises(AssertionError, match="index 1"):
            assert_path(run, ["greeting", "farewell"])

    def test_a_wrong_state_fails_with_both_names(self):
        _, run = self._run()
        with pytest.raises(AssertionError, match="greeting"):
            assert_state(run, "greeting")


class TestGroundedExtraction:
    """`basic/multi_turn_extraction` with `PromptGroundedLLM`.

    A fact comes back only when its evidence is in the prompt, so the path
    proves the data reached the model, not that a fixed dict was replayed.
    """

    _FACTS: ClassVar[dict] = {
        "full_name": ("Ada Lovelace", "Ada Lovelace"),
        "age": (36, "36"),
        "email": ("ada@example.org", "ada@example.org"),
        "phone": ("555-0100", "555-0100"),
        "confirmation": (True, "yes"),
    }
    _REPLIES: ClassVar[dict] = {
        "collect_personal": "Your name and age?",
        "collect_contact": "Your email and phone?",
        "confirm": "Is that correct?",
        "complete": "All done.",
    }

    def _api(self):
        llm = PromptGroundedLLM(facts=self._FACTS, responses=self._REPLIES)
        return llm, _api("basic/multi_turn_extraction/fsm.json", llm)

    def test_grounded_messages_reach_the_terminal_state(self):
        llm, api = self._api()
        run = ScriptedUser(
            [
                "I am Ada Lovelace, 36 years old",
                "ada@example.org and 555-0100",
                "yes",
            ]
        ).run(api)
        assert_path(run, ["collect_personal", "collect_contact", "confirm", "complete"])
        assert_context(run, full_name="Ada Lovelace", age=36)
        assert run.ended is True
        assert llm.calls("generate_response")

    def test_a_message_without_the_evidence_stays_in_the_state(self):
        _, api = self._api()
        run = ScriptedUser(["hello there", "still nothing"]).run(api)
        assert_path(run, ["collect_personal"] * 3)
        assert_state(run, "collect_personal")
        assert run.ended is False


class TestRulesOnAStory:
    """`basic/story_time`: rules answer the model's last reply."""

    _DATA: ClassVar[dict] = {
        "user_opinion_straw": "weak",
        "user_opinion_sticks": "so so",
        "user_prediction": "the brick house holds",
        "user_suggestion": "use the chimney",
        "user_feedback": "lovely",
    }

    def test_a_rule_keeps_the_user_talking_until_the_end(self):
        llm = ScriptedLLM(
            extraction_data=self._DATA, default_response="The story goes on."
        )
        api = _api("basic/story_time/fsm.json", llm)
        user = ScriptedUser(
            ["Tell me the story"], rules=[("story", "Go on.")], max_turns=20
        )
        run = user.run(api)
        assert_path(
            run,
            ["introduction", "first_pig_house", "wolf_attacks_straw"],
            exact=False,
        )
        assert_state(run, "end")
        assert run.ended is True
        assert len(run.messages) < 20

    def test_the_turn_cap_stops_a_user_that_never_runs_out(self):
        llm = ScriptedLLM(default_response="The story goes on.")  # extracts nothing
        api = _api("basic/story_time/fsm.json", llm)
        run = ScriptedUser(["Tell me"], fallback="more", max_turns=3).run(api)
        assert len(run.messages) == 3
        assert run.ended is False
        assert_state(run, run.final_state)


class TestClassifiedRouting:
    """`classification/classified_transitions`: a tie goes to the classifier."""

    def _llm(self) -> ScriptedLLM:
        verdict = {"reasoning": "an invoice", "intent": "billing", "confidence": 0.9}
        return ScriptedLLM(
            [final_reply(json.dumps(verdict))],
            extraction_data={"billing_issue_type": "refund"},
            default_response="Understood.",
        )

    def test_the_scripted_verdict_picks_the_branch(self):
        llm = self._llm()
        api = _api("classification/classified_transitions/fsm.json", llm)
        run = ScriptedUser(["I was charged twice", "a refund please"]).run(api)
        assert_path(run, ["welcome", "billing", "resolved"])
        assert_context(run, billing_issue_type="refund")
        llm.assert_exhausted()

    def test_a_path_recorder_sees_the_same_route(self):
        llm = self._llm()
        recorder = PathRecorder()
        api = _api(
            "classification/classified_transitions/fsm.json",
            llm,
            handlers=[recorder],
        )
        run = ScriptedUser(["I was charged twice", "a refund please"]).run(api)
        assert_path(
            recorder.path(run.conversation_id), ["welcome", "billing", "resolved"]
        )
        assert_path(run, recorder.path(run.conversation_id))

    def test_an_unused_verdict_fails_the_exhaustion_check(self):
        llm = self._llm()
        api = _api("classification/classified_transitions/fsm.json", llm)
        # a user who says nothing never reaches the tie, so the scripted
        # verdict stays in the queue and the exhaustion check says so
        ScriptedUser([]).run(api)
        with pytest.raises(AssertionError):
            llm.assert_exhausted()


class TestCompletionResult:
    """`assert_result` reads a completion result: no example FSM has one, so the
    scripted `complete` reply is checked directly."""

    def test_a_scripted_reply_has_the_expected_kind(self):
        from fsm_llm.llm import CompletionRequest

        llm = ScriptedLLM([tool_call_reply(("lookup", {"q": "rome"})), "It is sunny."])
        request = CompletionRequest(
            messages=[{"role": "user", "content": "weather?"}], call_type="completion"
        )
        assert_result(llm.complete(request), kind="calls")
        assert_result(llm.complete(request), kind="final")
        llm.assert_exhausted()

    def test_a_typo_in_the_expected_kind_raises_value_error(self):
        with pytest.raises(ValueError, match="final"):
            assert_result({"kind": "final"}, kind="finale")
