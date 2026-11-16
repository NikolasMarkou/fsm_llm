"""Contract of `ScriptedUser`, `ConversationRun` and `advance_until_terminal`.

The conversations run offline on two example FSM files (`simple_greeting` and
`form_filling`, read only, loaded with `API.from_file`) and on small inline
FSMs, with `ScriptedLLM` as the model. `ConversationRun.states` is the path:
the state after the start and after each reply, which the eval harness samples
the same way (an equivalence test below compares the two).
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
import fsm_llm.testing as public
from fsm_llm import API
from fsm_llm.core.definitions import FSMError, RunBudgetExceededError
from fsm_llm.eval import ConversationCase, EvalConfig, run_case_trial
from fsm_llm.eval.cases import Expectations
from fsm_llm.testing import (
    ConversationRun,
    ScriptedLLM,
    ScriptedUser,
    advance_until_terminal,
)
from tests.test_fsm_llm_testing.support import _chain, _state, _stuck

_EXAMPLES = Path(fsm_llm.__file__).resolve().parents[2] / "examples" / "basic"
_GREETER = _EXAMPLES / "simple_greeting" / "fsm.json"
_FORM = _EXAMPLES / "form_filling" / "fsm.json"

_GREETER_REPLIES = {
    "greeting": "Hi! How are you?",
    "conversation": "Glad to hear it.",
    "farewell": "Goodbye!",
}
_FORM_REPLIES = {
    "welcome": "Welcome to the form.",
    "collect_name": "What is your full name?",
    "collect_email": "What is your email address?",
    "collect_age": "How old are you?",
    "collect_occupation": "What is your occupation?",
    "confirm": "Is all of this correct? Please confirm.",
    "thank_you": "Thank you!",
}
_FORM_DATA = {
    "user_name": "Ada Lovelace",
    "user_email": "ada@example.org",
    "user_age": "36",
    "user_occupation": "Engineer",
    "user_confirmation": True,
}
_FORM_PATH = (
    "welcome",
    "collect_name",
    "collect_email",
    "collect_age",
    "collect_occupation",
    "confirm",
    "thank_you",
)


def _greeter() -> tuple[API, ScriptedLLM]:
    llm = ScriptedLLM(
        extraction_data={"mood": "happy", "intent": "chat"},
        responses=_GREETER_REPLIES,
    )
    return API.from_file(str(_GREETER), llm_interface=llm), llm


def _form() -> tuple[API, ScriptedLLM]:
    llm = ScriptedLLM(extraction_data=dict(_FORM_DATA), responses=_FORM_REPLIES)
    return API.from_file(str(_FORM), llm_interface=llm), llm


def _api(definition: dict[str, Any]) -> tuple[API, ScriptedLLM]:
    llm = ScriptedLLM(
        extraction_data={},
        responses={"a": "Still here." if "Stuck" == definition["name"] else "Say A"},
        default_response="Done.",
    )
    return API.from_definition(definition, llm_interface=llm), llm


class TestConversationRunRecord:
    def test_two_messages_on_the_greeter_record_the_whole_run(self) -> None:
        api, llm = _greeter()

        run = ScriptedUser(["hello", "good, thanks"]).run(api)

        assert run.greeting == "Hi! How are you?"
        assert run.messages == ("hello", "good, thanks")
        assert run.replies == ("Glad to hear it.", "Goodbye!")
        assert run.states == ("greeting", "conversation", "farewell")
        assert run.final_state == "farewell"
        assert run.ended is True
        assert run.api is api
        assert api.get_current_state(run.conversation_id) == "farewell"
        assert llm.unexpected == []

    def test_states_is_always_one_longer_than_replies(self) -> None:
        for count in (0, 1, 2):
            api, _ = _greeter()
            run = ScriptedUser(["a", "b"][:count]).run(api)
            assert len(run.states) == len(run.replies) + 1 == count + 1

    def test_the_record_is_frozen_and_made_of_tuples(self) -> None:
        api, _ = _greeter()
        run = ScriptedUser(["hello"]).run(api)

        assert isinstance(run, ConversationRun)
        assert dataclasses.is_dataclass(run)
        with pytest.raises(dataclasses.FrozenInstanceError):
            run.ended = False  # type: ignore[misc]
        for name in ("messages", "replies", "states"):
            assert isinstance(getattr(run, name), tuple)

    def test_the_api_is_left_out_of_repr_and_equality(self) -> None:
        first_api, _ = _greeter()
        second_api, _ = _greeter()
        first = ScriptedUser(["hello"]).run(first_api)
        second = ScriptedUser(["hello"]).run(second_api)

        assert "API" not in repr(first)
        # Different conversation ids, same everything else.
        assert dataclasses.replace(first, conversation_id="x") == dataclasses.replace(
            second, conversation_id="x"
        )

    def test_the_path_equals_what_the_eval_harness_samples(self) -> None:
        case = ConversationCase(
            id="greeter",
            fsm=str(_GREETER),
            turns=["hello", "good, thanks"],
            expect=Expectations(final_state="farewell"),
        )

        trial = run_case_trial(case, EvalConfig(), 0, lambda: _greeter()[1], checks=())
        api, _ = _greeter()
        run = ScriptedUser(list(case.turns)).run(api)

        assert trial.error is None
        assert tuple(trial.visited_states) == run.states
        assert tuple(trial.responses) == (run.greeting, *run.replies)


class TestStopping:
    def test_stops_when_the_conversation_ended_with_messages_left(self) -> None:
        api, llm = _greeter()

        run = ScriptedUser(["hello", "good", "never", "sent"]).run(api)

        assert run.messages == ("hello", "good")
        assert run.ended is True
        assert len(llm.calls("generate_response")) == 3  # start plus two replies

    def test_no_script_at_all_sends_nothing(self) -> None:
        api, _ = _greeter()

        run = ScriptedUser().run(api)

        assert run.messages == ()
        assert run.replies == ()
        assert run.states == ("greeting",)
        assert run.ended is False

    def test_max_turns_bounds_the_messages_sent(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(["1", "2", "3"], max_turns=2).run(api)

        assert run.messages == ("1", "2")

    def test_max_turns_zero_sends_nothing(self) -> None:
        api, _ = _greeter()

        run = ScriptedUser(["hello"], max_turns=0).run(api)

        assert run.messages == ()
        assert run.states == ("greeting",)

    def test_a_fallback_that_repeats_is_bounded_by_max_turns(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(fallback="again", max_turns=4).run(api)

        assert run.messages == ("again",) * 4
        assert run.states == ("a",) * 5
        assert run.ended is False

    def test_the_default_bound_is_twenty_five_turns(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(fallback="again").run(api)

        assert len(run.messages) == 25

    def test_no_fallback_stops_when_no_rule_fires(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(["go"], rules=[("never matches", "x")]).run(api)

        assert run.messages == ("go",)


class TestRules:
    def test_the_form_is_filled_by_rules_reading_each_reply(self) -> None:
        api, llm = _form()
        user = ScriptedUser(
            ["hi"],
            rules=[
                (r"full name", "Ada Lovelace"),
                (r"email", "ada@example.org"),
                (r"(?i)how old", "36"),
                (r"occupation", "Engineer"),
                (r"(?i)correct", "yes"),
            ],
        )

        run = user.run(api)

        assert run.states == _FORM_PATH
        assert run.messages == (
            "hi",
            "Ada Lovelace",
            "ada@example.org",
            "36",
            "Engineer",
            "yes",
        )
        assert run.replies[-1] == "Thank you!"
        assert run.ended is True
        assert api.get_data(run.conversation_id)["user_name"] == "Ada Lovelace"
        llm.assert_exhausted()

    def test_messages_come_before_rules(self) -> None:
        api, _ = _form()

        run = ScriptedUser(["one", "two"], rules=[(r".*", "rule")], max_turns=3).run(
            api
        )

        assert run.messages == ("one", "two", "rule")

    def test_the_first_matching_rule_wins(self) -> None:
        api, _ = _form()

        run = ScriptedUser(
            rules=[(r"Welcome", "first"), (r"Welcome", "second")], max_turns=1
        ).run(api)

        assert run.messages == ("first",)

    def test_the_greeting_is_what_the_first_rule_reads(self) -> None:
        api, _ = _form()

        run = ScriptedUser(rules=[(r"Welcome to the form", "hi")], max_turns=1).run(api)

        assert run.messages == ("hi",)

    def test_a_string_utterance_repeats_on_every_match(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(rules=[(r"Still here", "again")], max_turns=3).run(api)

        assert run.messages == ("again",) * 3

    def test_a_sequence_is_used_up_then_the_rule_stops_firing(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(
            rules=[(r"Still here", ("one", "two"))], fallback="rest", max_turns=4
        ).run(api)

        # The rule fires on the greeting and on each reply, twice; then the
        # fallback takes over.
        assert run.messages == ("one", "two", "rest", "rest")

    def test_an_exhausted_rule_lets_a_later_rule_fire(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(
            rules=[(r"Still", ("once",)), (r"Still", "then")], max_turns=3
        ).run(api)

        assert run.messages == ("once", "then", "then")

    def test_a_callable_matcher_reads_the_last_reply(self) -> None:
        seen: list[str] = []

        def matcher(reply: str) -> bool:
            seen.append(reply)
            return "Still" in reply

        api, _ = _api(_stuck())

        run = ScriptedUser(rules=[(matcher, "ping")], max_turns=2).run(api)

        assert run.messages == ("ping", "ping")
        assert seen == ["Still here.", "Still here."]

    def test_a_compiled_pattern_is_searched_not_matched(self) -> None:
        api, _ = _form()

        run = ScriptedUser(rules=[(re.compile(r"the form"), "found")], max_turns=1).run(
            api
        )

        assert run.messages == ("found",)

    def test_a_user_holds_no_state_between_runs(self) -> None:
        user = ScriptedUser(rules=[(r"Still", ("one", "two"))], max_turns=2)

        first = user.run(_api(_stuck())[0])
        second = user.run(_api(_stuck())[0])

        assert first.messages == second.messages == ("one", "two")


class TestStartAndClose:
    def test_initial_context_reaches_the_conversation_as_a_copy(self) -> None:
        api, _ = _greeter()
        context = {"user_name": "Ada"}

        run = ScriptedUser().run(api, initial_context=context)
        context["user_name"] = "changed"

        assert api.get_data(run.conversation_id)["user_name"] == "Ada"

    def test_a_bad_initial_context_is_the_apis_own_error(self) -> None:
        api, _ = _greeter()

        with pytest.raises(FSMError):
            ScriptedUser().run(api, initial_context=["not", "a", "dict"])  # type: ignore[arg-type]

    def test_close_ends_a_conversation_still_running(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(["x"]).run(api, close=True)

        assert run.ended is False  # read before the close
        assert api.list_active_conversations() == []

    def test_close_leaves_a_finished_conversation_closed_too(self) -> None:
        api, _ = _greeter()

        run = ScriptedUser(["hello", "good"]).run(api, close=True)

        assert run.ended is True
        assert api.list_active_conversations() == []

    def test_without_close_the_conversation_stays_open(self) -> None:
        api, _ = _api(_stuck())

        run = ScriptedUser(["x"]).run(api)

        assert api.list_active_conversations() == [run.conversation_id]


class TestExceptionsPropagate:
    def test_a_raising_matcher_propagates(self) -> None:
        def boom(reply: str) -> bool:
            raise RuntimeError("matcher failed")

        api, _ = _greeter()

        with pytest.raises(RuntimeError, match="matcher failed"):
            ScriptedUser(rules=[(boom, "x")]).run(api)

    def test_an_llm_failure_propagates(self) -> None:
        llm = ScriptedLLM(responses={"greeting": "Hi!"})  # no reply for the next state
        api = API.from_file(str(_GREETER), llm_interface=llm)

        with pytest.raises(FSMError):
            ScriptedUser(["hello"]).run(api)

    def test_close_still_runs_when_the_run_raised(self) -> None:
        def boom(reply: str) -> bool:
            raise RuntimeError("matcher failed")

        api, _ = _greeter()

        with pytest.raises(RuntimeError):
            ScriptedUser(rules=[(boom, "x")]).run(api, close=True)

        assert api.list_active_conversations() == []


class TestConstruction:
    def test_a_single_string_for_messages_is_refused(self) -> None:
        with pytest.raises(TypeError, match="not one string"):
            ScriptedUser("hello")

    def test_a_non_string_message_is_refused(self) -> None:
        with pytest.raises(TypeError, match="message is a string"):
            ScriptedUser(["ok", 3])  # type: ignore[list-item]

    def test_a_rule_matcher_of_another_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="rule matcher"):
            ScriptedUser(rules=[(3, "x")])  # type: ignore[list-item]

    def test_a_non_string_rule_utterance_is_refused(self) -> None:
        with pytest.raises(TypeError, match="rule utterance"):
            ScriptedUser(rules=[("a", ["ok", 3])])  # type: ignore[list-item]

    def test_a_bad_regex_raises_at_construction(self) -> None:
        with pytest.raises(re.error):
            ScriptedUser(rules=[("(", "x")])

    @pytest.mark.parametrize("value", [True, 2.5, "3", None])
    def test_a_non_int_max_turns_is_refused(self, value: Any) -> None:
        with pytest.raises(TypeError, match="max_turns"):
            ScriptedUser(max_turns=value)

    def test_a_negative_max_turns_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_turns"):
            ScriptedUser(max_turns=-1)

    def test_a_non_string_fallback_is_refused(self) -> None:
        with pytest.raises(TypeError, match="fallback"):
            ScriptedUser(fallback=3)  # type: ignore[arg-type]

    def test_the_options_are_keyword_only(self) -> None:
        params = inspect.signature(ScriptedUser).parameters
        assert [n for n, p in params.items() if p.kind is p.KEYWORD_ONLY] == [
            "rules",
            "fallback",
            "max_turns",
        ]


class TestAdvanceUntilTerminal:
    def test_a_message_free_fsm_runs_to_its_end(self) -> None:
        api, llm = _api(_chain())

        run = advance_until_terminal(api, max_steps=5)

        assert run.states == ("a", "b", "c")
        assert run.messages == ()
        assert run.greeting == "Say A"
        assert run.ended is True
        assert run.final_state == "c"
        assert llm.unexpected == []

    def test_a_silent_state_gives_an_empty_reply(self) -> None:
        api, _ = _api(_chain())

        run = advance_until_terminal(api, max_steps=5)

        # b is silent (empty response_instructions): "" like converse returns there.
        assert run.replies == ("", "Done.")
        assert len(run.states) == len(run.replies) + 1

    def test_the_path_equals_the_advance_results(self) -> None:
        api, _ = _api(_chain())
        steps_api, _ = _api(_chain())
        conversation_id, _ = steps_api.start_conversation()
        steps = steps_api.run_until_terminal(conversation_id, max_steps=5)

        run = advance_until_terminal(api, max_steps=5)

        assert run.states == ("a", *(s.state_after for s in steps))

    def test_a_spent_budget_raises(self) -> None:
        api, _ = _api(_chain())

        with pytest.raises(RunBudgetExceededError):
            advance_until_terminal(api, max_steps=1)

    def test_a_stuck_fsm_spends_the_budget_and_close_still_ends_it(self) -> None:
        api, _ = _api(_stuck())

        with pytest.raises(RunBudgetExceededError):
            advance_until_terminal(api, max_steps=2, close=True)

        assert api.list_active_conversations() == []

    def test_close_ends_the_finished_conversation(self) -> None:
        api, _ = _api(_chain())

        run = advance_until_terminal(api, max_steps=5, close=True)

        assert run.ended is True
        assert api.list_active_conversations() == []

    def test_initial_context_is_passed_as_a_copy(self) -> None:
        api, _ = _api(_chain())
        context = {"seed": 1}

        run = advance_until_terminal(api, max_steps=5, initial_context=context)

        assert api.get_data(run.conversation_id)["seed"] == 1

    def test_an_fsm_that_starts_terminal_runs_no_step(self) -> None:
        definition = {
            "name": "One",
            "description": "one state",
            "initial_state": "only",
            "persona": "a test persona",
            "states": {"only": _state("only", say="Done.")},
        }
        api, _ = _api({**definition})

        run = advance_until_terminal(api, max_steps=3)

        assert run.states == ("only",)
        assert run.replies == ()
        assert run.ended is True

    def test_max_steps_is_required_and_keyword_only(self) -> None:
        params = inspect.signature(advance_until_terminal).parameters
        assert params["max_steps"].default is inspect.Parameter.empty
        assert params["max_steps"].kind is inspect.Parameter.KEYWORD_ONLY


class TestExports:
    @pytest.mark.parametrize(
        "name", ["ScriptedUser", "ConversationRun", "advance_until_terminal"]
    )
    def test_public_with_an_interface_contract(self, name: str) -> None:
        assert name in public.__all__
        item = getattr(public, name)
        assert item.__module__ == "fsm_llm.testing.user"
        assert "Interface contract" in (inspect.getdoc(item) or "")

    def test_the_package_ships_no_pytest_dependency(self) -> None:
        source = inspect.getsource(inspect.getmodule(ScriptedUser))
        assert "import pytest" not in source
        assert "litellm" not in source
