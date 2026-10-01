"""Core fixes of review round 1 (plan 07ad3f8c, step 22.1, D-044).

Pinned here: the Pass-1 sibling call sites keep "no user message" as ``None``
(each test fails when its site passes ``user_message or ""``), a conversation
closed under a bounded run ends the run, the seconds budget is asked again
after the hook, ``max_seconds`` is type-checked, ``build_fsm_graph`` refuses
malformed shapes with ``ValueError`` and the runner prints nothing for a
silent reply.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm import RunBudgetExceededError, build_fsm_graph, constants
from fsm_llm.classification import Classifier
from fsm_llm.definitions import FieldExtractionRequest, FieldExtractionResponse
from fsm_llm.llm import LiteLLMInterface
from tests.test_fsm_llm.test_advance import (
    _always,
    _ScriptedLLM,
    _start,
    _state,
    _StreamingLLM,
    _trip_fsm,
    _when_set,
)
from tests.test_fsm_llm.test_llm_request_path import _response, _schema
from tests.test_fsm_llm.test_run_until_terminal import _FakeClock, _loop_fsm, clock

__all__ = ["clock"]  # the fixture, re-exported for this module

_INTENTS = {
    "field_name": "intent",
    "intents": [
        {"name": "buy", "description": "User wants to purchase"},
        {"name": "browse", "description": "User is just looking"},
    ],
    "fallback_intent": "browse",
    "confidence_threshold": 0.7,
}


class _NullThenValueLLM(_ScriptedLLM):
    """``_ScriptedLLM`` whose first answer for every field is null."""

    def __init__(self, fields: dict[str, Any]) -> None:
        super().__init__(fields)
        self._asked: set[str] = set()

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        if request.field_name not in self._asked:
            self._asked.add(request.field_name)
            self.field_requests.append(request)
            return FieldExtractionResponse(
                field_name=request.field_name,
                value=None,
                confidence=0.0,
                is_valid=False,
            )
        return super().extract_field(request)


def _classifying_scripted_llm() -> _ScriptedLLM:
    """A ``_ScriptedLLM`` whose ``complete`` is a real ``LiteLLMInterface``'s
    for the same model: the classifier sends through the conversation's
    interface (D-006 of plan 944e2692), so this is how its request reaches
    the provider binding ``_ClassifierProvider`` scripts."""
    llm = _ScriptedLLM()
    llm.complete = LiteLLMInterface(model=llm.model).complete
    return llm


class _ClassifierProvider:
    """Scripted provider for the classifier's LLM interface: replies with
    the queued ``(intent, confidence)`` pairs and records every request."""

    def __init__(self, *replies: tuple[str, float]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def _completion(self, **kwargs: Any) -> MagicMock:
        self.calls.append(kwargs)
        intent, confidence = self._replies.pop(0)
        return _response(
            json.dumps({"reasoning": "r", "intent": intent, "confidence": confidence})
        )

    def __enter__(self) -> _ClassifierProvider:
        self._patches = [
            patch("fsm_llm.llm.completion", side_effect=self._completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in self._patches:
            p.stop()

    def assert_every_call_is_message_free(self, expected_calls: int) -> None:
        assert len(self.calls) == expected_calls
        for call in self.calls:
            system, user = call["messages"]
            assert "(there is no user message)" in system["content"]
            assert "the user's message" not in system["content"]
            assert user["content"] == constants.NEUTRAL_USER_TURN


def _assert_message_free_field_requests(
    requests: list[FieldExtractionRequest], names: list[str]
) -> None:
    assert [request.field_name for request in requests] == names
    for request in requests:
        assert request.user_message is None
        assert "There is no user message" in request.system_prompt
        assert "User message" not in request.system_prompt


class TestNoMessageSiblingSites:
    """One test per Pass-1 call site that must pass ``None`` on (review
    concern 2; mutations N32-N36 and N38 of the core review)."""

    def test_retry_field_pass(self):
        llm = _NullThenValueLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        result = api.advance(conv_id)

        assert result.state_after == "plan"
        # The first ask answered null; the retry pass asked again.
        _assert_message_free_field_requests(llm.field_requests, ["city", "city"])

    def test_post_transition_missing_key_ask(self):
        fsm = {
            "name": "two_keys",
            "description": "d",
            "initial_state": "start",
            "states": {
                "start": _state("start", transitions=[_always("collect")]),
                "collect": _state(
                    "collect",
                    required_context_keys=["city"],
                    transitions=[_when_set("done", "city")],
                ),
                "done": _state("done", speaks=True),
            },
        }
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(fsm, llm)

        result = api.advance(conv_id)

        assert (result.state_before, result.state_after) == ("start", "collect")
        assert api.get_data(conv_id)["city"] == "Paris"
        _assert_message_free_field_requests(llm.field_requests, ["city"])

    def test_revisit_re_run(self):
        fsm = {
            "name": "revisit",
            "description": "d",
            "initial_state": "collect",
            "states": {
                "collect": _state(
                    "collect",
                    required_context_keys=["city"],
                    extraction_instructions="Extract the city and the season.",
                    transitions=[_when_set("review", "city")],
                ),
                "review": _state(
                    "review",
                    transitions=[
                        _when_set("collect", "redo", priority=10),
                        _when_set("done", "approved", priority=20),
                    ],
                ),
                "done": _state("done", speaks=True),
            },
        }
        llm = _ScriptedLLM({"city": "Paris"}, bulk={"season": "spring"})
        api, conv_id = _start(fsm, llm)
        assert api.advance(conv_id).state_after == "review"
        api.update_context(conv_id, {"redo": True})
        llm.reset()

        result = api.advance(conv_id)

        assert (result.state_before, result.state_after) == ("review", "collect")
        # `city` was extracted by the pipeline, so entering `collect` again
        # re-runs its whole Pass 1 (21cd7f8e/D-034): one bulk call here.
        (bulk,) = llm.bulk_requests
        assert bulk.user_message is None
        assert "There is no user message" in bulk.system_prompt
        assert "User message" not in bulk.system_prompt

    def test_post_transition_classification(self):
        fsm = {
            "name": "classify_next",
            "description": "d",
            "initial_state": "start",
            "states": {
                "start": _state("start", transitions=[_always("triage")]),
                "triage": _state(
                    "triage",
                    classification_extractions=[_INTENTS],
                    transitions=[_when_set("done", "never")],
                ),
                "done": _state("done", speaks=True),
            },
        }
        api, conv_id = _start(fsm, _classifying_scripted_llm())

        with _ClassifierProvider(("buy", 0.95)) as provider:
            result = api.advance(conv_id)

        assert (result.state_before, result.state_after) == ("start", "triage")
        assert api.get_data(conv_id)["intent"] == "buy"
        provider.assert_every_call_is_message_free(1)

    def test_retry_classification(self):
        fsm = {
            "name": "classify_retry",
            "description": "d",
            "initial_state": "triage",
            "states": {
                "triage": _state(
                    "triage",
                    classification_extractions=[{**_INTENTS, "required": True}],
                    transitions=[_when_set("done", "never")],
                ),
                "done": _state("done", speaks=True),
            },
        }
        api, conv_id = _start(fsm, _classifying_scripted_llm())

        # A non-fallback intent below the threshold is discarded, so the key
        # is still missing and the retry pass classifies again.
        with _ClassifierProvider(("buy", 0.2), ("buy", 0.95)) as provider:
            api.advance(conv_id)

        assert api.get_data(conv_id)["intent"] == "buy"
        provider.assert_every_call_is_message_free(2)

    def test_multi_intent_prompt_without_a_message(self):
        reply = _response(
            json.dumps(
                {"reasoning": "r", "intents": [{"intent": "buy", "confidence": 0.9}]}
            )
        )
        with (
            patch("fsm_llm.llm.completion", return_value=reply) as completion,
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
        ):
            result = Classifier(_schema(), model="gpt-4o").classify_multi(
                None, context={"data": {"cart": "phone"}}
            )

        assert result.primary.intent == "buy"
        system = completion.call_args.kwargs["messages"][0]["content"]
        assert "(there is no user message)" in system
        assert "one or more of the following intents" in system
        assert "If several intents apply, return them ranked" in system


class TestRunEndsWhenTheConversationIsClosed:
    """Review concern 3: a conversation closed under a bounded run ends the
    run; it never surfaces as ``ValueError("Unknown conversation ID")``."""

    def test_before_step_closing_the_conversation_returns_the_results_so_far(self):
        llm = _ScriptedLLM()
        api, conv_id = _start(_loop_fsm(), llm)

        def _hook(step: int) -> None:
            if step == 2:
                api.end_conversation(conv_id)

        results = api.run_until_terminal(conv_id, max_steps=5, before_step=_hook)

        assert [(r.state_before, r.state_after) for r in results] == [("ping", "pong")]
        assert api.has_conversation_ended(conv_id) is True

    def test_before_step_closing_the_conversation_ends_the_stream(self):
        api, conv_id = _start(_loop_fsm(), _StreamingLLM())
        seen: list[int] = []

        def _hook(step: int) -> None:
            seen.append(step)
            if step == 2:
                api.end_conversation(conv_id)

        stream = api.run_until_terminal_stream(conv_id, max_steps=5, before_step=_hook)

        assert list(stream) == []
        assert seen == [1, 2]

    def test_closed_between_the_check_and_the_step_returns_the_results_so_far(self):
        """What another thread's ``end_conversation`` looks like to the run:
        the round saw a live conversation, the step finds none."""
        api, conv_id = _start(_loop_fsm(), _ScriptedLLM())
        real_advance = api.advance
        steps: list[int] = []

        def _advance(conversation_id: str) -> Any:
            steps.append(len(steps) + 1)
            if len(steps) == 2:
                api.end_conversation(conversation_id)
            return real_advance(conversation_id)

        with patch.object(api, "advance", _advance):
            results = api.run_until_terminal(conv_id, max_steps=5)

        assert [(r.state_before, r.state_after) for r in results] == [("ping", "pong")]
        assert steps == [1, 2]

    def test_closed_between_the_check_and_the_step_ends_the_stream(self):
        api, conv_id = _start(_loop_fsm(), _StreamingLLM())
        real_stream = api.advance_stream
        steps: list[int] = []

        def _advance_stream(conversation_id: str) -> Any:
            steps.append(len(steps) + 1)
            if len(steps) == 2:
                api.end_conversation(conversation_id)
            return real_stream(conversation_id)

        with patch.object(api, "advance_stream", _advance_stream):
            assert list(api.run_until_terminal_stream(conv_id, max_steps=5)) == []

        assert steps == [1, 2]

    def test_a_failing_step_of_a_live_conversation_still_raises(self):
        api, conv_id = _start(_loop_fsm(), _ScriptedLLM())

        with (
            patch.object(api, "advance", side_effect=ValueError("boom")),
            pytest.raises(ValueError, match="boom"),
        ):
            api.run_until_terminal(conv_id, max_steps=5)

    def test_an_id_that_was_never_started_raises_value_error_in_both_forms(self):
        api, _ = _start(_loop_fsm(), _ScriptedLLM())

        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.run_until_terminal("never-started", max_steps=1)
        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.run_until_terminal_stream("never-started", max_steps=1)


class TestSecondsBudgetAfterTheHook:
    """Review concern 4: a hook that uses up the time starts no step."""

    def test_slow_hook_starts_no_step(self, clock: _FakeClock):
        llm = _ScriptedLLM()
        api, conv_id = _start(_loop_fsm(), llm)

        with pytest.raises(RunBudgetExceededError) as raised:
            api.run_until_terminal(
                conv_id,
                max_steps=50,
                max_seconds=0.2,
                before_step=lambda step: clock.advance(0.3),
            )

        assert (raised.value.budget, raised.value.steps_done) == ("seconds", 0)
        assert api.get_current_state(conv_id) == "ping"

    def test_slow_hook_starts_no_stream_step(self, clock: _FakeClock):
        api, conv_id = _start(_loop_fsm(), _StreamingLLM())
        stream = api.run_until_terminal_stream(
            conv_id,
            max_steps=50,
            max_seconds=0.2,
            before_step=lambda step: clock.advance(0.3),
        )

        with pytest.raises(RunBudgetExceededError) as raised:
            list(stream)

        assert (raised.value.budget, raised.value.steps_done) == ("seconds", 0)
        assert api.get_current_state(conv_id) == "ping"


class TestMaxSecondsType:
    """Review concern 5: ``max_seconds`` is checked like ``max_steps``."""

    @pytest.mark.parametrize("max_seconds", [True, False, "5", [1], 1 + 2j])
    def test_non_numbers_and_bools_raise_value_error(self, max_seconds):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))

        with pytest.raises(ValueError, match="max_seconds"):
            api.run_until_terminal(conv_id, max_steps=1, max_seconds=max_seconds)
        with pytest.raises(ValueError, match="max_seconds"):
            api.run_until_terminal_stream(conv_id, max_steps=1, max_seconds=max_seconds)
        assert api.get_current_state(conv_id) == "collect"

    @pytest.mark.parametrize("max_seconds", [5, 0.5, float("inf"), None])
    def test_numbers_and_none_are_accepted(self, max_seconds):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))

        results = api.run_until_terminal(conv_id, max_steps=5, max_seconds=max_seconds)

        assert results[-1].ended is True


class TestBuildFsmGraphRefusesMalformedShapes:
    """Review concern 6: the documented ``ValueError``, not ``AttributeError``."""

    @pytest.mark.parametrize(
        ("definition", "message"),
        [
            (
                {"name": "n", "initial_state": "a", "states": {"a": "oops"}},
                "State 'a' must be a mapping",
            ),
            (
                {"name": "n", "initial_state": "a", "states": ["a"]},
                "'states' must be a mapping",
            ),
            (
                {
                    "name": "n",
                    "initial_state": "a",
                    "states": {"a": {"transitions": ["b"]}, "b": {}},
                },
                "A transition of state 'a' must be a mapping",
            ),
            (
                {
                    "name": "n",
                    "initial_state": "a",
                    "states": {"a": {"transitions": {"target_state": "a"}}},
                },
                "'transitions' of state 'a' must be a list",
            ),
        ],
    )
    def test_malformed_shape_is_a_value_error(self, definition, message):
        with pytest.raises(ValueError, match=message):
            build_fsm_graph(definition)

    def test_explicit_null_transitions_still_mean_terminal(self):
        graph = build_fsm_graph(
            {"name": "n", "initial_state": "a", "states": {"a": {"transitions": None}}}
        )

        assert [(node.id, node.is_terminal) for node in graph.nodes] == [("a", True)]


class TestRunnerPrintsNothingForASilentReply:
    """The runner CLI logs no empty ``System:`` line (consumers review)."""

    def _system_lines(self, greeting: str, reply: str) -> list[str]:
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", greeting)
        mock_api.has_conversation_ended.side_effect = [False, True]
        mock_api.converse.return_value = reply
        mock_api.get_data.return_value = {}
        lines: list[str] = []
        with (
            patch.dict(os.environ, {"LLM_MODEL": "test-model"}, clear=True),
            patch("fsm_llm.runner.dotenv.load_dotenv"),
            patch("fsm_llm.runner.API.from_file", return_value=mock_api),
            patch("fsm_llm.runner.setup_file_logging"),
            patch("builtins.input", return_value="hello"),
            patch("fsm_llm.runner.logger") as logger,
        ):
            from fsm_llm.runner import main

            assert main("/tmp/t.json", 5, 1000) == 0
            lines = [
                call.args[0]
                for call in logger.info.call_args_list
                if call.args and str(call.args[0]).startswith("System:")
            ]
        return lines

    def test_silent_greeting_and_silent_reply_print_no_system_line(self):
        assert self._system_lines("", "") == []

    def test_spoken_replies_are_still_printed(self):
        assert self._system_lines("Hello!", "Sure.") == [
            "System: Hello!",
            "System: Sure.",
        ]
