"""The generic turn loop ``fsm_llm.simulation.driver.play`` (offline, scripted double).

The loop moved out of ``eval.cases.run_case_trial`` (plan c720601d step 4): these
rows pin it without eval, and the eval suites pin the public wrapper.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from fsm_llm.core.codes import MessageSource, current_message_source
from fsm_llm.engine.api.conversation import API
from fsm_llm.simulation.driver import Trial, play
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.support import _state

_CHAT: dict[str, Any] = {
    "name": "Chat",
    "description": "One speaking state that ends on the word stop",
    "initial_state": "chat",
    "states": {
        "chat": _state(
            "chat",
            response_instructions="Reply",
            transitions=[
                {
                    "target_state": "end",
                    "description": "the user said stop",
                    "conditions": [
                        {
                            "description": "stop given",
                            "requires_context_keys": ["stop"],
                            "logic": {"!!": [{"var": "stop"}]},
                        }
                    ],
                }
            ],
        ),
        "end": _state("end"),
    },
}


def _api() -> API:
    return API(copy.deepcopy(_CHAT), llm_interface=ScriptedLLM(responses={"chat": "r"}))


class TestPlay:
    def test_a_clean_run_is_plain_data(self):
        trial = play(_api, {"seed": 1}, ["a", "b"])
        assert trial.error is None and trial.close_error is None
        assert trial.turns_sent == 2
        assert trial.visited_states == ["chat", "chat", "chat"]
        assert len(trial.responses) == 3  # the greeting and two replies
        assert trial.final_state == "chat" and trial.ended is False
        assert trial.context["seed"] == 1
        assert trial.duration >= 0.0

    def test_the_result_carries_no_expectation_field(self):
        names = {f.name for f in dataclasses.fields(Trial)}
        assert names == {
            "responses",
            "visited_states",
            "turns_sent",
            "final_state",
            "context",
            "ended",
            "error",
            "close_error",
            "duration",
        }

    def test_make_api_raising_is_data_and_nothing_is_closed(self):
        def boom() -> API:
            raise RuntimeError("no api")

        trial = play(boom, {}, ["a"])
        assert trial.error == "RuntimeError: no api"
        assert trial.close_error is None and trial.turns_sent == 0
        assert trial.final_state is None and trial.responses == []

    def test_a_start_failure_closes_the_api(self, monkeypatch: pytest.MonkeyPatch):
        made: list[API] = []
        closed: list[int] = []

        def make() -> API:
            api = _api()
            monkeypatch.setattr(
                api, "start_conversation", lambda *a, **k: 1 / 0, raising=True
            )
            real = api.close
            monkeypatch.setattr(api, "close", lambda: (closed.append(1), real()))
            made.append(api)
            return api

        trial = play(make, {}, ["a"])
        assert trial.error is not None and trial.error.startswith("ZeroDivisionError")
        assert closed == [1] and len(made) == 1

    def test_a_turn_failure_closes_the_api_and_keeps_the_partial_run(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        closed: list[int] = []

        def make() -> API:
            api = _api()
            real_converse, real_close = api.converse, api.close
            seen = {"n": 0}

            def converse(*a: Any, **k: Any) -> str:
                seen["n"] += 1
                if seen["n"] == 2:
                    raise RuntimeError("model down")
                return real_converse(*a, **k)

            monkeypatch.setattr(api, "converse", converse)
            monkeypatch.setattr(api, "close", lambda: (closed.append(1), real_close()))
            return api

        trial = play(make, {}, ["a", "b", "c"])
        assert trial.error == "RuntimeError: model down"
        assert trial.turns_sent == 1 and len(trial.responses) == 2
        assert trial.final_state is None  # the run did not reach its end
        assert closed == [1]

    def test_a_close_failure_is_its_own_field_and_not_a_turn_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def make() -> API:
            api = _api()
            real_close = api.close

            def close() -> None:
                real_close()
                raise OSError("close boom")

            monkeypatch.setattr(api, "close", close)
            return api

        trial = play(make, {}, ["a"])
        assert trial.error is None
        assert trial.close_error == "close failed: close boom"
        assert trial.turns_sent == 1 and trial.final_state == "chat"


class TestStop:
    def test_stop_is_read_before_every_turn(self):
        calls: list[int] = []

        def stop() -> bool:
            calls.append(1)
            return len(calls) == 3

        trial = play(_api, {}, ["a", "b", "c", "d"], stop=stop)
        assert calls == [1, 1, 1]
        assert trial.turns_sent == 2 and len(trial.responses) == 3

    def test_a_stop_true_before_the_first_turn_sends_none(self):
        trial = play(_api, {}, ["a", "b"], stop=lambda: True)
        assert trial.turns_sent == 0 and trial.error is None
        assert trial.responses and trial.final_state == "chat"

    def test_stop_is_not_asked_once_the_conversation_ended(self):
        calls: list[int] = []
        trial = play(
            _api,
            {"stop": True},
            ["a", "b", "c"],
            stop=lambda: calls.append(1) or False,
        )
        assert trial.ended is True and trial.turns_sent == 1
        assert calls == [1]  # before the first turn only

    def test_a_raising_stop_is_a_trial_error(self):
        def stop() -> bool:
            raise KeyError("stop boom")

        trial = play(_api, {}, ["a"], stop=stop)
        assert trial.error == "KeyError: 'stop boom'" and trial.turns_sent == 0


class TestMessageSource:
    def test_every_converse_runs_under_the_simulator_source_and_it_resets(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        seen: list[MessageSource] = []
        real = API.converse

        def converse(self: API, *a: Any, **k: Any) -> str:
            seen.append(current_message_source())
            return real(self, *a, **k)

        monkeypatch.setattr(API, "converse", converse)
        assert current_message_source() is MessageSource.USER
        play(_api, {}, ["a", "b", "c"])
        assert seen == [MessageSource.SIMULATOR] * 3
        assert current_message_source() is MessageSource.USER

    def test_a_failed_turn_does_not_leak_the_source(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def converse(self: API, *a: Any, **k: Any) -> str:
            raise RuntimeError("down")

        monkeypatch.setattr(API, "converse", converse)
        assert play(_api, {}, ["a"]).error == "RuntimeError: down"
        assert current_message_source() is MessageSource.USER
