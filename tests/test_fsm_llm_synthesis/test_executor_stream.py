"""The synthesize kind on core's streams (plan baa37f40, step 21, D-040).

``API.run_until_terminal_stream`` and ``API.converse_stream`` over a host whose state is a
``synthesize`` state with a ``SynthesizeExecutor``: the same text and end state as the plain
``run_until_terminal`` and ``converse``, the arguments checked at call time, nothing sent before
the first ``next()``, and the rule of a stream closed early (settle runs at the START of a
stream and at its normal end, never in a ``finally``: an abandoned stream leaves the ended child
for the next entry, 5a6a1f80 and baa37f40 D-026). The runner's own stream wrappers and their
settle-on-close rows stay on ``test_runner_stream.py`` until step 22 deletes them with the runner.
"""

from typing import Any

import pytest

from fsm_llm.core.definitions import RunBudgetExceededError
from fsm_llm.synthesis import SynthesisPolicy

from .rig import make_rig


def _rig(valid_fsm: dict[str, Any], tmp_path: Any = None, **kw: Any) -> Any:
    return make_rig(valid_fsm, tmp_path, **kw)


def _stream(rig: Any, **kw: Any) -> Any:
    return rig.api.run_until_terminal_stream(rig.conv, max_steps=20, **kw)


class TestRunStream:
    def test_the_text_and_the_end_state_are_those_of_run(self, valid_fsm, tmp_path):
        plain = _rig(valid_fsm, tmp_path / "a")
        results = plain.run(20)
        streamed = _rig(valid_fsm, tmp_path / "b")

        chunks = list(_stream(streamed))

        assert chunks == [r.response for r in results if r.response]
        assert chunks == ["What is the answer?", "Thanks", "Booked!"]
        for rig in (plain, streamed):
            assert (rig.state, rig.depth, rig.data["answer"]) == ("booked", 1, "42")
            assert rig.names == ["obtained", "pushed", "popped"]
        assert streamed.result == plain.result
        assert streamed.library.get_record(streamed.digest).stats == {
            "runs": 1,
            "successes": 1,
        }

    def test_the_arguments_are_checked_at_call_time(self, valid_fsm):
        rig = _rig(valid_fsm)
        with pytest.raises(ValueError):
            rig.api.run_until_terminal_stream(rig.conv, max_steps=0)
        with pytest.raises(ValueError):
            rig.api.run_until_terminal_stream("no-such-conversation", max_steps=3)

    def test_a_spent_budget_raises_while_iterating(self, valid_fsm):
        rig = _rig(valid_fsm)
        with pytest.raises(RunBudgetExceededError):
            list(rig.api.run_until_terminal_stream(rig.conv, max_steps=1))

    def test_nothing_runs_before_the_first_next(self, valid_fsm):
        rig = _rig(valid_fsm)
        _stream(rig)
        assert rig.strong.requests == [] and rig.events == []


class TestClosingEarly:
    def test_a_closed_stream_leaves_the_ended_child_for_the_next_entry(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path)
        stream = _stream(rig)
        assert next(stream) == "What is the answer?"
        assert next(stream) == "Thanks"  # the child's last turn: it has ended

        stream.close()

        # settle is not a closing action: nothing is popped, counted or written
        assert (rig.depth, rig.state) == (2, "done")
        assert rig.names == ["obtained", "pushed"]
        assert "synthesis_result" not in rig.data
        assert rig.library.get_record(rig.digest).stats == {"runs": 0, "successes": 0}
        # a later run pops it, counts it once and goes on
        rig.run(20)
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.names == ["obtained", "pushed", "popped"]
        assert rig.result["status"] == "ok"
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 1}

    def test_a_running_child_is_left_alone(self, valid_fsm):
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=5))
        stream = _stream(rig)
        assert next(stream) == "What is the answer?"

        stream.close()

        assert (rig.depth, rig.state) == (2, "ask")
        assert rig.names == ["obtained", "pushed"]


class TestConverseStream:
    def test_each_turn_reads_as_converse_does(self, valid_fsm, tmp_path):
        plain = _rig(valid_fsm, tmp_path / "a")
        streamed = _rig(valid_fsm, tmp_path / "b")
        for message in ("hello", "forty-two"):
            want = plain.api.converse(message, plain.conv)
            got = list(streamed.api.converse_stream(message, streamed.conv))
            assert "".join(got) == want
            assert (streamed.state, streamed.depth) == (plain.state, plain.depth)
        assert streamed.names == plain.names == ["obtained", "pushed", "popped"]
        assert streamed.result == plain.result

    def test_an_aborted_child_and_the_apology_are_streamed_too(self, valid_fsm):
        kw: dict[str, Any] = {"extraction": {}, "policy": SynthesisPolicy(max_turns=2)}
        plain, streamed = _rig(valid_fsm, **kw), _rig(valid_fsm, **kw)
        wants, gots = [], []
        for message in ("hello", "one", "two"):
            wants.append(plain.api.converse(message, plain.conv))
            gots.append("".join(streamed.api.converse_stream(message, streamed.conv)))
        assert gots == wants
        assert (
            streamed.names == plain.names == ["obtained", "pushed", "aborted", "popped"]
        )
        assert streamed.result == plain.result
        assert (streamed.result["status"], streamed.result["reason"]) == (
            "aborted",
            "budget_spent",
        )

    def test_a_stream_that_begins_in_the_synthesize_state_obtains_at_its_start(
        self, valid_fsm, monkeypatch
    ):
        # an interrupt during obtain leaves the parent in the state with no record (D-039);
        # the next turn, plain or streamed, begins in it and obtains again
        def pending() -> Any:
            rig = _rig(valid_fsm)
            real = rig.synth.obtain

            def interrupted(*args: Any, **kwargs: Any) -> Any:
                raise KeyboardInterrupt

            monkeypatch.setattr(rig.synth, "obtain", interrupted)
            with pytest.raises(KeyboardInterrupt):
                rig.api.converse("hello", rig.conv)
            monkeypatch.setattr(rig.synth, "obtain", real)
            assert rig.names == [] and "synthesis_result" not in rig.data
            return rig

        plain, streamed = pending(), pending()
        want = plain.api.converse("hello", plain.conv)
        got = "".join(streamed.api.converse_stream("hello", streamed.conv))
        assert got == want
        assert (streamed.state, streamed.depth) == (plain.state, plain.depth)
        assert streamed.names == plain.names == ["obtained", "pushed"]

    def test_closing_early_leaves_the_child_for_the_next_call(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path)
        list(rig.api.converse_stream("hello", rig.conv))
        stream = rig.api.converse_stream("forty-two", rig.conv)
        assert next(stream) == "Thanks"

        stream.close()

        assert rig.depth == 2  # ended, not yet popped: the stream was cut
        rig.run(20)
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 1}

    def test_nothing_runs_before_the_first_next(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.api.converse_stream("hello", rig.conv)
        assert rig.strong.requests == [] and rig.events == []
