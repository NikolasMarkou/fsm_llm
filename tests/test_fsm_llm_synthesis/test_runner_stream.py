"""
Tests for ``ProcedureRunner.run_stream`` and ``converse_stream`` (plan 51fde95b
D-027, part 6): a thin wrapper over core's streams that gives the same text and
end state as ``run`` and ``converse``, and settles a finished child when the
stream is closed early.
"""

from typing import Any

import pytest

from fsm_llm.core.definitions import RunBudgetExceededError
from fsm_llm.synthesis import SynthesisPolicy

from .test_runner import _rig


def _end(rig: Any) -> tuple[Any, ...]:
    return (rig.state, rig.depth, rig.data.get("answer"), rig.names, rig.library)


class TestRunStream:
    def test_the_text_and_the_end_state_are_those_of_run(self, valid_fsm, tmp_path):
        plain = _rig(valid_fsm, tmp_path / "a")
        results = plain.runner.run(plain.conv, max_steps=20)
        streamed = _rig(valid_fsm, tmp_path / "b")

        chunks = list(streamed.runner.run_stream(streamed.conv, max_steps=20))

        assert chunks == [r.response for r in results if r.response]
        assert _end(streamed)[:4] == _end(plain)[:4] == ("booked", 1, "42", plain.names)
        assert streamed.result == plain.result
        assert streamed.library.get_record(streamed.digest).stats == {
            "runs": 1,
            "successes": 1,
        }

    def test_the_arguments_are_checked_at_call_time(self, valid_fsm):
        rig = _rig(valid_fsm)
        with pytest.raises(ValueError):
            rig.runner.run_stream(rig.conv, max_steps=0)
        with pytest.raises(ValueError):
            rig.runner.run_stream("no-such-conversation", max_steps=3)

    def test_a_spent_budget_raises_while_iterating(self, valid_fsm):
        rig = _rig(valid_fsm)
        with pytest.raises(RunBudgetExceededError):
            list(rig.runner.run_stream(rig.conv, max_steps=1))

    def test_nothing_runs_before_the_first_next(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.runner.run_stream(rig.conv, max_steps=20)
        assert rig.strong.requests == [] and rig.events == []


class TestClosingEarly:
    def test_a_finished_child_is_popped_and_counted_once(self, valid_fsm, tmp_path):
        rig = _rig(valid_fsm, tmp_path)
        stream = rig.runner.run_stream(rig.conv, max_steps=20)
        assert next(stream) == "Thanks"  # the child's last turn: it has ended

        stream.close()

        assert (rig.depth, rig.state) == (1, "solve")
        assert rig.result["status"] == "ok" and rig.data["answer"] == "42"
        assert rig.names == ["obtained", "pushed", "popped"]
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 1}
        # a later run goes on from there and counts nothing twice
        rig.runner.run(rig.conv, max_steps=20)
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.names == ["obtained", "pushed", "popped"]
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 1}

    def test_the_same_close_on_the_core_stream_leaves_the_child_pushed(self, valid_fsm):
        rig = _rig(valid_fsm)  # the control: why the close needs a settle
        stream = rig.api.run_until_terminal_stream(
            rig.conv, max_steps=20, before_step=rig.runner.hook(rig.conv)
        )
        next(stream)
        stream.close()
        assert rig.depth == 2 and "popped" not in rig.names

    def test_a_running_child_is_left_alone(self, valid_fsm):
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=5))
        stream = rig.runner.run_stream(rig.conv, max_steps=20)
        assert next(stream) == "What is the answer?"

        stream.close()

        assert (rig.depth, rig.state) == (2, "ask")
        assert rig.names == ["obtained", "pushed"]

    def test_the_settle_on_close_starts_nothing(self, valid_fsm, monkeypatch):
        rig = _rig(valid_fsm)
        calls: list[dict[str, Any]] = []
        real = rig.runner._tick

        def spy(conversation_id: str, step: int | None, **kw: Any) -> list[str]:
            calls.append(kw)
            return real(conversation_id, step, **kw)

        monkeypatch.setattr(rig.runner, "_tick", spy)
        stream = rig.runner.run_stream(rig.conv, max_steps=20)
        next(stream)
        calls.clear()
        stream.close()
        assert calls == [{"start": False}]

    def test_a_tick_without_start_obtains_nothing(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.to_solve()  # a pending synthesize state, no result yet
        assert rig.runner._tick(rig.conv, None, start=False) == []
        assert rig.strong.requests == [] and "synthesis_result" not in rig.data
        rig.runner._tick(rig.conv, None)
        assert len(rig.strong.requests) == 1  # the default still starts it


class TestConverseStream:
    def test_each_turn_reads_as_converse_does(self, valid_fsm, tmp_path):
        plain = _rig(valid_fsm, tmp_path / "a")
        streamed = _rig(valid_fsm, tmp_path / "b")
        for message in ("hello", "forty-two"):
            want = plain.runner.converse(plain.conv, message)
            got = list(streamed.runner.converse_stream(streamed.conv, message))
            assert "".join(got) == want
            assert (streamed.state, streamed.depth) == (plain.state, plain.depth)
        assert want == "Thanks\nBooked!" and streamed.names == plain.names
        assert streamed.result == plain.result

    def test_an_aborted_child_and_the_apology_are_streamed_too(self, valid_fsm):
        kw: dict[str, Any] = {"extraction": {}, "policy": SynthesisPolicy(max_turns=2)}
        plain, streamed = _rig(valid_fsm, **kw), _rig(valid_fsm, **kw)
        for message in ("hello", "one", "two"):
            want = plain.runner.converse(plain.conv, message)
            got = "".join(streamed.runner.converse_stream(streamed.conv, message))
            assert got == want
        assert want == "What is the answer?\nSorry"
        assert (
            streamed.names == plain.names == ["obtained", "pushed", "aborted", "popped"]
        )

    def test_a_pending_synthesize_state_starts_before_the_message_is_sent(
        self, valid_fsm
    ):
        plain, streamed = _rig(valid_fsm), _rig(valid_fsm)
        for rig in (plain, streamed):
            rig.to_solve()  # the parent waits on its synthesize state
        want = plain.runner.converse(plain.conv, "hello")
        got = "".join(streamed.runner.converse_stream(streamed.conv, "hello"))
        assert got == want and "Booked!" in want
        assert (streamed.state, streamed.depth) == (plain.state, plain.depth)
        assert streamed.names == plain.names == ["obtained", "pushed", "popped"]

    def test_closing_early_leaves_the_child_for_the_next_call(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path)
        list(rig.runner.converse_stream(rig.conv, "hello"))
        stream = rig.runner.converse_stream(rig.conv, "forty-two")
        assert next(stream) == "Thanks"

        stream.close()

        assert rig.depth == 2  # ended, not yet popped: the stream was cut
        rig.runner.run(rig.conv, max_steps=20)
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 1}

    def test_nothing_runs_before_the_first_next(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.runner.converse_stream(rig.conv, "hello")
        assert rig.strong.requests == [] and rig.events == []
