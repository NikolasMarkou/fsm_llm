"""
Tests that pin ``fsm_llm.synthesis.runner.ProcedureRunner`` itself, on REAL ``API`` objects.

What any driver must do (the ok path, abort at ``max_turns``, an outage, return keys, a reason per
failure, nesting, the trace events) lives in ``test_driver_rows.py`` on the rig of ``rig.py``; the
rows here read the runner's own ``hook``, ``_runs`` and ``_refused``, build a runner by hand, patch
its modules or its push, send a signal during the compile, or pin the bare ``converse`` on a
synthesize state, and go with the runner (plan baa37f40, D-013).

The host is a concierge FSM with one ``synthesize`` state; the strong model is
``ScriptedLLM``, the cheap model under the pushed procedure is
``ScriptedRunnerLLM`` (both in conftest). The runner is driven the two ways it
supports: ``run`` (core ``run_until_terminal`` with the runner's hook) and
``converse`` (user turns). Failure tests read the result the PARENT sees.
"""

import copy
import json
from typing import Any

import pytest

from fsm_llm.core.definitions import (
    FSMDefinition,
    FSMError,
    LLMResponseError,
    RunBudgetExceededError,
)
from fsm_llm.engine.api.conversation import API
from fsm_llm.synthesis import (
    FSMSynthesizer,
    ProcedureRunner,
    SynthesisPolicy,
)
from fsm_llm.synthesis import runner as runner_module
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM
from tests.test_fsm_llm_synthesis.rig import (
    CANARIES,
    CONTEXT,
    REPLIES,
    SECRET,
    Rig,
    make_host,
    make_procedure,
    make_rig,
    serve_procedure,
)


class TestOkPath:
    def test_the_push_hands_over_only_the_declared_inputs(self, valid_fsm):
        rig = make_rig(valid_fsm)
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)

        hook(2)

        assert rig.depth == 2
        child = rig.data
        assert child["topic"] == "tables"
        for absent in ("noise_note", "db_password", "synthesis_result"):
            assert absent not in child
        hook(3)  # a second call while the child runs does nothing
        assert (rig.depth, len(rig.strong.requests)) == (2, 1)
        assert rig.names == ["obtained", "pushed"]
        # the run goes on from where the manual hook left it
        rig.runner.run(rig.conv, max_steps=20)
        assert (rig.state, rig.depth) == ("booked", 1)

    def test_a_settled_result_is_never_obtained_twice(self, valid_fsm):
        rig = make_rig(valid_fsm, replies=[LLMResponseError(SECRET)])
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)

        hook(2)
        hook(3)

        assert len(rig.strong.requests) == 1
        assert rig.result["reason"] == "provider_error"

    def test_a_host_definition_object_is_accepted(self, valid_fsm):
        rig = make_rig(valid_fsm)
        runner = ProcedureRunner(rig.api, FSMDefinition(**make_host()), rig.synth)
        runner.run(rig.conv, max_steps=20)
        assert rig.state == "booked"


class TestConverse:
    def test_a_bare_converse_on_a_synthesize_state_stays_and_obtains_nothing(
        self, valid_fsm
    ):
        rig = make_rig(valid_fsm)

        rig.api.converse("hello", rig.conv)
        rig.api.converse("again", rig.conv)

        assert (rig.state, rig.depth) == ("solve", 1)
        assert "synthesis_result" not in rig.data
        assert rig.strong.requests == [] and rig.events == []


class TestAbort:
    def test_a_spent_parent_budget_raises_and_a_later_end_cleans_up(
        self, valid_fsm, tmp_path
    ):
        rig = make_rig(valid_fsm, tmp_path, extraction={})

        with pytest.raises(RunBudgetExceededError):
            rig.runner.run(rig.conv, max_steps=3)

        assert rig.depth == 2  # the child is still on top
        rig.runner.end_conversation(rig.conv)  # does not raise
        with pytest.raises(ValueError):
            rig.api.get_stack_depth(rig.conv)
        assert rig.conv not in rig.runner._runs
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 0

    def test_ending_the_conversation_with_a_child_on_top_counts_no_run(
        self, valid_fsm, tmp_path
    ):
        rig = make_rig(valid_fsm, tmp_path)
        rig.to_solve()
        rig.runner.hook(rig.conv)(2)
        assert rig.depth == 2

        rig.api.end_conversation(rig.conv)  # straight on the API: no raise

        with pytest.raises(ValueError):
            rig.api.get_stack_depth(rig.conv)
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 0

    def test_a_child_popped_from_outside_is_trimmed_not_obtained_again(self, valid_fsm):
        rig = make_rig(valid_fsm)
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)
        hook(2)

        rig.api.pop_fsm(rig.conv)
        hook(3)

        assert rig.runner._runs[rig.conv] == []
        assert len(rig.strong.requests) == 1
        assert rig.result["status"] == "running"


class TestTheEngineBoundsTheChild:
    """The child frame is pushed with ``max_steps=policy.max_turns`` and the engine's
    lifetime count decides the abort (D-012 of plan 5a6a1f80). The runner keeps no
    count: a turn run by hand counts, and a spent budget of the HOST is never taken
    for the child's."""

    def test_the_child_frame_carries_the_policy_budget_and_the_host_none(
        self, valid_fsm
    ):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))
        rig.to_solve()
        rig.runner.hook(rig.conv)(2)
        manager, child = rig.api.fsm_manager, rig.api.get_sub_conversation_id(rig.conv)
        assert child != rig.conv
        assert (manager.turn_budget_steps(child), manager.turn_budget_spent(child)) == (
            0,
            None,
        )
        assert manager.turn_budget_steps(rig.conv) is None  # the host has no limit

        rig.api.advance(rig.conv)
        rig.api.advance(rig.conv)

        assert manager.turn_budget_steps(child) == 2
        assert manager.turn_budget_spent(child) == "steps"
        with pytest.raises(RunBudgetExceededError) as refused:
            rig.api.advance(rig.conv)  # the engine itself refuses a third turn
        assert (refused.value.budget, refused.value.limit) == ("steps", 2)

    def test_a_turn_run_by_hand_counts_and_the_next_hook_call_aborts(self, valid_fsm):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)
        hook(2)
        rig.api.advance(rig.conv)
        rig.api.advance(rig.conv)  # two turns the hook never saw

        hook(3)

        assert (rig.state, rig.depth) == ("solve", 1)
        assert rig.result["status"] == "aborted"
        assert rig.result["reason"] == "budget_spent"
        assert rig.names == ["obtained", "pushed", "aborted", "popped"]
        assert [f["step"] for f in rig.steps] == [2]  # the engine's count, one event
        assert "answer" not in rig.data and "procedure_abort" not in rig.data

    def test_an_ended_child_is_never_aborted_even_on_its_last_budgeted_turn(
        self, valid_fsm
    ):
        rig = make_rig(valid_fsm, policy=SynthesisPolicy(max_turns=1))

        rig.runner.converse(rig.conv, "hello")
        rig.runner.converse(rig.conv, "forty-two")  # the one turn the budget allows

        assert [f["step"] for f in rig.steps] == [1]  # counted, and not over
        assert rig.result["status"] == "ok" and rig.data["answer"] == "42"
        assert "aborted" not in rig.names

    def test_a_spent_host_budget_still_raises_and_is_not_the_childs(self, valid_fsm):
        host = make_host()
        cheap = ScriptedRunnerLLM({"answer": "42"}, dict(REPLIES))
        # the API default reaches the host's own frame; the child's explicit limit wins
        api = API.from_definition(definition=host, llm_interface=cheap, max_steps=1)
        runner = ProcedureRunner(
            api, host, FSMSynthesizer(ScriptedLLM([json.dumps(valid_fsm)]))
        )
        conv, _ = api.start_conversation(CONTEXT)
        runner.converse(conv, "hello")  # the host's one turn; the child is pushed

        with pytest.raises(RunBudgetExceededError) as spent:
            runner.converse(conv, "forty-two")  # the child ends, the host's second turn

        assert (spent.value.budget, spent.value.limit) == ("steps", 1)
        assert api.get_stack_depth(conv) == 1
        assert api.get_data(conv)["synthesis_result"]["status"] == "ok"
        assert api.get_data(conv)["answer"] == "42"


class TestFailuresAreFixedReasons:
    def test_a_push_that_raises_after_pushing_leaves_the_stack_as_it_was(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm)
        real_push = rig.api.push_fsm

        def push_then_fail(*args: Any, **kwargs: Any) -> Any:
            real_push(*args, **kwargs)
            raise RuntimeError(SECRET)

        monkeypatch.setattr(rig.api, "push_fsm", push_then_fail)
        rig.runner.run(rig.conv, max_steps=10)

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result["reason"] == "unexpected_error"
        assert rig.runner._runs[rig.conv] == []

    @pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
    def test_ctrl_c_and_exit_are_never_swallowed(self, valid_fsm, monkeypatch, signal):
        rig = make_rig(valid_fsm)

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            raise signal()

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(signal):
            rig.runner.run(rig.conv, max_steps=10)

    @pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
    def test_an_interrupted_compile_leaves_a_final_failed_result(
        self, valid_fsm, monkeypatch, signal
    ):
        rig = make_rig(valid_fsm)
        interrupted = signal()
        calls: list[int] = []

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            raise interrupted

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(signal) as caught:
            rig.runner.run(rig.conv, max_steps=10)

        assert caught.value is interrupted
        assert rig.result["status"] == "failed"
        assert rig.result["reason"] == "unexpected_error"
        assert (rig.state, rig.depth) == ("solve", 1)
        assert rig.names == ["failed"]
        # the host catches the signal and keeps the conversation: nothing obtains again
        rig.runner.run(rig.conv, max_steps=10)
        assert calls == [1]
        assert (rig.state, rig.depth) == ("apology", 1)

    @pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
    def test_an_interrupt_after_the_push_unwinds_and_ends_failed(
        self, valid_fsm, monkeypatch, signal
    ):
        rig = make_rig(valid_fsm)
        real_push = rig.api.push_fsm

        def push_then_interrupt(*args: Any, **kwargs: Any) -> Any:
            real_push(*args, **kwargs)
            raise signal()

        monkeypatch.setattr(rig.api, "push_fsm", push_then_interrupt)
        with pytest.raises(signal):
            rig.runner.run(rig.conv, max_steps=10)

        assert (rig.state, rig.depth) == ("solve", 1)
        assert rig.result["status"] == "failed"
        assert rig.result["reason"] == "unexpected_error"
        assert rig.runner._runs[rig.conv] == []
        calls = len(rig.strong.requests)
        rig.runner.run(rig.conv, max_steps=10)
        assert len(rig.strong.requests) == calls
        assert rig.state == "apology"

    def test_an_interrupted_converse_turn_is_final_too(self, valid_fsm, monkeypatch):
        rig = make_rig(valid_fsm)

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt()

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(KeyboardInterrupt):
            rig.runner.converse(rig.conv, "hi")
        assert rig.result["status"] == "failed"

    def test_a_failed_result_write_never_masks_the_interrupt(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm)
        monkeypatch.setattr(
            rig.synth, "obtain", lambda *a, **k: (_ for _ in ()).throw(SystemExit(3))
        )
        real_update = rig.api.update_context

        def refuse_failed(conversation_id: str, data: dict[str, Any]) -> Any:
            if data.get("synthesis_result", {}).get("status") == "failed":
                raise RuntimeError(SECRET)
            return real_update(conversation_id, data)

        monkeypatch.setattr(rig.api, "update_context", refuse_failed)
        with pytest.raises(SystemExit) as caught:
            rig.runner.run(rig.conv, max_steps=10)
        assert caught.value.code == 3

    def test_a_push_that_fails(self, valid_fsm, monkeypatch):
        rig = make_rig(valid_fsm)

        def refuse(*args: Any, **kwargs: Any) -> str:
            raise FSMError(SECRET)

        monkeypatch.setattr(rig.api, "push_fsm", refuse)
        rig.runner.run(rig.conv, max_steps=10)

        assert rig.result["reason"] == "push_failed"
        assert SECRET not in repr(rig.data) and SECRET not in repr(rig.events)
        assert rig.names == ["obtained", "failed"]


class TestConverseLeavesTheStateAfterARefusedStart:
    """A start refused before any push writes ``failed`` and changes no depth, so
    the parent used to stay on its silent synthesize state and eat the NEXT user
    message. ``run`` steps on at once; ``converse`` must too (plan 52ce5464 E4)."""

    @staticmethod
    def _refuse(
        kind: str, valid_fsm: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> Rig:
        policy = None
        kw: dict[str, Any] = {}
        if kind == "inputs_missing":
            kw["context"] = {"unrelated": 1}
        elif kind == "compile_failed":
            kw["replies"] = ["no json here"]
            policy = SynthesisPolicy(max_repair_rounds=0)
        elif kind == "approval_denied":
            policy = SynthesisPolicy(require_approval=True)
        rig = make_rig(valid_fsm, policy=policy, **kw)
        if kind == "approval_denied":
            rig.synth.reviewer = lambda request: False
        if kind == "push_failed":

            def refuse(*args: Any, **kwargs: Any) -> Any:
                raise FSMError(SECRET)

            monkeypatch.setattr(rig.api, "push_fsm", refuse)
        return rig

    @pytest.mark.parametrize(
        "kind", ["inputs_missing", "compile_failed", "approval_denied", "push_failed"]
    )
    def test_the_first_turn_returns_the_apology_and_matches_run(
        self, valid_fsm, monkeypatch, kind
    ):
        rig = self._refuse(kind, valid_fsm, monkeypatch)
        reply = rig.runner.converse(rig.conv, "hello")

        control = self._refuse(kind, valid_fsm, monkeypatch)
        control.runner.run(control.conv, max_steps=10)

        assert reply == "Sorry"
        assert (rig.state, rig.depth) == (control.state, control.depth)
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result == control.result and rig.result["reason"] == kind
        assert rig.names == control.names
        with pytest.raises(FSMError, match="ended"):
            rig.runner.converse(rig.conv, "again")

    def test_a_marker_set_by_a_run_never_steps_a_later_converse_turn(self, valid_fsm):
        rig = make_rig(valid_fsm, context={"unrelated": 1})
        rig.runner.run(rig.conv, max_steps=10)
        assert (rig.state, rig.result["reason"]) == ("apology", "inputs_missing")
        assert rig.runner._refused == set()

    def test_ending_the_conversation_drops_the_marker(self, valid_fsm):
        rig = make_rig(valid_fsm)
        rig.runner._refused.add(rig.conv)
        rig.runner.end_conversation(rig.conv)
        assert rig.runner._refused == set()


class TestTraceSink:
    def test_a_non_callable_sink_is_refused(self, valid_fsm):
        rig = make_rig(valid_fsm)
        with pytest.raises(TypeError):
            ProcedureRunner(rig.api, make_host(), rig.synth, trace_sink="not callable")  # type: ignore[arg-type]


def _tied(valid_fsm: dict[str, Any]) -> dict[str, Any]:
    """``ask`` has two edges at one priority: ``answer`` alone reaches ``done``,
    ``answer`` plus ``extra`` also reaches ``done2`` (a tie only when both are set)."""
    data = copy.deepcopy(valid_fsm)
    ask = data["states"]["ask"]
    ask["field_extractions"].append(
        {"field_name": "extra", "field_type": "str", "extraction_instructions": "x"}
    )
    second = copy.deepcopy(ask["transitions"][0])
    second["target_state"] = "done2"
    second["conditions"][0]["requires_context_keys"] = ["answer", "extra"]
    ask["transitions"].append(second)
    data["states"]["done2"] = copy.deepcopy(data["states"]["done"])
    data["states"]["done2"]["id"] = "done2"
    return data


def _abort_at_once(rig: Rig, monkeypatch: pytest.MonkeyPatch, fsm: dict[str, Any]):
    """Push ``fsm``, run the child's first turn by hand (the engine counts it) and
    return a ``step -> None`` hook: the next call traces that turn and, with a
    ``max_turns`` of 1, finds the child's budget spent and aborts it."""
    serve_procedure(rig, monkeypatch, make_procedure(fsm))
    rig.to_solve()
    hook = rig.runner.hook(rig.conv)
    hook(2)
    assert rig.depth == 2
    rig.api.advance(rig.conv)
    return hook


class TestTraceEvents:
    """The events a ``TraceRecorder`` reads: the conversation on every one, a
    ``stepped`` event per counted child step, the stuck state and live outcome
    at an abort, the error class and round codes at a failure. Names, type
    names, state ids, counts and codes only (D-038, D-044)."""

    def test_an_interrupt_names_its_class(self, valid_fsm, monkeypatch):
        rig = make_rig(valid_fsm)

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(KeyboardInterrupt):
            rig.runner.run(rig.conv, max_steps=10)

        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert failed["error"] == "KeyboardInterrupt"

    def test_an_undeclared_name_in_the_child_is_never_listed(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm, extraction={})
        hook = _abort_at_once(rig, monkeypatch, valid_fsm)
        rig.api.update_context(rig.conv, {"answer": "x", "minted": CANARIES[2]})

        hook(3)

        assert rig.steps[-1]["keys"] == "answer,topic"
        assert "minted" not in json.dumps(rig.stream)

    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            (True, "bool"),
            (3, "int"),
            (1.5, "float"),
            ("s", "str"),
            ([1], "list"),
            ({"a": 1}, "dict"),
            (type("S", (str,), {})("sub"), "other"),
            (object(), "other"),
        ],
    )
    def test_a_type_is_named_from_a_closed_table(
        self, valid_fsm, monkeypatch, value, kind
    ):
        rig = make_rig(valid_fsm, extraction={})
        hook = _abort_at_once(rig, monkeypatch, valid_fsm)
        rig.api.update_context(rig.conv, {"answer": value})

        hook(3)

        assert rig.steps[-1]["kinds"] == f"answer:{kind}"
        assert "SubclassOfStr" not in json.dumps(rig.stream)

    @pytest.mark.parametrize(
        ("context", "outcome"),
        [
            ({}, "blocked"),
            ({"answer": "a"}, "deterministic"),
            ({"answer": "a", "extra": "b"}, "ambiguous"),
        ],
    )
    def test_an_abort_names_the_stuck_state_and_the_live_outcome(
        self, valid_fsm, monkeypatch, context, outcome
    ):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=1))
        hook = _abort_at_once(rig, monkeypatch, _tied(valid_fsm))
        rig.api.update_context(rig.conv, context)

        hook(3)

        (aborted,) = [f for n, f in rig.events if n == "aborted"]
        assert aborted["state"] == "ask" and aborted["outcome"] == outcome
        assert aborted["conv"] == rig.conv and aborted["step"] == 3
        # the outcome was read from the child before it was popped, and no abort
        # key is written any more: the engine's budget ended the child
        assert rig.depth == 1 and "procedure_abort" not in rig.data
        assert rig.result["status"] == "aborted"

    def test_an_event_that_cannot_be_read_never_changes_the_run(
        self, valid_fsm, monkeypatch
    ):
        class Broken:
            def evaluate_transitions(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(CANARIES[2])

        monkeypatch.setattr(runner_module, "TransitionEvaluator", Broken)
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))

        rig.runner.run(rig.conv, max_steps=20)

        assert (rig.state, rig.result["reason"]) == ("apology", "budget_spent")
        (aborted,) = [f for n, f in rig.events if n == "aborted"]
        assert "outcome" not in aborted and aborted["conv"] == rig.conv
        assert CANARIES[2] not in json.dumps(rig.stream)

    def test_without_a_sink_nothing_is_read(self, valid_fsm, monkeypatch):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))
        quiet = ProcedureRunner(rig.api, make_host(), rig.synth)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("read without a sink")

        monkeypatch.setattr(runner_module, "TransitionEvaluator", boom)
        quiet.run(rig.conv, max_steps=20)
        assert rig.result["status"] == "aborted"


_STALE = {"status": "ok", "procedure_id": "stale", "version": "1.0", "reason": None}


class TestPushedChildOwnsItsSynthesisResult:
    """A pushed child inherited the parent's ``synthesis_result`` (step 55 of plan
    d454fd62, D-022): its synthesize state's edges read the key, so they routed
    on the stale result before any run wrote one. The drop is on the child's
    initial context only. (The runner itself pushes with
    ``inherit_context=False`` and serves no frame it did not push.)"""

    def _pushed_child(self, valid_fsm: dict[str, Any]) -> Rig:
        rig = make_rig(valid_fsm, context={**CONTEXT, "synthesis_result": _STALE})
        rig.api.push_fsm(rig.conv, make_host())  # inherit_context defaults to True
        return rig

    def test_the_child_waits_for_its_own_result_instead_of_routing_on_the_stale_one(
        self, valid_fsm
    ):
        rig = self._pushed_child(valid_fsm)
        assert rig.depth == 2
        rig.to_solve()  # the child's start -> solve

        rig.api.advance(rig.conv)

        # no edge passes before a run wrote a result: the stale `ok` would have
        # taken the child on to `booked` without any procedure ever running
        assert rig.state == "solve"
        assert rig.names == []

    def test_the_child_does_not_see_the_inherited_result(self, valid_fsm):
        rig = self._pushed_child(valid_fsm)
        assert "synthesis_result" not in rig.data
        assert rig.data["topic"] == "tables"  # the rest is inherited as before

    def test_an_explicitly_passed_result_is_kept(self, valid_fsm):
        rig = make_rig(valid_fsm, context={**CONTEXT, "synthesis_result": _STALE})
        handed = {**_STALE, "procedure_id": "handed"}
        rig.api.push_fsm(
            rig.conv, make_host(), context_to_pass={"synthesis_result": handed}
        )
        assert rig.data["synthesis_result"] == handed

    def test_a_result_key_the_child_does_not_own_is_still_inherited(self, valid_fsm):
        rig = make_rig(valid_fsm, context={**CONTEXT, "synthesis_result": _STALE})
        other = json.loads(
            json.dumps(make_host(result_key="other_result")).replace(
                "synthesis_result", "other_result"
            )
        )
        rig.api.push_fsm(rig.conv, other)
        assert rig.data["synthesis_result"] == _STALE

    def test_the_parent_keeps_its_own_result_through_a_push_and_a_pop(self, valid_fsm):
        rig = self._pushed_child(valid_fsm)
        rig.api.pop_fsm(rig.conv, context_to_return={"from_child": 1})
        assert rig.depth == 1
        assert rig.data["synthesis_result"] == _STALE
        assert rig.data["from_child"] == 1
