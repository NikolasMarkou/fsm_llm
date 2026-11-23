"""The rows of ``test_runner.py`` that carried an invariant of the ENGINE or of the trace, ported
to ``SynthesizeExecutor`` when ``ProcedureRunner`` was deleted (plan baa37f40, step 22, D-041).

What they pin, each on a real ``API``:

- the engine bounds the child, not the executor: the child frame carries ``max_steps=
  policy.max_turns`` and the host none, an ended child is never aborted on its last budgeted
  turn, a spent HOST budget still raises and is never taken for the child's (5a6a1f80/D-012);
- a refused start is the same turn's failure edge, in ``converse`` as in ``run``;
- a result that is settled is never obtained twice, and a conversation ended under a running
  child counts no run;
- the trace is value free: a type is named from a closed table, an undeclared name is never
  listed, an abort names the stuck state and the live outcome (read before the pop, 2cefeb61/
  D-044), an event that cannot be read never changes the run, a run without a sink reads nothing.
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
from fsm_llm.engine.kinds.requests import ChildView
from fsm_llm.synthesis import FSMSynthesizer, SynthesisPolicy, SynthesizeExecutor
from fsm_llm.synthesis import executor as executor_module
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


class TestTheEngineBoundsTheChild:
    def test_the_child_frame_carries_the_policy_budget_and_the_host_none(
        self, valid_fsm
    ):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))
        rig.converse("hello")  # the arrival obtains and settle pushes the child
        assert rig.depth == 2
        manager, child = rig.api.fsm_manager, rig.api.get_sub_conversation_id(rig.conv)
        assert child != rig.conv
        assert (manager.turn_budget_steps(child), manager.turn_budget_spent(child)) == (
            0,
            None,
        )
        assert manager.turn_budget_steps(rig.conv) is None  # the host has no limit

        rig.api.advance(rig.conv)  # the child's first turn, counted by the engine

        assert manager.turn_budget_steps(child) == 1
        assert manager.turn_budget_spent(child) is None and rig.depth == 2

        rig.api.advance(
            rig.conv
        )  # the second spends the budget: settle at its end pops the child

        assert rig.depth == 1 and rig.result["reason"] == "budget_spent"
        assert rig.result["status"] == "aborted"

    def test_an_ended_child_is_never_aborted_even_on_its_last_budgeted_turn(
        self, valid_fsm
    ):
        rig = make_rig(valid_fsm, policy=SynthesisPolicy(max_turns=1))

        rig.converse("hello")
        rig.converse("forty-two")  # the one turn the budget allows

        assert [f["step"] for f in rig.steps] == [1]  # counted, and not over
        assert rig.result["status"] == "ok" and rig.data["answer"] == "42"
        assert "aborted" not in rig.names

    def test_a_spent_host_budget_still_raises_and_is_not_the_childs(self, valid_fsm):
        host = make_host()
        cheap = ScriptedRunnerLLM({"answer": "42"}, dict(REPLIES))
        synth = FSMSynthesizer(ScriptedLLM([json.dumps(valid_fsm)]))
        # the API default reaches the host's own frame; the child's explicit limit wins
        api = API.from_definition(
            definition=host,
            llm_interface=cheap,
            max_steps=1,
            executors=[SynthesizeExecutor(synth)],
        )
        conv, _ = api.start_conversation(CONTEXT)
        api.converse("hello", conv)  # the host's one turn; the child is pushed
        assert api.get_stack_depth(conv) == 2

        with pytest.raises(RunBudgetExceededError) as spent:
            # the child ends, then the host steps on and finds its own budget spent
            api.converse("forty-two", conv)
            api.advance(conv)

        assert (spent.value.budget, spent.value.limit) == ("steps", 1)
        assert api.get_stack_depth(conv) == 1
        assert api.get_data(conv)["synthesis_result"]["status"] == "ok"
        assert api.get_data(conv)["answer"] == "42"


class TestASettledResultIsFinal:
    def test_a_settled_result_is_never_obtained_twice(self, valid_fsm):
        rig = make_rig(valid_fsm, replies=[LLMResponseError(SECRET)])
        rig.to_solve()

        rig.api.advance(rig.conv)  # the failure edge of the settled result
        assert rig.state == "apology" and rig.result["reason"] == "provider_error"

        rig.run(5)  # nothing obtains again
        assert len(rig.strong.requests) == 1

    def test_ending_the_conversation_with_a_child_on_top_counts_no_run(
        self, valid_fsm, tmp_path
    ):
        rig = make_rig(valid_fsm, tmp_path)
        rig.converse("hello")
        assert rig.depth == 2

        rig.api.end_conversation(rig.conv)  # straight on the API: no raise

        with pytest.raises(ValueError):
            rig.api.get_stack_depth(rig.conv)
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 0


class TestARefusedStartIsTheSameTurnsFailureEdge:
    """A start refused before any push writes ``failed`` and the SAME turn takes ``on_failure``,
    so the parent never sits on its silent synthesize state and eats the next user message; in
    ``converse`` as in ``run`` (plan 52ce5464 E4, now the executor's rule, D-038 (5))."""

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

            async def refuse(*args: Any, **kwargs: Any) -> Any:
                raise FSMError(SECRET)

            monkeypatch.setattr(rig.api, "apush_fsm", refuse)
        return rig

    @pytest.mark.parametrize(
        "kind", ["inputs_missing", "compile_failed", "approval_denied", "push_failed"]
    )
    def test_the_first_turn_returns_the_apology_and_matches_run(
        self, valid_fsm, monkeypatch, kind
    ):
        rig = self._refuse(kind, valid_fsm, monkeypatch)
        reply = rig.converse("hello")

        control = self._refuse(kind, valid_fsm, monkeypatch)
        control.run(10)

        assert reply == "Sorry"
        assert (rig.state, rig.depth) == (control.state, control.depth)
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result == control.result and rig.result["reason"] == kind
        assert rig.names == control.names
        with pytest.raises(FSMError, match="ended"):
            rig.converse("again")


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


def _child_ran_once(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, fsm: dict[str, Any]
) -> None:
    """Serve ``fsm`` as the procedure, push it and run the child's first turn by hand: with a
    ``max_turns`` of 1 that turn spent its budget, and the next settle (the one after the turn,
    at its end) pops it as aborted."""
    serve_procedure(rig, monkeypatch, make_procedure(fsm))
    rig.converse("hello")  # the arrival obtains and settle pushes the child
    assert rig.depth == 2


class TestTheTraceIsValueFree:
    """The events a ``TraceRecorder`` reads: names, type names, state ids, counts and codes only
    (D-038, D-044 of plan 2cefeb61)."""

    def test_an_undeclared_name_in_the_child_is_never_listed(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=3))
        _child_ran_once(rig, monkeypatch, valid_fsm)
        rig.api.update_context(rig.conv, {"answer": "x", "minted": CANARIES[2]})

        rig.api.advance(rig.conv)  # a turn the engine counts: one `stepped` event

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
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=3))
        _child_ran_once(rig, monkeypatch, valid_fsm)
        rig.api.update_context(rig.conv, {"answer": value})

        rig.api.advance(rig.conv)

        assert rig.steps[-1]["kinds"] == f"answer:{kind}"
        assert "SubclassOfStr" not in json.dumps(rig.stream)

    def test_an_abort_names_the_stuck_state_and_the_live_outcome(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=1))
        _child_ran_once(rig, monkeypatch, _tied(valid_fsm))

        rig.api.advance(
            rig.conv
        )  # the one budgeted turn; settle at its end aborts the child

        (aborted,) = [f for n, f in rig.events if n == "aborted"]
        assert aborted["state"] == "ask" and aborted["outcome"] == "blocked"
        assert aborted["conv"] == rig.conv
        # the outcome was read from the child before it was popped, and no abort
        # key is written any more: the engine's budget ended the child
        assert rig.depth == 1 and "procedure_abort" not in rig.data
        assert rig.result["status"] == "aborted"

    @pytest.mark.parametrize(
        ("context", "outcome"),
        [
            ({}, "blocked"),
            ({"answer": "a"}, "deterministic"),
            ({"answer": "a", "extra": "b"}, "ambiguous"),
        ],
    )
    def test_the_live_outcome_is_one_of_the_three_names(
        self, valid_fsm, context, outcome
    ):
        """The engine never leaves a child stuck on a passing edge, so the three names are read
        off ``_view`` on a hand-made view of the child (the abort row above is the live one)."""
        rig = make_rig(valid_fsm, extraction={})
        procedure = make_procedure(_tied(valid_fsm))
        config = FSMDefinition(**make_host()).states["solve"].synthesize
        assert config is not None
        run = executor_module._Run(procedure, config, converse=False)
        child = ChildView(
            "c", None, "ask", {"topic": "t", **context}, False, "steps", 1
        )

        seen = rig.executor._view(run, child, outcome=True)

        assert seen["state"] == "ask" and seen["outcome"] == outcome

    def test_an_event_that_cannot_be_read_never_changes_the_run(
        self, valid_fsm, monkeypatch
    ):
        class Broken:
            def evaluate_transitions(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(CANARIES[2])

        monkeypatch.setattr(executor_module, "TransitionEvaluator", Broken)
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))

        rig.run(20)

        assert (rig.state, rig.result["reason"]) == ("apology", "budget_spent")
        (aborted,) = [f for n, f in rig.events if n == "aborted"]
        assert "outcome" not in aborted and aborted["conv"] == rig.conv
        assert CANARIES[2] not in json.dumps(rig.stream)

    def test_without_a_sink_nothing_is_read(self, valid_fsm, monkeypatch):
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("read without a sink")

        monkeypatch.setattr(executor_module, "TransitionEvaluator", boom)
        quiet = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))
        quiet.executor._sink = None  # the same executor, with no sink

        quiet.run(20)

        assert quiet.result["status"] == "aborted"
        assert quiet.stream == []
