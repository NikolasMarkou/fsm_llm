"""The shared edge rule of the result-key kinds (`fsm_llm.core.kind_call`).

Plan baa37f40, step 6 (K1.0, D-014): the loader rule of the `synthesize` kind (the edges read the
result key or a handler-only key, run on four sample results by the real evaluator, host keys do
not collide with a result key) is one body, `check_result_edges`, that the `synthesize` kind calls
in its own wording and the `call` kind of step 7 calls in its own. The synthesize wording is pinned
byte for byte by the exact-message rows of `test_state_kinds`; these rows pin what the lift added:
the four wording parameters, the lazy `owned` callable and the rule order.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fsm_llm.core.definitions import logic_referenced_keys
from fsm_llm.core.kind_call import check_result_edges
from fsm_llm.core.state_kinds import SYNTHESIZE_SAMPLES
from fsm_llm.core.transition_evaluator import TransitionEvaluator

RESULT_KEY = "call_result"
_STATUS_IN_FAILED = {
    "in": [{"var": f"{RESULT_KEY}.status"}, ["failed", "aborted"]],
}


def _edge(
    target: str,
    logic: dict[str, Any] | None,
    requires: tuple[str, ...] = (),
    priority: int = 100,
) -> Any:
    from fsm_llm.core.definitions import Transition, TransitionCondition

    conditions = (
        []
        if logic is None
        else [
            TransitionCondition(
                description="c", requires_context_keys=list(requires), logic=logic
            )
        ]
    )
    return Transition(
        target_state=target,
        description="go",
        conditions=conditions,
        priority=priority,
    )


def _ok_edge() -> Any:
    return _edge(
        "done",
        {"==": [{"var": f"{RESULT_KEY}.status"}, "ok"]},
        (RESULT_KEY,),
    )


def _failure_edge(priority: int = 100) -> Any:
    return _edge("apology", _STATUS_IN_FAILED, (RESULT_KEY,), priority)


def _check(
    transitions: list[Any],
    *,
    inputs: tuple[str, ...] = (),
    returns: tuple[str, ...] = (),
    handler_only: tuple[str, ...] = (),
    owned: Any = None,
    **words: str,
) -> str | None:
    state = SimpleNamespace(transitions=transitions)
    definition = SimpleNamespace(handler_only_keys=list(handler_only), states={})
    params: dict[str, Any] = {
        "kind": "call",
        "producer": "the child run",
        "outcome": "the call",
        "return_entry": "returns",
    }
    params.update(words)
    return check_result_edges(
        definition,
        state,
        TransitionEvaluator(),
        logic_referenced_keys,
        result_key=RESULT_KEY,
        on_failure="apology",
        inputs=inputs,
        return_names=returns,
        samples=SYNTHESIZE_SAMPLES,
        owned=owned if owned is not None else dict,
        **params,
    )


class TestTheWordingIsTheCallers:
    def test_a_valid_pair_of_edges_is_accepted(self):
        assert _check([_ok_edge(), _failure_edge()]) is None

    def test_an_edge_that_ignores_the_result_names_the_kind_and_the_producer(self):
        message = _check([_edge("done", None), _failure_edge()])
        assert message == (
            "the transition to 'done' must read the call result key 'call_result' "
            "(a condition that requires it or reads it in its logic), or it fires "
            "before the child run has produced a result"
        )

    def test_an_edge_that_passes_on_running_names_the_producer(self):
        edge = _edge(
            "done",
            {"!=": [{"var": f"{RESULT_KEY}.status"}, "ok"]},
            (RESULT_KEY,),
        )
        message = _check([edge, _failure_edge()])
        assert message == (
            "the transition to 'done' fires before the child run has produced a "
            "result (it passes when 'call_result' is running)"
        )

    def test_a_missing_failure_edge_names_the_outcome(self):
        message = _check([_ok_edge()])
        assert message is not None
        assert message.startswith(
            "no transition to 'apology' fires when the call failed or aborted "
            "(it does not pass when 'call_result' is failed, "
        )

    def test_a_return_name_that_collides_names_the_entry_label(self):
        message = _check(
            [_ok_edge(), _failure_edge()],
            returns=("flag",),
            handler_only=("flag",),
        )
        assert (
            message
            == "call returns entry 'flag' collides with a handler_only_keys entry"
        )

    def test_an_input_that_collides_with_an_owned_key_names_its_label(self):
        message = _check(
            [_ok_edge(), _failure_edge()],
            inputs=("other",),
            owned=lambda: {"other": "result_key of a call state"},
        )
        assert message == (
            "call inputs entry 'other' collides with result_key of a call state"
        )

    def test_the_synthesize_words_reproduce_the_old_text(self):
        message = _check(
            [_edge("done", None), _failure_edge()],
            kind="synthesize",
            producer="the runner",
            outcome="the procedure",
            return_entry="return_keys",
        )
        assert message == (
            "the transition to 'done' must read the synthesize result key "
            "'call_result' (a condition that requires it or reads it in its logic), "
            "or it fires before the runner has produced a result"
        )


class TestTheOwnedKeysAreLazy:
    def test_an_edge_refusal_never_asks_for_the_owned_keys(self):
        asked: list[bool] = []

        def owned() -> dict[str, str]:
            asked.append(True)
            return {}

        assert _check([_edge("done", None), _failure_edge()], owned=owned) is not None
        assert asked == []

    def test_a_clean_edge_set_asks_once(self):
        asked: list[bool] = []

        def owned() -> dict[str, str]:
            asked.append(True)
            return {}

        assert _check([_ok_edge(), _failure_edge()], owned=owned) is None
        assert asked == [True]

    def test_a_handler_only_return_name_is_not_a_return_collision_for_an_input(self):
        """The handler-only exemption belongs to the return entry only, as before."""
        assert (
            _check(
                [_ok_edge(), _failure_edge()],
                inputs=("flag",),
                handler_only=("flag",),
            )
            is None
        )


def _status_in(*statuses: str) -> dict[str, Any]:
    return {"in": [{"var": f"{RESULT_KEY}.status"}, list(statuses)]}


class TestTheFailureEdgeMustWin:
    """Plan 8525c797 step 3 (D-004): passing is not enough, the failure edge is the
    unique winner of the real ranking at the failed and the aborted sample."""

    def test_an_earlier_edge_that_also_passes_the_failure_is_refused(self):
        rival = _edge(
            "elsewhere", _status_in("failed", "aborted", "ok"), (RESULT_KEY,), 50
        )
        message = _check([rival, _failure_edge(100)])
        assert message is not None
        assert message.startswith(
            "the transition to 'elsewhere' wins over the failure edge to 'apology' "
            "when the call failed or aborted (the lowest priority number wins and "
            "the failure edge must win outright; it wins when 'call_result' is "
            "failed, "
        )

    def test_an_earlier_edge_that_passes_only_the_aborted_sample_is_refused(self):
        rival = _edge("elsewhere", _status_in("aborted"), (RESULT_KEY,), 1)
        message = _check([_ok_edge(), rival, _failure_edge(100)])
        assert message is not None
        assert "the transition to 'elsewhere' wins over" in message
        assert "it wins when 'call_result' is aborted, " in message

    def test_a_tie_at_the_lowest_priority_is_refused_and_names_the_other_edge(self):
        rival = _edge("elsewhere", _status_in("failed", "aborted"), (RESULT_KEY,))
        message = _check([rival, _failure_edge()])
        assert message is not None
        assert message.startswith(
            "the transition to 'apology' ties with the transition to 'elsewhere' "
            "when the call failed or aborted (a tie goes to the classifier model; "
            "the failure edge must win outright; they tie when 'call_result' is "
            "failed, "
        )

    def test_the_failure_edge_with_the_lowest_number_wins_over_a_passing_rival(self):
        rival = _edge("elsewhere", _status_in("failed", "aborted"), (RESULT_KEY,), 50)
        assert _check([_failure_edge(1), rival]) is None

    def test_two_edges_to_the_failure_state_are_one_candidate_not_a_tie(self):
        twin = _edge("apology", _status_in("failed"), (RESULT_KEY,))
        assert _check([_ok_edge(), _failure_edge(), twin]) is None

    def test_the_documented_retry_shape_loads(self):
        """The authoring rule: the gate sits on the retry edge, at a LOWER number;
        the failure edge reads the result key only and has the higher number."""
        retry = _edge(
            "again",
            {"and": [_status_in("failed"), {"<": [{"var": "attempts"}, 3]}]},
            (RESULT_KEY, "attempts"),
            1,
        )
        assert _check([_ok_edge(), retry, _failure_edge(2)]) is None

    def test_a_failure_edge_gated_on_another_key_stays_refused_as_a_known_limit(self):
        """The sampler fills the other keys with "x", so `attempts > 2` fails there
        (D-005: no sampler change). The refusal is the old, unchanged wording."""
        gated = _edge(
            "apology",
            {"and": [_status_in("failed", "aborted"), {">": [{"var": "attempts"}, 2]}]},
            (RESULT_KEY, "attempts"),
            2,
        )
        message = _check([_ok_edge(), gated])
        assert message is not None
        assert message.startswith("no transition to 'apology' fires when the call")

    def test_a_retry_and_failure_pair_gated_on_a_counter_stays_refused(self):
        retry = _edge(
            "again",
            {"and": [_status_in("failed"), {"<": [{"var": "attempts"}, 3]}]},
            (RESULT_KEY, "attempts"),
            1,
        )
        fail = _edge(
            "apology",
            {"and": [_status_in("failed"), {">=": [{"var": "attempts"}, 3]}]},
            (RESULT_KEY, "attempts"),
            2,
        )
        message = _check([_ok_edge(), retry, fail])
        assert message is not None
        assert message.startswith("no transition to 'apology' fires when the call")

    def test_the_synthesize_wording_of_the_new_refusal_names_the_procedure(self):
        rival = _edge("elsewhere", _status_in("failed", "aborted"), (RESULT_KEY,), 5)
        message = _check(
            [rival, _failure_edge()],
            kind="synthesize",
            producer="the runner",
            outcome="the procedure",
            return_entry="return_keys",
        )
        assert message is not None
        assert message.startswith(
            "the transition to 'elsewhere' wins over the failure edge to "
            "'apology' when the procedure failed or aborted"
        )


def _random_transitions(rng: Any) -> list[Any]:
    from fsm_llm.core.definitions import Transition, TransitionCondition

    transitions = []
    for _ in range(rng.randint(1, 6)):
        gate = rng.choice(["a", "b", None])
        conditions = (
            []
            if gate is None
            else [
                TransitionCondition(description="c", logic={"==": [{"var": gate}, 1]})
            ]
        )
        transitions.append(
            Transition(
                target_state=rng.choice("wxyz"),
                description="go",
                priority=rng.randint(1, 4),
                conditions=conditions,
            )
        )
    return transitions


class TestWinningTargetsIsTheRuntimeRanking:
    """`winning_targets` and `evaluate_transitions` share one ranking helper; the
    parity table (seeded) makes a copy with another comparison fail."""

    def test_it_equals_what_evaluate_transitions_decides_over_random_edges(self):
        import random

        from fsm_llm.core.definitions import (
            FSMContext,
            State,
            TransitionEvaluationResult,
        )

        evaluator = TransitionEvaluator()
        rng = random.Random(8525)
        seen: set[str] = set()
        for _ in range(400):
            transitions = _random_transitions(rng)
            data = {key: rng.choice([0, 1]) for key in "ab"}
            state = State(id="s", description="d", purpose="p", transitions=transitions)
            decided = evaluator.evaluate_transitions(state, FSMContext(data=data))
            winners = evaluator.winning_targets(transitions, data)
            seen.add(decided.result_type.value)
            if decided.result_type is TransitionEvaluationResult.BLOCKED:
                assert winners == []
            elif decided.result_type is TransitionEvaluationResult.DETERMINISTIC:
                assert winners == [decided.deterministic_transition]
            else:
                assert winners == [o.target_state for o in decided.available_options]
        assert len(seen) == 3, "the seeded table must reach all three outcomes"

    def test_a_lower_number_wins_a_tie_collapses_by_target_in_definition_order(self):
        evaluator = TransitionEvaluator()
        edges = [
            _edge("late", None, priority=9),
            _edge("b", None, priority=2),
            _edge("a", None, priority=2),
            _edge("b", None, priority=2),
        ]
        assert evaluator.winning_targets(edges, {}) == ["b", "a"]
        assert evaluator.winning_targets([], {}) == []
