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
    target: str, logic: dict[str, Any] | None, requires: tuple[str, ...] = ()
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
    return Transition(target_state=target, description="go", conditions=conditions)


def _ok_edge() -> Any:
    return _edge(
        "done",
        {"==": [{"var": f"{RESULT_KEY}.status"}, "ok"]},
        (RESULT_KEY,),
    )


def _failure_edge() -> Any:
    return _edge("apology", _STATUS_IN_FAILED, (RESULT_KEY,))


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
