"""
Tests for the best-effort witness of an edge outside the G6a fragment (plan
51fde95b D-027, part 4b): it fills ``edge_witnesses(best_effort=True)`` for
``happy_path`` and never decides anything G6a reports.
"""

from typing import Any

import pytest

from fsm_llm.core.definitions import FSMContext, FSMDefinition
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.synthesis import dryrun
from fsm_llm.synthesis.dryrun import analyze_dryrun, edge_witnesses

from .test_dryrun import _edge, _field, _fsm, _fuzz_case, _state, _var
from .test_happy_path import _fsm as _plain_fsm
from .test_happy_path import _go, _simulate, _step

SUM = {">": [{"+": [_var("a"), _var("b")]}, 10]}
AB = {">": [_var("a"), _var("b")]}


def _ask(logic: Any, kind: str = "int", names=("a", "b"), **extra: Any):
    return _fsm(
        _state(
            "ask",
            [_edge("done", logic, requires=names)],
            field_extractions=[_field(n, kind) for n in names],
            **extra,
        ),
        _state("done"),
    )


def _opens(definition: FSMDefinition, found: dict[str, Any] | None) -> bool:
    state = definition.states["ask"]
    result = TransitionEvaluator().evaluate_transitions(
        state, FSMContext(data=found or {})
    )
    return result.deterministic_transition == "done"


class TestTheWitnessOpensTheEdge:
    @pytest.mark.parametrize("logic", [SUM, AB], ids=["sum", "key-against-key"])
    def test_an_edge_outside_the_fragment_gets_a_witness_the_evaluator_accepts(
        self, logic
    ):
        definition = _ask(logic)
        assert edge_witnesses(definition)["ask"] == {0: None}  # G6a stays silent
        found = edge_witnesses(definition, best_effort=True)["ask"][0]
        assert found is not None and set(found) == {"a", "b"}
        assert _opens(definition, found)

    def test_a_typed_key_gets_a_number(self):
        found = edge_witnesses(_ask(AB), best_effort=True)["ask"][0]
        assert all(isinstance(v, int | float) for v in found.values())

    def test_a_key_another_state_types_otherwise_is_not_narrowed(self):
        # the narrowing of G6a: `a` is also read by `done`, so it may be any value
        definition = _fsm(
            _state(
                "ask",
                [_edge("done", AB, requires=("a", "b"))],
                field_extractions=[_field("a", "int"), _field("b", "int")],
            ),
            _state("done", field_extractions=[_field("a", "str")]),
        )
        found = edge_witnesses(definition, best_effort=True)["ask"][0]
        assert found is not None and _opens(definition, found)

    def test_the_edge_is_judged_by_the_whole_state(self):
        early = _edge("ask", {"==": [{"var": "a.b"}, None]}, priority=0)
        definition = _fsm(
            _state(
                "ask",
                [early, _edge("done", SUM, requires=("a", "b"), priority=1)],
                field_extractions=[_field("a", "int"), _field("b", "int")],
            ),
            _state("done"),
        )
        assert edge_witnesses(definition, best_effort=True)["ask"] == {0: None, 1: None}


class TestWhatTheFallbackDoesNotDo:
    def test_a_dotted_read_keeps_the_filler(self):
        gate = {"==": [{"var": "user.tier"}, "gold"]}
        definition = _fsm(
            _state("ask", [_edge("done", gate)], field_extractions=[_field("user")]),
            _state("done"),
        )
        assert edge_witnesses(definition, best_effort=True)["ask"] == {0: None}

    def test_a_computed_key_name_is_unknown(self):
        gate = {"==": [{"var": {"cat": ["a", "b"]}}, 1]}
        definition = _fsm(_state("ask", [_edge("done", gate)]), _state("done"))
        assert edge_witnesses(definition, best_effort=True)["ask"] == {0: None}

    def test_a_dead_edge_stays_absent(self):
        dead = {"==": [_var("ok"), "yes"]}  # a bool key can never equal "yes"
        definition = _ask(dead, "bool", ("ok",))
        assert edge_witnesses(definition, best_effort=True)["ask"] == {}

    def test_the_search_stops_at_the_cap(self, monkeypatch):
        calls = []
        real = TransitionEvaluator.evaluate_transitions

        def spy(self, state, context):
            calls.append(1)
            return real(self, state, context)

        monkeypatch.setattr(TransitionEvaluator, "evaluate_transitions", spy)
        # three untyped keys: far more assignments than the cap; one is never true
        never = {
            "and": [
                {"==": [1, 2]},
                {">": [{"+": [_var("a"), _var("b"), _var("c")]}, 1]},
            ]
        }
        definition = _fsm(
            _state("ask", [_edge("done", never, requires=("a", "b", "c"))]),
            _state("done"),
        )
        assert edge_witnesses(definition, best_effort=True)["ask"] == {0: None}
        assert len(calls) == dryrun.MAX_ASSIGNMENTS

    def test_a_pool_of_many_literals_stays_bounded(self, monkeypatch):
        sizes = []
        real = dryrun._pool

        def spy(narrow, lits):
            sizes.append(len(lits))
            return real(narrow, lits)

        monkeypatch.setattr(dryrun, "_pool", spy)
        many = {"or": [{">": [{"+": [_var("a"), _var("b")]}, n]} for n in range(300)]}
        edge_witnesses(_ask(many), best_effort=True)
        assert sizes and max(sizes) <= 300 + 2 * dryrun.FALLBACK_LITERALS**2

    def test_a_value_only_a_difference_of_literals_reaches_is_tried(self):
        gate = {
            "and": [{"==": [_var("a"), 3]}, {"==": [{"+": [_var("a"), _var("b")]}, 37]}]
        }
        definition = _ask(gate)
        found = edge_witnesses(definition, best_effort=True)["ask"][0]
        assert found == {"a": 3, "b": 34}

    def test_g6a_never_reads_the_fallback(self, monkeypatch):
        def boom(*args: Any, **kw: Any):
            raise AssertionError("G6a consulted the fallback")

        monkeypatch.setattr(dryrun, "fallback_witness", boom)
        bools = _ask(SUM, "bool")  # unknown to G6a, truly unfireable: no finding
        assert analyze_dryrun(bools) == []
        assert analyze_dryrun(_ask(AB)) == []
        for seed in range(40):  # the fuzz set: G6a never reaches the fallback
            try:
                definition = _fuzz_case(seed)[0]
            except ValueError:  # a random case the loader refuses
                continue
            analyze_dryrun(definition)


class TestTheHappyPathUsesIt:
    def test_a_compile_gated_on_a_sum_passes_g6b(self):
        fsm = _plain_fsm(
            "ask",
            ask=_step(
                "ask", [("a", "int"), ("b", "int")], [_go("done", ["a", "b"], SUM)]
            ),
            done=_step("done"),
        )
        report, _ = _simulate(fsm, ("a", "b"))
        assert report.passed, report.repair_text()
        assert report.happy["k"] == 2

    def test_a_compile_gated_on_one_key_against_another_passes_g6b(self):
        fsm = _plain_fsm(
            "ask",
            ask=_step(
                "ask", [("a", "int"), ("b", "int")], [_go("done", ["a", "b"], AB)]
            ),
            done=_step("done"),
        )
        report, _ = _simulate(fsm, ("a", "b"))
        assert report.passed, report.repair_text()
