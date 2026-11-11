"""
Tests for ``fsm_llm.synthesis.diagnose``: the fixed rule table over run traces.

Every rule has a hand trace and its exclusions; an outage or an unknown reason
is never a procedure fault; the table order is checked by swapping two rules.
"""

import dataclasses
import random
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm.synthesis import Diagnosis, RunTrace, SynthesisPolicy, diagnose
from fsm_llm.synthesis.constants import (
    DIAGNOSIS_CODES,
    PROCEDURE_FAULT_REASONS,
    REASON_BUDGET_SPENT,
    REASON_RETURN_KEYS_MISSING,
    REASONS,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_OK,
)

# the package attribute `diagnose` is the function, so take the module by name
diagnose_module = sys.modules["fsm_llm.synthesis.diagnose"]

D = "a" * 64
POLICY = SynthesisPolicy()
NPF = "not-procedure-fault"


def _state(sid: str, **extra: Any) -> dict[str, Any]:
    return {"id": sid, "description": sid, "purpose": sid, **extra}


def _edge(target: str, key: str, logic: Any, priority: int = 100) -> dict[str, Any]:
    return {
        "target_state": target,
        "description": f"to {target}",
        "priority": priority,
        "conditions": [
            {"description": "gate", "requires_context_keys": [key], "logic": logic}
        ],
    }


def _field(name: str, kind: str) -> dict[str, Any]:
    return {"field_name": name, "field_type": kind, "extraction_instructions": "ask"}


DEFINITION = FSMDefinition(
    **{
        "name": "t",
        "description": "t",
        "initial_state": "hub",
        "states": {
            "hub": _state(
                "hub",
                transitions=[
                    _edge(s, "go", {"==": [{"var": "go"}, s]})
                    for s in ("ask", "rate", "both", "pick")
                ],
            ),
            "ask": _state(
                "ask",
                field_extractions=[_field("age", "int"), _field("anything", "any")],
                transitions=[_edge("done", "age", {">=": [{"var": "age"}, 18]})],
            ),
            "rate": _state(
                "rate",
                field_extractions=[_field("score", "float")],
                transitions=[_edge("done", "score", {">": [{"var": "score"}, 0.5]})],
            ),
            "both": _state(
                "both",
                field_extractions=[_field("age", "int"), _field("nick", "str")],
                transitions=[
                    {
                        **_edge("done", "age", {">=": [{"var": "age"}, 18]}),
                        "conditions": [
                            {
                                "description": "gate",
                                "requires_context_keys": ["age", "nick"],
                                "logic": {"==": [{"var": "nick"}, "x"]},
                            }
                        ],
                    }
                ],
            ),
            "pick": _state(
                "pick",
                transitions=[
                    _edge("done", "age", {"==": [1, 1]}),
                    _edge("rate", "age", {"==": [1, 1]}),
                ],
            ),
            "done": _state("done"),
        },
    }
)


def _t(**kw: Any) -> RunTrace:
    """An aborted run stuck on ``ask``, blocked, no key yet."""
    base: dict[str, Any] = {
        "digest": D,
        "status": STATUS_ABORTED,
        "reason": REASON_BUDGET_SPENT,
        "stuck": "ask",
        "outcome": "blocked",
        "states": ("ask", "ask"),
        "steps": 2,
    }
    return RunTrace(**{**base, **kw})


def _codes(traces: list[RunTrace], **kw: Any) -> list[tuple[str, str]]:
    policy = kw.pop("policy", POLICY)
    return [(d.code, d.state) for d in diagnose(traces, DEFINITION, policy, **kw)]


class TestNotProcedureFault:
    def test_an_ok_run_is_set_aside(self):
        got = diagnose([_t(status=STATUS_OK, reason=None)], DEFINITION, POLICY)
        assert got == (Diagnosis(NPF, "", (), 1, 1.0),)

    @pytest.mark.parametrize(
        "reason",  # literals, not PROCEDURE_FAULT_REASONS: a mutation of it must show
        [r for r in REASONS if r not in ("budget_spent", "return_keys_missing")],
    )
    def test_every_infrastructure_reason_is_never_a_fault(self, reason):
        """The trace would match extraction-miss on its other fields."""
        for status in (STATUS_ABORTED, STATUS_FAILED):
            got = diagnose([_t(status=status, reason=reason)], DEFINITION, POLICY)
            assert [d.code for d in got] == [NPF]

    @pytest.mark.parametrize("reason", ["error", "check_failed", "future_reason", None])
    def test_an_unknown_reason_fails_closed(self, reason):
        got = diagnose([_t(reason=reason)], DEFINITION, POLICY)
        assert [d.code for d in got] == [NPF]

    def test_the_whitelist_is_the_two_budget_and_return_reasons(self):
        assert set(PROCEDURE_FAULT_REASONS) == {
            REASON_BUDGET_SPENT,
            REASON_RETURN_KEYS_MISSING,
        }
        assert set(PROCEDURE_FAULT_REASONS) <= set(REASONS)

    def test_a_mixed_batch_counts_each_side(self):
        got = diagnose(
            [_t(), _t(reason="provider_error"), _t(status=STATUS_OK, reason=None)],
            DEFINITION,
            POLICY,
        )
        by_code = {d.code: d for d in got}
        assert by_code[NPF].runs == 2 and by_code[NPF].share == 0.667
        assert by_code["extraction-miss"].runs == 1


class TestRules:
    def test_extraction_miss_names_the_absent_gate_keys(self):
        got = diagnose([_t(keys=("anything",))], DEFINITION, POLICY)
        assert got == (Diagnosis("extraction-miss", "ask", ("age",), 1, 1.0),)

    def test_extraction_miss_needs_a_blocked_stay(self):
        assert _codes([_t(outcome="deterministic", states=("ask", "rate"))]) == [
            ("path-too-long", "ask")
        ]
        assert _codes([_t(outcome=None)]) == []

    def test_type_mismatch_compares_to_the_declared_field_type(self):
        got = diagnose([_t(keys=("age",), kinds=(("age", "str"),))], DEFINITION, POLICY)
        assert got == (Diagnosis("type-mismatch", "ask", ("age",), 1, 1.0),)

    def test_an_int_for_a_float_field_is_no_mismatch(self):
        t = _t(stuck="rate", keys=("score",), kinds=(("score", "int"),))
        assert _codes([t]) == [("value-gate-unmet", "rate")]
        t = _t(stuck="rate", keys=("score",), kinds=(("score", "str"),))
        assert _codes([t]) == [("type-mismatch", "rate")]

    def test_a_key_with_no_declared_type_is_no_mismatch(self):
        t = _t(stuck="hub", keys=("go",), kinds=(("go", "int"),))
        assert _codes([t]) == [("value-gate-unmet", "hub")]

    def test_value_gate_unmet_when_the_keys_are_there_and_typed(self):
        got = diagnose([_t(keys=("age",), kinds=(("age", "int"),))], DEFINITION, POLICY)
        assert got == (Diagnosis("value-gate-unmet", "ask", ("age",), 1, 1.0),)

    def test_tie_loop_needs_the_dwell(self):
        def tie(n: int) -> RunTrace:
            return _t(stuck="pick", outcome="ambiguous", states=("pick",) * n)

        assert _codes([tie(3)]) == [("tie-loop", "pick")]
        assert _codes([tie(2)]) == []
        assert _codes([tie(2)], policy=SynthesisPolicy(dwell_threshold=2)) == [
            ("tie-loop", "pick")
        ]
        moved = _t(stuck="pick", outcome="ambiguous", states=("pick",) * 3 + ("x",))
        assert _codes([moved]) == []

    def test_a_cycle_needs_a_period_of_two_to_four_repeated(self):
        def loop(*block: str, times: int = 3) -> RunTrace:
            return _t(outcome="deterministic", states=block * times, stuck=block[-1])

        assert _codes([loop("a", "b")]) == [("cycle", "b")]
        assert _codes([loop("a", "b", "c", "d")]) == [("cycle", "d")]
        assert _codes([loop("a", "b", "c", "d", "e")]) == [("path-too-long", "e")]
        assert _codes([loop("a", "b", times=2)]) == [("path-too-long", "b")]
        assert _codes(
            [loop("a", "b", times=2)], policy=SynthesisPolicy(dwell_threshold=2)
        ) == [("cycle", "b")]
        assert _codes([loop("a", times=4)]) == []  # a stay is no cycle

    def test_path_too_long_needs_a_move_on_every_step(self):
        t = _t(outcome="deterministic", states=("a", "b", "c", "d", "e"), stuck="e")
        assert _codes([t]) == [("path-too-long", "e")]
        assert _codes([dataclasses.replace(t, states=("a", "b", "b", "c"))]) == []
        assert _codes([dataclasses.replace(t, states=())]) == []

    def test_return_keys_missing_names_the_absent_return_keys(self):
        t = _t(
            status=STATUS_FAILED,
            reason=REASON_RETURN_KEYS_MISSING,
            stuck=None,
            outcome=None,
            keys=("x",),
            states=("ask", "done"),
        )
        got = diagnose([t], DEFINITION, POLICY, return_keys=("x", "y"))
        assert got == (Diagnosis("return-keys-missing", "done", ("y",), 1, 1.0),)
        assert _codes([t]) == [("return-keys-missing", "done")]
        assert _codes([dataclasses.replace(t, states=())]) == [
            ("return-keys-missing", "")
        ]

    def test_an_unexplained_run_matches_no_rule_at_all(self):
        assert _codes([_t(stuck="nowhere")]) == []
        assert _codes([_t(stuck=None, states=())]) == []


class TestAbandonment:
    def _stray(self, state: str = "ask", mode: str = "converse") -> RunTrace:
        return _t(mode=mode, outcome=None, stuck=state, states=(state,))

    def test_it_needs_the_failures_and_the_share(self):
        three = [self._stray() for _ in range(3)]
        got = diagnose(three, DEFINITION, POLICY)
        assert got == (Diagnosis("abandonment-concentration", "ask", (), 3, 1.0),)
        assert _codes(three[:2]) == []

    def test_a_diffuse_set_is_no_concentration(self):
        spread = [self._stray() for _ in range(3)] + [
            self._stray(s) for s in ("rate", "pick", "done", "x")
        ]
        assert _codes(spread) == []
        half = [self._stray() for _ in range(3)] + [self._stray("rate")] * 3
        assert _codes(half) == [
            ("abandonment-concentration", "ask"),
            ("abandonment-concentration", "rate"),
        ]

    def test_run_mode_traces_are_not_abandonment(self):
        assert _codes([self._stray(mode="run") for _ in range(5)]) == []

    def test_the_failure_count_is_the_policy_field(self):
        two = [self._stray(), self._stray()]
        assert _codes(two, policy=SynthesisPolicy(repair_min_failures=2)) == [
            ("abandonment-concentration", "ask")
        ]

    def test_a_traced_cause_is_not_counted_again(self):
        blocked = [_t(mode="converse", keys=("anything",)) for _ in range(3)]
        assert [c for c, _ in _codes(blocked)] == ["extraction-miss"]


class TestRuleOrder:
    """Overlapping hand traces: the earlier rule must win, and a table with
    two rules swapped must answer differently."""

    ORDER = (
        "extraction-miss",
        "type-mismatch",
        "value-gate-unmet",
        "tie-loop",
        "cycle",
        "path-too-long",
        "return-keys-missing",
        "abandonment-concentration",
    )

    def test_the_table_is_the_fixed_order(self):
        assert tuple(code for code, _ in diagnose_module.RULES) == self.ORDER[:7]
        assert DIAGNOSIS_CODES == (NPF, *self.ORDER)

    OVERLAPS: ClassVar[list[Any]] = [
        (  # nick is absent and age has the wrong type: the miss comes first
            "extraction-miss",
            "type-mismatch",
            _t(stuck="both", keys=("age",), kinds=(("age", "str"),)),
        ),
        (
            "type-mismatch",
            "value-gate-unmet",
            _t(keys=("age",), kinds=(("age", "str"),)),
        ),
        (
            "cycle",
            "path-too-long",
            _t(outcome="deterministic", states=("a", "b") * 3, stuck="b"),
        ),
    ]

    @pytest.mark.parametrize("first,second,trace", OVERLAPS)
    def test_the_earlier_rule_wins_an_overlap(self, first, second, trace):
        assert [c for c, _ in _codes([trace])] == [first]
        swapped = [c for c, _ in diagnose_module.RULES]
        i, j = swapped.index(first), swapped.index(second)
        table = list(diagnose_module.RULES)
        table[i], table[j] = table[j], table[i]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(diagnose_module, "RULES", tuple(table))
            assert [c for c, _ in _codes([trace])] == [second]


class TestShape:
    def test_the_answer_is_sorted_and_independent_of_input_order(self):
        traces = (
            [_t(keys=("anything",)) for _ in range(2)]
            + [_t(keys=("age",), kinds=(("age", "int"),))]
            + [_t(stuck="rate", keys=())]
            + [_t(status=STATUS_OK, reason=None)]
        )
        want = diagnose(traces, DEFINITION, POLICY)
        assert [(d.code, d.state, d.runs) for d in want] == [
            ("extraction-miss", "ask", 2),
            (NPF, "", 1),
            ("value-gate-unmet", "ask", 1),
            ("extraction-miss", "rate", 1),
        ]
        for seed in range(5):
            shuffled = list(traces)
            random.Random(seed).shuffle(shuffled)
            assert diagnose(shuffled, DEFINITION, POLICY) == want

    def test_a_diagnosis_is_frozen_plain_data(self):
        d = diagnose([_t()], DEFINITION, POLICY)[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.runs = 9  # type: ignore[misc]
        assert isinstance(d.keys, tuple) and all(type(k) is str for k in d.keys)
        assert type(d.runs) is int and type(d.share) is float

    def test_no_traces_no_diagnosis(self):
        assert diagnose([], DEFINITION, POLICY) == ()

    def test_a_canary_in_a_trace_never_reaches_a_finding(self):
        canary = "CANARY-7f3a"
        t = _t(keys=("anything",), scenario=canary, version=canary)
        sim = _t(mode="sim", scenario=canary, reason="check_failed")
        got = diagnose([t, sim], DEFINITION, POLICY)
        assert got and canary not in repr(got)

    def test_diagnose_is_pure_over_its_arguments(self):
        t = _t(keys=("anything",))
        before = t.to_row()
        diagnose([t], DEFINITION, POLICY)
        assert t.to_row() == before

    def test_the_module_takes_no_clock_no_random_no_file(self):
        text = Path(diagnose_module.__file__).read_text()
        for banned in ("import time", "import random", "open(", "import os", "logger"):
            assert banned not in text
