"""Golden hashes of the reasoning FSM definitions and strategy tables.

Captured from the code before the prompt sentences, the per-state ``id`` and the
strategy tables were generated from one place (plan 5980fbb7 step 18). A hash
that moves means a prompt, an alias, a description, a merged key or the solution
order changed: update it only on purpose.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from fsm_llm.reasoning.constants import ContextKeys, ReasoningType
from fsm_llm.reasoning.handlers import ContextManager, OutputFormatter
from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS
from fsm_llm.reasoning.utilities import (
    get_available_reasoning_types,
    map_reasoning_type,
)

# Every key the strategy table knew an alias for, plus case, space and unknown probes.
_ALIAS_PROBES = (
    *"analytical deductive inductive creative critical hybrid simple_calculator "
    "abductive analogical arithmetic calculation math calculator compute numeric "
    "analyze analysis deduce deduction logic logical syllogism induce induction "
    "pattern observation generalize create innovation innovative brainstorm "
    "divergent critique evaluation evaluate assessment judge mixed combined multi "
    "explain explanation hypothesis abduction diagnose analogy comparison compare "
    "similar".split(),
    "direct computation",
    "direct calculation",
    "  Deductive ",
    "MATH",
    "no such type",
    "",
)

_GOLDEN_FSMS = {
    "orchestrator": "0c07a006f8422ae033be38263f3c08919fd6aa661ee927db0852c9f19d558cd9",
    "classifier": "1ada9e1547adb115ae5ad902c4e612dd2290f355ed7b6d897679ef053465b6c7",
    "simple_calculator": "4277a46339b2373db1fbb2b802a339f1f096e7390a4b6b52031c34e911826bde",
    "analytical": "13fd110501336bf248d06cc4376c1fb0798fcc84713e0fb655f2cefc154f4a20",
    "deductive": "d0d5c2cfb7978714284bbc47f3c71415c274e6a07ca15345bb8b9a0d3435174e",
    "inductive": "aedd000dba3f897707b774a8bcc2ffdf0f15a3edf7d8c831d6aa05cc2f250e72",
    "abductive": "fdc87ca4ead2f1652f9aedb26e13ede13dc92539b886307846067bbf36e73a6d",
    "analogical": "e26790bfab6c106a281d6fbd2928854ce97e1c1f5dc1c0d63c040dbcf3509c86",
    "creative": "3929fa9e3361b55efbef4789eaebae4e53467403f59190bf81a91ec5ccd1026c",
    "critical": "6c50fac4c6ddb083cf3486ae029fa3aa5be6227e4bb364351f1c3b916fa390cc",
    "hybrid": "08257b872ec74cf329a37e451e98cf414098212a5753e02417c7b728443774ac",
}
_GOLDEN_TABLES = {
    "enum": "31951ae8800944462c4c7a9694df81db56fd68dddf414125bf6fef0b4353561d",
    "fsm_registry_keys": "dfd445d472c78bb0f4fb0202001cf5029c4302154c3dd1c3b5fc7457c05c7829",
    "aliases": "553514a177ee4c9459754fae261202c64e49a4a0fc6c35a7398ff8da5ec1153a",
    "descriptions": "d96fb4b12d07b85ed0c9c1d24f597e07aef08f373d468a0c0c3e05351118099f",
    "merge_ladder": "cfc304b3878488d08da2e6145631950facd2be075fc1f26bf7fcb676c1bb722a",
    "merge_none_values": "48393bdd2d664c2d77301a09d8b4b67f218613c7ae02399199171dccba08f767",
    "merge_calculator_cases": "12a0c4a62c50622ea1a6ac3e0c8053a7c1f201d777ac70b39e98ab1d39665853",
    "solution_order": "d036c8b670b52e7a851e527724decf1ddffbac72a3f894cdbffef1e6ebc1132f",
}


def _hash(value: object) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _every_key_filled() -> dict[str, str]:
    return {
        value: f"v:{name}"
        for name, value in vars(ContextKeys).items()
        if name.isupper() and isinstance(value, str)
    }


def _merge_ladder() -> dict[str, dict]:
    sub = _every_key_filled()
    types = [t.value for t in ReasoningType] + ["bogus"]
    return {t: ContextManager.merge_reasoning_results({}, sub, t) for t in types}


def _merge_none_values() -> dict[str, dict]:
    sub = {value: None for value in _every_key_filled()}
    types = [t.value for t in ReasoningType]
    return {t: ContextManager.merge_reasoning_results({}, sub, t) for t in types}


def _merge_calculator_cases() -> list[dict]:
    calc = ReasoningType.SIMPLE_CALCULATOR.value
    cases = [
        {},
        {ContextKeys.CALCULATION_RESULT: 0},
        {ContextKeys.CALCULATION_RESULT: None},
        {ContextKeys.CALCULATION_ERROR: "boom"},
        {ContextKeys.CALCULATION_ERROR: None},
        {ContextKeys.CALCULATION_RESULT: 4, ContextKeys.CALCULATION_ERROR: "e"},
    ]
    return [ContextManager.merge_reasoning_results({}, c, calc) for c in cases]


def _solution_order() -> list[str]:
    """The keys ``extract_final_solution`` reads, in the order it reads them."""
    context = {
        key: f"answer:{key}"
        for key in vars(ContextKeys).values()
        if isinstance(key, str)
    }
    order = []
    for _ in range(len(context)):
        answer = OutputFormatter.extract_final_solution(context)
        if not answer.startswith("answer:"):
            break
        key = answer.removeprefix("answer:")
        order.append(key)
        del context[key]
    return order


def _tables() -> dict[str, object]:
    return {
        "enum": [(m.name, m.value) for m in ReasoningType],
        "fsm_registry_keys": list(ALL_REASONING_FSMS),
        "aliases": {p: map_reasoning_type(p) for p in _ALIAS_PROBES},
        "descriptions": list(get_available_reasoning_types().items()),
        "merge_ladder": _merge_ladder(),
        "merge_none_values": _merge_none_values(),
        "merge_calculator_cases": _merge_calculator_cases(),
        "solution_order": _solution_order(),
    }


@pytest.mark.parametrize("name", sorted(_GOLDEN_FSMS))
def test_fsm_definition_is_byte_identical(name):
    assert _hash(ALL_REASONING_FSMS[name]) == _GOLDEN_FSMS[name]


@pytest.mark.parametrize("name", sorted(_GOLDEN_TABLES))
def test_strategy_table_is_identical(name):
    assert _hash(_tables()[name]) == _GOLDEN_TABLES[name]


def test_solution_order_starts_final_proposed_calculation():
    order = _solution_order()
    assert order[:3] == [
        ContextKeys.FINAL_SOLUTION,
        ContextKeys.PROPOSED_SOLUTION,
        ContextKeys.CALCULATION_RESULT,
    ]
    assert len(order) == 11


def test_every_reasoning_type_has_an_fsm_a_description_and_an_alias():
    types = {t.value for t in ReasoningType}
    assert types <= set(ALL_REASONING_FSMS)
    assert types == set(get_available_reasoning_types())
    assert all(map_reasoning_type(t) == t for t in types)
