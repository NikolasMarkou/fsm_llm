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
from fsm_llm.reasoning.handlers import OutputFormatter
from fsm_llm.reasoning.strategy_files import FSM_NAMES, STRATEGY_ORDER, strategy_facts
from fsm_llm.reasoning.utilities import (
    get_available_reasoning_types,
    get_reasoning_fsm,
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
    # Re-baselined on purpose (D-031 of plan baa37f40, step 15): `execute_reasoning` is a `call`
    # state with `choose`, `reasoning_push_pending` left `handler_only_keys`; the 51fde95b D-052
    # re-baseline (calculator keys in `handler_only_keys`) is inside it.
    "orchestrator": "82c88f57cb8bd0f8f442315a9f63825c3da47adf09c84639ea2d2f6e12136437",
    # Re-baselined on purpose (D-019 of plan baa37f40, step 5b): `recommend_strategy` classifies
    # through the nine closed intents; `test_classifier_intents` pins the file to its derivation.
    "classifier": "fe187a5446ca7564f7de07a9335169ff24c3de91b2f2ceee94818cff69bc29a5",
    "simple_calculator": "5e74e1cb5164c203afafedc292131cc5553554c36d68fb2183f92a22bdf2f32b",
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
    # The three merge tables went with `merge_reasoning_results` (D-031); the keys a strategy
    # returns are its `results` map, which `settle` merges (`test_call_settle`), and this one
    # table pins them.
    "results_maps": "eb887e6e853ef3d13671574b49f5f6fd8e0f10e3fe0ce0ad4fda1f2e59ba2f78",
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


def _results_maps() -> dict[str, dict[str, str]]:
    """Each strategy's parent key to child key map, the keys a call state merges on its pop."""
    return {name: strategy_facts(name)["results"] for name in STRATEGY_ORDER}


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
        "fsm_registry_keys": list(FSM_NAMES),
        "aliases": {p: map_reasoning_type(p) for p in _ALIAS_PROBES},
        "descriptions": list(get_available_reasoning_types().items()),
        "results_maps": _results_maps(),
        "solution_order": _solution_order(),
    }


@pytest.mark.parametrize("name", sorted(_GOLDEN_FSMS))
def test_fsm_definition_is_byte_identical(name):
    assert _hash(get_reasoning_fsm(name)) == _GOLDEN_FSMS[name]


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
    assert types <= set(FSM_NAMES)
    assert types == set(get_available_reasoning_types())
    assert all(map_reasoning_type(t) == t for t in types)


class TestOneStrategyTable:
    """F-RE-02, F-RE-03, F-RE-07: tables derive from one registry."""

    def test_strategies_cover_exactly_the_enum(self):
        assert set(STRATEGY_ORDER) == {t.value for t in ReasoningType}

    def test_every_state_id_is_its_key(self):
        for name in FSM_NAMES:
            fsm = get_reasoning_fsm(name)
            assert all(s["id"] == key for key, s in fsm["states"].items())

    def test_get_reasoning_fsm_returns_a_copy_and_names_the_missing(self):
        copy = get_reasoning_fsm("orchestrator")
        copy["states"].clear()
        assert get_reasoning_fsm("orchestrator")["states"]
        with pytest.raises(KeyError, match="FSM definition not found: nope"):
            get_reasoning_fsm("nope")
