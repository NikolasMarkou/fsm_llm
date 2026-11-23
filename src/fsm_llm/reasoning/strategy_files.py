"""The reasoning FSMs as package data (plan baa37f40 step 1, roadmap E3.3 job A).

The nine strategies, the classifier and the orchestrator are JSON envelopes under
``reasoning/strategies/<name>.json`` (the ``fsm-llm/executable`` format of
``fsm_llm.core.definitions.executable``): ``body.fsm`` is the FSM the engine runs, and for a
strategy ``description`` and ``body.strategy`` (``aliases``, ``results``, ``solution_key``)
are the four facts the table ``STRATEGIES`` held. Every file is canonical text
(``json.dumps(indent=2, ensure_ascii=True)``, one final newline, LF), read here as UTF-8
through ``importlib.resources`` and never imported as code.

Interface contract:

- ``FSM_NAMES``: every FSM name, in the order of the former ``ALL_REASONING_FSMS``.
- ``STRATEGY_ORDER``: the nine strategy names, in the order of the former ``STRATEGIES``;
  ``extract_final_solution`` reads the solution keys in this order, and the glob order of the
  directory is alphabetical, so the order is written here and a test pins it.
- ``load_document(name)``: a deep copy of the envelope; ``KeyError`` for a name not in
  ``FSM_NAMES`` (a name is never joined into a path unchecked). ``load_fsm(name)`` is the deep
  copy of its ``body.fsm``. A file that is missing or unparsable raises its own error: it is a
  packaging bug, not an input.
- ``strategy_facts(name)``: ``{description, aliases, results, solution_key}`` of a strategy,
  fresh on every call (a caller may mutate it); ``KeyError`` for a name not in
  ``STRATEGY_ORDER``.
- ``solution_keys()``: the ``solution_key`` of every strategy, in ``STRATEGY_ORDER``.
- ``strategy_id(name)``: the executable id of a strategy (``reasoning.<name>``, the target id a
  ``call`` state names). The one spelling of the id: ``executables`` and the engine both read it
  here (a leaf they may import).
"""

from __future__ import annotations

import copy
import json
from functools import cache
from importlib import resources
from typing import Any

FSM_NAMES: tuple[str, ...] = (
    "orchestrator",
    "classifier",
    "simple_calculator",
    "analytical",
    "deductive",
    "inductive",
    "creative",
    "critical",
    "abductive",
    "analogical",
    "hybrid",
)

STRATEGY_ORDER: tuple[str, ...] = (
    "simple_calculator",
    "analytical",
    "deductive",
    "inductive",
    "creative",
    "critical",
    "hybrid",
    "abductive",
    "analogical",
)


#: The prefix of a built-in strategy's executable id.
_ID_PREFIX = "reasoning."


def strategy_id(name: str) -> str:
    """The executable id of the strategy *name* (a ``ReasoningType`` value)."""
    return f"{_ID_PREFIX}{name}"


@cache
def _document(name: str) -> dict[str, Any]:
    """The parsed envelope of *name*, shared: never hand it out, copy it."""
    path = resources.files("fsm_llm.reasoning").joinpath("strategies", f"{name}.json")
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def load_document(name: str) -> dict[str, Any]:
    """The envelope of the FSM *name* as a deep copy (``KeyError`` when unknown)."""
    if name not in FSM_NAMES:
        raise KeyError(name)
    return copy.deepcopy(_document(name))


# DECISION plan-2026-10-01T093600-944e2692/D-054: the orchestrator's validation verdict,
# retry counters, confidence and strategy choice are handler-owned (the
# ``handler_only_keys`` of ``strategies/orchestrator.json``, written from
# ``ORCHESTRATOR_HANDLER_ONLY_KEYS``). Do NOT take a key out of that list to let the
# model "help" (a bulk reply with validation_result: true would open the gate) and do
# NOT list them in a state's required_context_keys (never extracted, the validator
# warns). The rule moved here with the file from ``reasoning_modes/orchestrator.py``
# (D-009 of plan baa37f40): a JSON file cannot hold the comment.
def load_fsm(name: str) -> dict[str, Any]:
    """The FSM definition dict of *name* as a deep copy (``KeyError`` when unknown)."""
    if name not in FSM_NAMES:
        raise KeyError(name)
    fsm: dict[str, Any] = copy.deepcopy(_document(name)["body"]["fsm"])
    return fsm


def strategy_facts(name: str) -> dict[str, Any]:
    """The four facts of the strategy *name*, fresh (``KeyError`` when not a strategy)."""
    if name not in STRATEGY_ORDER:
        raise KeyError(name)
    document = _document(name)
    facts: dict[str, Any] = copy.deepcopy(document["body"]["strategy"])
    facts["description"] = document["description"]
    return facts


def solution_keys() -> tuple[str, ...]:
    """Each strategy's ``solution_key``, in ``STRATEGY_ORDER``."""
    return tuple(
        _document(name)["body"]["strategy"]["solution_key"] for name in STRATEGY_ORDER
    )
