"""The closed intents of the classifier envelope and the writer that puts them there.

Plan baa37f40 step 5b (decision D-019, owner order D-017, roadmap E3.4 slice b): the
``recommend_strategy`` state of ``reasoning/strategies/classifier.json`` classifies through core
``classification_extractions`` with the nine strategies as closed intents, one structured
``complete`` call in place of the free-text per-field call. The file is generated here, never
typed: ``python -m tests.test_fsm_llm_reasoning.classifier_file --write`` rewrites only
``body.fsm.states.recommend_strategy`` of the envelope as canonical text (``json.dumps(indent=2,
ensure_ascii=True)``, one final newline, LF), and the rows of ``test_classifier_intents.py`` pin
file equals derivation. Step 17b moves the derivation into ``reasoning/selection.py`` (from the
registry); this module then derives the same entry from the built-in registry.

Interface contract:

- ``intents()``: the nine ``{name, description}`` dicts in ``STRATEGY_ORDER``; ``description`` is
  ``capped_line(strategy_facts(name)["description"], DESCRIPTION_CAP)``.
- ``entry()``: the one ``classification_extractions`` entry (a fresh dict).
- ``derive(document)``: a deep copy of an envelope with ``recommend_strategy`` rewritten;
  idempotent. ``KeyError`` when the state is missing.
- ``write(path=CLASSIFIER_PATH)``: rewrites the file; returns ``None``.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from fsm_llm.reasoning.strategy_files import STRATEGY_ORDER, strategy_facts
from fsm_llm.text import capped_line

CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "fsm_llm"
    / "reasoning"
    / "strategies"
    / "classifier.json"
)

STATE_ID = "recommend_strategy"
FIELD_NAME = "recommended_reasoning_type"
FALLBACK_INTENT = "analytical"
CONFIDENCE_THRESHOLD = 0.6
DESCRIPTION_CAP = 300
CONTEXT_KEYS = (
    "problem_statement",
    "problem_type",
    "problem_components",
    "problem_domain",
    "problem_structure",
    "reasoning_requirements",
    "key_challenges",
)

PURPOSE = (
    "Choose the reasoning strategy that best fits the problem, justify it, and name "
    "1-2 alternative approaches"
)
EXTRACTION_INSTRUCTIONS = (
    "\n            Recommend the best reasoning strategy for the problem. "
    "Provide clear justification and identify 1-2 alternative approaches that could "
    "also work.\n\n"
    "            IMPORTANT: Do not ask questions or request additional input from the "
    "user. Make recommendation based on your complete analysis.\n            "
)


def intents() -> list[dict[str, str]]:
    """The strategies as closed intents, in ``STRATEGY_ORDER``."""
    return [
        {
            "name": name,
            "description": capped_line(
                strategy_facts(name)["description"], DESCRIPTION_CAP
            ),
        }
        for name in STRATEGY_ORDER
    ]


def entry() -> dict[str, Any]:
    """The one ``classification_extractions`` entry of ``recommend_strategy``."""
    return {
        "field_name": FIELD_NAME,
        "intents": intents(),
        "fallback_intent": FALLBACK_INTENT,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "required": False,
        "prompt_config": {"include_entities": False},
        "context_keys": list(CONTEXT_KEYS),
    }


def derive(document: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of the envelope *document* with ``recommend_strategy`` rewritten."""
    result = copy.deepcopy(document)
    state = result["body"]["fsm"]["states"][STATE_ID]
    state["purpose"] = PURPOSE
    state["extraction_instructions"] = EXTRACTION_INSTRUCTIONS
    state["classification_extractions"] = [entry()]
    return result


def render(document: dict[str, Any]) -> str:
    """The canonical text of an envelope (the form every strategy file is in)."""
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def write(path: Path = CLASSIFIER_PATH) -> None:
    """Rewrite the classifier envelope at *path* from its own content and the derivation."""
    document = json.loads(path.read_text(encoding="utf-8"))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render(derive(document)))


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit(
            "usage: python -m tests.test_fsm_llm_reasoning.classifier_file --write"
        )
    write()
