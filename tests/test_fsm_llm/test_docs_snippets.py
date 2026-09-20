"""Every full FSM snippet in the first-touch docs must load.

The docs audit (GD-01, plan-2026-09-19T175721-21cd7f8e D-037) found that every
copy-paste FSM in the README, quickstart and CLAUDE.md was rejected by the
loader (missing required ``description``, mis-nested classification schema).
This test extracts each fenced block that names ``"initial_state"`` and loads
it through ``FSMDefinition``, so a docs edit that breaks a first-touch snippet
fails here. JSON blocks are parsed as JSON; Python blocks contribute the first
dict literal that has ``initial_state`` (via ``ast``, nothing is executed).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from fsm_llm import FSMDefinition

_ROOT = Path(__file__).resolve().parents[2]
# Files that each carry at least one full FSM snippet (the first-touch docs).
_FULL_FSM_DOCS = [
    "README.md",
    "docs/quickstart.md",
    "src/fsm_llm/README.md",
    "CLAUDE.md",
]
# Reference docs scanned too (plan-2026-09-19T175721-21cd7f8e D-052 (5)). At the
# time of writing none carries a block naming "initial_state" (docs/fsm_design.md
# has state-level JSON fragments only), so they contribute no case today; a full
# FSM added to one of them is picked up and must load.
_REFERENCE_DOCS = [
    "docs/api_reference.md",
    "docs/architecture.md",
    "docs/fsm_design.md",
    "docs/handlers.md",
]
_DOCS = [*_FULL_FSM_DOCS, *_REFERENCE_DOCS]
_FENCE = re.compile(r"```(json|python)\n(.*?)```", re.DOTALL)


def _snippets() -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    for rel in _DOCS:
        text = (_ROOT / rel).read_text(encoding="utf-8")
        for n, match in enumerate(_FENCE.finditer(text)):
            lang, body = match.groups()
            if '"initial_state"' not in body:
                continue
            label = f"{rel}#block{n}"
            if lang == "json":
                found.append((label, json.loads(body)))
                continue
            for node in ast.walk(ast.parse(body)):
                if isinstance(node, ast.Dict):
                    try:
                        data = ast.literal_eval(node)
                    except ValueError:
                        continue
                    if isinstance(data, dict) and "initial_state" in data:
                        found.append((label, data))
                        break
    return found


_SNIPPETS = _snippets()


def test_the_extractor_found_the_first_touch_snippets():
    # Guard against passing vacuously if a fence style or file moves.
    assert len(_SNIPPETS) >= 4
    found = {label.split("#")[0] for label, _ in _SNIPPETS}
    assert set(_FULL_FSM_DOCS) <= found
    assert found <= set(_DOCS)
    assert all((_ROOT / rel).is_file() for rel in _REFERENCE_DOCS)


@pytest.mark.parametrize(
    ("label", "definition"), _SNIPPETS, ids=[s[0] for s in _SNIPPETS]
)
def test_doc_fsm_snippet_loads(label, definition):
    FSMDefinition(**definition)
