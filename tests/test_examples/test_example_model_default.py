"""The example scripts' model default (plan 51fde95b D-058 of step 15, review D R-D5).

The five meta examples read ``LLM_MODEL`` so that an empty value counts as unset
(the library's ``llm.resolve_model`` also treats whitespace as blank). No example script is run;
only the ``model = ...`` line is evaluated, and every printed hint names the same
default as the code.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_DEFAULT = "ollama_chat/bonsai-2:27b"
_META = [
    "build_agent",
    "build_fsm",
    "build_workflow",
    "meta_from_spec",
    "meta_review_loop",
]


def _model_expression(script: Path) -> ast.expr:
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Assign)
            and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["model"]
            and "LLM_MODEL" in ast.unparse(node.value)
        ):
            return node.value
    raise AssertionError(f"no model = ...LLM_MODEL... line in {script}")


@pytest.mark.parametrize("name", _META)
@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, _DEFAULT), ("", _DEFAULT), ("x/y", "x/y")],
    ids=["unset", "empty", "given"],
)
def test_a_meta_example_reads_a_blank_model_as_unset(
    monkeypatch, name, value, expected
):
    expression = _model_expression(_EXAMPLES / "meta" / name / "run.py")
    if value is None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
    else:
        monkeypatch.setenv("LLM_MODEL", value)
    got = eval(compile(ast.Expression(expression), "<example>", "eval"), {"os": os})
    assert got == expected


def test_no_example_hint_names_the_old_default():
    stale = [
        str(p.relative_to(_EXAMPLES))
        for p in _EXAMPLES.rglob("*.py")
        if "ollama_chat/qwen3.5:4b" in p.read_text(encoding="utf-8")
    ]
    assert stale == []
