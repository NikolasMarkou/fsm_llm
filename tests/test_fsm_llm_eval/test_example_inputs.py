"""The stdin scripts of ``EXAMPLE_INPUTS`` against the examples' ``input()`` calls
(plan 0068b055 step 12.22, live finding D6).

The harness pipes one script per interactive example. A script that does not
match what the example asks for fails quietly: ``react_hitl_combined`` was sent
``"Search for weather in Paris\\ny\\n..."``; the task never called a gated tool, no
approval was asked, and the ``y`` ran as a second task. The scan is static
(AST over the example source, read only).

Shapes the scan understands:

- task loop: ``input("Task: ")`` in a ``while True`` that stops on ``quit``;
- chat loop: ``input("You: ")`` in a loop that stops on ``quit``/``exit`` or when the
  conversation ends;
- approval prompt: an ``input()`` inside a function whose name contains
  ``approval`` or an ``Approve`` prompt: it consumes one line, ``y`` or ``n``, only
  when the agent calls a gated tool;
- builder session: ``run_interactive()``, which reads until EOF (a ``quit`` line is
  sent to the builder as a message, so the script must not carry one).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from fsm_llm.eval.constants import EXAMPLE_INPUTS, EXAMPLE_SCRIPT_NAMES

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_QUIT = {"quit", "exit", "q"}


def _script_path(name: str) -> Path:
    base = name.removesuffix("_manual")
    script = "run_manual.py" if name.endswith("_manual") else "run.py"
    return _EXAMPLES / base / script


def _input_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "input"
    ]


def _prompt(call: ast.Call) -> str:
    arg = call.args[0] if call.args else None
    return arg.value if isinstance(arg, ast.Constant) else ""


def _is_approval(call: ast.Call) -> bool:
    return "approv" in _prompt(call).lower()


def _uses_run_interactive(tree: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run_interactive"
        for n in ast.walk(tree)
    )


def _lines(name: str) -> list[str]:
    script = EXAMPLE_INPUTS[name]
    assert script.endswith("\n"), "a script ends with a newline"
    return script.splitlines()


def _tree(name: str) -> ast.AST:
    return ast.parse(_script_path(name).read_text(encoding="utf-8"))


def _all_interactive_scripts() -> list[str]:
    found = []
    for script in EXAMPLE_SCRIPT_NAMES:
        for path in sorted(_EXAMPLES.glob(f"*/*/{script}")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if _input_calls(tree) or _uses_run_interactive(tree):
                directory = f"{path.parts[-3]}/{path.parts[-2]}"
                found.append(
                    directory + ("_manual" if script == "run_manual.py" else "")
                )
    return found


_NAMES = sorted(EXAMPLE_INPUTS)
_BUILDER = [n for n in _NAMES if _uses_run_interactive(_tree(n))]
_LOOPS = [n for n in _NAMES if n not in _BUILDER]
# Examples with an approval prompt -> a regex the first task line must match so the
# agent calls a gated tool.
_GATED = {
    "agents/hitl_approval": r"\bsend\b.*\bemail\b",
    "agents/react_hitl_combined": r"\bpublish\b|\barchive\b",
}


class TestTableMatchesTheExamples:
    @pytest.mark.parametrize("name", _NAMES)
    def test_every_script_names_an_interactive_example(self, name):
        path = _script_path(name)
        assert path.is_file(), f"{name}: no {path.name}"
        tree = _tree(name)
        assert _input_calls(tree) or _uses_run_interactive(tree), (
            f"{name} never reads stdin; its script is dead"
        )

    def test_every_interactive_example_has_a_script(self):
        assert sorted(set(_all_interactive_scripts()) - set(EXAMPLE_INPUTS)) == []

    @pytest.mark.parametrize("name", _NAMES)
    def test_script_has_no_blank_line(self, name):
        assert all(line.strip() for line in _lines(name))


class TestLoopExamplesEndOnTheirExitCommand:
    @pytest.mark.parametrize("name", _LOOPS)
    def test_loop_script_ends_on_quit_after_a_turn(self, name):
        lines = _lines(name)
        assert lines[-1].strip().lower() in _QUIT
        assert len(lines) >= 2

    @pytest.mark.parametrize("name", _BUILDER)
    def test_builder_script_has_no_quit_line(self, name):
        """run_interactive reads to EOF and sends every line to the builder."""
        assert all(line.strip().lower() not in _QUIT for line in _lines(name))

    def test_both_shapes_exist(self):
        assert len(_BUILDER) == 3
        assert len(_LOOPS) >= 15


class TestApprovalPrompts:
    """An example with an approval prompt: the line after the gated task is the answer."""

    def test_approval_examples_are_all_known(self):
        found = sorted(
            n for n in _NAMES if any(_is_approval(c) for c in _input_calls(_tree(n)))
        )
        assert found == sorted(_GATED)

    @pytest.mark.parametrize("name", sorted(_GATED))
    def test_first_task_calls_a_gated_tool(self, name):
        assert re.search(_GATED[name], _lines(name)[0], re.IGNORECASE)

    @pytest.mark.parametrize("name", sorted(_GATED))
    def test_answer_follows_the_task_then_quit(self, name):
        lines = _lines(name)
        assert lines[1].lower() in {"y", "yes", "n", "no"}
        assert lines[2].lower() in _QUIT
        assert len(lines) == 3

    def test_react_hitl_combined_script_is_pinned(self):
        assert EXAMPLE_INPUTS["agents/react_hitl_combined"] == (
            "Research recent AI papers and publish a summary report\ny\nquit\n"
        )
