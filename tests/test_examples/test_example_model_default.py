"""No example names a model the caller did not set (plan ace7a8c8 step 18.4, D-035, D-039).

Bonsai is the default of the owner's machine through configuration (a `.env`, the shell), never
code: no file under ``examples/`` holds the literal, no example reads ``LLM_MODEL`` with a default,
and an example that needs a model and finds none exits 1 with one stderr line. The static rows scan
every example; the behaviour rows run eight scripts (every spelling of the rule) with the variable
unset or blank.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "examples"
_BONSAI = "bonsai"
_META = [
    "build_agent",
    "build_fsm",
    "build_workflow",
    "meta_from_spec",
    "meta_review_loop",
]
# a script per spelling of the rule: `os.environ.get`, `os.getenv`, and a def-time default
_OTHERS = [
    "agents/react_search",
    "intermediate/adaptive_quiz",
    "intermediate/book_recommendation",
]


def _scripts() -> list[Path]:
    return sorted(_EXAMPLES.rglob("*.py"))


def _is_llm_model_read(node: ast.AST) -> bool:
    """``os.environ.get("LLM_MODEL", ...)`` or ``os.getenv("LLM_MODEL", ...)``."""
    if not (isinstance(node, ast.Call) and node.args):
        return False
    first = node.args[0]
    name = ast.unparse(node.func)
    return (
        isinstance(first, ast.Constant)
        and first.value == "LLM_MODEL"
        and name in ("os.environ.get", "os.getenv")
    )


def _defaults(script: Path) -> list[str]:
    """The ways a script gives ``LLM_MODEL`` a value of its own: a second argument, or ``or x``."""
    found = []
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if _is_llm_model_read(node) and (len(node.args) > 1 or node.keywords):
            found.append(f"{script.name}:{node.lineno} get with a default")
        if (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and _is_llm_model_read(node.values[0])
        ):
            found.append(f"{script.name}:{node.lineno} read `or` a fallback")
    return found


def test_no_example_file_names_the_machine_default():
    named = [
        str(p.relative_to(_EXAMPLES))
        for p in _EXAMPLES.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".md", ".txt", ".sh"}
        and _BONSAI in p.read_text(encoding="utf-8", errors="replace").lower()
    ]
    assert named == []


def test_no_example_reads_llm_model_with_a_default():
    assert [line for s in _scripts() for line in _defaults(s)] == []


def test_the_scan_finds_a_planted_default(tmp_path):
    planted = tmp_path / "run.py"
    planted.write_text(
        'import os\na = os.environ.get("LLM_MODEL", "x/y")\n'
        'b = os.getenv("LLM_MODEL", "x/y")\n'
        'c = os.environ.get("LLM_MODEL") or "x/y"\n'
        'd = os.environ.get("LLM_MODEL")\n',
        encoding="utf-8",
    )
    assert len(_defaults(planted)) == 3


def test_no_example_hint_names_the_old_default():
    stale = [
        str(p.relative_to(_EXAMPLES))
        for p in _EXAMPLES.rglob("*.py")
        if "ollama_chat/qwen3.5:4b" in p.read_text(encoding="utf-8")
    ]
    assert stale == []


def _run(script: Path, model: str | None) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items() if k not in ("LLM_MODEL", "OPENAI_API_KEY")
    }
    env.update(
        PYTHONPATH=str(_ROOT / "src"),
        PYTHONUTF8="1",
        OLLAMA_API_BASE="http://127.0.0.1:9",
        OLLAMA_HOST="http://127.0.0.1:9",
    )
    if model is not None:
        env["LLM_MODEL"] = model
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        stdin=subprocess.DEVNULL,
        cwd=_ROOT,
    )


def _exits_asking_for_a_model(done: subprocess.CompletedProcess[str]) -> bool:
    lines = [line for line in done.stderr.splitlines() if line.strip()]
    return done.returncode == 1 and len(lines) == 1 and "LLM_MODEL" in lines[0]


@pytest.mark.parametrize("name", _META)
def test_a_meta_example_reads_a_blank_model_as_unset(name):
    # the meta examples call load_dotenv(), which never replaces a variable that exists, blank
    # included: a `.env` on the machine cannot turn a blank value into a model here
    done = _run(_EXAMPLES / "meta" / name / "run.py", "  ")

    assert _exits_asking_for_a_model(done), (done.returncode, done.stderr[-300:])


@pytest.mark.parametrize("name", _OTHERS)
def test_an_example_without_a_model_exits_one_with_one_line(name):
    done = _run(_EXAMPLES / name / "run.py", None)

    assert _exits_asking_for_a_model(done), (done.returncode, done.stderr[-300:])
