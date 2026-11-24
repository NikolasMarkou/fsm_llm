"""The reasoning CLI follows the registry (plan baa37f40, step 18, D-035, roadmap E3.4c).

``--executables DIR`` reads the strategy documents of a directory into the registry the engine
uses, ``--type`` accepts the names of that registry and refuses any other with exit 2,
``--list-types`` lists it, the engine is closed on every way out of a solve, and Ctrl-C is 130.
Every row is in process with a stub engine (or a real engine on a scripted model): no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

import fsm_llm
from fsm_llm.reasoning import ReasoningEngine
from fsm_llm.reasoning import __main__ as cli
from fsm_llm.reasoning.selection import strategy_choices
from fsm_llm.reasoning.strategy_files import STRATEGY_ORDER, load_document

from .test_engine_scripted import _ScriptedLLM
from .test_selection import _PIRATE_ANSWER, _document, _script_for_user_strategy

REPO = Path(__file__).resolve().parents[2]
PROG = "python -m fsm_llm.reasoning"


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(cli.__file__).resolve().is_relative_to(REPO / "src")


def _write(directory: Path, name: str, document: Any) -> Path:
    path = directory / name
    text = document if isinstance(document, str) else json.dumps(document, indent=2)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def strategies(tmp_path: Path) -> Path:
    """A directory holding one valid user strategy, ``pirate`` (id ``user.pirate``)."""
    directory = tmp_path / "strategies"
    directory.mkdir()
    _write(directory, "pirate.json", _document("pirate"))
    return directory


class _StubEngine:
    """Stands in for ``ReasoningEngine`` in ``cli``: records its use, answers on demand."""

    instances: ClassVar[list[_StubEngine]] = []
    outcome: ClassVar[Any] = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = 0
        self.solved: tuple[str, Any] | None = None
        type(self).instances.append(self)

    def __enter__(self) -> _StubEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed += 1

    def solve_problem(self, problem: str, initial_context: Any = None) -> Any:
        self.solved = (problem, initial_context)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return "four", {"reasoning_trace": {"steps": []}, "summary": "s"}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[_StubEngine]:
    _StubEngine.instances = []
    _StubEngine.outcome = None
    monkeypatch.setattr(cli, "ReasoningEngine", _StubEngine)
    return _StubEngine


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    return cli.main()


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


class TestExecutablesAndType:
    def test_a_user_strategy_of_the_directory_is_accepted_by_type(
        self, monkeypatch, capsys, stub, strategies
    ) -> None:
        code = _main(
            monkeypatch,
            "what is 2+2?",
            "--executables",
            str(strategies),
            "--type",
            "pirate",
        )
        assert code == 0, capsys.readouterr().err
        (engine,) = stub.instances
        names = [c.name for c in strategy_choices(engine.kwargs["registry"])]
        assert names == [*STRATEGY_ORDER, "pirate"]
        assert engine.solved is not None
        assert engine.solved[1]["preferred_reasoning_type"] == "pirate"

    def test_the_engine_gets_the_registry_as_the_explicit_keyword(
        self, monkeypatch, stub, strategies
    ) -> None:
        _main(monkeypatch, "p", "--executables", str(strategies), "-m", "x/y")
        (engine,) = stub.instances
        assert engine.kwargs["model"] == "x/y"
        assert engine.kwargs["registry"].get("user.pirate").kind == "strategy"

    def test_without_the_option_the_engine_builds_its_own_builtin_registry(
        self, monkeypatch, stub
    ) -> None:
        assert _main(monkeypatch, "p", "--type", "analytical") == 0
        (engine,) = stub.instances
        assert engine.kwargs["registry"] is None

    def test_a_user_type_without_the_directory_is_a_usage_error(
        self, monkeypatch, capsys, stub
    ) -> None:
        with pytest.raises(SystemExit) as info:
            _main(monkeypatch, "p", "--type", "pirate")
        assert info.value.code == 2
        err = capsys.readouterr().err
        assert "argument --type/-t: invalid choice: 'pirate'" in err
        assert all(name in err for name in STRATEGY_ORDER)
        assert stub.instances == []

    def test_an_unknown_type_with_the_directory_lists_the_user_strategy_too(
        self, monkeypatch, capsys, stub, strategies
    ) -> None:
        with pytest.raises(SystemExit) as info:
            _main(monkeypatch, "p", "--executables", str(strategies), "--type", "bogus")
        assert info.value.code == 2
        err = capsys.readouterr().err
        assert "invalid choice: 'bogus'" in err and "pirate" in err
        assert stub.instances == []

    def test_an_alias_is_not_a_type(
        self, monkeypatch, capsys, stub, strategies
    ) -> None:
        with pytest.raises(SystemExit) as info:
            _main(monkeypatch, "p", "--executables", str(strategies), "--type", "arr")
        assert info.value.code == 2

    def test_the_builtin_types_still_work_as_before(self, monkeypatch, stub) -> None:
        for name in STRATEGY_ORDER:
            assert _main(monkeypatch, "p", "--type", name) == 0
        assert len(stub.instances) == 9


class TestListTypes:
    def test_the_table_lists_the_user_strategy_after_the_nine(
        self, monkeypatch, capsys, strategies
    ) -> None:
        assert _main(monkeypatch, "--list-types", "--executables", str(strategies)) == 0
        out = capsys.readouterr().out
        assert "pirate" in out and "Answers like a pirate" in out
        assert "Total: 10 reasoning types available" in out
        assert out.index("hybrid") < out.index("pirate")

    def test_without_the_option_it_is_the_nine(self, monkeypatch, capsys) -> None:
        assert _main(monkeypatch, "--list-types") == 0
        out = capsys.readouterr().out
        assert "Total: 9 reasoning types available" in out and "pirate" not in out

    def test_the_help_names_the_option_and_does_not_freeze_the_nine(
        self, monkeypatch, capsys
    ) -> None:
        with pytest.raises(SystemExit) as info:
            _main(monkeypatch, "--help")
        assert info.value.code == 0
        out = capsys.readouterr().out
        assert "--executables" in out and "--list-types" in out
        assert "Available reasoning types:" not in out


class TestADirectoryThatCannotBeRead:
    @pytest.mark.parametrize("kind", ["missing", "a-file"])
    def test_one_error_line_exit_one_no_engine(
        self, monkeypatch, capsys, stub, tmp_path, kind: str
    ) -> None:
        target = tmp_path / "nothing"
        if kind == "a-file":
            target = _write(tmp_path, "file.json", "{}")
        assert _main(monkeypatch, "p", "--executables", str(target)) == 1
        lines = _lines(capsys.readouterr().err)
        assert len(lines) == 1 and lines[0].startswith(f"{PROG}: error: ")
        assert "--executables" in lines[0] and "Traceback" not in lines[0]
        assert stub.instances == []

    def test_a_broken_file_is_one_warning_and_the_solve_goes_on(
        self, monkeypatch, capsys, stub, strategies
    ) -> None:
        _write(strategies, "broken.json", "{not json")
        assert _main(monkeypatch, "p", "--executables", str(strategies)) == 0
        warnings = [ln for ln in _lines(capsys.readouterr().err) if "warning" in ln]
        (line,) = warnings
        assert line.startswith(f"{PROG}: warning: ") and "broken.json" in line
        (engine,) = stub.instances
        assert "pirate" in [c.name for c in strategy_choices(engine.kwargs["registry"])]

    def test_a_file_that_collides_with_a_builtin_never_replaces_it(
        self, monkeypatch, capsys, stub, strategies
    ) -> None:
        clash = load_document("analytical")
        clash["description"] = "hijacked"
        _write(strategies, "analytical.json", clash)
        assert _main(monkeypatch, "p", "--executables", str(strategies)) == 0
        out = capsys.readouterr()
        assert any("warning" in ln for ln in _lines(out.err))
        (engine,) = stub.instances
        choices = {c.name: c for c in strategy_choices(engine.kwargs["registry"])}
        assert choices["analytical"].description != "hijacked"


class TestTheEngineIsClosedOnEveryWayOut:
    def test_after_a_solve(self, monkeypatch, stub) -> None:
        assert _main(monkeypatch, "p") == 0
        assert [e.closed for e in stub.instances] == [1]

    def test_after_a_failed_solve_with_one_error_line(
        self, monkeypatch, capsys, stub
    ) -> None:
        stub.outcome = RuntimeError("engine exploded")
        assert _main(monkeypatch, "p") == 1
        assert [e.closed for e in stub.instances] == [1]
        lines = _lines(capsys.readouterr().err)
        assert len(lines) == 1 and "error: engine exploded" in lines[0]

    def test_on_ctrl_c_with_exit_130_and_one_line(
        self, monkeypatch, capsys, stub
    ) -> None:
        stub.outcome = KeyboardInterrupt()
        assert _main(monkeypatch, "p") == 130
        assert [e.closed for e in stub.instances] == [1]
        lines = _lines(capsys.readouterr().err)
        assert lines == [f"{PROG}: interrupted by user"]


class TestARealSolveThroughTheCli:
    def test_a_user_strategy_chosen_by_type_answers(
        self, monkeypatch, capsys, strategies
    ) -> None:
        script = _script_for_user_strategy()

        def build(**kwargs: Any) -> ReasoningEngine:
            return ReasoningEngine(llm_interface=_ScriptedLLM(script), **kwargs)

        monkeypatch.setattr(cli, "ReasoningEngine", build)
        code = _main(
            monkeypatch,
            "Is the argument valid?",
            "--executables",
            str(strategies),
            "--type",
            "pirate",
            "--quiet",
        )
        out = capsys.readouterr()
        assert code == 0, out.err
        assert out.out.strip() == _PIRATE_ANSWER
