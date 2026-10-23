"""Tests for fsm_llm.agents.python_tools.PythonToolLoader file discovery."""

from __future__ import annotations

import dataclasses
import importlib.util
import textwrap
import warnings
from pathlib import Path

import pytest

from fsm_llm.agents.python_tools import PythonToolLoader, PythonToolLoadReport

_MODULE_BOTH_SOURCES = textwrap.dedent(
    """
    from fsm_llm.agents.python_tools import PythonToolDefinition
    from fsm_llm.agents.tools import tool


    @tool
    def lookup(query: str) -> str:
        \"\"\"Look something up.\"\"\"
        return query


    PYTHON_TOOLS = [
        PythonToolDefinition(
            name="lookup",
            description="Look something up (explicit)",
            execute=lookup,
        )
    ]
    """
)


class TestPythonToolLoaderDedupe:
    """PT-07: a tool in PYTHON_TOOLS and also @tool-decorated is loaded once."""

    def test_python_tools_list_and_decorator_yield_one_tool(self, tmp_path: Path):
        path = tmp_path / "dup_tool.py"
        path.write_text(_MODULE_BOTH_SOURCES)

        tools = PythonToolLoader.from_directory(tmp_path)

        names = [t.name for t in tools]
        assert names == ["lookup"]
        # The explicit PYTHON_TOOLS entry wins over the decorator scan.
        assert tools[0].description == "Look something up (explicit)"

    def test_decorator_only_module_still_loads(self, tmp_path: Path):
        path = tmp_path / "deco_only.py"
        path.write_text(
            textwrap.dedent(
                """
                from fsm_llm.agents.tools import tool


                @tool
                def ping(value: str) -> str:
                    \"\"\"Echo.\"\"\"
                    return value
                """
            )
        )

        tools = PythonToolLoader.from_directory(tmp_path)

        assert [t.name for t in tools] == ["ping"]


_MODULE_ALIASED_TOOL = textwrap.dedent(
    """
    from fsm_llm.agents.tools import tool


    @tool
    def fast_add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b


    fast_add_alias = fast_add
    another_alias = fast_add
    """
)

_MODULE_NAME_COLLISION = textwrap.dedent(
    """
    from fsm_llm.agents.tools import tool


    @tool(name="lookup")
    def a_first(query: str) -> str:
        \"\"\"First lookup.\"\"\"
        return "first"


    @tool(name="lookup")
    def b_second(query: str) -> str:
        \"\"\"Second lookup.\"\"\"
        return "second"
    """
)


def _capture_warnings(fn):
    from fsm_llm.logging import logger

    logger.enable("fsm_llm")
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        result = fn()
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")
    return result, captured


class TestPythonToolLoaderToolAliasDedupe:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-012: the @tool scan dedupes on
    the tool name against PYTHON_TOOLS and against @tool tools it already loaded."""

    def test_one_tool_under_three_names_loads_once(self, tmp_path: Path):
        (tmp_path / "aliased.py").write_text(_MODULE_ALIASED_TOOL)

        tools, warnings = _capture_warnings(
            lambda: PythonToolLoader.from_directory(tmp_path)
        )

        assert [t.name for t in tools] == ["fast_add"]
        assert tools[0].execute(a=2, b=3) == 5
        # The same function under another name is not a conflict: no warning.
        assert warnings == []

    def test_two_functions_with_one_tool_name_first_wins_and_warns(
        self, tmp_path: Path
    ):
        (tmp_path / "collision.py").write_text(_MODULE_NAME_COLLISION)

        tools, warnings = _capture_warnings(
            lambda: PythonToolLoader.from_directory(tmp_path)
        )

        assert [t.name for t in tools] == ["lookup"]
        # dir() order is alphabetical, so the first attribute wins.
        assert tools[0].execute(query="q") == "first"
        assert len(warnings) == 1
        assert "lookup" in warnings[0]
        assert "b_second" in warnings[0]


class TestPythonToolLoaderRemovedNames:
    """D-039: a tool file that still imports a removed name loads nothing and
    raises nothing. Edited on purpose by plan 93d78536 step 4 (I-126): the file
    is still skipped quietly (no exception, no stdout or stderr), and it now ALSO
    warns the caller, because a trace that is off until logging is enabled is no
    trace at all."""

    def test_a_file_importing_a_removed_name_is_skipped_and_warns(
        self, tmp_path: Path, capsys
    ):
        removed = "Skill" + "Definition"
        (tmp_path / "old.py").write_text(f"from fsm_llm.agents import {removed}\n")

        with pytest.warns(UserWarning, match=removed) as caught:
            assert PythonToolLoader.from_directory(tmp_path) == []
        assert len(caught) == 1
        assert capsys.readouterr() == ("", "")
        with pytest.warns(UserWarning):
            _, logged = _capture_warnings(
                lambda: PythonToolLoader.from_directory(tmp_path)
            )
        assert len(logged) == 1 and removed in logged[0]


def _write_broken_and_good(tmp_path: Path) -> Path:
    (tmp_path / "a_good.py").write_text(
        "from fsm_llm.agents.tools import tool\n\n\n"
        "@tool\ndef ping(value: str) -> str:\n"
        '    """Echo."""\n    return value\n'
    )
    broken = tmp_path / "b_broken.py"
    broken.write_text("raise RuntimeError('boom')\n")
    return broken


class TestPythonToolLoadReport:
    """I-126: ``load_directory`` returns the tools AND one (file, message) entry
    for every file it could not load; nothing is swallowed into a log line."""

    def test_a_clean_directory_has_tools_and_no_errors(self, tmp_path: Path):
        (tmp_path / "a_good.py").write_text(_MODULE_BOTH_SOURCES)

        report = PythonToolLoader.load_directory(tmp_path)

        assert isinstance(report, PythonToolLoadReport)
        assert [t.name for t in report.tools] == ["lookup"]
        assert isinstance(report.tools, tuple) and report.errors == ()

    def test_the_report_is_frozen(self, tmp_path: Path):
        report = PythonToolLoader.load_directory(tmp_path)

        with pytest.raises(dataclasses.FrozenInstanceError):
            report.errors = ()  # type: ignore[misc]

    def test_an_exception_in_a_file_is_an_error_and_the_rest_loads(
        self, tmp_path: Path
    ):
        broken = _write_broken_and_good(tmp_path)

        report = PythonToolLoader.load_directory(tmp_path)

        assert [t.name for t in report.tools] == ["ping"]
        assert report.errors == ((str(broken), "RuntimeError: boom"),)

    def test_an_oversize_file_is_an_error_and_is_never_executed(self, tmp_path: Path):
        big = tmp_path / "big.py"
        big.write_text("raise RuntimeError('executed')\n" + "#" * 1_000_001)

        report = PythonToolLoader.load_directory(tmp_path)

        assert report.tools == ()
        assert [file for file, _ in report.errors] == [str(big)]
        assert "1 MB" in report.errors[0][1]
        assert "executed" not in report.errors[0][1]

    def test_an_unloadable_spec_is_an_error(self, tmp_path: Path, monkeypatch):
        odd = tmp_path / "odd.py"
        odd.write_text("X = 1\n")
        monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a: None)

        report = PythonToolLoader.load_directory(tmp_path)

        assert report.tools == ()
        assert [file for file, _ in report.errors] == [str(odd)]
        assert "spec" in report.errors[0][1]

    def test_an_unreadable_stat_is_an_error(self, tmp_path: Path, monkeypatch):
        locked = tmp_path / "locked.py"
        locked.write_text("X = 1\n")
        real_stat = Path.stat

        def stat(self, *args, **kwargs):
            if self.name == "locked.py":
                raise PermissionError(13, "Permission denied")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)

        report = PythonToolLoader.load_directory(tmp_path)

        assert report.tools == ()
        assert [file for file, _ in report.errors] == [str(locked)]
        assert "PermissionError" in report.errors[0][1]

    def test_a_missing_directory_is_an_error(self, tmp_path: Path):
        missing = tmp_path / "nope"

        report = PythonToolLoader.load_directory(missing)

        assert report.tools == ()
        assert report.errors == ((str(missing), "directory not found"),)

    def test_private_and_non_python_files_are_ignored_not_errors(self, tmp_path: Path):
        (tmp_path / "_private.py").write_text("raise RuntimeError('never run')\n")
        (tmp_path / "notes.txt").write_text("not python\n")

        report = PythonToolLoader.load_directory(tmp_path, pattern="*")

        assert report.tools == () and report.errors == ()

    @pytest.mark.parametrize("exit_type", [SystemExit, KeyboardInterrupt])
    def test_exit_and_interrupt_from_a_tool_file_still_propagate(
        self, tmp_path: Path, exit_type
    ):
        (tmp_path / "quits.py").write_text(f"raise {exit_type.__name__}()\n")

        with pytest.raises(exit_type):
            PythonToolLoader.load_directory(tmp_path)
        with pytest.raises(exit_type):
            PythonToolLoader.from_directory(tmp_path)


class TestFromDirectoryWarns:
    """``from_directory`` keeps returning a plain list and warns the CALLER once
    per failed file (``stacklevel=2``, so ``fsm_llm.disable_warnings()`` cannot
    hide it)."""

    def test_the_result_is_the_report_tools_as_a_list(
        self, tmp_path: Path, monkeypatch
    ):
        (tmp_path / "a_good.py").write_text(_MODULE_BOTH_SOURCES)
        report = PythonToolLoader.load_directory(tmp_path)
        monkeypatch.setattr(PythonToolLoader, "load_directory", lambda *a: report)

        tools = PythonToolLoader.from_directory(tmp_path)

        assert type(tools) is list and len(tools) == 1
        assert tools[0] is report.tools[0]

    def test_one_warning_per_failed_file_attributed_to_the_caller(self, tmp_path: Path):
        broken = _write_broken_and_good(tmp_path)
        (tmp_path / "c_syntax.py").write_text("def (:\n")

        with pytest.warns(UserWarning) as caught:
            PythonToolLoader.from_directory(tmp_path)

        assert len(caught) == 2
        assert {w.filename for w in caught} == {__file__}
        text = "\n".join(str(w.message) for w in caught)
        assert str(broken) in text and "RuntimeError: boom" in text
        assert "c_syntax.py" in text and "SyntaxError" in text

    def test_a_missing_directory_warns_once_and_returns_an_empty_list(
        self, tmp_path: Path
    ):
        with pytest.warns(UserWarning, match="not found") as caught:
            assert PythonToolLoader.from_directory(tmp_path / "nope") == []
        assert len(caught) == 1

    def test_disable_warnings_does_not_hide_it(self, tmp_path: Path):
        import fsm_llm

        _write_broken_and_good(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fsm_llm.disable_warnings()
            PythonToolLoader.from_directory(tmp_path)

        assert len(caught) == 1

    def test_one_failed_file_is_still_one_logged_warning(self, tmp_path: Path):
        _write_broken_and_good(tmp_path)

        with pytest.warns(UserWarning):
            _, logged = _capture_warnings(
                lambda: PythonToolLoader.from_directory(tmp_path)
            )

        assert len(logged) == 1 and "boom" in logged[0]

    def test_a_clean_directory_warns_nothing(self, tmp_path: Path):
        (tmp_path / "a_good.py").write_text(_MODULE_BOTH_SOURCES)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tools = PythonToolLoader.from_directory(tmp_path)

        assert [t.name for t in tools] == ["lookup"]
