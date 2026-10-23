"""Tests for fsm_llm.agents.python_tools.PythonToolLoader file discovery."""

from __future__ import annotations

import textwrap
from pathlib import Path

from fsm_llm.agents.python_tools import PythonToolLoader

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
    raises nothing; the only trace is a warning, off until logging is enabled."""

    def test_a_file_importing_a_removed_name_is_skipped_quietly(
        self, tmp_path: Path, capsys
    ):
        removed = "Skill" + "Definition"
        (tmp_path / "old.py").write_text(f"from fsm_llm.agents import {removed}\n")

        assert PythonToolLoader.from_directory(tmp_path) == []
        assert capsys.readouterr() == ("", "")
        _, warnings = _capture_warnings(
            lambda: PythonToolLoader.from_directory(tmp_path)
        )
        assert len(warnings) == 1 and removed in warnings[0]
