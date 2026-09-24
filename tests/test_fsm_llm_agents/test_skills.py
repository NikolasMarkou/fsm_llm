"""Tests for fsm_llm_agents.skills.SkillLoader file discovery."""

from __future__ import annotations

import textwrap
from pathlib import Path

from fsm_llm_agents.skills import SkillLoader

_MODULE_BOTH_SOURCES = textwrap.dedent(
    """
    from fsm_llm_agents.skills import SkillDefinition
    from fsm_llm_agents.tools import tool


    @tool
    def lookup(query: str) -> str:
        \"\"\"Look something up.\"\"\"
        return query


    SKILLS = [
        SkillDefinition(
            name="lookup",
            description="Look something up (explicit)",
            execute=lookup,
        )
    ]
    """
)


class TestSkillLoaderDedupe:
    """PT-07: a skill in SKILLS and also @tool-decorated is loaded once."""

    def test_skills_list_and_decorator_yield_one_skill(self, tmp_path: Path):
        path = tmp_path / "dup_skill.py"
        path.write_text(_MODULE_BOTH_SOURCES)

        skills = SkillLoader.from_directory(tmp_path)

        names = [s.name for s in skills]
        assert names == ["lookup"]
        # The explicit SKILLS entry wins over the decorator scan.
        assert skills[0].description == "Look something up (explicit)"

    def test_decorator_only_module_still_loads(self, tmp_path: Path):
        path = tmp_path / "deco_only.py"
        path.write_text(
            textwrap.dedent(
                """
                from fsm_llm_agents.tools import tool


                @tool
                def ping(value: str) -> str:
                    \"\"\"Echo.\"\"\"
                    return value
                """
            )
        )

        skills = SkillLoader.from_directory(tmp_path)

        assert [s.name for s in skills] == ["ping"]


_MODULE_ALIASED_TOOL = textwrap.dedent(
    """
    from fsm_llm_agents.tools import tool


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
    from fsm_llm_agents.tools import tool


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


class TestSkillLoaderToolAliasDedupe:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-012: the @tool scan dedupes on
    the tool name against SKILLS and against @tool skills it already loaded."""

    def test_one_tool_under_three_names_loads_once(self, tmp_path: Path):
        (tmp_path / "aliased.py").write_text(_MODULE_ALIASED_TOOL)

        skills, warnings = _capture_warnings(
            lambda: SkillLoader.from_directory(tmp_path)
        )

        assert [s.name for s in skills] == ["fast_add"]
        assert skills[0].execute(a=2, b=3) == 5
        # The same function under another name is not a conflict: no warning.
        assert warnings == []

    def test_two_functions_with_one_tool_name_first_wins_and_warns(
        self, tmp_path: Path
    ):
        (tmp_path / "collision.py").write_text(_MODULE_NAME_COLLISION)

        skills, warnings = _capture_warnings(
            lambda: SkillLoader.from_directory(tmp_path)
        )

        assert [s.name for s in skills] == ["lookup"]
        # dir() order is alphabetical, so the first attribute wins.
        assert skills[0].execute(query="q") == "first"
        assert len(warnings) == 1
        assert "lookup" in warnings[0]
        assert "b_second" in warnings[0]
