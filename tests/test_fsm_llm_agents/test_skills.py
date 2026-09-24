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
