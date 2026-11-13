"""
Agent Skills: folders with a ``SKILL.md`` (YAML frontmatter plus markdown body).

The format layer, a layer package, read-only: ``catalog`` (the tolerant parser,
bounded loading, the progressive-disclosure catalog and confined reads) and ``cli``
(the ``fsm-llm-skills`` command). The agents-side tools are
``fsm_llm.agents.agent_skills``. Skill text is untrusted prose: content never raises
(a defect becomes a ``SkillIssue``), nothing in a skill folder is executed or
imported, ``allowed-tools`` is reported, never enforced, the body is never kept.

Imports the kernel (``constants``, ``frontmatter``, ``security``, ``text``,
``cli_support``) and ``core`` only, never ``okf``, the engine or a subpackage.

This ``__init__`` is the one import surface of the package (a static ``__all__``);
a private name is imported from the module that defines it.
"""

from __future__ import annotations

from fsm_llm.core.definitions import SkillError

from .catalog import (
    AgentSkill,
    SkillCatalog,
    SkillIssue,
    SkillLimits,
    incomplete_note,
    load_skills,
    parse_skill_text,
    skipped_note,
)

__all__ = [
    "AgentSkill",
    "SkillCatalog",
    "SkillError",
    "SkillIssue",
    "SkillLimits",
    "incomplete_note",
    "load_skills",
    "parse_skill_text",
    "skipped_note",
]
