"""
Agent Skills for agents: three read-only tools over a loaded ``SkillCatalog``.

The format itself (parser, loader, catalog, confined reads) is core
``fsm_llm.agent_skills``; this module only adapts it to agents and imports core
and sibling agents modules, never litellm.

Skill text is untrusted third-party prose. It reaches a model in exactly one
way here: as the OUTPUT of a read-only tool, capped, control-stripped and under
the fixed label ``SKILL_UNTRUSTED_LABEL``. It is NEVER a tool ``description``,
a parameter description, an ``AgentConfig.instructions`` value, a persona or a
state instruction: the descriptions below are module constants. A caller who
wants the catalog in the task string uses ``SkillCatalog.render_catalog()``.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.agent_skills import AgentSkill, SkillCatalog, incomplete_note
from fsm_llm.constants import (
    SKILL_CATALOG_DESCRIPTION_CHARS,
    SKILL_LIST_MAX_ITEMS,
    SKILL_MAX_FILE_BYTES,
    SKILL_READ_PAGE_CHARS,
    SKILL_UNTRUSTED_LABEL,
)
from fsm_llm.definitions import SkillError
from fsm_llm.okf import capped_line, text_page

from .definitions import ToolAnnotations
from .okf import _budget, _finish, _fit, _int
from .tools import ToolRegistry

__all__ = ["create_skill_tools"]

SKILL_TOOL_DESCRIPTIONS: dict[str, str] = {
    "skill_list": (
        "List the available skills (name, short description, number of files). "
        "Use offset to see the next page."
    ),
    "skill_load": (
        "Load one skill by name: a header, its file list, then a page of its "
        "instructions. Use the 'next offset' line to read on."
    ),
    "skill_read": (
        "Read one file bundled with a skill, by skill name and relative path, "
        "a page at a time; an empty path or a directory lists its files. "
        "Use the 'next offset' line to read on."
    ),
}

_NAME = {"type": "string", "description": "Skill name, as shown by skill_list."}
_OFFSET = {
    "type": "integer",
    "description": "Character offset to read from (default 0).",
}

SKILL_TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "skill_list": {
        "properties": {
            "offset": {
                "type": "integer",
                "description": "Number of skills to skip (default 0).",
            },
        },
    },
    "skill_load": {
        "properties": {"name": _NAME, "offset": _OFFSET},
        "required": ["name"],
    },
    "skill_read": {
        "properties": {
            "name": _NAME,
            "path": {
                "type": "string",
                "description": "File or directory path relative to the skill ('' lists all).",
            },
            "offset": _OFFSET,
        },
        "required": ["name", "path"],
    },
}

_FOOTER_RESERVE = 40  # room for the "next offset: N" line (own value, see D-031)
_FILES_CHARS = 400  # first-page file list of skill_load
_LINE_DESC_CHARS = 120  # skill_list: one description per line
_CLAIM_CHARS = 120


# DECISION plan-2026-10-23T043147-4c2042f3/D-030
# The label is the FIRST line of every reply and the tool descriptions are the
# constants above. Do NOT build a description, a parameter description or an
# instruction from a skill, do NOT drop the label or the incomplete note to
# save reply budget, and add NO tool that runs a script (D-005). A skill's
# `allowed-tools` is only ever shown, as a claim. See decisions.md D-030.
def _label(catalog: SkillCatalog) -> str:
    """The untrusted label, plus the fixed incomplete-catalog note if any."""
    note = incomplete_note(catalog)
    return f"{SKILL_UNTRUSTED_LABEL}\n{note}" if note else SKILL_UNTRUSTED_LABEL


def _start(offset: Any, high: int) -> int:
    """*offset* clamped to ``0..high``; a bool or a non-int is 0."""
    return 0 if isinstance(offset, bool) else _int(offset, 0, 0, high)


def _claims(skill: AgentSkill) -> list[str]:
    """Header lines for what the skill says about itself (shown, never enforced)."""
    tools = " ".join(skill.allowed_tools)
    return [
        f"{key} (claimed by the skill, not enforced): {capped_line(value, _CLAIM_CHARS)}"
        for key, value in (
            ("license", skill.license),
            ("compatibility", skill.compatibility),
            ("allowed-tools", tools),
        )
        if value
    ]


def _under(skill: AgentSkill, path: Any) -> list[str]:
    """The bundled files under directory *path* (``""`` is the root); ``[]`` for a
    file, a non-string or an unknown directory (the read then refuses it)."""
    if not isinstance(path, str):
        return []
    prefix = path.strip("/") + "/" if path.strip("/") else ""
    return [r for r in skill.resources if r.startswith(prefix)]


def _paged(
    head: str, page: str, following: int | None, total: int, kind: str, budget: int
) -> str:
    """*head*, a page of text, and the ``next offset`` line; never over *budget*."""
    out = head + (page or (f"(end of {kind})" if total else f"(empty {kind})"))
    if following is not None:
        out += f"\nnext offset: {following}"
    return _finish(out, budget)


def create_skill_tools(
    catalog: SkillCatalog, *, registry: ToolRegistry | None = None
) -> ToolRegistry:
    """Register three read-only tools over *catalog*; return the registry.

    Interface contract:
        - ``skill_list(offset=0)``, ``skill_load(name, offset=0)``,
          ``skill_read(name, path, offset=0)``, annotated read-only and
          idempotent, registered into *registry* (a new one when ``None``; a name
          already there is replaced). Descriptions and parameter descriptions are
          the module constants.
        - Every reply starts with ``SKILL_UNTRUSTED_LABEL``, then the fixed
          ``incomplete_note`` (codes only) for a partial catalog, is
          control-stripped and sized with the OKF ``_budget``. ``skill_load``:
          name, then on the first page the description, the claims (never
          enforced) and the files; ``skill_read``: a file page, or the file list
          of a directory (``""`` is all). Pages end with ``next offset: N``
          while more follows and add up to the stripped text.
        - Parameters are clamped, never raised on: a bool, non-int or negative
          offset is 0; a wrong-typed or unknown name gives a short message.
        - Nothing is executed; the tools only read the catalog's folders.
    """
    tools = registry if registry is not None else ToolRegistry()

    def failed(why: str, budget: int) -> str:
        return _finish(f"{_label(catalog)}\n{capped_line(why, 300)}", budget)

    def not_found(name: Any, budget: int) -> str:
        shown = capped_line(name, 80)
        close = catalog.closest(shown)
        hint = f" Close matches: {', '.join(close)}" if close else ""
        return failed(f"No skill named {shown!r}.{hint}", budget)

    def skill_list(offset: Any = 0) -> str:
        budget = _budget("skill_list", offset=offset)
        skills = list(catalog)
        start = _start(offset, len(skills))
        head = f"{_label(catalog)}\n{len(skills)} skill(s), from {start}:"
        lines, _ = _fit(
            [
                f"{capped_line(s.name, 64)} | "
                f"{capped_line(s.description, _LINE_DESC_CHARS)} | "
                f"{len(s.resources)} file(s)"
                for s in skills[start : start + SKILL_LIST_MAX_ITEMS]
            ],
            budget - len(head) - _FOOTER_RESERVE,
        )
        out = [head, *lines]
        if start + len(lines) < len(skills):
            out.append(f"next offset: {start + len(lines)}")
        return _finish("\n".join(out), budget)

    def skill_load(name: Any = "", offset: Any = 0) -> str:
        budget = _budget("skill_load", name=name, offset=offset)
        skill = catalog.get(name)
        if skill is None:
            return not_found(name, budget)
        start = _start(offset, SKILL_MAX_FILE_BYTES)
        lines = [_label(catalog), f"name: {capped_line(skill.name, 64)}"]
        if start:  # later pages repeat only the name: more room for the text
            lines[-1] += f" (from character {start})"
        else:
            desc = capped_line(skill.description, SKILL_CATALOG_DESCRIPTION_CHARS)
            lines += [f"description: {desc}", *_claims(skill)]
        if start == 0 and skill.resources:
            files, hidden = _fit([f"  {r}" for r in skill.resources], _FILES_CHARS)
            lines += [f"files ({len(skill.resources)}):", *files]
            if hidden:
                lines.append(f'  ... {hidden} more: skill_read with path "" lists all')
        head = "\n".join([*lines, "", ""])
        room = min(SKILL_READ_PAGE_CHARS, budget - len(head) - _FOOTER_RESERVE)
        try:
            page, following, total = catalog.read_body(skill.name, start, size=room)
        except SkillError as refused:
            return failed(str(refused), budget)
        return _paged(head, page, following, total, "body", budget)

    def skill_read(name: Any = "", path: Any = "", offset: Any = 0) -> str:
        budget = _budget("skill_read", name=name, path=path, offset=offset)
        skill = catalog.get(name)
        if skill is None:
            return not_found(name, budget)
        files = _under(skill, path)
        what = (
            f"files under {capped_line(path, 300)!r} ({len(files)})"
            if files
            else (f"file: {capped_line(path, 300)}")
        )
        head = "\n".join(
            [_label(catalog), f"skill: {capped_line(skill.name, 64)}", what, "", ""]
        )
        room = min(SKILL_READ_PAGE_CHARS, budget - len(head) - _FOOTER_RESERVE)
        start = _start(offset, SKILL_MAX_FILE_BYTES)
        try:
            if files:
                page, following, total = text_page(
                    "\n".join(files), start, max(room, 1)
                )
            else:
                page, following, total = catalog.read_resource(
                    skill.name, path, start, size=room
                )
        except SkillError as refused:
            return failed(str(refused), budget)
        return _paged(head, page, following, total, "file", budget)

    annotations = ToolAnnotations(
        read_only=True, destructive=False, idempotent=True, open_world=False
    )
    for fn in (skill_list, skill_load, skill_read):
        tools.register_function(
            fn,
            name=fn.__name__,
            description=SKILL_TOOL_DESCRIPTIONS[fn.__name__],
            parameter_schema=SKILL_TOOL_PARAMETERS[fn.__name__],
            annotations=annotations,
        )
    return tools
