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

import sys
from typing import Any

from fsm_llm.agent_skills import (
    AgentSkill,
    SkillCatalog,
    incomplete_note,
    skipped_note,
)
from fsm_llm.constants import (
    SKILL_CATALOG_DESCRIPTION_CHARS,
    SKILL_LIST_MAX_ITEMS,
    SKILL_READ_HINT,
    SKILL_READ_PAGE_CHARS,
    SKILL_UNTRUSTED_LABEL,
)
from fsm_llm.definitions import SkillError
from fsm_llm.okf import capped_line, strip_unsafe_text, text_page

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
_MIN_PAGE = 200  # text a page keeps before the claims and files give way
_FILES_CHARS = 400  # first-page file list of skill_load
_MORE_LINE = 60  # the "... N more" line closing a cut file list
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


def _first_page(skill: AgentSkill, spare: int) -> list[str]:
    """Header lines of the first ``skill_load`` page that fit in *spare* characters:
    the description, then the files (count, names that fit, the fixed read hint),
    then the claims. Nothing is cut mid-line; what does not fit is left out."""
    desc = capped_line(skill.description, SKILL_CATALOG_DESCRIPTION_CHARS)
    out, _ = _fit([f"description: {desc}"], spare)
    spare -= sum(len(line) + 1 for line in out)
    count = f"files ({len(skill.resources)}):"
    names = min(_FILES_CHARS, spare - len(count) - len(SKILL_READ_HINT) - _MORE_LINE)
    if skill.resources and names > 0:
        shown, hidden = _fit([f"  {r}" for r in skill.resources], names)
        more = [f'  ... {hidden} more: skill_read with path "" lists all']
        files = [count, *shown, *(more if hidden else []), SKILL_READ_HINT]
        out += files
        spare -= sum(len(line) + 1 for line in files)
    return out + _fit(_claims(skill), spare)[0]


def _tail(text: str, following: int | None) -> str:
    """Control-stripped *text*, then the ``next offset`` line when more follows.
    The footer is appended after the strip, never cut: callers size *text* to leave
    it room (a page is at least one character, so a reply always moves on)."""
    out = strip_unsafe_text(text, multiline=True)
    return out if following is None else f"{out}\nnext offset: {following}"


def _reply(head: str, page: str, following: int | None, total: int, kind: str) -> str:
    """*head*, a page of text (or the end marker) and the footer, via ``_tail``."""
    return _tail(
        head + (page or (f"(end of {kind})" if total else f"(empty {kind})")), following
    )


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
        notes = [_label(catalog), skipped_note(catalog)]
        head = "\n".join(
            [*filter(None, notes), f"{len(skills)} skill(s), from {start}:"]
        )
        entries = [
            f"{capped_line(s.name, 64)} | "
            f"{capped_line(s.description, _LINE_DESC_CHARS)} | "
            f"{len(s.resources)} file(s)"
            for s in skills[start : start + SKILL_LIST_MAX_ITEMS]
        ]
        lines, _ = _fit(entries, budget - len(head) - _FOOTER_RESERVE)
        lines = lines or entries[:1]  # a page always moves on
        more = start + len(lines) < len(skills)
        return _tail("\n".join([head, *lines]), start + len(lines) if more else None)

    def skill_load(name: Any = "", offset: Any = 0) -> str:
        budget = _budget("skill_load", name=name, offset=offset)
        skill = catalog.get(name)
        if skill is None:
            return not_found(name, budget)
        try:  # the catalog clamps the offset to the text, whatever its size limit
            page, following, total = catalog.read_body(
                skill.name, _start(offset, sys.maxsize), size=SKILL_READ_PAGE_CHARS
            )
        except SkillError as refused:
            return failed(str(refused), budget)
        begin = (total if following is None else following) - len(page)
        shown = f"name: {capped_line(skill.name, 64)}"
        lines = [
            _label(catalog),
            shown + (f" (from character {begin})" if begin else ""),
        ]
        if not begin:  # later pages repeat only the name: more room for the text
            spare = budget - _FOOTER_RESERVE - _MIN_PAGE - len("\n".join(lines)) - 2
            lines += _first_page(skill, spare)
        head = "\n".join([*lines, "", ""])
        room = max(min(SKILL_READ_PAGE_CHARS, budget - _FOOTER_RESERVE - len(head)), 1)
        if len(page) > room:
            page, following = page[:room], begin + room
        return _reply(head, page, following, total, "body")

    def skill_read(name: Any = "", path: Any = "", offset: Any = 0) -> str:
        budget = _budget("skill_read", name=name, path=path, offset=offset)
        skill = catalog.get(name)
        if skill is None:
            return not_found(name, budget)
        files = _under(skill, path)
        top = [_label(catalog), f"skill: {capped_line(skill.name, 64)}"]
        if not files and isinstance(path, str) and not path.strip("/"):
            return _finish("\n".join([*top, "(no files)"]), budget)
        what = (
            f"files under {capped_line(path, 300)!r} ({len(files)})"
            if files
            else (f"file: {capped_line(path, 300)}")
        )
        head = "\n".join([*top, what, "", ""])
        room = max(min(SKILL_READ_PAGE_CHARS, budget - len(head) - _FOOTER_RESERVE), 1)
        start = _start(offset, sys.maxsize)
        try:
            if files:
                page, following, total = text_page("\n".join(files), start, room)
            else:
                page, following, total = catalog.read_resource(
                    skill.name, path, start, size=room
                )
        except SkillError as refused:
            return failed(str(refused), budget)
        return _reply(head, page, following, total, "file")

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
