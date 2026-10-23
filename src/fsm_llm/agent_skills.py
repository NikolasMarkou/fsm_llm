"""
Agent Skills: folders with a ``SKILL.md`` (YAML frontmatter plus markdown body).

The format layer, read-only: a tolerant parser, bounded loading and a
progressive-disclosure catalog. Skill text is untrusted third-party prose:
content never raises (a defect becomes a ``SkillIssue``), nothing a skill folder
holds is executed or imported, ``allowed-tools`` is a claim that is reported and
never enforced, and the body is never retained (re-read from disk on demand).

Spec status: the open standard at agentskills.io, read on 2026-10-23, unversioned
and mutable. Names are ASCII (``[a-z0-9]`` and single hyphens), as Anthropic's
docs and API say; the standard's prose allows unicode letters, so a unicode name
is skipped here with ``name-invalid``, never loaded under a lookalike.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    SKILL_BODY_MAX_LINES,
    SKILL_COMPATIBILITY_MAX_CHARS,
    SKILL_DESCRIPTION_MAX_CHARS,
    SKILL_MAX_FRONTMATTER_BYTES,
    SKILL_MAX_YAML_DEPTH,
    SKILL_NAME_MAX_CHARS,
)
from .frontmatter import Note, load_yaml, split_frontmatter
from .okf import capped_line, strip_unsafe_text

__all__ = ["AgentSkill", "SkillIssue", "parse_skill_text"]

# The six keys of the open standard; any other key is reported, never an error.
_KNOWN_KEYS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)
_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# Anthropic's API refuses these inside a name; kept as a portability warning.
_RESERVED_WORDS = ("anthropic", "claude")


class SkillIssue(BaseModel):
    """One problem found while reading a skill; ``message`` and ``path`` are
    untrusted, so control-stripped and capped here."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    path: str | None = None

    @field_validator("message", "path")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        return None if value is None else strip_unsafe_text(value)[:300]


def _clean(value: Any) -> Any:
    """Control-strip every string in a str, tuple or dict value; others as is."""
    if isinstance(value, str):
        return strip_unsafe_text(value)
    if isinstance(value, tuple):
        return tuple(_clean(item) for item in value)
    if isinstance(value, dict):
        return {_clean(key): _clean(item) for key, item in value.items()}
    return value


# DECISION plan-2026-10-23T043147-4c2042f3/D-022
# Do NOT add a ``body`` (or ``text``) field here, and do NOT keep the body in a
# cache next to the catalog to "save a disk read". Progressive disclosure is the
# point of the format: the catalog holds metadata only, and ``read_body`` re-reads
# the file, bounded and paged. A retained body costs memory per skill and puts
# untrusted prose where no reader decided to read it. See decisions.md D-022.
class AgentSkill(BaseModel):
    """The metadata of one skill: no body, only its length in lines.

    Interface contract: built by ``parse_skill_text`` (``directory`` and
    ``resources`` are set by the loader). Every string it holds is
    control-stripped by this model, whoever constructs it. ``extra_keys`` names
    the frontmatter keys outside the standard; ``allowed_tools``, ``license`` and
    ``compatibility`` are claims the skill makes, reported and never enforced.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    directory: Path | None = None
    skill_md: str = "SKILL.md"
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    extra_keys: tuple[str, ...] = ()
    line_count: int = 0
    resources: tuple[str, ...] = ()

    @field_validator("*")
    @classmethod
    def _strip_untrusted(cls, value: Any) -> Any:
        return _clean(value)


def _text(data: dict[Any, Any], key: str, code: str, note: Note) -> str:
    """The stripped string at *key*; ``""`` when absent, null or not a string
    (a non-string adds a ``code`` warning)."""
    value = data.get(key)
    if isinstance(value, str):
        return strip_unsafe_text(value).strip()
    if value is not None:
        note("warning", code, f"{key!r} must be a string; ignored")
    return ""


def parse_skill_text(
    text: str,
    folder_name: str,
    *,
    directory: Path | None = None,
    skill_md: str = "SKILL.md",
    resources: tuple[str, ...] = (),
) -> tuple[AgentSkill | None, list[SkillIssue]]:
    """Parse the text of one ``SKILL.md`` into ``(skill or None, issues)``.

    Interface contract: pure and total, no filesystem access, never raises on
    content (any YAML shape, non-string value, huge value, lone surrogate or
    duplicate key). ``None`` comes with at least one ``error`` issue: no,
    unterminated, invalid, too large, too deep or non-mapping frontmatter, a
    missing or non-string description, or a name that is not ``[a-z0-9]`` words
    joined by single hyphens in 1 to 64 characters (after defaulting to
    *folder_name*). Everything else loads with a warning; ``body-long`` is info.
    The body is counted in ``line_count`` and discarded."""
    issues: list[SkillIssue] = []
    where = f"{folder_name}/{skill_md}"

    def note(
        severity: Literal["error", "warning", "info"], code: str, msg: str
    ) -> None:
        issues.append(SkillIssue(severity=severity, code=code, message=msg, path=where))

    raw, body, problem = split_frontmatter(text)
    if raw is None:
        note("error", problem or "frontmatter-missing", "no complete frontmatter block")
        return None, issues
    data = load_yaml(
        raw, note, max_bytes=SKILL_MAX_FRONTMATTER_BYTES, max_depth=SKILL_MAX_YAML_DEPTH
    )
    if any(issue.severity == "error" for issue in issues):
        return None, issues

    given_name = _text(data, "name", "value-invalid", note)
    if not given_name:
        note("warning", "name-missing", "no usable 'name'; using the folder name")
    name = given_name or strip_unsafe_text(folder_name).strip()
    raw_description = data.get("description")
    description = (
        strip_unsafe_text(raw_description).strip()
        if isinstance(raw_description, str)
        else ""
    )
    if not description:
        missing = raw_description is None or isinstance(raw_description, str)
        note(
            "error",
            "description-missing" if missing else "description-invalid",
            "'description' is required and must be a non-empty string",
        )
    if len(name) > SKILL_NAME_MAX_CHARS or not _NAME_RE.fullmatch(name):
        note(
            "error",
            "name-invalid",
            f"name {name[:80]!r} must be lowercase letters and digits joined by "
            f"single hyphens, at most {SKILL_NAME_MAX_CHARS} characters",
        )
    if any(issue.severity == "error" for issue in issues):
        return None, issues

    if given_name and given_name != folder_name:
        note("warning", "name-dir-mismatch", f"name {name!r} differs from its folder")
    if any(word in name for word in _RESERVED_WORDS):
        note("warning", "name-reserved", "name holds a word reserved by Anthropic")
    if "<" in description or ">" in description:  # the name regex excludes them
        note("warning", "xml-chars", "description holds '<' or '>' (XML tags)")
    if len(description) > SKILL_DESCRIPTION_MAX_CHARS:
        note(
            "warning",
            "description-long",
            f"description has {len(description)} characters; "
            f"the standard allows {SKILL_DESCRIPTION_MAX_CHARS}",
        )
    compatibility = _text(data, "compatibility", "compatibility-invalid", note)
    if len(compatibility) > SKILL_COMPATIBILITY_MAX_CHARS:
        note(
            "warning", "compatibility-long", "compatibility is over the allowed length"
        )

    given_metadata = data.get("metadata")
    metadata = (
        {
            k: v
            for k, v in given_metadata.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        if isinstance(given_metadata, dict)
        else {}
    )
    if given_metadata is not None and (
        not isinstance(given_metadata, dict) or len(metadata) != len(given_metadata)
    ):
        note("warning", "metadata-invalid", "'metadata' must map strings to strings")
    tools = data.get("allowed-tools")
    allowed: tuple[str, ...] = ()
    if isinstance(tools, str):
        allowed = tuple(strip_unsafe_text(tools).split())
    elif isinstance(tools, list) and all(isinstance(tool, str) for tool in tools):
        allowed = tuple(tools)
    elif tools is not None:
        note("warning", "allowed-tools-format", "'allowed-tools' ignored: not text")
    extras = tuple(
        capped_line(key, 64) or f"<{type(key).__name__}>"
        for key in data
        if key not in _KNOWN_KEYS
    )
    if extras:
        note(
            "warning", "key-unknown", f"keys outside the standard: {', '.join(extras)}"
        )

    line_count = body.count("\n") + (1 if body and not body.endswith("\n") else 0)
    if line_count > SKILL_BODY_MAX_LINES:
        note("info", "body-long", f"body has {line_count} lines; guidance is 500")
    skill = AgentSkill(
        name=name,
        description=description,
        directory=directory,
        skill_md=skill_md,
        license=_text(data, "license", "license-invalid", note) or None,
        compatibility=compatibility or None,
        metadata=metadata,
        allowed_tools=allowed,
        extra_keys=extras,
        line_count=line_count,
        resources=resources,
    )
    return skill, issues
