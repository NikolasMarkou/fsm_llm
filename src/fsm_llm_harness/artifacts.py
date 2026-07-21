"""
Typed models and Markdown (de)serializers for the protocol's on-disk artifacts.

The protocol's memory is a directory of Markdown files ("Context Window = RAM,
Filesystem = Disk").  This module is the boundary between that text and typed
Python: every artifact kind gets a pydantic model with :meth:`Artifact.to_markdown`
and :meth:`Artifact.from_markdown`.

Three rules shape everything here.

1. **Isomorphism, not byte-compatibility** (decisions.md D-010).  The models
   target the same section NAMES, the same ORDER, and the same strict grammars
   the protocol documents -- ``## D-NNN | PHASE | YYYY-MM-DD``, the 5-bullet
   Verdict, the 8-field changelog line, the 11 ``plan.md`` sections of
   :data:`~.constants.PlanSchema.SECTIONS`.  They do NOT chase the external
   ``validate-plan.mjs`` parser byte-for-byte; that tool lives outside this
   repository and no test here could prove parity with it.  The property that
   IS tested is ``from_markdown(to_markdown(model)) == model`` for every kind.

2. **Fail closed.**  A parser either returns a fully-populated model or raises
   :class:`~.exceptions.HarnessArtifactError`.  It never returns a half-filled
   model, and it never silently drops a section it did not understand -- a
   protocol gate that reads a half-parsed artifact is a gate that opens on
   noise.

3. **Contracts are data.**  The 6 Presentation Contracts, the 9 decision entry
   types, the 3 mandatory Additional Checks and the 5 Verdict bullets live here
   as required-field/floor tables, not as rendering logic.  ``harness.py``
   renders; this module only says what a rendering must contain.

Serialization is CANONICAL: ``to_markdown`` emits one normal form, and
``from_markdown`` is its inverse on that form.  Reading a hand-written file and
re-serializing it may move a stray line (a changelog note written below the
entries is re-emitted above them, for instance) -- that is isomorphism doing its
job, not a defect.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, NoReturn, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from fsm_llm.logging import logger

from .constants import ArtifactNames, Defaults, HarnessStates, PlanSchema
from .exceptions import HarnessArtifactError

__all__ = [
    "ARTIFACT_MODELS",
    "DECISION_ENTRY_SCHEMAS",
    "MANDATORY_ADDITIONAL_CHECKS",
    "PRESENTATION_CONTRACTS",
    "REJECTED_EVIDENCE",
    "VERDICT_BULLETS",
    "VERDICT_RECOMMENDATIONS",
    "Artifact",
    "ChangelogDoc",
    "ChangelogEntry",
    "CheckpointDoc",
    "ChecklistItem",
    "ConsolidatedDoc",
    "CriterionRow",
    "DecisionEntry",
    "DecisionsDoc",
    "FindingsIndexDoc",
    "FindingsTopicDoc",
    "IndexDoc",
    "IndexRow",
    "LessonsDoc",
    "PlanDoc",
    "PlanStep",
    "PresentationContract",
    "ProgressDoc",
    "Section",
    "SectionedArtifact",
    "StateDoc",
    "SummaryDoc",
    "SystemAtlasDoc",
    "VerificationDoc",
    "compression_marker_issues",
    "evidence_is_acceptable",
    "lesson_importance",
    "missing_entry_fields",
    "missing_floor_fields",
    "parse_changelog_line",
    "parse_markdown_table",
]

A = TypeVar("A", bound="Artifact")


# ---------------------------------------------------------------------------
# Failure funnel
# ---------------------------------------------------------------------------


def _fail(artifact: str, message: str, cause: Exception | None = None) -> NoReturn:
    """Raise :class:`HarnessArtifactError`; the single exit for every parser.

    ``exceptions.HarnessArtifactError``'s docstring names ``storage.py`` as its
    raiser.  It is raised here too: a malformed artifact IS a read/parse/schema
    failure, which is exactly what that type is for.
    """
    logger.debug(f"Artifact '{artifact}' rejected: {message}")
    raise HarnessArtifactError(artifact, message, cause=cause)


# ---------------------------------------------------------------------------
# Shared Markdown primitives
# ---------------------------------------------------------------------------

_H1_RE = re.compile(r"^#\s+(.+?)\s*$")
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^-\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\d+[a-z]?\.\s+")
_CHECKBOX_RE = re.compile(r"^\[([ xX])\]\s+(.*)$")
_COMMENT_OPEN = "<!--"
_COMMENT_CLOSE = "-->"


def _heading_key(name: str) -> str:
    """Normalize a ``## `` heading for required-section matching.

    ``## Iterations: 3`` and ``## Iterations`` are the same section; so are
    ``## Lockfiles snapshotted:`` and ``## Lockfiles snapshotted``.  The
    protocol's own templates use both shapes for the same artifact.
    """
    return name.split(":", 1)[0].strip()


def _lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").rstrip("\n").split("\n")


def _mask_comments(lines: Sequence[str]) -> list[bool]:
    """Return a per-line "inside an HTML comment" mask.

    ``decisions.md`` ships a schema EXAMPLE inside an HTML comment whose body
    contains a literal ``## D-001 | EXPLORE -> PLAN | YYYY-MM-DD`` header.  A
    parser that does not mask comments reads that example as a real entry with
    an unparseable date and rejects a perfectly valid file.
    """
    mask: list[bool] = []
    inside = False
    for line in lines:
        opened = _COMMENT_OPEN in line
        closed = _COMMENT_CLOSE in line
        mask.append(inside or (opened and not closed))
        if opened and not closed:
            inside = True
        elif closed:
            inside = False
    return mask


def _render_bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _parse_bullets(body: str) -> list[str]:
    """Parse ``- item`` bullets; indented lines continue the previous item."""
    items: list[str] = []
    for line in body.split("\n"):
        match = _BULLET_RE.match(line)
        if match is not None:
            items.append(match.group(1).rstrip())
        elif line.strip() and line[:1].isspace() and items:
            items[-1] = f"{items[-1]}\n{line.rstrip()}"
    return items


def _parse_numbered(body: str) -> list[str]:
    """Parse ``N. ...`` numbered items; indented lines continue the previous one."""
    items: list[str] = []
    for line in body.split("\n"):
        if _NUMBERED_RE.match(line) is not None:
            items.append(line.rstrip())
        elif items and line.strip():
            items[-1] = f"{items[-1]}\n{line.rstrip()}"
    return items


def parse_markdown_table(body: str) -> list[list[str]]:
    """Return a pipe table's data rows as cell lists (header/separator dropped)."""
    rows: list[list[str]] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set("-: ") and cell for cell in cells):
            continue  # separator row
        rows.append(cells)
    return rows[1:] if rows else rows


def compression_marker_issues(text: str) -> list[str]:
    """Report unbalanced / nested / duplicate ``<!-- COMPRESSED-SUMMARY -->`` markers."""
    open_marker = Defaults.COMPRESSED_SUMMARY_MARKER
    close_marker = open_marker.replace("<!-- ", "<!-- /")
    issues: list[str] = []
    depth = 0
    seen = 0
    for number, line in enumerate(text.split("\n"), start=1):
        if open_marker in line:
            if depth:
                issues.append(f"nested compression marker at line {number}")
            depth += 1
            seen += 1
        elif close_marker in line:
            if not depth:
                issues.append(f"unmatched closing marker at line {number}")
            else:
                depth -= 1
    if depth:
        issues.append("unclosed compression marker block")
    if seen > 1:
        issues.append(f"{seen} compression blocks (expected at most 1)")
    return issues


def _parse_sectioned(artifact: str, text: str) -> dict[str, Any]:
    """Split ``# H1`` + preamble + ``## `` sections; the shared document reader.

    Interface contract: returns ``{"title", "preamble", "sections"}`` where
    ``sections`` is an ordered list of ``{"name", "body"}`` dicts.  Raises
    :class:`HarnessArtifactError` (tagged with ``artifact``) when the text does
    not begin with an H1.  Headings inside HTML comments are not sections.
    """
    lines = _lines(text)
    head = _H1_RE.match(lines[0]) if lines else None
    if head is None:
        _fail(artifact, "first line must be an '# ' H1 heading")
    mask = _mask_comments(lines)
    preamble: list[str] = []
    sections: list[dict[str, Any]] = []
    for index in range(1, len(lines)):
        line = lines[index]
        heading = None if mask[index] else _H2_RE.match(line)
        if heading is not None:
            sections.append({"name": heading.group(1), "body": []})
        elif sections:
            sections[-1]["body"].append(line)
        else:
            preamble.append(line)
    return {
        "title": head.group(1),
        "preamble": "\n".join(preamble),
        "sections": [
            {"name": section["name"], "body": "\n".join(section["body"])}
            for section in sections
        ],
    }


# ---------------------------------------------------------------------------
# Base artifact
# ---------------------------------------------------------------------------


class Artifact(BaseModel):
    """Base for every on-disk artifact model.

    Subclasses implement :meth:`to_markdown` and :meth:`_parse`; the public
    :meth:`from_markdown` funnels every failure -- parse error OR schema
    violation -- into :class:`HarnessArtifactError`, so a caller never has to
    catch two error families to know an artifact was unusable.
    """

    model_config = ConfigDict(extra="forbid")

    #: The artifact's filename, used as the error's ``artifact`` field.
    ARTIFACT: ClassVar[str] = "artifact"

    def to_markdown(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def from_markdown(cls: type[A], text: str) -> A:
        try:
            data = cls._parse(text)
        except HarnessArtifactError:
            raise
        except Exception as exc:  # deliberate fail-closed funnel
            _fail(cls.ARTIFACT, f"could not be parsed: {exc}", exc)
        try:
            return cls(**data)
        except ValidationError as exc:
            _fail(cls.ARTIFACT, f"failed schema validation: {exc}", exc)


class Section(BaseModel):
    """A ``## `` section: its heading text and its (stripped) body."""

    model_config = ConfigDict(extra="forbid")

    name: str
    body: str = ""

    @field_validator("name", "body", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @property
    def key(self) -> str:
        return _heading_key(self.name)


class SectionedArtifact(Artifact):
    """An ``# H1`` + optional preamble + ordered ``## `` sections document.

    Interface contract for subclasses: set :attr:`REQUIRED_SECTIONS` to the
    heading keys (see :func:`_heading_key`) that must be present, and
    :attr:`REQUIRE_ORDER` when their relative order is load-bearing.
    :meth:`section_issues` reports every deviation; :attr:`ADVISORY_SECTIONS`
    decides whether a deviation also REJECTS the document.

    The split mirrors the source protocol's own severities and is deliberate.
    ``plan.md``'s 11 sections are an ERROR there, so :class:`PlanDoc` rejects --
    a plan missing "Failure Modes" is not a plan.  Topic-findings sections,
    ``progress.md``'s four sections and the Verdict bullets are WARNs there, so
    those models parse and report instead: the document is well-formed
    Markdown carrying a policy deviation, and a reader that refuses it cannot
    show anyone what is wrong with it.  Fail-closed applies to text that cannot
    be understood, not to text that can be understood and found wanting.
    """

    title: str
    preamble: str = ""
    sections: list[Section] = Field(default_factory=list)

    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = ()
    REQUIRE_ORDER: ClassVar[bool] = False
    #: When True, :meth:`section_issues` findings are reported, not raised.
    ADVISORY_SECTIONS: ClassVar[bool] = False

    @field_validator("title", "preamble", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def section_issues(self) -> list[str]:
        """Every way the section set deviates from the artifact's schema."""
        keys = [section.key for section in self.sections]
        issues: list[str] = []
        positions: list[int] = []
        for required in self.REQUIRED_SECTIONS:
            found = [index for index, key in enumerate(keys) if key == required]
            if not found:
                issues.append(f"missing required section '## {required}'")
                continue
            if len(found) > 1:
                issues.append(f"duplicate section '## {required}'")
            positions.append(found[0])
        if self.REQUIRE_ORDER and positions != sorted(positions):
            expected = ", ".join(self.REQUIRED_SECTIONS)
            issues.append(f"sections out of order; required order: {expected}")
        return issues

    @model_validator(mode="after")
    def _check_sections(self) -> SectionedArtifact:
        if not self.ADVISORY_SECTIONS:
            issues = self.section_issues()
            if issues:
                raise ValueError("; ".join(issues))
        return self

    def section(self, key: str) -> Section | None:
        """Return the section whose normalized heading is ``key``, if present."""
        for candidate in self.sections:
            if candidate.key == key:
                return candidate
        return None

    def body_of(self, key: str) -> str:
        section = self.section(key)
        return section.body if section is not None else ""

    def to_markdown(self) -> str:
        parts = [f"# {self.title}"]
        if self.preamble:
            parts.append(self.preamble)
        for section in self.sections:
            heading = f"## {section.name}"
            parts.append(f"{heading}\n{section.body}" if section.body else heading)
        return "\n\n".join(parts) + "\n"

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any]:
        return _parse_sectioned(cls.ARTIFACT, text)


# ---------------------------------------------------------------------------
# state.md
# ---------------------------------------------------------------------------

_STATE_H1_PREFIX = "Current State: "
_STATE_ITERATION = "Iteration"
_STATE_STEP = "Current Plan Step"
_STATE_CHECKLIST = "Pre-Step Checklist (reset before each EXECUTE step)"
_STATE_ATTEMPTS = "Fix Attempts (resets per plan step)"
_STATE_MANIFEST = "Change Manifest (current iteration)"
_STATE_LAST_TRANSITION = "Last Transition"
_STATE_HISTORY = "Transition History"
_STATE_NO_ATTEMPTS = "(none yet for current step)"
_SKILL_STAMP_RE = re.compile(r"^\*Skill:\s*(.+?)\s*\*$")

#: The protocol's Fix-Attempts line grammar, applied to the bullet text.
#: Comma-optional and plural-tolerant, matching both enforcement tiers.
#:
#: The optional letter suffix (`Step 4b, attempt 1`) is a deliberate widening of
#: the source grammar, and the direction matters: a leash counter may over-count
#: safely (it only halts sooner) but must never under-count, and plans DO carry
#: inserted labels like `4b`.  Do not narrow this back to `\d+`.
_ATTEMPT_RE = re.compile(
    r"^(Step\s+\d+[a-z]?[,\s]+attempts?\s+\d+|Attempts?\s+\d+)", re.I
)


class ChecklistItem(BaseModel):
    """One ``- [ ]`` / ``- [x]`` pre-step checklist row."""

    model_config = ConfigDict(extra="forbid")

    checked: bool = False
    text: str


class StateDoc(Artifact):
    """``state.md`` -- the single source of truth for "where am I?"."""

    ARTIFACT: ClassVar[str] = ArtifactNames.STATE

    state: str
    skill_version: str | None = None
    iteration: int = Field(default=0, ge=0)
    current_step: str = ""
    checklist: list[ChecklistItem] = Field(default_factory=list)
    fix_attempts: list[str] = Field(default_factory=list)
    change_manifest: list[str] = Field(default_factory=list)
    last_transition: str = ""
    transition_history: list[str] = Field(default_factory=list)

    @field_validator("state", mode="before")
    @classmethod
    def _known_state(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() not in HarnessStates.ALL:
            raise ValueError(f"unknown protocol state '{value}'")
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("fix_attempts", mode="before")
    @classmethod
    def _drop_placeholder(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [item for item in value if item != _STATE_NO_ATTEMPTS]
        return value

    @property
    def fix_attempt_count(self) -> int:
        """Recorded fix attempts, counted by the protocol's own line grammar."""
        return sum(1 for line in self.fix_attempts if _ATTEMPT_RE.match(line))

    def to_markdown(self) -> str:
        attempts = self.fix_attempts or [_STATE_NO_ATTEMPTS]
        parts = [f"# {_STATE_H1_PREFIX}{self.state.upper()}"]
        if self.skill_version:
            parts.append(f"*Skill: {self.skill_version}*")
        body = [
            f"## {_STATE_ITERATION}: {self.iteration}",
            f"## {_STATE_STEP}: {self.current_step}",
            f"## {_STATE_CHECKLIST}",
            _render_bullets(
                [
                    f"[{'x' if item.checked else ' '}] {item.text}"
                    for item in self.checklist
                ]
            ),
            f"## {_STATE_ATTEMPTS}",
            _render_bullets(attempts),
            f"## {_STATE_MANIFEST}",
            _render_bullets(self.change_manifest),
            f"## {_STATE_LAST_TRANSITION}: {self.last_transition}",
            f"## {_STATE_HISTORY}:",
            _render_bullets(self.transition_history),
        ]
        parts.extend(chunk for chunk in body if chunk)
        return "\n".join(parts) + "\n"

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any]:
        raw = _parse_sectioned(cls.ARTIFACT, text)
        title = str(raw["title"])
        if not title.startswith(_STATE_H1_PREFIX):
            _fail(cls.ARTIFACT, f"H1 must start with '# {_STATE_H1_PREFIX}'")
        skill = _SKILL_STAMP_RE.match(str(raw["preamble"]).strip())
        blocks = {
            _heading_key(str(section["name"])): section for section in raw["sections"]
        }

        def value_of(key: str) -> str:
            section = blocks.get(key)
            if section is None:
                _fail(cls.ARTIFACT, f"missing '## {key}:' line")
            name = str(section["name"])
            return name.split(":", 1)[1].strip() if ":" in name else ""

        def body_of(key: str) -> str:
            section = blocks.get(key)
            return str(section["body"]) if section is not None else ""

        iteration_text = value_of(_STATE_ITERATION)
        if not iteration_text.isdigit():
            _fail(cls.ARTIFACT, f"'## {_STATE_ITERATION}:' must be an integer")

        checklist: list[dict[str, Any]] = []
        for item in _parse_bullets(body_of(_heading_key(_STATE_CHECKLIST))):
            box = _CHECKBOX_RE.match(item)
            if box is None:
                _fail(cls.ARTIFACT, f"checklist row is not a checkbox: {item!r}")
            checklist.append(
                {"checked": box.group(1).lower() == "x", "text": box.group(2)}
            )

        return {
            "state": title[len(_STATE_H1_PREFIX) :].strip(),
            "skill_version": skill.group(1) if skill is not None else None,
            "iteration": int(iteration_text),
            "current_step": value_of(_STATE_STEP),
            "checklist": checklist,
            "fix_attempts": _parse_bullets(body_of(_heading_key(_STATE_ATTEMPTS))),
            "change_manifest": _parse_bullets(body_of(_heading_key(_STATE_MANIFEST))),
            "last_transition": value_of(_STATE_LAST_TRANSITION),
            "transition_history": _parse_bullets(body_of(_STATE_HISTORY)),
        }


# ---------------------------------------------------------------------------
# plan.md
# ---------------------------------------------------------------------------

#: DOTALL: a plan step's annotations (`[RISK: ...]`, `[deps: ...]`) routinely sit
#: on a continuation line, so the text group must span the whole item.
_STEP_RE = re.compile(r"^(\d+[a-z]?)\.\s+\[([ x])\]\s+(.*)$", re.I | re.S)
_RISK_RE = re.compile(r"\[RISK:\s*([^\]]+)\]", re.I)
_DEPS_RE = re.compile(r"\[deps:\s*([^\]]*)\]", re.I)
_IRREVERSIBLE = "[IRREVERSIBLE]"


class PlanStep(BaseModel):
    """A projection of one ``## Steps`` entry -- never stored, always derived."""

    model_config = ConfigDict(extra="forbid")

    number: str
    done: bool
    text: str
    risk: str | None = None
    deps: str | None = None
    irreversible: bool = False


class PlanDoc(SectionedArtifact):
    """``plan.md`` -- the 11 required sections, in the required order."""

    ARTIFACT: ClassVar[str] = ArtifactNames.PLAN
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = PlanSchema.SECTIONS
    REQUIRE_ORDER: ClassVar[bool] = True

    def steps(self) -> list[PlanStep]:
        """Parse the ``## Steps`` body into annotated step records."""
        parsed: list[PlanStep] = []
        for item in _parse_numbered(self.body_of("Steps")):
            match = _STEP_RE.match(item)
            if match is None:
                continue
            text = match.group(3)
            risk = _RISK_RE.search(text)
            deps = _DEPS_RE.search(text)
            parsed.append(
                PlanStep(
                    number=match.group(1),
                    done=match.group(2).lower() == "x",
                    text=text,
                    risk=risk.group(1).strip() if risk else None,
                    deps=deps.group(1).strip() if deps else None,
                    irreversible=_IRREVERSIBLE in text,
                )
            )
        return parsed


# ---------------------------------------------------------------------------
# decisions.md
# ---------------------------------------------------------------------------

_DECISION_HEADER_RE = re.compile(
    r"^##\s+(D-\d{3})\s*\|\s*([^|]+?)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*$"
)
_DECISION_FIELD_RE = re.compile(r"^\*\*([^*]+)\*\*:(.*)$")
_PLAN_PREAMBLE_RE = re.compile(r"^\*Plan:\s*(\S+?)\s*\*$")
_TRADE_OFF = "Trade-off"
_AT_THE_COST_OF = "at the cost of"

#: Required field sets for the 9 documented decision entry types.  Every type
#: also carries the header and a ``**Trade-off**:`` line; those are enforced by
#: :class:`DecisionEntry` itself rather than repeated in each row.
DECISION_ENTRY_SCHEMAS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "explore-to-plan": ("Context", "Decision", "Trade-off", "Reasoning"),
        "reflect-to-pivot": (
            "Context",
            "What Failed",
            "What Was Learned",
            "Root Cause Analysis",
            "Complexity Assessment",
            "Decision",
            "Trade-off",
            "Reasoning",
        ),
        "reflect-no-pivot": (
            "Context",
            "Devil's Advocate Note",
            "Decision",
            "Trade-off",
            "Reasoning",
        ),
        "scope-drift": (
            "Context",
            "Unplanned Files",
            "Justification",
            "Decision",
            "Trade-off",
        ),
        "falsification-signal": (
            "Context",
            "Signal Fired",
            "Observation",
            "Decision",
            "Trade-off",
        ),
        "ghost-constraint": (
            "Context",
            "Constraint",
            "Why No Longer Applies",
            "Solution-Space Change",
            "Decision",
            "Trade-off",
        ),
        "three-strike": (
            "Context",
            "3-STRIKE TRIGGERED",
            "Three Attempts",
            "Decision",
            "Trade-off",
        ),
        "simplification-check": (
            "Context",
            "6 Check Answers",
            "Blocker Found",
            "Decision",
            "Trade-off",
        ),
        "devils-advocate": (
            "Context",
            "Strongest Counter-argument",
            "Why Pursuing Anyway",
            "Decision",
            "Trade-off",
        ),
    }
)


class DecisionEntry(BaseModel):
    """One ``## D-NNN | PHASE | YYYY-MM-DD`` entry with its ordered fields."""

    model_config = ConfigDict(extra="forbid")

    id: str
    phase: str
    date: str
    fields: list[tuple[str, str]] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if re.fullmatch(r"D-\d{3}", value) is None:
            raise ValueError(f"decision id '{value}' is not D-NNN")
        return value

    @field_validator("date")
    @classmethod
    def _date_shape(cls, value: str) -> str:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError(f"decision date '{value}' is not YYYY-MM-DD")
        return value

    @field_validator("phase")
    @classmethod
    def _phase_shape(cls, value: str) -> str:
        if not value.strip() or "|" in value:
            raise ValueError(f"decision phase '{value}' is empty or contains '|'")
        return value.strip()

    @model_validator(mode="after")
    def _trade_off_required(self) -> DecisionEntry:
        for name, value in self.fields:
            if name == _TRADE_OFF:
                # Whitespace-normalized: the phrase routinely wraps across a
                # line break in real entries, and a hard-wrapped trade-off is
                # still a trade-off.
                if _AT_THE_COST_OF not in " ".join(value.split()):
                    raise ValueError(
                        f"{self.id}: **Trade-off**: must phrase 'X "
                        f"**{_AT_THE_COST_OF}** Y'"
                    )
                return self
        raise ValueError(f"{self.id}: missing required '**{_TRADE_OFF}**:' line")

    @property
    def number(self) -> int:
        return int(self.id.split("-", 1)[1])

    def field(self, name: str) -> str | None:
        for candidate, value in self.fields:
            if candidate == name:
                return value
        return None

    def to_markdown(self) -> str:
        head = f"## {self.id} | {self.phase} | {self.date}"
        rendered = [head]
        for name, value in self.fields:
            # A value that opens on its own line (a numbered RCA block, say)
            # keeps its leading newline; anything else is separated by one space.
            suffix = value if value.startswith("\n") else (f" {value}" if value else "")
            rendered.append(f"**{name}**:{suffix}")
        return "\n".join(rendered)


def missing_entry_fields(entry: DecisionEntry, entry_type: str) -> tuple[str, ...]:
    """Required field names absent from ``entry`` for a documented entry type.

    Matching is by prefix so a field written as ``**3-STRIKE TRIGGERED on
    `roles.py`**:`` still satisfies the ``3-STRIKE TRIGGERED`` requirement.
    """
    required = DECISION_ENTRY_SCHEMAS.get(entry_type)
    if required is None:
        raise KeyError(f"unknown decision entry type '{entry_type}'")
    present = [name for name, _ in entry.fields]
    return tuple(
        name
        for name in required
        if not any(candidate.startswith(name) for candidate in present)
    )


class DecisionsDoc(Artifact):
    """``decisions.md`` -- append-only, plan-id preambled, D-NNN sequential."""

    ARTIFACT: ClassVar[str] = ArtifactNames.DECISIONS

    title: str = "Decision Log"
    plan_id: str
    skill_version: str | None = None
    preamble: str = ""
    entries: list[DecisionEntry] = Field(default_factory=list)

    @field_validator("title", "plan_id", "preamble", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _sequential(self) -> DecisionsDoc:
        for position, entry in enumerate(self.entries, start=1):
            if entry.number != position:
                raise ValueError(
                    f"decision ids must run D-001.. with no gaps; "
                    f"found {entry.id} at position {position}"
                )
        return self

    def entry(self, decision_id: str) -> DecisionEntry | None:
        for candidate in self.entries:
            if candidate.id == decision_id:
                return candidate
        return None

    def to_markdown(self) -> str:
        parts = [f"# {self.title}", f"*Plan: {self.plan_id}*"]
        if self.skill_version:
            parts.append(f"*Skill: {self.skill_version}*")
        head = "\n".join(parts)
        blocks = [head]
        if self.preamble:
            blocks.append(self.preamble)
        blocks.extend(entry.to_markdown() for entry in self.entries)
        return "\n\n".join(blocks) + "\n"

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any]:
        lines = _lines(text)
        if not lines or _H1_RE.match(lines[0]) is None:
            _fail(cls.ARTIFACT, "first line must be an '# ' H1 heading")
        title = _H1_RE.match(lines[0]).group(1)  # type: ignore[union-attr]
        if len(lines) < 2 or _PLAN_PREAMBLE_RE.match(lines[1].strip()) is None:
            _fail(cls.ARTIFACT, "second line must be the '*Plan: <plan-id>*' preamble")
        plan_id = _PLAN_PREAMBLE_RE.match(lines[1].strip()).group(1)  # type: ignore[union-attr]
        cursor = 2
        skill = None
        if cursor < len(lines):
            stamp = _SKILL_STAMP_RE.match(lines[cursor].strip())
            if stamp is not None:
                skill = stamp.group(1)
                cursor += 1

        mask = _mask_comments(lines)
        preamble: list[str] = []
        entries: list[dict[str, Any]] = []
        for index in range(cursor, len(lines)):
            line = lines[index]
            if not mask[index] and line.startswith("## "):
                header = _DECISION_HEADER_RE.match(line)
                if header is None:
                    _fail(
                        cls.ARTIFACT,
                        "entry header must be exactly "
                        f"'## D-NNN | PHASE | YYYY-MM-DD'; got {line!r}",
                    )
                entries.append(
                    {
                        "id": header.group(1),
                        "phase": header.group(2),
                        "date": header.group(3),
                        "fields": [],
                    }
                )
            elif entries:
                field_match = _DECISION_FIELD_RE.match(line)
                if field_match is not None:
                    value = field_match.group(2)
                    entries[-1]["fields"].append(
                        [
                            field_match.group(1),
                            value[1:] if value.startswith(" ") else value,
                        ]
                    )
                elif entries[-1]["fields"]:
                    name, value = entries[-1]["fields"][-1]
                    entries[-1]["fields"][-1] = [name, f"{value}\n{line.rstrip()}"]
                elif line.strip():
                    _fail(
                        cls.ARTIFACT,
                        f"{entries[-1]['id']}: text before the first '**Field**:' line",
                    )
            else:
                preamble.append(line)

        return {
            "title": title,
            "plan_id": plan_id,
            "skill_version": skill,
            "preamble": "\n".join(preamble).strip(),
            "entries": [
                {
                    "id": entry["id"],
                    "phase": entry["phase"],
                    "date": entry["date"],
                    "fields": [
                        (name, value.rstrip()) for name, value in entry["fields"]
                    ],
                }
                for entry in entries
            ],
        }


# ---------------------------------------------------------------------------
# findings.md + findings/{topic}.md
# ---------------------------------------------------------------------------

_INDEX_ENTRY_RE = re.compile(r"^(?:\d+\.|[-*])\s+(.+)$")


class FindingsIndexDoc(SectionedArtifact):
    """``findings.md`` -- summary and index only; detail lives in ``findings/``."""

    ARTIFACT: ClassVar[str] = ArtifactNames.FINDINGS_INDEX
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = ("Index", "Key Constraints")
    ADVISORY_SECTIONS: ClassVar[bool] = True

    def index_entries(self) -> list[str]:
        body = self.body_of("Index")
        return [
            match.group(1).strip()
            for match in (_INDEX_ENTRY_RE.match(line) for line in body.split("\n"))
            if match is not None
        ]

    @property
    def findings_count(self) -> int:
        """Indexed findings -- the EXPLORE -> PLAN gate's counted quantity."""
        return len(self.index_entries())


class FindingsTopicDoc(SectionedArtifact):
    """``findings/{topic}.md`` -- all five required sections, plus an optional sixth."""

    ARTIFACT: ClassVar[str] = f"{ArtifactNames.FINDINGS_DIR}/<topic>.md"
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "Summary",
        "Key Findings",
        "Constraints",
        "Code Patterns",
        "Risks & Unknowns",
    )
    ADVISORY_SECTIONS: ClassVar[bool] = True
    #: The documented optional sixth section, promoted by the orchestrator to a
    #: ``[CONTRADICTED iter-N]`` line in ``findings.md``.
    OPTIONAL_SECTION: ClassVar[str] = "Atlas Contradictions"


# ---------------------------------------------------------------------------
# progress.md
# ---------------------------------------------------------------------------


class ProgressDoc(SectionedArtifact):
    """``progress.md`` -- a flat checklist in four fixed sections."""

    ARTIFACT: ClassVar[str] = ArtifactNames.PROGRESS
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "Completed",
        "In Progress",
        "Remaining",
        "Blocked",
    )
    REQUIRE_ORDER: ClassVar[bool] = True
    ADVISORY_SECTIONS: ClassVar[bool] = True

    def items(self, key: str) -> list[str]:
        return _parse_bullets(self.body_of(key))


# ---------------------------------------------------------------------------
# verification.md
# ---------------------------------------------------------------------------

#: Rows that must appear in ``## Additional Checks`` on every REFLECT pass.
MANDATORY_ADDITIONAL_CHECKS: tuple[str, ...] = (
    "Regression",
    "Scope drift",
    "Diff review",
)

#: The 5 Verdict bullets, in the order the protocol requires them.
VERDICT_BULLETS: tuple[str, ...] = (
    "Criteria passed",
    "Regressions",
    "Scope drift",
    "Simplification blockers",
    "Recommendation",
)

#: The 4 legal Verdict recommendations.
VERDICT_RECOMMENDATIONS: tuple[str, ...] = ("CLOSE", "PIVOT", "EXPLORE", "EXECUTE")

#: Evidence strings the protocol rejects outright.
REJECTED_EVIDENCE: frozenset[str] = frozenset(
    {"looks good", "seems to work", "lgtm", "yes", "ok", "done", "fine", "n/a", "-", ""}
)

_EVIDENCE_TEST_COUNT_RE = re.compile(r"\d+\s*/\s*\d+")
_EVIDENCE_EXIT_RE = re.compile(r"\bexit\s+(?:code\s+)?\d+", re.I)
_EVIDENCE_MANUAL_RE = re.compile(r"manual review\s*[-—:]\s*\S+", re.I)


def evidence_is_acceptable(evidence: str) -> bool:
    """True when ``evidence`` matches one of the 3 accepted shapes.

    Accepted: (a) a test-output count (``47/47``), (b) an exit code with an
    excerpt (``exit 0; "Build succeeded"``), (c) an explicit
    ``manual review - observed X``.  Everything else -- including ``looks
    good``, ``LGTM``, a bare word, or an empty cell -- is rejected.
    """
    text = evidence.strip().strip("*_` ")
    if not text or text.lower() in REJECTED_EVIDENCE or len(text.split()) < 2:
        return False
    return bool(
        _EVIDENCE_TEST_COUNT_RE.search(text)
        or _EVIDENCE_EXIT_RE.search(text)
        or _EVIDENCE_MANUAL_RE.search(text)
    )


class CriterionRow(BaseModel):
    """One ``## Criteria Verification`` row."""

    model_config = ConfigDict(extra="forbid")

    number: str
    criterion: str
    method: str = ""
    command: str = ""
    result: str = ""
    evidence: str = ""

    @property
    def evidence_ok(self) -> bool:
        return evidence_is_acceptable(self.evidence)


class VerificationDoc(SectionedArtifact):
    """``verification.md`` -- rewritten each REFLECT, never appended to."""

    ARTIFACT: ClassVar[str] = ArtifactNames.VERIFICATION
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "Criteria Verification",
        "Additional Checks",
        "Not Verified",
        "Verdict",
    )
    REQUIRE_ORDER: ClassVar[bool] = True
    ADVISORY_SECTIONS: ClassVar[bool] = True

    def criteria(self) -> list[CriterionRow]:
        rows: list[CriterionRow] = []
        for cells in parse_markdown_table(self.body_of("Criteria Verification")):
            padded = list(cells) + [""] * (6 - len(cells))
            rows.append(
                CriterionRow(
                    number=padded[0],
                    criterion=padded[1],
                    method=padded[2],
                    command=padded[3],
                    result=padded[4],
                    evidence=padded[5],
                )
            )
        return rows

    def additional_check_names(self) -> list[str]:
        return [
            cells[0]
            for cells in parse_markdown_table(self.body_of("Additional Checks"))
        ]

    def missing_additional_checks(self) -> tuple[str, ...]:
        present = {name.strip().lower() for name in self.additional_check_names()}
        return tuple(
            check
            for check in MANDATORY_ADDITIONAL_CHECKS
            if check.lower() not in present
        )

    def verdict_bullets(self) -> list[tuple[str, str]]:
        parsed: list[tuple[str, str]] = []
        for item in _parse_bullets(self.body_of("Verdict")):
            label, _, value = item.partition(":")
            parsed.append((label.strip().strip("*"), value.strip()))
        return parsed

    def verdict_issues(self) -> list[str]:
        """Report every way the Verdict deviates from the 5-bullet contract."""
        labels = [label for label, _ in self.verdict_bullets()]
        issues: list[str] = []
        positions: list[int] = []
        for required in VERDICT_BULLETS:
            if required not in labels:
                issues.append(f"Verdict is missing the '{required}' bullet")
            else:
                positions.append(labels.index(required))
        if positions != sorted(positions):
            issues.append("Verdict bullets are out of the required order")
        recommendation = dict(self.verdict_bullets()).get("Recommendation", "")
        target = recommendation.lstrip("→> ").split()[:1]
        if target and target[0].upper() not in VERDICT_RECOMMENDATIONS:
            issues.append(
                f"Recommendation '{recommendation}' is not one of {VERDICT_RECOMMENDATIONS}"
            )
        return issues


# ---------------------------------------------------------------------------
# changelog.md
# ---------------------------------------------------------------------------

_CHANGELOG_FIELDS = 8
_CHANGELOG_SEP = " | "
_CHANGELOG_SPEC: Mapping[str, re.Pattern[str]] = MappingProxyType(
    {
        "timestamp": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
        # `step` is `iter-N/step-M` with a NUMERIC M. Do NOT relax this to
        # accept `step-4b` "because the plan has a step 4b": this repo hit that
        # exact constraint in iteration 1 and resolved it the way the protocol
        # intends -- by recording step 4b as `iter-1/step-18` and letting the
        # `4b` label stay positional (recorded in this plan's own changelog.md
        # header note, and used by the predecessor for its 7a-7f = 18-23).
        # Relaxing the grammar here would make the ledger's step field
        # unjoinable with state.md's numeric step counter, which is the only
        # thing that field is for.
        "step": re.compile(r"^iter-\d+/step-\d+$"),
        "commit": re.compile(r"^(?:[0-9a-f]{7,40}|uncommitted)$"),
        "path": re.compile(r"^\S.*$"),
        "op": re.compile(
            r"^(?:CREATE\(\+\d+\)|EDIT\(\+\d+,-\d+\)|DELETE\(-\d+\)"
            r"|RENAME\(.+\)|REVERT\(.+\))$"
        ),
        "radius": re.compile(r"^radius:(?:(?:LOW|MED|HIGH)\(\d+\)|UNKNOWN\(.+\))$"),
        "decision_ref": re.compile(r"^(?:D-\d{3}|-)$"),
        "reason": re.compile(r"^\S.*$"),
    }
)
_CHANGELOG_FIELD_ORDER: tuple[str, ...] = tuple(_CHANGELOG_SPEC)


class ChangelogEntry(BaseModel):
    """One 8-field, pipe-delimited per-edit ledger line."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    step: str
    commit: str
    path: str
    op: str
    radius: str
    decision_ref: str = "-"
    reason: str

    @model_validator(mode="after")
    def _fields_match_spec(self) -> ChangelogEntry:
        for name, pattern in _CHANGELOG_SPEC.items():
            value = str(getattr(self, name))
            if pattern.match(value) is None:
                raise ValueError(f"changelog field '{name}' is malformed: {value!r}")
        return self

    def to_markdown(self) -> str:
        return _CHANGELOG_SEP.join(
            str(getattr(self, name)) for name in _CHANGELOG_FIELD_ORDER
        )


def parse_changelog_line(line: str) -> ChangelogEntry:
    """Parse one ledger line, or raise :class:`HarnessArtifactError`.

    The reason field absorbs any trailing ``|`` characters (the protocol
    tolerates pipes in prose), so the split is bounded at 7 separators.
    """
    parts = line.strip().split(_CHANGELOG_SEP, _CHANGELOG_FIELDS - 1)
    if len(parts) != _CHANGELOG_FIELDS:
        _fail(
            ArtifactNames.CHANGELOG,
            f"line has {len(parts)} of {_CHANGELOG_FIELDS} fields: {line.strip()!r}",
        )
    try:
        return ChangelogEntry(
            **dict(
                zip(
                    _CHANGELOG_FIELD_ORDER,
                    (part.strip() for part in parts),
                    strict=True,
                )
            )
        )
    except ValidationError as exc:
        _fail(ArtifactNames.CHANGELOG, f"malformed line {line.strip()!r}: {exc}", exc)


class ChangelogDoc(Artifact):
    """``changelog.md`` -- header, optional notes, then append-only entries.

    Notes (compression markers, inline ``- (compressed: ...)`` summaries) are
    re-emitted between the header and the entries regardless of where they were
    read from.  That is the one place this module trades position fidelity for
    a canonical form; entry ORDER, which is the ledger's actual semantics, is
    always preserved.
    """

    ARTIFACT: ClassVar[str] = ArtifactNames.CHANGELOG

    title: str = "Changelog"
    header: str = ""
    notes: list[str] = Field(default_factory=list)
    entries: list[ChangelogEntry] = Field(default_factory=list)

    @field_validator("title", "header", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def to_markdown(self) -> str:
        parts = [f"# {self.title}"]
        if self.header:
            parts.append(self.header)
        parts.extend(self.notes)
        parts.extend(entry.to_markdown() for entry in self.entries)
        return "\n".join(parts) + "\n"

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any]:
        lines = _lines(text)
        if not lines or _H1_RE.match(lines[0]) is None:
            _fail(cls.ARTIFACT, "first line must be an '# ' H1 heading")
        title = _H1_RE.match(lines[0]).group(1)  # type: ignore[union-attr]
        header: list[str] = []
        notes: list[str] = []
        entries: list[ChangelogEntry] = []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            # Header lines are matched FIRST: the protocol's own changelog
            # header quotes the pipe-delimited FORMAT, so an entry-shaped test
            # applied first would read documentation as a malformed ledger line.
            if stripped.startswith("*") and not entries and not notes:
                header.append(stripped)
            elif _CHANGELOG_SEP in stripped:
                entries.append(parse_changelog_line(stripped))
            else:
                notes.append(stripped)
        return {
            "title": title,
            "header": "\n".join(header),
            "notes": notes,
            "entries": [entry.model_dump() for entry in entries],
        }


# ---------------------------------------------------------------------------
# checkpoints/cp-NNN-iterN.md and summary.md
# ---------------------------------------------------------------------------


class CheckpointDoc(SectionedArtifact):
    """``checkpoints/cp-NNN-iterN.md`` -- a restore point.

    ``## Lockfiles snapshotted:`` is mandatory even when it holds the single
    line ``- none (no package manager touched)``; its absence is what marks a
    checkpoint as malformed rather than merely manifest-free.
    """

    ARTIFACT: ClassVar[str] = f"{ArtifactNames.CHECKPOINTS_DIR}/cp-NNN-iterN.md"
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "Created",
        "Git State",
        "Lockfiles snapshotted",
        "Rollback",
    )
    ADVISORY_SECTIONS: ClassVar[bool] = True


class SummaryDoc(SectionedArtifact):
    """``summary.md`` -- written once, at CLOSE."""

    ARTIFACT: ClassVar[str] = ArtifactNames.SUMMARY
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "Outcome",
        "Key Decisions",
        "Files Changed",
        "Decision Anchors Registry",
        "Lessons",
    )
    ADVISORY_SECTIONS: ClassVar[bool] = True


# ---------------------------------------------------------------------------
# Cross-plan tier
# ---------------------------------------------------------------------------

_PLAN_ID_RE = re.compile(r"^plan[-_]\d{4}-\d{2}-\d{2}(?:T\d{6})?[-_][0-9a-f]{8}$")
_IMPORTANCE_RE = re.compile(r"\[I:([1-5])\]")


def lesson_importance(line: str) -> int:
    """Return a LESSONS bullet's ``[I:N]`` tag; untagged bullets are ``[I:3]``."""
    match = _IMPORTANCE_RE.search(line)
    return int(match.group(1)) if match is not None else 3


class ConsolidatedDoc(SectionedArtifact):
    """``plans/FINDINGS.md`` / ``plans/DECISIONS.md`` -- newest plan first."""

    ARTIFACT: ClassVar[str] = ArtifactNames.CROSS_FINDINGS

    #: Most-recent plan sections kept by the sliding window.
    WINDOW: ClassVar[int] = Defaults.SLIDING_WINDOW_PLANS
    #: Line count above which a compressed summary block is inserted.
    COMPRESS_LINES: ClassVar[int] = Defaults.CONSOLIDATED_COMPRESS_LINES

    def plan_ids(self) -> list[str]:
        return [
            section.name
            for section in self.sections
            if _PLAN_ID_RE.match(section.name) is not None
        ]

    def marker_issues(self) -> list[str]:
        return compression_marker_issues(self.to_markdown())


class LessonsDoc(SectionedArtifact):
    """``plans/LESSONS.md`` -- rewritten at CLOSE, hard-capped, ``[I:N]`` tagged."""

    ARTIFACT: ClassVar[str] = ArtifactNames.LESSONS
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "Recurring Patterns",
        "Failed Approaches (+ why)",
        "Successful Strategies",
        "Codebase Gotchas",
    )
    ADVISORY_SECTIONS: ClassVar[bool] = True

    #: Hard line cap. Enforcement (trim + eviction) is ``storage.py``'s job;
    #: this module only carries the number and the ordering key.
    LINE_CAP: ClassVar[int] = Defaults.LESSONS_LINE_CAP
    #: Importance level that is never evicted, even over the cap.
    PROTECTED_IMPORTANCE: ClassVar[int] = Defaults.LESSONS_PROTECTED_IMPORTANCE

    def lessons(self) -> list[tuple[int, str]]:
        """Every bullet as ``(importance, text)``, in document order."""
        collected: list[tuple[int, str]] = []
        for section in self.sections:
            collected.extend(
                (lesson_importance(item), item) for item in _parse_bullets(section.body)
            )
        return collected

    @property
    def line_count(self) -> int:
        return len(self.to_markdown().rstrip("\n").split("\n"))

    def over_cap(self) -> bool:
        return self.line_count > self.LINE_CAP


class SystemAtlasDoc(SectionedArtifact):
    """``plans/SYSTEM.md`` -- the domain-neutral 6-section system atlas."""

    ARTIFACT: ClassVar[str] = ArtifactNames.SYSTEM
    REQUIRED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "Identity",
        "Components",
        "Boundaries",
        "Invariants",
        "Flows",
        "Known Patterns",
    )
    REQUIRE_ORDER: ClassVar[bool] = True
    ADVISORY_SECTIONS: ClassVar[bool] = True
    #: Present only when the system's domain is a codebase.
    OPTIONAL_SECTION: ClassVar[str] = "Codebase Specialization"
    LINE_CAP: ClassVar[int] = Defaults.SYSTEM_LINE_CAP

    @property
    def line_count(self) -> int:
        return len(self.to_markdown().rstrip("\n").split("\n"))

    def over_cap(self) -> bool:
        return self.line_count > self.LINE_CAP


class IndexRow(BaseModel):
    """One ``plans/INDEX.md`` row."""

    model_config = ConfigDict(extra="forbid")

    plan: str
    date: str = ""
    goal: str = ""
    topics: str = ""


class IndexDoc(Artifact):
    """``plans/INDEX.md`` -- the topic-to-directory table that survives trimming."""

    ARTIFACT: ClassVar[str] = ArtifactNames.INDEX

    title: str = "Plan Index"
    preamble: str = ""
    rows: list[IndexRow] = Field(default_factory=list)

    HEADER: ClassVar[tuple[str, ...]] = ("Plan", "Date", "Goal", "Key Topics")

    @field_validator("title", "preamble", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def to_markdown(self) -> str:
        parts = [f"# {self.title}"]
        if self.preamble:
            parts.append(self.preamble)
        table = [
            "| " + " | ".join(self.HEADER) + " |",
            "|" + "|".join("------" for _ in self.HEADER) + "|",
        ]
        table.extend(
            f"| {row.plan} | {row.date} | {row.goal} | {row.topics} |"
            for row in self.rows
        )
        parts.append("\n".join(table))
        return "\n\n".join(parts) + "\n"

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any]:
        lines = _lines(text)
        if not lines or _H1_RE.match(lines[0]) is None:
            _fail(cls.ARTIFACT, "first line must be an '# ' H1 heading")
        title = _H1_RE.match(lines[0]).group(1)  # type: ignore[union-attr]
        preamble = [line for line in lines[1:] if not line.strip().startswith("|")]
        table = "\n".join(line for line in lines[1:] if line.strip().startswith("|"))
        rows: list[dict[str, str]] = []
        for cells in parse_markdown_table(table):
            padded = list(cells) + [""] * (4 - len(cells))
            rows.append(
                {
                    "plan": padded[0],
                    "date": padded[1],
                    "goal": padded[2],
                    "topics": padded[3],
                }
            )
        return {
            "title": title,
            "preamble": "\n".join(preamble).strip(),
            "rows": rows,
        }


# ---------------------------------------------------------------------------
# The 6 Presentation Contracts -- data, not rendering
# ---------------------------------------------------------------------------


class PresentationContract(BaseModel):
    """A user-facing chat block's required fields and its non-negotiable floor.

    Interface contract: ``required`` is the ORDERED full field list;
    ``floor`` is the subset that must render even when the rest is condensed
    for token cost.  ``floor`` is always a subset of ``required`` -- enforced
    below, because a floor naming a field the contract does not have would be
    unfalsifiable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    when_emitted: str
    fidelity: str
    required: tuple[str, ...]
    floor: frozenset[str]

    @model_validator(mode="after")
    def _floor_subset(self) -> PresentationContract:
        unknown = self.floor - set(self.required)
        if unknown:
            raise ValueError(
                f"{self.name}: floor names unknown fields {sorted(unknown)}"
            )
        return self


def _contract(
    name: str,
    when_emitted: str,
    fidelity: str,
    required: Sequence[str],
    floor: Iterable[str] | None = None,
) -> PresentationContract:
    return PresentationContract(
        name=name,
        when_emitted=when_emitted,
        fidelity=fidelity,
        required=tuple(required),
        floor=frozenset(required if floor is None else floor),
    )


PRESENTATION_CONTRACTS: Mapping[str, PresentationContract] = MappingProxyType(
    {
        contract.name: contract
        for contract in (
            _contract(
                "PC-EXPLORE",
                "EXPLORE -> PLAN handoff, before transitioning state",
                "digest; index and constraints verbatim from disk",
                (
                    "findings-index",
                    "key-constraints",
                    "exploration-confidence",
                    "synthesis",
                ),
                floor=("findings-index", "key-constraints"),
            ),
            _contract(
                "PC-PLAN",
                "PLAN -> EXECUTE handoff, before requesting approval",
                "verbatim for the 11 plan sections",
                (
                    "goal",
                    "problem-statement",
                    "context",
                    "files-to-modify",
                    "steps",
                    "assumptions",
                    "failure-modes",
                    "pre-mortem",
                    "success-criteria",
                    "verification-strategy",
                    "complexity-budget",
                    "approval-prompt",
                ),
                floor=(
                    "steps",
                    "success-criteria",
                    "verification-strategy",
                    "failure-modes",
                    "assumptions",
                ),
            ),
            _contract(
                "PC-EXECUTE-STEP",
                "after each successful EXECUTE step's post-step gate",
                "digest, but every field is mandatory",
                ("step", "files", "commit", "surprises", "next-preview"),
            ),
            _contract(
                "PC-EXECUTE-LEASH",
                "after 2 failed fix attempts, before EXECUTE -> REFLECT",
                "verbatim for step intent and checkpoints; digest for the rest",
                (
                    "step-intent",
                    "attempts",
                    "root-cause-guess",
                    "checkpoints",
                    "prompt",
                ),
            ),
            _contract(
                "PC-REFLECT",
                "after REFLECT evaluation, before the routing decision",
                "verbatim for completed/remaining/verification table",
                (
                    "completed",
                    "remaining",
                    "verification-results",
                    "issues",
                    "recommendation",
                ),
            ),
            _contract(
                "PC-PIVOT",
                "at the REFLECT -> PIVOT routing decision, before PLAN",
                "verbatim for checkpoints and ghost constraints",
                (
                    "pivot-reason",
                    "checkpoints",
                    "ghost-constraints",
                    "candidate-directions",
                    "prompt",
                ),
                floor=("checkpoints", "candidate-directions"),
            ),
        )
    }
)


def missing_floor_fields(
    contract_name: str, provided: Iterable[str]
) -> tuple[str, ...]:
    """Floor fields of ``contract_name`` absent from ``provided``, in contract order."""
    contract = PRESENTATION_CONTRACTS.get(contract_name)
    if contract is None:
        raise KeyError(f"unknown presentation contract '{contract_name}'")
    supplied = set(provided)
    return tuple(
        field
        for field in contract.required
        if field in contract.floor and field not in supplied
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Artifact name (as in :class:`~.constants.ArtifactNames`) -> model.  The two
#: directory entries map to the per-file model used INSIDE that directory.
ARTIFACT_MODELS: Mapping[str, type[Artifact]] = MappingProxyType(
    {
        ArtifactNames.STATE: StateDoc,
        ArtifactNames.PLAN: PlanDoc,
        ArtifactNames.DECISIONS: DecisionsDoc,
        ArtifactNames.FINDINGS_INDEX: FindingsIndexDoc,
        ArtifactNames.FINDINGS_DIR: FindingsTopicDoc,
        ArtifactNames.PROGRESS: ProgressDoc,
        ArtifactNames.VERIFICATION: VerificationDoc,
        ArtifactNames.CHANGELOG: ChangelogDoc,
        ArtifactNames.CHECKPOINTS_DIR: CheckpointDoc,
        ArtifactNames.SUMMARY: SummaryDoc,
        ArtifactNames.CROSS_FINDINGS: ConsolidatedDoc,
        ArtifactNames.CROSS_DECISIONS: ConsolidatedDoc,
        ArtifactNames.LESSONS: LessonsDoc,
        ArtifactNames.SYSTEM: SystemAtlasDoc,
        ArtifactNames.INDEX: IndexDoc,
    }
)
