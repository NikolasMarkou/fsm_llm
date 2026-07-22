"""
Two-tier validation of a plan directory: a hot-path gate and a full audit.

``artifacts.py`` says what an artifact IS; ``storage.py`` says where it lives.
This module is the layer that reads a whole plan directory and says whether the
protocol was actually followed.  It has exactly two entry points, and the split
between them is deliberate rather than cosmetic:

* :func:`pre_step_gate` is the REAL-TIME gate.  It opens one file --
  ``state.md`` -- decides one of four HARD failures in a fixed order, and
  returns on the first hit.  It runs before every EXECUTE step, so it must stay
  cheap and must never walk the directory.
* :func:`audit` is the RETROSPECTIVE pass.  It reads every artifact, returns a
  list of typed :class:`Issue` records with severities, and never raises for a
  finding.  It is what a finished plan directory is judged by.

Three rules shape everything here.

1. **Isomorphism, not byte-compatibility** (decisions.md D-010).  The check
   TAGS and the logical predicates mirror the source protocol's
   ``validate-plan.mjs``; the message strings and the parser do not.  Only the
   ESSENTIAL check set is ported -- the source's own Markdown release-hygiene
   meta-checks are excluded by construction, because this package's per-state
   rules are Python objects rather than prose files citing each other.

2. **Fail closed.**  An artifact that cannot be read, cannot be parsed, or was
   truncated by the read cap is an ERROR issue, never a silent pass.  A check
   function that raises is caught and reported as an ERROR against its own tag,
   so one unreadable file cannot suppress the rest of the audit.

3. **Nothing here decides policy.**  Thresholds come from
   :class:`~.constants.Defaults`, slugs from :class:`~.constants.GateSlug`,
   severities from :class:`~.constants.Severity`, the plan sections from
   :class:`~.constants.PlanSchema`, and artifact ownership from
   ``rules.OWNERSHIP`` -- read-only, never rewritten.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from fsm_llm.logging import logger

from .artifacts import (
    ChangelogEntry,
    CheckpointDoc,
    DecisionsDoc,
    FindingsIndexDoc,
    FindingsTopicDoc,
    LessonsDoc,
    PlanDoc,
    ProgressDoc,
    StateDoc,
    SystemAtlasDoc,
    VerificationDoc,
    compression_marker_issues,
    lesson_importance,
    parse_changelog_line,
)
from .constants import (
    ArtifactNames,
    Defaults,
    GateSlug,
    HarnessStates,
    PlanSchema,
    Role,
    Severity,
)
from .exceptions import HarnessArtifactError, HarnessError
from .rules import OWNERSHIP
from .storage import PlanDirectory
from .tools import Workspace

__all__ = [
    "CHECKS",
    "GateResult",
    "Issue",
    "audit",
    "pre_step_gate",
]

#: Every check tag :func:`audit` can emit.  A tag is the stable, greppable
#: identity of a rule; the message is free prose that may be reworded.
CHECKS: tuple[str, ...] = (
    "anchor-badprefix",
    "anchor-orphan",
    "anchor-refs-missing",
    "anchor-refs-stale",
    "anchor-unqualified",
    "atlas-absent",
    "atlas-cap",
    "changelog-dref-orphan",
    "changelog-malformed",
    "checkpoints",
    "complexity",
    "compress-markers",
    "decisions-schema",
    "evidence",
    "findings",
    "findings-index",
    "findings-topic",
    "iteration",
    "leash",
    "lessons-absent",
    "lessons-cap",
    "lessons-eviction",
    "ownership",
    "plan",
    "plan-section",
    "preamble-mismatch",
    "preamble-missing",
    "progress",
    "state",
    "verdict",
)


# ---------------------------------------------------------------------------
# Result types (the plan's 3rd and last pre-allocated harness abstraction)
# ---------------------------------------------------------------------------


class Issue(BaseModel):
    """One retrospective finding: a severity, a stable tag, and a message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: str
    check: str
    message: str
    artifact: str = ""

    @field_validator("severity")
    @classmethod
    def _known_severity(cls, value: str) -> str:
        if value not in Severity.ORDER:
            raise ValueError(f"unknown severity '{value}'")
        return value

    @field_validator("check")
    @classmethod
    def _known_check(cls, value: str) -> str:
        if value not in CHECKS:
            raise ValueError(f"unknown check tag '{value}'")
        return value

    @property
    def is_error(self) -> bool:
        return self.severity == Severity.ERROR

    def __str__(self) -> str:
        where = f" {self.artifact}:" if self.artifact else ""
        return f"[{self.check}]{where} {self.message}"


class GateResult(BaseModel):
    """The pre-step gate's verdict: pass, or one HARD slug with its detail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    slug: str | None = None
    hard: bool = True
    detail: str = ""

    @model_validator(mode="after")
    def _slug_matches_outcome(self) -> GateResult:
        if self.passed and self.slug is not None:
            raise ValueError(f"a passing gate cannot carry the slug '{self.slug}'")
        if not self.passed:
            if self.slug not in GateSlug.ORDER:
                raise ValueError(f"'{self.slug}' is not a pre-step gate slug")
            if not self.hard:
                # All four slugs are HARD by definition; a soft failure would be
                # an issue, and issues are `audit()`'s business, not this one's.
                raise ValueError(f"gate failure '{self.slug}' must be hard")
        return self

    @property
    def exit_code(self) -> int:
        """``0`` on pass, ``2`` on a HARD fail -- the protocol's reserved code."""
        return 0 if self.passed else 2

    def __str__(self) -> str:
        if self.passed:
            return "GATE:PASS"
        detail = f" {self.detail}" if self.detail else ""
        return f"GATE:FAIL [{self.slug}]{detail}"


# ---------------------------------------------------------------------------
# Shared readers
# ---------------------------------------------------------------------------


def _issue(severity: str, check: str, message: str, artifact: str = "") -> Issue:
    return Issue(severity=severity, check=check, message=message, artifact=artifact)


def _iteration_of(doc: StateDoc) -> int:
    """The effective iteration: the declared count, or the derived one if higher.

    The protocol increments ``## Iteration:`` by hand at PLAN -> EXECUTE, so a
    forgotten bump would silently lower the safety cap.  Deriving a second
    count from the ``EXECUTE -> REFLECT`` arrows in the transition history and
    taking the maximum makes the counter able to over-count (which only halts
    sooner) but never to under-count.
    """
    derived = sum(1 for line in doc.transition_history if _EXECUTE_REFLECT_RE.search(line))
    return max(doc.iteration, derived)


#: An ``EXECUTE -> REFLECT`` transition, in either arrow spelling the protocol
#: uses.  One such arrow closes one iteration.
_EXECUTE_REFLECT_RE = re.compile(r"EXECUTE\s*(?:->|→|=>)\s*REFLECT", re.I)


def _absolute(directory: PlanDirectory, name: str) -> Path:
    """The resolved path of *name*, addressed exactly as a read would address it."""
    return directory.root / directory.memory.locate(name)


def _read(directory: PlanDirectory, name: str) -> str | None:
    """Read an artifact's full text, ``None`` if absent.

    The read goes through ``PlanDirectory.read_text``, the DRIVER's read path
    (storage.py D-037), so a real ``decisions.md`` is audited whole instead of
    coming back clipped at the agent-facing 64 KB cap.  A truncated artifact
    must never be audited as if it were whole -- the missing tail is exactly
    where an appended entry lives.

    Raises:
        HarnessArtifactError: If the file is over the driver read bound, or
            cannot be read at all.  Either way the audit fails CLOSED: the
            caller records it as an ERROR against the check's own tag.
    """
    if not _absolute(directory, name).is_file():
        return None
    return directory.read_text(name)


def _parse(
    text: str, model: type[Any], issues: list[Issue], check: str, artifact: str
) -> Any | None:
    """Parse *text* into *model*, or record an ERROR and return ``None``."""
    try:
        return model.from_markdown(text)
    except HarnessArtifactError as exc:
        issues.append(_issue(Severity.ERROR, check, str(exc), artifact))
        return None


def _section_issues(
    doc: Any, issues: list[Issue], check: str, artifact: str, severity: str
) -> None:
    """Relay a sectioned artifact's own deviation report as issues."""
    for problem in doc.section_issues():
        issues.append(_issue(severity, check, problem, artifact))


# ---------------------------------------------------------------------------
# Markdown-aware text views
# ---------------------------------------------------------------------------

_CODE_SPAN_RE = re.compile(r"`[^`]*`")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
#: An unfilled template slot, e.g. ``<one-paragraph background>``.  Applied
#: only AFTER code spans are stripped, so a documented literal like
#: ``` `<topic>.md` ``` or ``` `<think>` ``` is prose about a shape, not a hole.
_TEMPLATE_SLOT_RE = re.compile(r"<[a-zA-Z][^<>\n]{2,}>")
_PLACEHOLDER_LINE_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?[<(\[]?\s*(?:tbd|todo|fixme|fill in|placeholder"
    r"|none yet|coming soon|\.\.\.)\b",
    re.I,
)


def _strip_code(text: str) -> str:
    """Blank out fenced blocks and inline code spans, PRESERVING line numbers.

    Every check below that scans prose for a literal -- a template slot, a
    compression marker -- must ignore the same literal quoted as code, or the
    protocol's own documentation of a marker becomes a violation of it.
    Measured: ``decisions.md`` D-020 quotes ``<!-- COMPRESSED-SUMMARY -->`` in
    its Reasoning, and a naive scan reports that plan as having an unclosed
    compression block.  Line structure is kept because
    :func:`~.artifacts.compression_marker_issues` reports line numbers.
    """
    rendered: list[str] = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            rendered.append("")
            continue
        rendered.append("" if fenced else _CODE_SPAN_RE.sub("", line))
    return "\n".join(rendered)


def _visible(body: str) -> str:
    """*body* with HTML comments, fenced blocks and code spans removed."""
    return _COMMENT_RE.sub("", _strip_code(body)).strip()


def _is_placeholder(body: str) -> bool:
    """Whether a section still holds template text instead of content."""
    visible = _visible(body)
    if not visible:
        return True
    if _TEMPLATE_SLOT_RE.search(visible):
        return True
    lines = [line for line in visible.split("\n") if line.strip()]
    return bool(lines) and all(_PLACEHOLDER_LINE_RE.match(line) for line in lines)


# ---------------------------------------------------------------------------
# The real-time pre-step gate
# ---------------------------------------------------------------------------

#: A gate predicate: returns the failure detail, or ``None`` when it passes.
#: Every predicate takes the same four arguments and ignores what it does not
#: need, so the dispatch loop below can be driven straight off
#: :data:`GateSlug.ORDER` and a test can prove short-circuiting by spying.
_GateCheck = Callable[[StateDoc, str, int, int], "str | None"]


def _gate_wrong_state(
    doc: StateDoc, expected_state: str, max_fix_attempts: int, iteration_cap: int
) -> str | None:
    if doc.state == expected_state:
        return None
    return f"expected={expected_state.upper()} actual={doc.state.upper()}"


def _gate_leash_cap(
    doc: StateDoc, expected_state: str, max_fix_attempts: int, iteration_cap: int
) -> str | None:
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-023
    # This fires at `attempts >= max_fix_attempts` (2), which HARD-blocks the
    # THIRD spawn -- and `audit()`'s `_check_state` deliberately uses DIFFERENT
    # thresholds for the same counter (silent at 2, WARN at 3, ERROR at 4+).
    # Do NOT "align" the two. They answer different questions: this gate runs
    # while a step is live and 2 recorded attempts means the budget is spent,
    # whereas the audit runs over a finished plan where a step is ALLOWED to
    # have used both of its attempts, so ERRORing at 2 would false-positive on
    # every plan that correctly spent its leash and then pivoted. A 3rd recorded
    # attempt means this gate was passed; a 4th means it was bypassed.
    # See decisions.md D-023.
    attempts = doc.fix_attempt_count
    if attempts < max_fix_attempts:
        return None
    return f"attempts={attempts} cap={max_fix_attempts}"


def _gate_iteration_cap(
    doc: StateDoc, expected_state: str, max_fix_attempts: int, iteration_cap: int
) -> str | None:
    iteration = _iteration_of(doc)
    if iteration < iteration_cap:
        return None
    return f"iteration={iteration} hard-cap={iteration_cap}"


#: Slug -> predicate.  ``no-plan`` is absent on purpose: it is decided by
#: whether ``state.md`` could be read at all, so it cannot take a ``StateDoc``.
_GATE_CHECKS: Mapping[str, _GateCheck] = MappingProxyType(
    {
        GateSlug.WRONG_STATE: _gate_wrong_state,
        GateSlug.LEASH_CAP: _gate_leash_cap,
        GateSlug.ITERATION_CAP: _gate_iteration_cap,
    }
)


def pre_step_gate(
    plan_dir: str | Path,
    *,
    expected_state: str = HarnessStates.EXECUTE,
    max_fix_attempts: int = Defaults.MAX_FIX_ATTEMPTS,
    iteration_cap: int = Defaults.ITERATION_HARD_CAP,
) -> GateResult:
    """Decide whether an EXECUTE step may be dispatched.

    Interface contract (2 call sites: ``harness.py``'s pre-step handler and the
    CLI):
        - The four slugs are evaluated in :data:`GateSlug.ORDER` --
          ``no-plan``, ``wrong-state``, ``leash-cap``, ``iteration-cap`` -- and
          the FIRST failure returns; later predicates are not evaluated.
        - Every failure is HARD and carries ``exit_code == 2``.
        - Reads exactly one file, ``state.md``, and writes nothing.
        - Never raises: an unreadable or unparseable ``state.md`` IS the
          ``no-plan`` answer, not an exception.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-024
    # This reads `state.md` with a plain `Path.read_text`, NOT through
    # `PlanDirectory`/`PlanMemory`, and that is deliberate on two counts.
    # (1) `PlanMemory.__init__` does `mkdir(parents=True, exist_ok=True)`, so
    # routing the gate through it would CREATE the very directory whose absence
    # the `no-plan` slug exists to report -- the check would manufacture the
    # state it is asked about, and a second run would then report a different
    # answer than the first. (2) The gate runs before every EXECUTE step and its
    # stated budget is one file open; the confined accessor exists to bound what
    # a MODEL may address, and `state.md` here is a fixed name under a
    # driver-supplied root, so there is no path-injection surface to bound.
    # Do NOT "tidy" this into `PlanDirectory.read_artifact`.
    # See decisions.md D-024.
    state_path = Path(plan_dir).expanduser() / ArtifactNames.STATE
    try:
        doc = StateDoc.from_markdown(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, HarnessArtifactError) as exc:
        logger.debug(f"pre-step gate [{GateSlug.NO_PLAN}]: {state_path} unreadable")
        return GateResult(
            passed=False,
            slug=GateSlug.NO_PLAN,
            detail=f"{ArtifactNames.STATE} is unreadable: {exc}",
        )

    for slug in GateSlug.ORDER:
        check = _GATE_CHECKS.get(slug)
        if check is None:
            continue  # no-plan, already decided above
        detail = check(doc, expected_state, max_fix_attempts, iteration_cap)
        if detail is not None:
            logger.warning(f"pre-step gate [{slug}]: {detail}")
            return GateResult(passed=False, slug=slug, detail=detail)
    return GateResult(passed=True)


# ---------------------------------------------------------------------------
# Audit checks -- each is `(directory, issues) -> None`
# ---------------------------------------------------------------------------


def _check_state(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``state.md``: parseability, the leash tiers, and the iteration tiers."""
    text = _read(directory, ArtifactNames.STATE)
    if text is None:
        issues.append(
            _issue(Severity.ERROR, "state", "is missing", ArtifactNames.STATE)
        )
        return
    doc = _parse(text, StateDoc, issues, "state", ArtifactNames.STATE)
    if doc is None:
        return

    # See D-022 on `_gate_leash_cap`: these thresholds are intentionally NOT
    # the gate's. 2 recorded attempts is a step that legally spent its leash.
    attempts = doc.fix_attempt_count
    if attempts >= Defaults.LEASH_AUDIT_ERROR_ATTEMPTS:
        issues.append(
            _issue(
                Severity.ERROR,
                "leash",
                f"{attempts} fix attempts recorded; the {Defaults.MAX_FIX_ATTEMPTS}"
                "-attempt gate was bypassed",
                ArtifactNames.STATE,
            )
        )
    elif attempts >= Defaults.LEASH_AUDIT_WARN_ATTEMPTS:
        issues.append(
            _issue(
                Severity.WARNING,
                "leash",
                f"{attempts} fix attempts recorded; a 3rd attempt means the "
                f"{Defaults.MAX_FIX_ATTEMPTS}-attempt gate was passed",
                ArtifactNames.STATE,
            )
        )

    iteration = _iteration_of(doc)
    if iteration >= Defaults.ITERATION_HARD_CAP:
        issues.append(
            _issue(
                Severity.ERROR,
                "iteration",
                f"iteration {iteration} exceeds the hard limit "
                f"({Defaults.ITERATION_HARD_CAP}): decompose into smaller goals",
                ArtifactNames.STATE,
            )
        )
    elif iteration >= Defaults.ITERATION_WARN:
        issues.append(
            _issue(
                Severity.WARNING,
                "iteration",
                f"iteration {iteration}: a decomposition analysis is required",
                ArtifactNames.STATE,
            )
        )


def _check_plan(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``plan.md``: the 11 sections in order, and unfilled template sections."""
    text = _read(directory, ArtifactNames.PLAN)
    if text is None:
        issues.append(_issue(Severity.ERROR, "plan", "is missing", ArtifactNames.PLAN))
        return
    # `PlanDoc` REJECTS a missing or out-of-order section rather than reporting
    # it (artifacts.py's severity split): a plan without "Failure Modes" is not
    # a plan.  So a parse failure here IS the section error.
    doc = _parse(text, PlanDoc, issues, "plan-section", ArtifactNames.PLAN)
    if doc is None:
        return
    for name in PlanSchema.SECTIONS:
        if not _is_placeholder(doc.body_of(name)):
            continue
        check = "complexity" if name == "Complexity Budget" else "plan-section"
        issues.append(
            _issue(
                Severity.WARNING,
                check,
                f"section '## {name}' still holds placeholder content",
                ArtifactNames.PLAN,
            )
        )


def _check_decisions(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``decisions.md``: plan-id preamble, header grammar, sequencing, Trade-off."""
    text = _read(directory, ArtifactNames.DECISIONS)
    if text is None:
        issues.append(
            _issue(
                Severity.ERROR, "decisions-schema", "is missing", ArtifactNames.DECISIONS
            )
        )
        return
    lines = text.replace("\r\n", "\n").split("\n")
    preamble = lines[1].strip() if len(lines) > 1 else ""
    match = re.match(r"^\*Plan:\s*(\S+?)\s*\*$", preamble)
    if match is None:
        issues.append(
            _issue(
                Severity.ERROR,
                "preamble-missing",
                "line 2 must be the '*Plan: <plan-id>*' preamble",
                ArtifactNames.DECISIONS,
            )
        )
        return
    if match.group(1) != directory.plan_id:
        issues.append(
            _issue(
                Severity.ERROR,
                "preamble-mismatch",
                f"preamble names '{match.group(1)}', not '{directory.plan_id}'",
                ArtifactNames.DECISIONS,
            )
        )
    # Header grammar, D-NNN sequencing and the `**Trade-off**:` line are all
    # enforced by the model itself, so a parse failure is the schema issue.
    _parse(text, DecisionsDoc, issues, "decisions-schema", ArtifactNames.DECISIONS)


def _check_findings(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``findings.md`` index count, plus the 5-section schema of each topic file."""
    text = _read(directory, ArtifactNames.FINDINGS_INDEX)
    if text is None:
        issues.append(
            _issue(
                Severity.WARNING,
                "findings",
                "is missing; EXPLORE -> PLAN needs "
                f"{Defaults.FINDINGS_THRESHOLD} indexed findings",
                ArtifactNames.FINDINGS_INDEX,
            )
        )
    else:
        doc = _parse(
            text,
            FindingsIndexDoc,
            issues,
            "findings-index",
            ArtifactNames.FINDINGS_INDEX,
        )
        if doc is not None:
            _section_issues(
                doc,
                issues,
                "findings-index",
                ArtifactNames.FINDINGS_INDEX,
                Severity.WARNING,
            )
            if doc.findings_count < Defaults.FINDINGS_THRESHOLD:
                issues.append(
                    _issue(
                        Severity.WARNING,
                        "findings",
                        f"only {doc.findings_count} indexed findings "
                        f"(minimum {Defaults.FINDINGS_THRESHOLD} before PLAN)",
                        ArtifactNames.FINDINGS_INDEX,
                    )
                )
    for name in _markdown_files(directory, ArtifactNames.FINDINGS_DIR):
        body = _read(directory, name)
        if body is None:  # pragma: no cover - listed then vanished
            continue
        topic = _parse(body, FindingsTopicDoc, issues, "findings-topic", name)
        if topic is not None:
            # WARN, not ERROR: the source protocol reports topic sections
            # advisorily, and a findings file that can be read and found
            # wanting is more useful than one that is refused.
            _section_issues(topic, issues, "findings-topic", name, Severity.WARNING)


def _check_progress(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``progress.md``: the four sections, in order."""
    text = _read(directory, ArtifactNames.PROGRESS)
    if text is None:
        issues.append(
            _issue(
                Severity.WARNING, "progress", "is missing", ArtifactNames.PROGRESS
            )
        )
        return
    doc = _parse(text, ProgressDoc, issues, "progress", ArtifactNames.PROGRESS)
    if doc is not None:
        _section_issues(
            doc, issues, "progress", ArtifactNames.PROGRESS, Severity.WARNING
        )


#: Criterion results that are not yet a claim.  Evidence quality is only
#: meaningful once a row asserts an outcome; demanding a measurement from a row
#: that says PENDING would report the protocol working as designed.
_UNCLAIMED_RESULTS: frozenset[str] = frozenset({"", "PENDING", "BLOCKED", "N/A", "-"})
_VERDICT_WORD_RE = re.compile(r"[A-Za-z/]+")


def _claimed_result(cell: str) -> str:
    """The verdict word a Result cell asserts, e.g. ``PASS`` or ``BLOCKED``.

    Real cells carry emphasis and a qualifier -- ``**PASS (gating proof
    only)**``, ``**BLOCKED - criterion untestable**`` -- so the leading word is
    the claim and the rest is commentary.
    """
    match = _VERDICT_WORD_RE.search(cell)
    return match.group(0).upper() if match is not None else ""


def _check_verification(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``verification.md``: the 5-bullet Verdict, mandatory checks, evidence shape."""
    text = _read(directory, ArtifactNames.VERIFICATION)
    if text is None:
        issues.append(
            _issue(
                Severity.WARNING, "verdict", "is missing", ArtifactNames.VERIFICATION
            )
        )
        return
    doc = _parse(text, VerificationDoc, issues, "verdict", ArtifactNames.VERIFICATION)
    if doc is None:
        return
    _section_issues(
        doc, issues, "verdict", ArtifactNames.VERIFICATION, Severity.WARNING
    )
    for problem in doc.verdict_issues():
        issues.append(
            _issue(Severity.WARNING, "verdict", problem, ArtifactNames.VERIFICATION)
        )
    for missing in doc.missing_additional_checks():
        issues.append(
            _issue(
                Severity.WARNING,
                "verdict",
                f"'## Additional Checks' is missing the mandatory '{missing}' row",
                ArtifactNames.VERIFICATION,
            )
        )
    for row in doc.criteria():
        verdict = _claimed_result(row.result)
        if verdict in _UNCLAIMED_RESULTS or row.evidence_ok:
            continue
        issues.append(
            _issue(
                Severity.WARNING,
                "evidence",
                f"criterion {row.number} claims '{verdict}' but its evidence is "
                "not a test count, an exit code, or a manual review",
                ArtifactNames.VERIFICATION,
            )
        )


#: A ledger line candidate: not a heading, a header stamp, a note or a table.
_LEDGER_SKIP_PREFIXES = ("#", "*", "-", "<", "|", ">")


def _changelog_entries(
    directory: PlanDirectory, issues: list[Issue]
) -> list[ChangelogEntry]:
    """Parse ``changelog.md`` line by line, reporting each malformed line once."""
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-025
    # Do NOT replace this loop with `ChangelogDoc.from_markdown(text)`. That
    # method is correct and is the right API for a WRITER, but it funnels the
    # whole file into one `HarnessArtifactError` on the first bad line -- so a
    # single malformed entry would hide every later entry from the audit,
    # including the decision-ref join below, and would report one issue where
    # the ledger has one defect per line. Measured on this repository's own
    # `changelog.md`: exactly one line carries `radius:LOW(-1)`, which the
    # 8-field spec rejects; the whole-file API reported nothing else about the
    # other 30 lines. The field grammar itself is NOT re-implemented here --
    # `parse_changelog_line` stays the single source of that fact, and this
    # function only adds a line DISCRIMINATOR. See decisions.md D-025.
    text = _read(directory, ArtifactNames.CHANGELOG)
    if text is None:
        issues.append(
            _issue(
                Severity.WARNING,
                "changelog-malformed",
                "is missing",
                ArtifactNames.CHANGELOG,
            )
        )
        return []
    entries: list[ChangelogEntry] = []
    for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), start=1):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(_LEDGER_SKIP_PREFIXES)
            or " | " not in stripped
        ):
            continue
        try:
            entries.append(parse_changelog_line(stripped))
        except HarnessArtifactError as exc:
            issues.append(
                _issue(
                    Severity.WARNING,
                    "changelog-malformed",
                    f"line {number}: {exc}",
                    ArtifactNames.CHANGELOG,
                )
            )
    return entries


def _check_changelog(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``changelog.md``: the 8-field shape, and the decision-ref join."""
    entries = _changelog_entries(directory, issues)
    known = _decision_ids(directory)
    if known is None:
        return  # decisions.md is unreadable; `_check_decisions` already said so
    for entry in entries:
        if entry.decision_ref == "-" or entry.decision_ref in known:
            continue
        issues.append(
            _issue(
                Severity.WARNING,
                "changelog-dref-orphan",
                f"'{entry.path}' references {entry.decision_ref}, which is not a "
                f"'## {entry.decision_ref}' entry in {ArtifactNames.DECISIONS}",
                ArtifactNames.CHANGELOG,
            )
        )


_CHECKPOINT_NAME_RE = re.compile(r"^cp-\d{3}-iter\d+\.md$")
#: The nuclear-fallback restore point every iteration-1 EXECUTE must create.
_NUCLEAR_CHECKPOINT = "cp-000-iter1.md"


def _check_checkpoints(directory: PlanDirectory, issues: list[Issue]) -> None:
    """``checkpoints/``: naming, the mandatory cp-000, and the 4 sections each."""
    names = _markdown_files(directory, ArtifactNames.CHECKPOINTS_DIR)
    if not names:
        issues.append(
            _issue(
                Severity.WARNING,
                "checkpoints",
                "no checkpoint files; the iteration-1 nuclear fallback "
                f"'{_NUCLEAR_CHECKPOINT}' is mandatory",
                ArtifactNames.CHECKPOINTS_DIR,
            )
        )
        return
    basenames = {Path(name).name for name in names}
    if _NUCLEAR_CHECKPOINT not in basenames:
        issues.append(
            _issue(
                Severity.WARNING,
                "checkpoints",
                f"'{_NUCLEAR_CHECKPOINT}' is missing (the nuclear-fallback "
                "restore point for the whole run)",
                ArtifactNames.CHECKPOINTS_DIR,
            )
        )
    for name in names:
        if _CHECKPOINT_NAME_RE.match(Path(name).name) is None:
            issues.append(
                _issue(
                    Severity.WARNING,
                    "checkpoints",
                    "name does not match 'cp-NNN-iterN.md'",
                    name,
                )
            )
        body = _read(directory, name)
        if body is None:  # pragma: no cover - listed then vanished
            continue
        doc = _parse(body, CheckpointDoc, issues, "checkpoints", name)
        if doc is not None:
            _section_issues(doc, issues, "checkpoints", name, Severity.WARNING)


def _check_cross_plan(directory: PlanDirectory, issues: list[Issue]) -> None:
    """The cross-plan tier: line caps, eviction protection, compression markers."""
    lessons = _read(directory, ArtifactNames.LESSONS)
    if lessons is None:
        issues.append(
            _issue(
                Severity.WARNING,
                "lessons-absent",
                "the cross-plan lessons file does not exist",
                ArtifactNames.LESSONS,
            )
        )
    else:
        doc = _parse(lessons, LessonsDoc, issues, "lessons-cap", ArtifactNames.LESSONS)
        if doc is not None and doc.over_cap():
            issues.append(
                _issue(
                    Severity.ERROR,
                    "lessons-cap",
                    f"is {doc.line_count} lines, over its {doc.LINE_CAP}-line cap",
                    ArtifactNames.LESSONS,
                )
            )

    archive = _read(directory, ArtifactNames.LESSONS_ARCHIVE)
    if archive is not None:
        protected = [
            line
            for line in archive.split("\n")
            if line.lstrip().startswith("- ")
            and lesson_importance(line) >= LessonsDoc.PROTECTED_IMPORTANCE
        ]
        for line in protected:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "lessons-eviction",
                    f"an [I:{LessonsDoc.PROTECTED_IMPORTANCE}] lesson was evicted "
                    f"and must never be: {line.strip()[:120]}",
                    ArtifactNames.LESSONS_ARCHIVE,
                )
            )

    atlas = _read(directory, ArtifactNames.SYSTEM)
    if atlas is None:
        issues.append(
            _issue(
                Severity.WARNING,
                "atlas-absent",
                "the cross-plan system atlas does not exist",
                ArtifactNames.SYSTEM,
            )
        )
    else:
        doc = _parse(atlas, SystemAtlasDoc, issues, "atlas-cap", ArtifactNames.SYSTEM)
        if doc is not None and doc.over_cap():
            issues.append(
                _issue(
                    Severity.ERROR,
                    "atlas-cap",
                    f"is {doc.line_count} lines, over its {doc.LINE_CAP}-line cap; "
                    "the atlas must be rewritten, not truncated",
                    ArtifactNames.SYSTEM,
                )
            )

    for name in (
        ArtifactNames.CROSS_FINDINGS,
        ArtifactNames.CROSS_DECISIONS,
        ArtifactNames.DECISIONS,
        ArtifactNames.CHANGELOG,
    ):
        try:
            body = _read(directory, name)
        except HarnessArtifactError as exc:
            # An oversized consolidated file is precisely what the sliding
            # window exists to prevent, and it is not this plan's ERROR: say
            # the markers could not be checked and keep auditing.
            issues.append(
                _issue(
                    Severity.WARNING,
                    "compress-markers",
                    f"markers could not be checked: {exc}",
                    name,
                )
            )
            continue
        if body is None:
            continue
        for problem in compression_marker_issues(_strip_code(body)):
            issues.append(_issue(Severity.ERROR, "compress-markers", problem, name))


def _check_ownership(directory: PlanDirectory, issues: list[Issue]) -> None:
    """Every file in the plan directory must be an artifact some role owns.

    ``rules.OWNERSHIP`` is READ here and nowhere written: it is the whole of
    what the protocol may put under a plan directory, so an entry it does not
    name is a stray file the protocol has no writer for and no reader of.
    """
    for entry in directory.list_dir("."):
        name = entry.rstrip("/")
        if name.startswith("."):
            continue
        if name not in OWNERSHIP:
            issues.append(
                _issue(
                    Severity.WARNING,
                    "ownership",
                    f"'{entry}' is not a protocol artifact; no role owns it",
                    directory.plan_id,
                )
            )


# ---------------------------------------------------------------------------
# The bounded decision-anchor scan
# ---------------------------------------------------------------------------

#: Matches an anchor in ANY comment syntax -- `#`, `//`, `--`, `/* */` -- because
#: the marker word plus a `D-NNN` is the whole of the anchor grammar.
_ANCHOR_RE = re.compile(r"\bDECISION\s+(?:(\S+)/)?(D-\d{3})\b")
_ANCHOR_GREP = r"\bDECISION\s+(\S+/)?D-[0-9]{3}\b"
#: Cap on anchors read in one scan.  A repository with more than this is not
#: silently truncated -- the overflow is reported as an issue.
_ANCHOR_MAX_HITS = 500
#: The prefix ``Workspace.grep`` appends when it stopped early.
_GREP_TRUNCATED = "... [truncated"
_HIT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+): (?P<text>.*)$")
#: `plan-YYYY-MM-DDTHHMMSS-hex8` -> the COMMIT-TAG form `plan-YYYY-MM-DD-hex8`,
#: which is what a `# DECISION` anchor must NOT use (D-014).
_PLAN_ID_RE = re.compile(r"^(plan-\d{4}-\d{2}-\d{2})T\d{6}(-[0-9a-f]{8})$")
_ANCHOR_REF_RE = re.compile(r"`?([\w./-]+\.\w+):(\d+)(?:-\d+)?`?")


def _commit_tag_form(plan_id: str) -> str | None:
    """The plan id with its ``THHMMSS`` segment dropped, or ``None``."""
    match = _PLAN_ID_RE.match(plan_id)
    return None if match is None else f"{match.group(1)}{match.group(2)}"


def _check_anchors(
    directory: PlanDirectory, issues: list[Issue], *, workspace_root: str | Path
) -> None:
    """Scan the workspace for ``# DECISION <plan-id>/D-NNN`` anchors.

    Bounded by construction: the walk skips VCS/build directories, caps the
    number of files opened and the number of matches, and never leaves
    *workspace_root*.  The plan directory itself is excluded -- protocol memory
    quotes anchors as documentation and is not source.
    """
    root = Path(workspace_root).expanduser()
    if not root.is_dir():
        issues.append(
            _issue(
                Severity.ERROR,
                "anchor-orphan",
                f"workspace root '{root}' does not exist, so anchors are unverified",
            )
        )
        return
    hits = Workspace(root).grep(_ANCHOR_GREP, ".", max_hits=_ANCHOR_MAX_HITS)
    if hits and hits[-1].startswith(_GREP_TRUNCATED):
        hits = hits[:-1]
        # `Workspace.grep` stops on EITHER budget -- matches or files walked --
        # and reports both the same way.  Fewer hits than the cap therefore
        # means the FILE budget ran out, which is what happens when a whole
        # repository root is passed instead of its source tree.
        reason = (
            f"{_ANCHOR_MAX_HITS} matches"
            if len(hits) >= _ANCHOR_MAX_HITS
            else "the confined walker's file budget"
        )
        issues.append(
            _issue(
                Severity.WARNING,
                "anchor-orphan",
                f"the anchor scan of '{root}' stopped at {reason} after "
                f"{len(hits)} matches; the remainder is UNVERIFIED. Pass the "
                "source root rather than a whole repository",
            )
        )

    plan_id = directory.plan_id
    commit_tag = _commit_tag_form(plan_id)
    #: D-NNN -> every ``(path, line)`` in the scan where it is anchored.
    anchored: dict[str, set[tuple[str, int]]] = {}
    for hit in hits:
        parsed = _HIT_RE.match(hit)
        if parsed is None:  # pragma: no cover - grep's own output shape
            continue
        where = f"{parsed.group('path')}:{parsed.group('line')}"
        if _under_plan_dir(root, parsed.group("path"), directory):
            continue
        match = _ANCHOR_RE.search(parsed.group("text"))
        if match is None:  # pragma: no cover - grep matched, so this cannot fail
            continue
        prefix, decision_id = match.group(1), match.group(2)
        if prefix is None:
            issues.append(
                _issue(
                    Severity.WARNING,
                    "anchor-unqualified",
                    f"anchor '{decision_id}' carries no plan id and is invisible "
                    "to the anchor audit",
                    where,
                )
            )
        elif prefix == commit_tag:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "anchor-badprefix",
                    f"anchor uses the COMMIT-TAG form '{prefix}'; anchors keep the "
                    f"timestamp, so it must be '{plan_id}'",
                    where,
                )
            )
        elif prefix == plan_id:
            anchored.setdefault(decision_id, set()).add(
                (parsed.group("path"), int(parsed.group("line")))
            )

    known = _decision_ids(directory)
    if known is None:
        return
    for decision_id in sorted(set(anchored) - known):
        issues.append(
            _issue(
                Severity.ERROR,
                "anchor-orphan",
                f"anchor '{plan_id}/{decision_id}' has no matching entry in "
                f"{ArtifactNames.DECISIONS}",
                ArtifactNames.DECISIONS,
            )
        )
    _check_anchor_refs(
        directory,
        issues,
        {key: value for key, value in anchored.items() if key in known},
    )


def _same_file(reference: str, scanned: str) -> bool:
    """Whether an Anchor-Ref path and a scanned path name the same file.

    Compared by suffix in BOTH directions on purpose: the Anchor-Ref is written
    repository-relative (``src/fsm_llm_harness/roles.py``) while the scan is
    relative to whatever source root the caller supplied, which may be a
    subtree of it (``fsm_llm_harness/roles.py``).  Neither frame is wrong; only
    their overlap is checkable.
    """
    left = reference.replace("\\", "/").split("/")
    right = scanned.replace("\\", "/").split("/")
    depth = min(len(left), len(right))
    # Compared COMPONENT-wise, not as a raw string suffix: `heroles.py` ends
    # with `roles.py` and is a different file.
    return left[-depth:] == right[-depth:]


def _check_anchor_refs(
    directory: PlanDirectory,
    issues: list[Issue],
    anchored: Mapping[str, set[tuple[str, int]]],
) -> None:
    """Every anchored decision must carry an ``**Anchor-Refs**:`` back-link.

    The back-link is checked against the anchors the scan actually FOUND, not
    against mere file existence: "D-013 says it is anchored in a file where it
    is not" is the defect worth reporting, and it is the only one that can be
    stated without knowing which frame the reference was written in.
    """
    text = _read(directory, ArtifactNames.DECISIONS)
    if text is None:  # pragma: no cover - `_decision_ids` already returned
        return
    try:
        doc = DecisionsDoc.from_markdown(text)
    except HarnessArtifactError:  # pragma: no cover - `_check_decisions` reported it
        return
    for decision_id, sites in sorted(anchored.items()):
        entry = doc.entry(decision_id)
        raw = entry.field("Anchor-Refs") if entry is not None else None
        # FIRST LINE only: `Anchor-Refs` is a one-line field, and a following
        # `**Outcome ...**` line with no colon is absorbed into its value by the
        # decisions parser, dragging prose (and prose-shaped `a/b:4` tokens)
        # into the reference list.
        refs = _ANCHOR_REF_RE.findall((raw or "").strip().split("\n")[0])
        if not refs:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "anchor-refs-missing",
                    f"{decision_id} is anchored at "
                    f"{', '.join(f'{path}:{line}' for path, line in sorted(sites))} "
                    "but its entry carries no '**Anchor-Refs**:' back-link",
                    ArtifactNames.DECISIONS,
                )
            )
            continue
        for path, line in refs:
            matching = [site for site in sites if _same_file(path, site[0])]
            if not matching:
                issues.append(
                    _issue(
                        Severity.WARNING,
                        "anchor-refs-stale",
                        f"{decision_id} back-links '{path}', which carries no "
                        f"'{directory.plan_id}/{decision_id}' anchor",
                        ArtifactNames.DECISIONS,
                    )
                )
            elif int(line) not in {site[1] for site in matching}:
                issues.append(
                    _issue(
                        Severity.INFO,
                        "anchor-refs-stale",
                        f"{decision_id} back-links '{path}:{line}' but the anchor "
                        f"is at line {', '.join(str(s[1]) for s in sorted(matching))}",
                        ArtifactNames.DECISIONS,
                    )
                )


def _under_plan_dir(root: Path, relative: str, directory: PlanDirectory) -> bool:
    """Whether a workspace-relative hit lives inside the plan directory."""
    try:
        (root / relative).resolve().relative_to(directory.path)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _markdown_files(directory: PlanDirectory, subdir: str) -> list[str]:
    """Plan-relative ``.md`` paths directly under *subdir*, non-recursively."""
    if not directory.exists(subdir):
        return []
    return [
        f"{subdir}/{entry}"
        for entry in directory.list_dir(subdir)
        if entry.endswith(".md")
    ]


def _decision_ids(directory: PlanDirectory) -> set[str] | None:
    """Every ``D-NNN`` header in ``decisions.md``, or ``None`` if unreadable.

    Deliberately regex-scanned rather than model-parsed: the join below must
    still work when the document as a whole fails its schema, and a header this
    loose can only make the join more permissive, never less.
    """
    try:
        text = _read(directory, ArtifactNames.DECISIONS)
    except HarnessArtifactError:
        return None
    if text is None:
        return None
    return set(re.findall(r"^##\s+(D-\d{3})\b", text, re.M))


#: The audit's check functions, run in this order.  Each takes
#: ``(directory, issues)`` and appends; none raises for a finding.
_AUDIT_CHECKS: tuple[Callable[[PlanDirectory, list[Issue]], None], ...] = (
    _check_state,
    _check_plan,
    _check_decisions,
    _check_findings,
    _check_progress,
    _check_verification,
    _check_changelog,
    _check_checkpoints,
    _check_cross_plan,
    _check_ownership,
)


def audit(
    plan_dir: str | Path, *, workspace_root: str | Path | None = None
) -> list[Issue]:
    """Audit a whole plan directory against the ESSENTIAL protocol check set.

    Interface contract (2 call sites: ``harness.py``'s CLOSE gate and the CLI):
        - Returns a list of :class:`Issue`; an empty list means clean.
        - NEVER raises for a finding and never prints.  A check that raises is
          itself reported as an ERROR, so one unreadable artifact cannot
          suppress the others.
        - Reads only; nothing under *plan_dir* is created or modified.
        - ``workspace_root`` enables the bounded decision-anchor scan.  Omitted,
          the anchor checks are simply not run -- there is no source tree to
          scan, so there is nothing to say about it.
    """
    path = Path(plan_dir).expanduser()
    if not path.is_dir():
        return [_issue(Severity.ERROR, "state", f"plan directory '{path}' is absent")]

    issues: list[Issue] = []
    directory = PlanDirectory(path, role=Role.ORCHESTRATOR)
    checks: list[Callable[[PlanDirectory, list[Issue]], None]] = list(_AUDIT_CHECKS)
    if workspace_root is not None:
        checks.append(partial(_check_anchors, workspace_root=workspace_root))
    for check in checks:
        name = getattr(check, "func", check).__name__
        try:
            check(directory, issues)
        except (HarnessError, OSError, ValueError) as exc:
            # Fail closed: a check that could not complete is a finding, not a
            # pass.  The tag is `state` because "the audit could not run" is a
            # statement about the plan directory, not about one artifact.
            issues.append(
                _issue(
                    Severity.ERROR,
                    "state",
                    f"check '{name}' could not complete: {exc}",
                    directory.plan_id,
                )
            )
    logger.debug(
        f"audit({directory.plan_id}): {len(issues)} issues, "
        f"{sum(1 for issue in issues if issue.is_error)} of them errors"
    )
    return issues
