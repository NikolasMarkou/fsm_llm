"""
The plan directory: minting, atomic writes, line caps and the sliding window.

``artifacts.py`` turns Markdown into typed models.  This module is the layer
below it: it decides WHERE a file lives, gets bytes onto disk without ever
leaving a half-written artifact behind, and enforces the three size policies the
protocol states as numbers -- LESSONS' 200-line ``[I:N]`` eviction, SYSTEM's
300-line cap, and the 4-plan sliding window over the consolidated cross-plan
files.

Three rules shape everything here.

1. **Confinement and ownership are NOT reimplemented.**  Every path this module
   touches is resolved by :class:`~.tools.PlanMemory`, which composes
   ``Workspace`` (the single confinement chokepoint) and checks
   ``rules.OWNERSHIP`` before any write.  This module adds atomicity on top of
   that authorisation; it never decides for itself whether a path is legal.

2. **A write is atomic or it did not happen.**  ``Workspace.write_text`` is a
   plain ``Path.write_text``: a crash mid-write leaves a truncated artifact,
   and a truncated artifact is worse than a missing one because a gate will
   happily parse it.  :func:`_atomic_write_text` writes a temp file beside the
   target and ``os.replace``\\ s it into position, so a reader sees the old
   content or the new content and never a blend.

3. **Refuse rather than mangle.**  A cap this module cannot enforce by a rule
   the protocol actually states is raised as
   :class:`~.exceptions.HarnessArtifactError`, not resolved by truncating
   somebody's memory at line 300.
"""

from __future__ import annotations

import os
import re
import secrets
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from fsm_llm.logging import logger

from .artifacts import (
    ARTIFACT_MODELS,
    Artifact,
    ConsolidatedDoc,
    LessonsDoc,
    Section,
    StateDoc,
    SystemAtlasDoc,
    _parse_bullets,
    _render_bullets,
    lesson_importance,
)
from .constants import ArtifactNames, Defaults, Role
from .exceptions import HarnessArtifactError
from .tools import PlanMemory

__all__ = [
    "COMPRESSED_SUMMARY_CLOSE",
    "COMPRESSED_SUMMARY_OPEN",
    "COMPRESSED_SUMMARY_SECTION",
    "PLAN_ID_RE",
    "CapReport",
    "PlanDirectory",
    "RunState",
    "WindowReport",
    "apply_sliding_window",
    "check_system_cap",
    "evict_lessons",
    "mint_plan_id",
]

#: The id shape this module MINTS.  It is deliberately narrower than
#: ``artifacts``' recogniser, which also accepts the protocol's legacy
#: ``plan_YYYY-MM-DD_hex8`` directories: a reader must tolerate what history
#: left on disk, a writer must emit exactly one form.
PLAN_ID_RE = re.compile(r"^plan-\d{4}-\d{2}-\d{2}T\d{6}-[0-9a-f]{8}$")

_MINT_STAMP = "%Y-%m-%dT%H%M%S"
_MINT_ENTROPY_BYTES = 4
_MINT_MAX_ATTEMPTS = 8

COMPRESSED_SUMMARY_OPEN = Defaults.COMPRESSED_SUMMARY_MARKER
COMPRESSED_SUMMARY_CLOSE = COMPRESSED_SUMMARY_OPEN.replace("<!-- ", "<!-- /")
#: The ``## `` heading the protocol's own consolidated files use for the block
#: (see ``plans/DECISIONS.md``).  It is not a plan id, so the window below can
#: never select it for trimming -- which is the failsafe, expressed structurally.
COMPRESSED_SUMMARY_SECTION = "Summary (compressed)"


# ---------------------------------------------------------------------------
# Plan-id minting
# ---------------------------------------------------------------------------


def mint_plan_id(*, now: datetime | None = None) -> str:
    """Mint a fresh ``plan-YYYY-MM-DDTHHMMSS-<hex8>`` id.

    Interface contract (2 call sites: :meth:`PlanDirectory.create` and callers
    that need an id before a directory):
        - ``now``: the timestamp to stamp; defaults to UTC now.  A naive
          datetime is stamped verbatim, so a caller controlling the clock in a
          test gets exactly the id it asked for.
        - Returns a string matching :data:`PLAN_ID_RE`.  Uniqueness comes from
          32 bits of :mod:`secrets` entropy, NOT from the timestamp: two plans
          minted in the same second must not collide.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime(_MINT_STAMP)
    return f"plan-{stamp}-{secrets.token_hex(_MINT_ENTROPY_BYTES)}"


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write_text(target: Path, content: str, *, artifact: str) -> Path:
    """Write *content* to *target* atomically, or raise and change nothing.

    Interface contract (2 call sites: :meth:`PlanDirectory.write_text` and
    :meth:`PlanDirectory.append_text`):
        - ``target`` must already be an AUTHORISED absolute path -- this
          function performs no confinement or ownership check of its own.
        - On success the file's content is exactly ``content``.  On failure the
          file is untouched: a reader concurrent with either outcome sees the
          old bytes or the new bytes, never a truncated blend.
        - Leaves no temp file behind on either path.
        - Raises :class:`HarnessArtifactError` (tagged ``artifact``) for any
          ``OSError`` -- a full disk, a read-only mount, a vanished parent.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-019
    # The temp file MUST be created in `target.parent`, not in the system temp
    # directory. `os.replace` is only atomic within one filesystem; across a
    # mount boundary it degrades to copy-then-unlink, which reintroduces exactly
    # the torn-write window this function exists to close -- and on many systems
    # `/tmp` is a different filesystem (tmpfs) from a repository checkout.
    # Do NOT "tidy" the `dir=` argument away, and do NOT reach for
    # `tempfile.NamedTemporaryFile()` without it.
    # The `finally` shape is `FileSessionStore.save`'s (session.py:151-173),
    # copied deliberately: an `except OSError: raise` shape leaks the temp file
    # on every non-OSError exit, and the existence check is what makes the
    # cleanup a no-op after a successful `os.replace` consumed the temp name.
    # See decisions.md D-019.
    directory = target.parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle_fd, tmp_name = tempfile.mkstemp(
            dir=str(directory), prefix=f".{target.name}.", suffix=".tmp"
        )
    except OSError as exc:
        raise HarnessArtifactError(
            artifact, f"could not open a temp file beside '{target}'", cause=exc
        ) from exc
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, str(target))
    except OSError as exc:
        raise HarnessArtifactError(
            artifact, f"could not be written to '{target}'", cause=exc
        ) from exc
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - cleanup is best-effort
                logger.debug(f"could not remove temp file {tmp_name}")
    logger.debug(f"atomically wrote {target} ({len(content)} chars)")
    return target


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class CapReport(BaseModel):
    """What a line-cap enforcement pass did, and what it removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: str
    cap: int
    lines_before: int
    lines_after: int
    evicted: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether enforcement actually removed anything."""
        return bool(self.evicted)

    @property
    def over_cap(self) -> bool:
        """Whether the artifact is STILL over cap after enforcement.

        True means the protected content alone exceeds the cap -- a human
        signal, not a failure: an ``[I:5]``-only LESSONS.md over 200 lines is
        the protocol asking for a rewrite, not for a deletion.
        """
        return self.lines_after > self.cap


class WindowReport(BaseModel):
    """What the cross-plan sliding window trimmed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: str
    keep: int
    kept_plans: tuple[str, ...] = ()
    trimmed_plans: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.trimmed_plans)


class RunState(BaseModel):
    """The resumable slice of a run: which plan, and that plan's ``state.md``."""

    # DECISION plan-2026-07-21T191807-bf7ffe24/D-018
    # Do NOT add a `run.json` (or any other sidecar) next to the artifacts to
    # carry resume data. Two reasons, and the first is mechanical:
    # `rules.OWNERSHIP` is the WHOLE of what may be written under a plan
    # directory, and `PlanMemory._classify` returns None -- refused -- for any
    # path that is not a listed artifact. A sidecar is therefore unwritable
    # without widening the ownership table, i.e. without weakening the very
    # invariant (I7, exactly one writing role per artifact) that makes the
    # protocol auditable. The second reason is that it would be a SECOND source
    # of truth for "where am I?", and the two would drift the first time a
    # human edited state.md by hand -- which the protocol expects them to do.
    # See decisions.md D-018.
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    doc: StateDoc

    @property
    def state(self) -> str:
        return self.doc.state

    @property
    def iteration(self) -> int:
        return self.doc.iteration

    @property
    def current_step(self) -> str:
        return self.doc.current_step

    @property
    def fix_attempts(self) -> int:
        """Recorded fix attempts, counted by ``state.md``'s own line grammar."""
        return self.doc.fix_attempt_count


# ---------------------------------------------------------------------------
# LESSONS.md -- 200 lines, [I:N] eviction
# ---------------------------------------------------------------------------


def _rebuild_lessons(doc: LessonsDoc, bullets: Sequence[Sequence[str]]) -> LessonsDoc:
    """Return *doc* with each section's bullet list replaced, in order."""
    return LessonsDoc(
        title=doc.title,
        preamble=doc.preamble,
        sections=[
            Section(name=section.name, body=_render_bullets(items))
            for section, items in zip(doc.sections, bullets, strict=True)
        ],
    )


def evict_lessons(
    doc: LessonsDoc, *, cap: int | None = None
) -> tuple[LessonsDoc, CapReport]:
    """Trim ``LESSONS.md`` to its line cap, lowest ``[I:N]`` and oldest first.

    Interface contract (2 call sites: :meth:`PlanDirectory.enforce_lessons_cap`
    and direct use by the archivist's CLOSE pass):
        - Returns ``(document, report)``.  Under cap, the SAME document is
          returned untouched and ``report.changed`` is False.
        - Eviction order is importance ascending, then document order
          ascending, so the least important and oldest bullet goes first.
        - A bullet at :attr:`LessonsDoc.PROTECTED_IMPORTANCE` (``[I:5]``) is
          NEVER evicted, even when that leaves the file over cap; the report
          says so via :attr:`CapReport.over_cap`.
        - Raises :class:`HarnessArtifactError` when a section carries content
          that is not part of a bullet (see the anchor below).
    """
    limit = LessonsDoc.LINE_CAP if cap is None else cap
    before = doc.line_count

    # DECISION plan-2026-07-21T191807-bf7ffe24/D-017
    # This is the precondition, and it is deliberately strict: a section is
    # only rewritable if `_render_bullets(_parse_bullets(body))` reproduces the
    # body BYTE FOR BYTE. Do NOT relax this to "parse what you can and rewrite
    # anyway". `_parse_bullets` silently DISCARDS every line that is not a
    # bullet or its indented continuation, so a rewrite of a section containing
    # a paragraph, a table or a fenced block would delete that content while
    # reporting success -- the protocol would lose institutional memory to a
    # size policy, which is the opposite of what the cap is for.
    # Verified against the real `plans/LESSONS.md`: all 4 sections reproduce
    # exactly. `plans/SYSTEM.md` does NOT (its Identity section is prose), which
    # is the measured reason SYSTEM is checked and refused rather than evicted.
    # See decisions.md D-017.
    for section in doc.sections:
        items = _parse_bullets(section.body)
        if _render_bullets(items) != section.body:
            raise HarnessArtifactError(
                doc.ARTIFACT,
                f"section '## {section.name}' holds content that is not a "
                "bullet; refusing to rewrite it to enforce the line cap",
            )

    bullets: list[list[str]] = [
        _parse_bullets(section.body) for section in doc.sections
    ]
    evicted: list[str] = []
    current = doc
    while current.line_count > limit:
        candidate = _weakest_lesson(bullets)
        if candidate is None:
            break
        section_index, bullet_index = candidate
        evicted.append(bullets[section_index].pop(bullet_index))
        current = _rebuild_lessons(doc, bullets)

    report = CapReport(
        artifact=doc.ARTIFACT,
        cap=limit,
        lines_before=before,
        lines_after=current.line_count,
        evicted=tuple(evicted),
    )
    if report.changed:
        logger.debug(
            f"LESSONS.md evicted {len(evicted)} bullets "
            f"({before} -> {report.lines_after} lines, cap {limit})"
        )
    return current, report


def _weakest_lesson(bullets: Sequence[Sequence[str]]) -> tuple[int, int] | None:
    """Index of the next bullet to evict, or ``None`` when all are protected."""
    # (importance, section index, bullet index).  The comparison below is
    # STRICTLY less-than, so among equally unimportant bullets the first one
    # reached in document order wins -- i.e. oldest-first within a tier.
    weakest: tuple[int, int, int] | None = None
    for section_index, items in enumerate(bullets):
        for bullet_index, item in enumerate(items):
            importance = lesson_importance(item)
            if importance >= LessonsDoc.PROTECTED_IMPORTANCE:
                continue
            if weakest is None or importance < weakest[0]:
                weakest = (importance, section_index, bullet_index)
    return None if weakest is None else (weakest[1], weakest[2])


# ---------------------------------------------------------------------------
# SYSTEM.md -- 300 lines, checked but never auto-trimmed
# ---------------------------------------------------------------------------


def check_system_cap(doc: SystemAtlasDoc, *, cap: int | None = None) -> CapReport:
    """Measure ``SYSTEM.md`` against its 300-line cap.

    Interface contract (2 call sites: :meth:`PlanDirectory.enforce_system_cap`
    and :meth:`PlanDirectory.write_artifact`'s pre-write gate):
        - Never modifies the document; ``report.evicted`` is always empty and
          ``report.changed`` is always False.
        - ``report.over_cap`` is the answer the caller acts on.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-017
    # SYSTEM.md is CHECKED, not evicted, and that asymmetry with LESSONS.md is
    # the point. LESSONS carries an explicit, protocol-defined eviction ORDER
    # (`[I:N]` importance, then recency); SYSTEM carries none, and all six of
    # its sections are required. Any automatic trim would therefore have to
    # invent a priority the protocol never stated and delete part of a required
    # section -- so the honest enforcement is to refuse the write and make the
    # archivist rewrite the atlas, which is what the source protocol asks for.
    # Do NOT "finish the job" by adding a tail-truncate here.
    # See decisions.md D-017.
    limit = SystemAtlasDoc.LINE_CAP if cap is None else cap
    count = doc.line_count
    return CapReport(
        artifact=doc.ARTIFACT, cap=limit, lines_before=count, lines_after=count
    )


# ---------------------------------------------------------------------------
# The 4-plan sliding window over FINDINGS.md / DECISIONS.md
# ---------------------------------------------------------------------------

Summariser = Callable[[str, Section], str]


def _default_summary_line(plan_id: str, section: Section) -> str:
    """One honest bullet per trimmed plan: what it was, and where it still is."""
    lines = len(section.body.split("\n")) if section.body else 0
    return (
        f"**{plan_id}** — {lines} lines trimmed from the cross-plan window; "
        f"full content remains in `{plan_id}/`."
    )


def apply_sliding_window(
    doc: ConsolidatedDoc,
    *,
    keep: int | None = None,
    summarise: Summariser | None = None,
) -> tuple[ConsolidatedDoc, WindowReport]:
    """Keep only the *keep* most recent ``## <plan-id>`` sections.

    Interface contract (2 call sites: :meth:`PlanDirectory.apply_sliding_window`
    and direct use by the archivist's CLOSE pass):
        - Sections are newest-first on disk, so the kept set is the first
          *keep* plan-id sections in document order.
        - Trimmed plans are recorded as bullets inside the single
          ``<!-- COMPRESSED-SUMMARY -->`` block, which is created if absent.
        - Non-plan-id sections (the summary block itself, and anything a human
          added) are never candidates and are left in place.
        - Raises :class:`HarnessArtifactError` if the result's compression
          markers are unbalanced, nested or duplicated.
    """
    window = ConsolidatedDoc.WINDOW if keep is None else keep
    if window < 1:
        raise HarnessArtifactError(
            doc.ARTIFACT, f"sliding window must keep at least 1 plan (got {window})"
        )
    plan_ids = doc.plan_ids()
    if len(plan_ids) <= window:
        return doc, WindowReport(
            artifact=doc.ARTIFACT, keep=window, kept_plans=tuple(plan_ids)
        )

    kept_ids = plan_ids[:window]
    trimmed_ids = plan_ids[window:]
    summariser = summarise or _default_summary_line
    by_name = {section.name: section for section in doc.sections}
    new_bullets = [summariser(plan_id, by_name[plan_id]) for plan_id in trimmed_ids]

    trimmed = set(trimmed_ids)
    survivors = [section for section in doc.sections if section.name not in trimmed]
    preamble, sections = _record_compression(doc.preamble, survivors, new_bullets)

    result = ConsolidatedDoc(title=doc.title, preamble=preamble, sections=sections)
    issues = result.marker_issues()
    if issues:
        raise HarnessArtifactError(
            doc.ARTIFACT, f"compression markers are malformed: {'; '.join(issues)}"
        )
    logger.debug(f"{doc.ARTIFACT}: window kept {kept_ids}, trimmed {trimmed_ids}")
    return result, WindowReport(
        artifact=doc.ARTIFACT,
        keep=window,
        kept_plans=tuple(kept_ids),
        trimmed_plans=tuple(trimmed_ids),
    )


def _record_compression(
    preamble: str, sections: Sequence[Section], bullets: Sequence[str]
) -> tuple[str, list[Section]]:
    """Add *bullets* to the compressed-summary block, creating it if absent."""
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-020
    # The source protocol's explicit failsafe is that a compressed summary must
    # never be summarised into itself. That is enforced STRUCTURALLY here, in
    # two places, and neither may be replaced by a check-afterwards:
    #   1. the window's candidate set is exactly the sections whose heading is a
    #      plan id (`ConsolidatedDoc.plan_ids`), and this block's heading is
    #      "Summary (compressed)", so it can never be selected for trimming;
    #   2. new bullets are inserted INSIDE the existing block, above its closing
    #      marker -- the block is never wrapped in a fresh one, which is the
    #      only way nesting could arise.
    # Do NOT "simplify" this to prepending a new marker pair each pass: two
    # passes would then nest, `compression_marker_issues` would report it, and
    # the file would need hand repair. Live examples of the shape being
    # preserved: `plans/DECISIONS.md:3-68`.
    # See decisions.md D-020.
    rendered = _render_bullets(bullets)
    remaining = list(sections)
    for index, section in enumerate(remaining):
        if section.name != COMPRESSED_SUMMARY_SECTION:
            continue
        body = section.body.split("\n")
        close = len(body)
        for offset in range(len(body) - 1, -1, -1):
            if COMPRESSED_SUMMARY_CLOSE in body[offset]:
                close = offset
                break
        merged = [*body[:close], rendered, *body[close:]]
        remaining[index] = Section(
            name=section.name, body="\n".join(line for line in merged if line)
        )
        return preamble, remaining
    if COMPRESSED_SUMMARY_OPEN in preamble:
        opened = preamble
    else:
        opened = f"{preamble}\n{COMPRESSED_SUMMARY_OPEN}".lstrip("\n")
    block = Section(
        name=COMPRESSED_SUMMARY_SECTION,
        body=f"{rendered}\n{COMPRESSED_SUMMARY_CLOSE}",
    )
    return opened, [block, *remaining]


# ---------------------------------------------------------------------------
# The plan directory
# ---------------------------------------------------------------------------


class PlanDirectory:
    """A plan directory, addressed through one role's ``PlanMemory``.

    Every read and write goes through :class:`~.tools.PlanMemory`, so this
    class inherits confinement (``Workspace.resolve``) and ownership
    (``rules.OWNERSHIP``) rather than restating them.  What it adds is
    atomicity, the protocol's path layout, and the three size policies.

    Args:
        plan_dir: The plan directory.  Created if absent.
        role: The role every write is authorised against.  Defaults to
            ``Role.ORCHESTRATOR``, the driver itself.

    Example::

        directory = PlanDirectory.create("plans", role=Role.ORCHESTRATOR)
        directory.write_text(ArtifactNames.STATE, state_doc.to_markdown())
        resumed = directory.load_run_state()
    """

    #: Cross-plan files the sliding window applies to.
    WINDOWED: ClassVar[tuple[str, ...]] = (
        ArtifactNames.CROSS_FINDINGS,
        ArtifactNames.CROSS_DECISIONS,
    )

    def __init__(
        self, plan_dir: str | os.PathLike[str], *, role: str = Role.ORCHESTRATOR
    ) -> None:
        self._memory = PlanMemory(plan_dir, role=role)

    @classmethod
    def create(
        cls,
        parent: str | os.PathLike[str],
        *,
        role: str = Role.ORCHESTRATOR,
        now: datetime | None = None,
    ) -> PlanDirectory:
        """Mint a fresh plan id under *parent* and open its directory.

        Raises:
            HarnessArtifactError: If a free id could not be minted, which means
                :mod:`secrets` returned the same 32 bits eight times running.
        """
        base = Path(parent).expanduser()
        for _ in range(_MINT_MAX_ATTEMPTS):
            plan_id = mint_plan_id(now=now)
            if not (base / plan_id).exists():
                logger.debug(f"minted plan directory {plan_id}")
                return cls(base / plan_id, role=role)
        raise HarnessArtifactError(
            str(base), f"could not mint an unused plan id in {_MINT_MAX_ATTEMPTS} tries"
        )

    # -- properties -----------------------------------------------------

    @property
    def memory(self) -> PlanMemory:
        """The confined, ownership-scoped accessor every path goes through."""
        return self._memory

    @property
    def plan_id(self) -> str:
        return self._memory.plan_id

    @property
    def path(self) -> Path:
        """The resolved plan directory."""
        return self._memory.plan_dir

    @property
    def root(self) -> Path:
        """The cross-plan tier's directory: the plan directory's parent."""
        return self._memory.root

    @property
    def role(self) -> str:
        return self._memory.role

    def __repr__(self) -> str:
        return f"PlanDirectory(plan_id={self.plan_id!r}, role={self.role!r})"

    # -- layout ---------------------------------------------------------

    @staticmethod
    def finding_path(topic: str) -> str:
        """The plan-relative path of a topic finding: ``findings/<slug>.md``."""
        slug = re.sub(r"[^a-z0-9]+", "-", topic.strip().lower()).strip("-")
        if not slug:
            raise HarnessArtifactError(
                ArtifactNames.FINDINGS_DIR, f"topic {topic!r} has no usable slug"
            )
        return f"{ArtifactNames.FINDINGS_DIR}/{slug}.md"

    @staticmethod
    def checkpoint_path(index: int, iteration: int) -> str:
        """The plan-relative path of a checkpoint: ``checkpoints/cp-NNN-iterN.md``."""
        if index < 0 or iteration < 0:
            raise HarnessArtifactError(
                ArtifactNames.CHECKPOINTS_DIR,
                f"checkpoint index/iteration must be non-negative "
                f"(got {index}/{iteration})",
            )
        return f"{ArtifactNames.CHECKPOINTS_DIR}/cp-{index:03d}-iter{iteration}.md"

    # -- reads ----------------------------------------------------------

    def exists(self, path: str) -> bool:
        return self._memory.exists(path)

    def list_dir(self, path: str = ".") -> list[str]:
        return self._memory.list_dir(path)

    def read_text(self, path: str) -> str:
        """Read a protocol artifact.

        Raises:
            HarnessArtifactError: If the file is absent or unreadable.
        """
        try:
            return self._memory.read_text(path)
        except OSError as exc:
            raise HarnessArtifactError(path, "could not be read", cause=exc) from exc

    def read_artifact(self, path: str) -> Artifact:
        """Read and parse an artifact into its typed model.

        The model is chosen by ``PlanMemory.artifact_for``, the same
        classification the ownership check uses, so a path that is not a
        protocol artifact is refused here rather than parsed as a guess.
        """
        model = self._model_for(path)
        return model.from_markdown(self.read_text(path))

    def _model_for(self, path: str) -> type[Artifact]:
        artifact = self._memory.artifact_for(path)
        model = ARTIFACT_MODELS.get(artifact) if artifact is not None else None
        if model is None:
            raise HarnessArtifactError(
                path, "is not a protocol artifact, so it has no schema"
            )
        return model

    # -- writes ---------------------------------------------------------

    def write_text(self, path: str, content: str) -> str:
        """Authorise *path*, then write it atomically.

        Interface contract (3 call sites: :meth:`write_artifact`,
        :meth:`append_text`, and direct callers):
            - Returns the memory-root-relative path written.
            - Raises ``HarnessConfinementError`` / ``HarnessOwnershipError``
              BEFORE touching the filesystem, and :class:`HarnessArtifactError`
              if the write itself fails.
        """
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-019
        # Do NOT replace these two lines with `PlanMemory.write_text`. That
        # method is correct about WHO may write and WHERE, and wrong about HOW:
        # it delegates to `Workspace.write_text`, a plain `Path.write_text`
        # that truncates the target before writing a byte. A crash, a full
        # disk or a killed process at that instant leaves a half-written
        # artifact that still parses -- a gate then opens on a document nobody
        # wrote. Splitting the operation into `authorise` (whose own contract
        # promises it performs no write) plus `_atomic_write_text` keeps a
        # single confinement/ownership implementation while making the write
        # atomic. Do NOT "fix" this by making `Workspace.write_text` atomic
        # instead: that class is the AGENT-facing tool surface, and its writes
        # go to the user's source tree where a temp file appearing beside every
        # edited file is a visible side effect. See decisions.md D-019.
        target = self._memory.authorise(path)
        _atomic_write_text(target, content, artifact=path)
        return self._memory.locate(path)

    def append_text(self, path: str, content: str) -> str:
        """Append to an owned artifact, atomically.

        Read-modify-write rather than ``O_APPEND``: the whole point of this
        module is that a reader never sees a partial artifact, and an
        interrupted ``O_APPEND`` leaves exactly that.
        """
        existing = self.read_text(path) if self.exists(path) else ""
        return self.write_text(path, existing + content)

    def write_artifact(self, path: str, artifact: Artifact) -> str:
        """Serialize and atomically write a typed artifact.

        Raises:
            HarnessArtifactError: If *artifact* is not the model *path* holds,
                or if it is a ``SYSTEM.md`` over its line cap (D-017).
        """
        model = self._model_for(path)
        if not isinstance(artifact, model):
            raise HarnessArtifactError(
                path,
                f"expects a {model.__name__}, got {type(artifact).__name__}",
            )
        if isinstance(artifact, SystemAtlasDoc):
            report = check_system_cap(artifact)
            if report.over_cap:
                raise HarnessArtifactError(
                    path,
                    f"is {report.lines_before} lines, over its {report.cap}-line "
                    "cap; the atlas must be rewritten, not truncated",
                )
        return self.write_text(path, artifact.to_markdown())

    # -- resumable run state --------------------------------------------

    def load_run_state(self) -> RunState | None:
        """Read ``state.md`` back into a :class:`RunState`, or ``None`` if absent."""
        if not self.exists(ArtifactNames.STATE):
            return None
        doc = self.read_artifact(ArtifactNames.STATE)
        if not isinstance(doc, StateDoc):  # pragma: no cover - registry invariant
            raise HarnessArtifactError(
                ArtifactNames.STATE, "did not parse as a StateDoc"
            )
        return RunState(plan_id=self.plan_id, doc=doc)

    def save_run_state(self, run_state: RunState) -> str:
        """Write a :class:`RunState` back to ``state.md``.

        Raises:
            HarnessArtifactError: If the state belongs to a different plan --
                writing plan A's position into plan B's memory is a bug, never
                an intention.
        """
        if run_state.plan_id != self.plan_id:
            raise HarnessArtifactError(
                ArtifactNames.STATE,
                f"belongs to plan '{run_state.plan_id}', not '{self.plan_id}'",
            )
        return self.write_artifact(ArtifactNames.STATE, run_state.doc)

    # -- size policies ---------------------------------------------------

    def enforce_lessons_cap(self, *, cap: int | None = None) -> CapReport:
        """Evict ``LESSONS.md`` down to its cap, archiving what was removed.

        Evicted bullets are appended to ``LESSONS-archive.md`` -- the protocol's
        append-only overflow file -- so the cap costs the cross-plan tier
        nothing but its working-set size.  Nothing is written when the file is
        already under cap.
        """
        doc = self.read_artifact(ArtifactNames.LESSONS)
        if not isinstance(doc, LessonsDoc):  # pragma: no cover - registry invariant
            raise HarnessArtifactError(
                ArtifactNames.LESSONS, "did not parse as LESSONS"
            )
        trimmed, report = evict_lessons(doc, cap=cap)
        if not report.changed:
            return report
        self.append_text(
            ArtifactNames.LESSONS_ARCHIVE,
            f"\n## Evicted from {ArtifactNames.LESSONS} "
            f"({datetime.now(timezone.utc).strftime('%Y-%m-%d')})\n"
            f"{_render_bullets(report.evicted)}\n",
        )
        self.write_artifact(ArtifactNames.LESSONS, trimmed)
        return report

    def enforce_system_cap(self, *, cap: int | None = None) -> CapReport:
        """Measure ``SYSTEM.md`` against its cap; never rewrites it (D-017)."""
        doc = self.read_artifact(ArtifactNames.SYSTEM)
        if not isinstance(doc, SystemAtlasDoc):  # pragma: no cover - registry
            raise HarnessArtifactError(ArtifactNames.SYSTEM, "did not parse as SYSTEM")
        return check_system_cap(doc, cap=cap)

    def apply_sliding_window(
        self,
        artifact: str,
        *,
        keep: int | None = None,
        summarise: Summariser | None = None,
    ) -> WindowReport:
        """Trim a consolidated cross-plan file to the *keep* most recent plans.

        ``INDEX.md`` is what preserves a trimmed plan's discoverability; keeping
        it current is the archivist's job, not this method's.  Nothing is
        written when the file is already within the window.
        """
        if artifact not in self.WINDOWED:
            raise HarnessArtifactError(
                artifact,
                f"has no sliding window; windowed files are {', '.join(self.WINDOWED)}",
            )
        doc = self.read_artifact(artifact)
        if not isinstance(doc, ConsolidatedDoc):  # pragma: no cover - registry
            raise HarnessArtifactError(artifact, "did not parse as a consolidated file")
        trimmed, report = apply_sliding_window(doc, keep=keep, summarise=summarise)
        if report.changed:
            self.write_text(artifact, trimmed.to_markdown())
        return report
