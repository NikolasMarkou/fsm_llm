"""
``HarnessAgent`` -- the iterative-planner protocol driver.

The driver is a :class:`~fsm_llm_agents.base.BaseAgent` subclass.  It owns no
conversation loop, no budget enforcement, no trace building and no API
construction: all of that is ``BaseAgent``'s.  What it adds is the *protocol*:

* one worker dispatch per ``(state, iteration, step_number)`` key (invariant I4),
* the ``iteration`` / ``step_number`` / ``fix_attempts`` counters and their
  reset rules,
* the three human approval points, routed through ``HumanInTheLoop`` (I6),
* a re-entrancy guard so a dispatched worker cannot become a second
  coordinator (I5),
* the autonomy-leash and iteration-cap halt paths,
* fail-closed handling of every worker reply (I8).

Artifact I/O
------------
The driver reads its own protocol memory freely and writes exactly ONE artifact,
``state.md`` -- the only one ``rules.OWNERSHIP`` gives it alone.  Everything
else under the plan directory belongs to a role, and a driver write to one
raises ``HarnessOwnershipError`` from ``PlanMemory.authorise`` rather than being
policed here.  Four seams carry it:

* :meth:`HarnessAgent._sync_state_doc` records the driver's position on every
  state ENTRY and after every loop-turn dispatch -- never inside
  :meth:`HarnessAgent._dispatch_if_needed`, so the pre-step gate always reads a
  document written by an earlier event than itself (D-038).
* :meth:`HarnessAgent._pre_step_gate` runs ``plan_validator.pre_step_gate``
  over that document and routes each of its four slugs to its OWN action
  (D-040).  Only ``leash-cap`` emits a leash block; only ``iteration-cap`` ends
  the run; neither recovery slug writes ``state.md``.
* The five presentation contracts are filled from the artifacts they quote --
  ``findings.md``, ``plan.md``, ``changelog.md``, ``progress.md``,
  ``verification.md`` -- rather than from what a worker says it did.
* :meth:`HarnessAgent._derive_gate_counts` COUNTS the files behind a
  disk-derived gate key after a dispatch (D-032).  A gate value the driver
  reads off the filesystem itself is evidence; one a worker reports is
  testimony, and the two are handled differently on purpose.

Every read goes through ``storage.PlanDirectory``, which is the DRIVER's
accessor: confined like every other path, but uncapped, because the 64 KB
read cap bounds what an untrusted worker pulls into a prompt and a real
``decisions.md`` outgrows it (storage.py D-037).  Every write goes through it
too, which makes it atomic (storage.py D-019).

Dispatch mechanism
------------------
Three call sites share ONE ledger-guarded entry point,
:meth:`HarnessAgent._dispatch_if_needed`:

1. a ``START_CONVERSATION`` handler, for the initial state;
2. one ``on_state_entry(S)`` handler per state, for every real entry into ``S``;
3. ``BaseAgent._on_loop_iteration``, for a state the FSM is *holding*.

Measured firing semantics of the core handler system (verified against
``fsm_llm.API`` with a mocked LLM, 2026-07-21), which is why all three are
needed:

======================================  =====================================
Event                                   ``on_state_entry(S)`` fires?
======================================  =====================================
conversation start, initial state == S  **no** (START_CONVERSATION only)
transition into S                       yes, exactly once
BLOCKED turn while holding S            **no**
re-entry into S later                   yes, exactly once
======================================  =====================================

A ``START_CONVERSATION`` handler additionally cannot read ``_current_state``:
the pipeline seeds that key during the first transition, so at start it is
absent.  The initial-state handler therefore names
``HarnessStates.INITIAL`` directly.

Bounded EXPLORE re-dispatch
---------------------------
One exception to "one dispatch per key" is authorised EXPLICITLY, by the driver,
and bounded by a counter no worker can reach: while the EXPLORE -> PLAN findings
gate is still BLOCKED, :meth:`HarnessAgent._after_explore_dispatch` removes the
EXPLORE key from the ledger so the next loop turn dispatches another explorer.
The source protocol asks for exactly this ("if gate fails: spawn additional
explorers for gaps"); see :meth:`HarnessAgent._after_explore_dispatch` and
decisions.md D-028 / D-029.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger
from fsm_llm_agents.base import BaseAgent
from fsm_llm_agents.definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_agents.hitl import ApprovalCallback, HumanInTheLoop

from .artifacts import (
    PRESENTATION_CONTRACTS,
    Artifact,
    FindingsIndexDoc,
    PlanDoc,
    ProgressDoc,
    Section,
    StateDoc,
    VerificationDoc,
    missing_floor_fields,
    parse_changelog_line,
)
from .constants import (
    DRIVER_OWNED_SEEDS,
    DRIVER_OWNED_UNSET,
    ArtifactNames,
    ContextKeys,
    Defaults,
    GateSlug,
    HandlerNames,
    HandlerPriorities,
    HarnessStates,
    PlanSchema,
    Role,
    Severity,
)
from .exceptions import HarnessError, HarnessReentrancyError
from .fsm_definition import build_harness_fsm
from .hardening import as_int, coerce_worker_output
from .plan_validator import GateResult, Issue, _is_placeholder, audit, pre_step_gate
from .rules import ROLE_BY_STATE, explore_topics, get_rules
from .storage import PlanDirectory
from .tools import (
    DISK_DERIVED_COUNTS,
    PlanMemory,
    derive_disk_counts,
    gate_files,
)

__all__ = [
    "HarnessAgent",
    "Presentation",
    "RevertCallback",
    "RevertDirective",
    "RoleRequest",
    "WorkerFactory",
    "derive_execute_target",
]

_ExcT = TypeVar("_ExcT", bound=BaseException)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: ``agent_type`` tag handed to ``BaseAgent._standard_run`` (logs, lifecycle
#: handler names, the ``_agent_type`` context marker).
_AGENT_TYPE = "harness"

#: Consecutive turns the protocol may make no progress at all -- no dispatch,
#: no state change, no counter change -- before the driver halts.  Without it a
#: permanently BLOCKED gate spins until ``BaseAgent._check_budgets`` raises,
#: which on a live model is minutes of LLM calls that cannot change anything.
#: Since D-045 this is also the NORMAL termination of a ``worker_factory=None``
#: run: no worker means no evidence, every gate stays shut, and the run reports
#: which gate it is sitting behind instead of letting the FSM's own extraction
#: invent the evidence for it.
_DEFAULT_MAX_STALL_TURNS = 3

#: ``tool_name`` values for the three human approval points.  They are not tool
#: calls; ``HumanInTheLoop.request_approval`` takes a ``ToolCall`` as its
#: request envelope, so these name the *gate* being approved.
_APPROVAL_PLAN = "harness.approve_plan"
_APPROVAL_CLOSE = "harness.confirm_close"
_APPROVAL_LEASH = "harness.continue_after_leash"
#: The fourth approval point, added with the leash-cap revert (D-039).  It is
#: consulted ONLY when the caller supplied a ``revert_callback``; without one
#: there is nothing to approve, because nothing will be executed.
_APPROVAL_REVERT = "harness.revert_uncommitted"

#: Ledger entry prefixes.  Both live in the single ``dispatch_ledger`` list:
#: a ``dispatch:`` entry records "this key has been dispatched", an ``entry:``
#: entry records "this state-entry handler fire has been processed".
_LEDGER_DISPATCH = "dispatch"
_LEDGER_ENTRY = "entry"

# DECISION plan-2026-07-21T125237-191b2eb2/D-020
# The three caps below are NOT cosmetic tidiness -- they are what keeps the
# protocol runnable at all. `dispatch_ledger`, `role_results` and each recorded
# answer are ordinary (non-internal-prefixed) context keys, so every one of them
# is rendered into BOTH LLM prompts on EVERY turn. Uncapped, a measured
# EXECUTE <-> REFLECT remediation cycle grew the response-generation system
# prompt by ~145 chars per turn and CRASHED the run at turn ~150 on
# `ResponseGenerationRequest`'s 30,000-character pydantic bound -- a hard
# failure with no protocol meaning. Do NOT raise these caps to "keep more
# history": the plan directory is where protocol history belongs (a later step
# writes it); context is the working set. The ledger window is >> the reuse
# window of any single key (a key is only ever re-checked while its state is
# occupied), so eviction cannot resurrect a suppressed dispatch.
# See decisions.md D-020.

#: Most recent ledger entries retained; older ones are evicted.
_MAX_LEDGER_ENTRIES = 64

#: Most recent role results retained in ``role_results``.
_MAX_ROLE_RESULTS = 20

#: Longest worker answer recorded in a role result.
_MAX_ANSWER_CHARS = 400

#: Gate keys each state's worker is allowed to write, and the exact type each
#: must have.  A value of any other type is dropped, which leaves the gate
#: BLOCKED (invariant I8).
#:
#: ``plan_approved`` and ``close_confirmed`` appear nowhere: they record a
#: HUMAN decision and are written only by the approval path below.  A worker
#: that returns them is ignored.
_WORKER_WRITABLE: Mapping[str, Mapping[str, type]] = MappingProxyType(
    {
        HarnessStates.EXPLORE: MappingProxyType(
            {
                ContextKeys.FINDINGS_COUNT: int,
                ContextKeys.NEEDS_EXPLORE: bool,
            }
        ),
        HarnessStates.PLAN: MappingProxyType(
            {
                ContextKeys.NEEDS_EXPLORE: bool,
                ContextKeys.TOTAL_STEPS: int,
            }
        ),
        # EXECUTE writes no gate flag directly: the driver derives
        # execute_complete / step_number / fix_attempts from the worker's
        # success flag, so a worker cannot understate its own attempt count.
        HarnessStates.EXECUTE: MappingProxyType({}),
        HarnessStates.REFLECT: MappingProxyType(
            {
                ContextKeys.ALL_CRITERIA_PASS: bool,
                ContextKeys.NEEDS_PIVOT: bool,
                ContextKeys.COMPLETION_FIX: bool,
                ContextKeys.NEEDS_EXPLORE: bool,
                ContextKeys.CRITERIA_PASS_COUNT: int,
                ContextKeys.CRITERIA_TOTAL: int,
            }
        ),
        HarnessStates.PIVOT: MappingProxyType(
            {
                ContextKeys.PIVOT_RESOLVED: bool,
                ContextKeys.PIVOT_REASON: str,
            }
        ),
        HarnessStates.CLOSE: MappingProxyType({ContextKeys.HALT_REASON: str}),
    }
)

#: REFLECT's mutually exclusive routing flags.  ``rules.py`` tells the worker
#: to set exactly one; this makes it mechanical -- setting one clears the rest.
_REFLECT_ROUTING_FLAGS: tuple[str, ...] = (
    ContextKeys.COMPLETION_FIX,
    ContextKeys.NEEDS_PIVOT,
    ContextKeys.NEEDS_EXPLORE,
)

# DECISION plan-2026-07-21T125237-191b2eb2/D-052
# The two per-step leash counters reset TOGETHER, as one fact, at the three
# places a genuinely NEW step begins (a re-planned iteration, the step cursor
# advancing, a PIVOT). Do NOT split them back into two literal `0`s at three
# call sites: `fix_attempts` alone is exactly the escape review C3(b)
# reproduced -- `_after_reflect_dispatch` reset it on every granted
# `harness.continue_after_leash`, so an approving callback bought unbounded
# retries. `leash_grants` is the half that a leash grant must NOT reset, and a
# reader who sees `fix_attempts: 0` written on its own will assume the same is
# true here. See decisions.md D-052.
_LEASH_RESET: Mapping[str, Any] = MappingProxyType(
    {
        ContextKeys.FIX_ATTEMPTS: 0,
        ContextKeys.LEASH_GRANTS: 0,
    }
)


# ---------------------------------------------------------------------------
# Worker seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleRequest:
    """Everything a role worker needs to do one state's job.

    Interface contract for :data:`WorkerFactory` implementations:

    * the factory is called with exactly one ``RoleRequest`` and must return an
      :class:`~fsm_llm_agents.definitions.AgentResult`;
    * anything it raises is caught, recorded as a ``success=False`` role result
      and the protocol turn continues (the one exception is
      :class:`~fsm_llm_harness.exceptions.HarnessReentrancyError`, which is
      always re-raised);
    * gate flags are read back from ``AgentResult.final_context`` and, when
      present, ``AgentResult.structured_output``, filtered by
      :data:`_WORKER_WRITABLE`;
    * the worker must not call back into the driver (invariant I5).

    Attributes:
        role: The worker role being dispatched (a ``Role.WORKERS`` member).
        state: The protocol state this dispatch belongs to.
        goal: The run's goal, verbatim.
        operative_rules: The state's protocol rules, from ``rules.py``.
        gate_summary: One line describing the state's exit gate.
        iteration: Current plan iteration.
        step_number: Current EXECUTE step cursor.
        total_steps: Number of steps in the current plan.
        fix_attempts: Fix attempts already spent on ``step_number``.
        context: Read-only snapshot of the FSM context at dispatch time.
        plan_dir: This run's plan directory, or ``None`` when the caller did
            not supply one.  Driver-owned (``DRIVER_OWNED_UNSET``): a worker
            cannot redirect the protocol's own memory by inventing a path.
            ``None`` means a factory MUST NOT hand the role plan-directory
            tools -- there is no directory to confine them to.
        workspace_root: The code root the run is changing, or ``None``.
            Reported to the worker; the confinement itself lives in the
            ``Workspace`` the factory was built with.
        assigned_topic: For an EXPLORE dispatch, the ONE kebab-case slug this
            dispatch is to write ``findings/<slug>.md`` for; ``None`` for every
            other state, and for an EXPLORE dispatch with no plan directory or
            no slug left to assign.  Driver-owned and derived per dispatch from
            the files really on disk (:meth:`HarnessAgent._assign_explore_topic`);
            a factory renders it into the prompt and must not invent one.
        assigned_write_target: The ONE workspace-relative file an EXECUTE
            step edits (:func:`derive_execute_target`, driver-owned); ``None``
            otherwise -- the prompt then falls back unchanged.
        execute_target_reason: WHY ``assigned_write_target`` holds what it
            holds, for an EXECUTE dispatch: one of the ``EXECUTE_TARGET_*``
            literals below.  ``None`` for every other state.  DIAGNOSTIC only:
            no prompt builder reads it (D-010 fail-open is proven by test).
    """

    role: str
    state: str
    goal: str
    operative_rules: tuple[str, ...]
    gate_summary: str
    iteration: int
    step_number: int
    total_steps: int
    fix_attempts: int
    context: Mapping[str, Any] = field(default_factory=dict)
    plan_dir: str | None = None
    workspace_root: str | None = None
    assigned_topic: str | None = None
    assigned_write_target: str | None = None
    execute_target_reason: str | None = None


_TICKED_RE = re.compile(r"`([^`\s]+)`")
_TARGET_RE = re.compile(r"^(?!/)(?!.*\.\.)(?=.*[./])[\w./-]*\w$")

#: Diagnostic reasons ``_assign_execute_target`` reports alongside its target.
#: Purely observational (rubric rows, logs): the prompt half NEVER reads them,
#: so an unassigned dispatch renders byte-identically whatever the reason
#: (D-010 fail-open).  L6 B0 could not tell a run with no plan.md from one
#: whose Files-To-Modify simply had no path-shaped token -- all three causes
#: collapsed into one silent ``None``.
EXECUTE_TARGET_ASSIGNED = "assigned"
EXECUTE_TARGET_ASSIGNED_PROSE = "assigned-prose"
EXECUTE_TARGET_NO_PLAN_DIR = "no-plan-dir"
EXECUTE_TARGET_NO_PLAN_DOC = "no-plan-doc"
EXECUTE_TARGET_NO_TOKEN = "no-target-token"


def derive_execute_target(plan: PlanDoc, step_number: int) -> str | None:
    """The ONE workspace file plan step *step_number* should edit, or ``None``.

    Interface contract (2 call sites: ``HarnessAgent._assign_execute_target``,
    the live bench's ``_execute_request``): first path-shaped backticked token
    per Files-To-Modify line -> the one named in the step's own text, else the
    first, else ``None``.  Never raises; no I/O.
    """
    # DECISION plan-2026-07-22T114536-879d04a0/D-010
    # NEVER guess a root, and fail OPEN to the status quo: an unparseable
    # section returns None, which renders NO assignment line -- the dispatch
    # prompt stays byte-identical to the pre-D-010 text.  A guessed assignment
    # is WORSE than none (it points a live executor somewhere the plan never
    # said), which is why _TARGET_RE refuses absolutes, `..`, globs and bare
    # backticked words (`NEW`, `upload()`) rather than repairing them, and why
    # only the FIRST path-shaped token per line counts -- the file column leads
    # its row; prose cells must not compete.  Exactly ONE heuristic beyond
    # "first target": prefer a candidate the current step's own text names
    # (the per-step mapping, when the plan-writer provided one).  Do NOT add
    # more heuristics -- if this rule cannot decide, the fix is the
    # plan-writer's section, not cleverness here (plan.md Pre-Mortem #2).
    # Section lookup is positional on purpose: PlanSchema.SECTIONS order is
    # validator-enforced, so index 3 IS "Files To Modify".
    # See decisions.md D-010.
    targets: list[str] = [
        target
        for line in plan.body_of(PlanSchema.SECTIONS[3]).splitlines()
        if (target := next(filter(_TARGET_RE.match, _TICKED_RE.findall(line)), None))
    ]
    if not targets:
        return None
    text = next((s.text for s in plan.steps() if s.number == str(step_number)), "")
    named = [target for target in targets if target in text]
    return (named or targets)[0]


#: A token that ends in a real file extension (``.py``, ``.md`` ...): a dot then
#: 1-6 alnum chars at the very end.  Distinguishes ``uploader.py`` (a file) from
#: ``upload()`` (no extension); it does NOT by itself distinguish ``uploader.py``
#: from ``requests.post`` -- the workspace-existence gate in
#: :func:`_derive_prose_target` does that.
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,6}$")
#: Prose/markdown punctuation stripped from a whitespace token before it is
#: tested as a path (``uploader.py:`` -> ``uploader.py``, ``` `config.py` ``` ->
#: ``config.py``).  Only surrounding chars are stripped; interior ``.``/``/``/``-``
#: (part of a real path) survive because they are not in this set.
_PROSE_EDGE = "`*_,;:()[]{}\"'<>. "


def _derive_prose_target(
    plan: PlanDoc, step_number: int, existing_files: frozenset[str]
) -> str | None:
    """A Files-To-Modify target from UN-backticked prose, gated by existence.

    Interface contract (1 call site: ``HarnessAgent._assign_execute_target``,
    invoked ONLY when the strict :func:`derive_execute_target` returns ``None``):
    per Files-To-Modify line, take path-shaped, extension-bearing whitespace
    tokens (prose punctuation stripped) that name a file the caller says EXISTS
    under the workspace root; prefer one the step's own text names, else the
    first per line.  ``None`` when no line yields an existing-file token.  Never
    raises; no I/O (the existence set is supplied by the caller).

    # DECISION plan-2026-07-23T155204-fdc2d181/D-001
    # This is the ADDITIVE prose fallback S4a needs: a real 4b plan writes
    # `uploader.py`/`config.py` as PROSE (no backticks), so `derive_execute_target`
    # (backtick-only, `_TICKED_RE`) returns None, no target is assigned, and the
    # model's self-directed EXECUTE write produces a label the HASH-FROZEN L6
    # floor normalizer cannot credit (verified_write=False, B5 run-1).  Measured
    # levers: assigning `uploader.py` flips verified_write 3/3 live (F4); steering
    # 4b to backtick via the response_format field DESCRIPTION is a no-op over
    # Ollama -- the schema is sent as a grammar constraint, not prompt text (F5).
    # So the fix is DETERMINISTIC prose extraction here, NOT model-steering and
    # NOT loosening `derive_execute_target` (kept byte-frozen with its D-010
    # anchor + tests).  The `existing_files` gate is what keeps D-010's real
    # invariant ("never point a live executor somewhere the plan never said"):
    # a target MUST name a real workspace file, so `requests.post`, invented
    # names, and new-file tokens are all rejected -- order-independently (a file
    # named AFTER a method-call token still wins).  A step naming no existing
    # file yields None -> no assignment -> byte-identical prompt (fail-open).
    # Do NOT drop the existence gate (it becomes the guess D-010 forbids) and do
    # NOT feed it un-filtered tokens.  See decisions.md D-001, findings F1/F4/F5/F7.
    """
    if not existing_files:
        return None
    per_line: list[str] = []
    for line in plan.body_of(PlanSchema.SECTIONS[3]).splitlines():
        for token in line.split():
            candidate = token.strip(_PROSE_EDGE)
            if (
                _TARGET_RE.match(candidate)
                and _EXT_RE.search(candidate)
                and candidate in existing_files
            ):
                per_line.append(candidate)
                break
    if not per_line:
        return None
    text = next((s.text for s in plan.steps() if s.number == str(step_number)), "")
    named = [target for target in per_line if target in text]
    return (named or per_line)[0]


#: A line that ``artifacts._parse_sectioned`` would read as an ATX heading (its
#: ``_H2_RE`` = ``^##\s+``; ``_H1_RE`` = ``^#\s+``).  ``#{1,6}\s`` is the
#: superset, so escaping a match makes the line inert under BOTH.
_ATX_HEADING_RE = re.compile(r"^#{1,6}\s")


def _demote_heading_lines(body: str) -> str:
    """Escape any body line that would re-parse as a ``## `` / ``# `` heading.

    Interface contract (1 call site: :meth:`HarnessAgent._render_plan_from_structured`):
        - A section body authored by the model may embed a line that starts
          ``## `` (or any ``#{1,6} ``).  Rendered verbatim under a section, that
          line would be read back by ``PlanDoc.from_markdown`` as a SPURIOUS new
          section -- breaking the 11-section round-trip and the positional
          section check.  Each such line is prefixed with a backslash so it no
          longer matches at line start; the content stays readable and the
          escape is idempotent (an already-escaped ``\\## x`` does not match, so
          re-rendering never double-escapes).
        - Pure, deterministic, never raises.  An empty/whitespace body is
          returned unchanged (no line matches), preserving the no-filler ethos:
          an empty field stays an empty section that the gate denies.
    """
    return "\n".join(
        "\\" + line if _ATX_HEADING_RE.match(line) else line
        for line in body.split("\n")
    )


# DECISION plan-2026-07-23T095051-a6dcb40d/D-001
def _plan_is_approvable(plan: PlanDoc) -> bool:
    """Whether a parsed ``plan.md`` is APPROVABLE: EVERY one of the 11
    ``PlanSchema.SECTIONS`` bodies carries non-placeholder content.

    Interface contract (TWO call sites, deliberately ONE predicate so the two
    PLAN gates CANNOT drift):
        - :meth:`HarnessAgent._plan_has_content` -- the PLAN redispatch BUDGET
          gate (an un-approvable plan consumes the budget and halts on the
          honest ``plan-cap`` slug, never a slugless stall).
        - ``DiskEvidenceApprovals._decide`` -- the test-side APPROVAL gate
          (``test_live_ollama.py`` imports THIS function; see its D-013 anchor).
    The two MUST share one bar because approval DENIAL does NOT redispatch
    (predecessor D-005 / this plan's S2): a plan the budget gate PASSES but the
    approval gate DENIES falls through to ``_check_stall`` and stalls
    ``slug=None`` -- the exact slugless stall this iteration closes.  If the
    budget bar were LOOSER than the approval bar, that gap reopens, so BOTH read
    the SAME all-sections-non-placeholder predicate.  Reuses
    :func:`plan_validator._is_placeholder` (empty/whitespace/template-slot body
    -> placeholder); NEVER hand-roll a second content predicate (DRY, I1).  The
    seeded HEADERS-ONLY scaffold is ``PlanDoc``-valid but all-placeholder, so
    validity ALONE never gates it -- this predicate is what does.
    """
    return all(not _is_placeholder(plan.body_of(name)) for name in PlanSchema.SECTIONS)


#: A role worker: one :class:`RoleRequest` in, one ``AgentResult`` out.
#:
#: Deliberately WIDER than ``OrchestratorAgent``'s ``Callable[[str],
#: AgentResult]`` (``orchestrator.py:54``): that seam dispatches homogeneous
#: subtasks, while every harness dispatch is role- and state-specific and needs
#: the protocol counters to do its job at all (an executor cannot know which
#: step to run, or that it is on its second attempt, from a string).
WorkerFactory = Callable[[RoleRequest], AgentResult]


# ---------------------------------------------------------------------------
# Revert seam (the leash-cap recovery action)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RevertDirective:
    """What the ``leash-cap`` action asks to be reverted, and what to spare.

    Attributes:
        root: The workspace root -- the CODE the protocol is changing.  Never
            the plan directory (D-009).
        exclude: Paths under *root* that must survive the revert, relative to
            it.  Non-empty exactly when the plan directory lives inside the
            workspace, which is the common layout (``<repo>/plans/<plan-id>``).
        commands: The revert the protocol asks for, as shell text.  This class
            EXECUTES NOTHING; the strings are what a ``revert_callback`` (or a
            human reading the leash block) is being asked to run.
    """

    root: str
    exclude: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()

    def __str__(self) -> str:
        spared = f"; spare {', '.join(self.exclude)}" if self.exclude else ""
        return f"{' && '.join(self.commands)} (in {self.root}{spared})"


#: Executes a :class:`RevertDirective`.  Returns whether the revert was done.
RevertCallback = Callable[[RevertDirective], bool]


# ---------------------------------------------------------------------------
# Presentation contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Presentation:
    """One emitted user-facing block, plus what it failed to fill.

    ``missing_floor`` is the mechanism that makes "emitted with its floor
    fields" checkable rather than hoped for: it is the contract's floor minus
    what was actually supplied, in contract order, and it is ``()`` on every
    healthy emission.
    """

    name: str
    fields: Mapping[str, str]
    missing_floor: tuple[str, ...]
    block: str


#: ``state.md``'s ``## Current Plan Step:`` cursor, as the driver writes it
#: (``"7 of 17"``).  Only the leading integer is read back, so a hand-edited
#: cursor carrying prose (``"10 (wiring seam) + 12 (CLI)"``) still resumes.
_STEP_CURSOR_RE = re.compile(r"^(\d+)")

#: Rendered when a contract field has no value the driver can honestly supply.
#: Deliberately NOT counted as supplied by :meth:`HarnessAgent._emit_contract`,
#: so a placeholder can never satisfy a floor field.
_CONTRACT_ABSENT = "(not on record)"

#: Longest artifact excerpt rendered into one contract field.
_CONTRACT_FIELD_CHARS = 600

#: The protocol's Revert-First recovery, as text.  Nothing here is executed by
#: this module (D-039); a ``revert_callback`` or a human runs it.
_REVERT_COMMANDS: tuple[str, ...] = ("git checkout -- .", "git clean -fd")


# ---------------------------------------------------------------------------
# Re-entrancy guard (invariant I5)
# ---------------------------------------------------------------------------

# DECISION plan-2026-07-21T125237-191b2eb2/D-014
# The guard state is MODULE-level, not instance-level. Do NOT "tidy" it into
# `self._local`: the re-entry this defends against is a worker constructing a
# SECOND HarnessAgent and calling run() on it, which an instance-scoped flag
# cannot see. threading.local (not a plain global) keeps concurrent runs on
# different threads independent -- a process-wide lock would serialise them.
# See decisions.md D-014.
_DISPATCH_LOCAL = threading.local()


def _dispatch_in_flight() -> str | None:
    """Return the role currently being dispatched on this thread, if any."""
    role: str | None = getattr(_DISPATCH_LOCAL, "role", None)
    return role


def _reject_reentry(what: str) -> None:
    """Raise if a worker dispatched on this thread is re-entering the driver.

    Args:
        what: The driver surface being re-entered, for the error message.

    Raises:
        HarnessReentrancyError: If a dispatch is in flight on this thread.
    """
    role = _dispatch_in_flight()
    if role is not None:
        raise HarnessReentrancyError(role, details={"reentered": what})


@contextmanager
def _dispatch_in_progress(role: str) -> Iterator[None]:
    """Mark a dispatch as in flight on this thread for the duration."""
    _reject_reentry("dispatch")
    _DISPATCH_LOCAL.role = role
    try:
        yield
    finally:
        _DISPATCH_LOCAL.role = None


# ---------------------------------------------------------------------------
# Controlled halt
# ---------------------------------------------------------------------------


class _HarnessHalt(Exception):
    """Internal signal: stop the run now and report why.

    ``BaseAgent._run_conversation_loop`` exits only when the FSM reaches a
    terminal state, so a permanently BLOCKED gate has no other way out than
    burning the whole iteration budget.  Raising this from
    ``_on_loop_iteration`` unwinds the loop through its own
    ``finally: api.end_conversation(...)``; ``_standard_run`` re-raises it
    wrapped in an ``AgentError``, and :meth:`HarnessAgent.run` converts it back
    into a normal ``AgentResult``.  Private on purpose -- it is control flow,
    not an error a caller should ever see.
    """

    def __init__(self, reason: str, slug: str | None, context: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.slug = slug
        self.context = context


def _find_cause(error: BaseException, wanted: type[_ExcT]) -> _ExcT | None:
    """Return the first *wanted* exception in *error*'s cause chain.

    ``BaseAgent._standard_run`` wraps everything that is not a timeout or a
    budget failure in an ``AgentError``, so the driver's own control-flow and
    invariant signals only reach ``run()`` as a ``__cause__``.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, wanted):
            return current
        seen.add(id(current))
        current = current.__cause__
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class HarnessAgent(BaseAgent):
    """Drives the 6-state iterative-planner protocol over a real FSM.

    Usage::

        def worker(request: RoleRequest) -> AgentResult:
            ...  # do the role's job
            return AgentResult(answer="done", success=True, final_context={})

        agent = HarnessAgent(
            worker_factory=worker,
            approval_callback=lambda req: input(f"{req.tool_name}? ") == "y",
        )
        result = agent.run("Add retry logic to the uploader")

    The goal is the ``run(task)`` argument, not a constructor field, so the
    signature matches every other agent in the repo and one configured driver
    can serve several goals.
    """

    def __init__(
        self,
        *,
        worker_factory: WorkerFactory | None = None,
        approval_callback: ApprovalCallback | None = None,
        revert_callback: RevertCallback | None = None,
        config: AgentConfig | None = None,
        findings_threshold: int = Defaults.FINDINGS_THRESHOLD,
        max_fix_attempts: int = Defaults.MAX_FIX_ATTEMPTS,
        max_leash_grants: int = Defaults.MAX_LEASH_GRANTS,
        iteration_hard_cap: int = Defaults.ITERATION_HARD_CAP,
        max_explore_redispatches: int = Defaults.MAX_EXPLORE_REDISPATCHES,
        max_plan_redispatches: int = Defaults.MAX_PLAN_REDISPATCHES,
        max_reflect_redispatches: int = Defaults.MAX_REFLECT_REDISPATCHES,
        max_close_denials: int = Defaults.MAX_CLOSE_DENIALS,
        max_stall_turns: int = _DEFAULT_MAX_STALL_TURNS,
        **api_kwargs: Any,
    ) -> None:
        """Initialize the protocol driver.

        Args:
            worker_factory: Role dispatch seam.  ``None`` degrades to a
                conversation-only run: the FSM still turns, Pass-2 still
                answers, nothing crashes -- but no gate ever opens, because a
                gate flag records worker or human evidence and there is no
                worker to produce any (D-045).  Expect a stall halt naming the
                first shut gate.  ``None`` is a diagnostic mode, not a way to
                run the protocol.
            approval_callback: Consulted at every human gate.  Defaults to a
                callback that DENIES, so an unattended run cannot approve its
                own plan or close itself.
            revert_callback: Executes the ``leash-cap`` revert.  ``None`` (the
                default) means the driver COMPUTES the directive, reports it in
                the leash block, and executes nothing -- see D-039.  A supplied
                callback is still gated by *approval_callback* first.
            config: Agent configuration; defaults to the harness profile.
            findings_threshold: EXPLORE -> PLAN findings gate.
            max_fix_attempts: The autonomy leash -- unattended fix attempts per
                plan step.
            max_leash_grants: Human leash-continues honoured per plan step.
                Together with *max_fix_attempts* this bounds the executor
                dispatches one step can consume at
                ``max_fix_attempts * (1 + max_leash_grants)``, for ANY sequence
                of approvals.  The approval callback cannot raise it.
            iteration_hard_cap: PLAN -> EXECUTE iteration gate.
            max_explore_redispatches: EXTRA explorer dispatches spent per RUN
                while the findings gate is BLOCKED.  Bounds total EXPLORE
                dispatches at ``(genuine entries) + max_explore_redispatches``.
                Spending it without satisfying the gate is a HALT, never a pass.
            max_plan_redispatches: EXTRA plan-writer dispatches spent per RUN
                after a FAILED plan-writer reply.  Bounds total PLAN dispatches
                per ``(iteration, step)`` at ``1 + max_plan_redispatches``.
                Spending it is a HALT with the ``plan-cap`` slug, never a pass.
            max_reflect_redispatches: EXTRA verifier dispatches spent per RUN
                after a REFLECT reply that produced NO routable verdict.
                Bounds total REFLECT dispatches per ``(iteration, step)`` at
                ``1 + max_reflect_redispatches``.  Spending it is a HALT with
                the ``reflect-cap`` slug, never a pass.
            max_close_denials: EXTRA verifier dispatches spent per RUN after
                a DENIED human CLOSE approval on an all-criteria-pass
                verdict.  Bounds REFLECT dispatches (and approval
                consultations) on the denied-close path at
                ``1 + max_close_denials``.  Spending it is a HALT with the
                ``close-cap`` slug, never a pass.
            max_stall_turns: Consecutive no-progress turns before halting.
            api_kwargs: Forwarded to ``fsm_llm.API``.
        """
        super().__init__(config or self._default_config(), **api_kwargs)

        self.worker_factory = worker_factory
        self.findings_threshold = findings_threshold
        self.max_fix_attempts = max_fix_attempts
        self.max_leash_grants = max_leash_grants
        self.iteration_hard_cap = iteration_hard_cap
        self.max_explore_redispatches = max_explore_redispatches
        self.max_plan_redispatches = max_plan_redispatches
        self.max_reflect_redispatches = max_reflect_redispatches
        self.max_close_denials = max_close_denials
        self.max_stall_turns = max_stall_turns

        # DECISION plan-2026-07-21T125237-191b2eb2/D-015
        # A callback is ALWAYS supplied, and the default DENIES. Do NOT "simplify"
        # this to `approval_callback=None`: `HumanInTheLoop.request_approval`
        # RAISES ApprovalDeniedError when no callback is configured (hitl.py:95-98
        # -- its docstring's "defaults to auto-approve" is stale), which would turn
        # every unattended approval gate into a crashed turn instead of a denied
        # gate. Measured 2026-07-21: request_approval has NO confidence branch at
        # all and `ApprovalRequest` has no `confidence` field, so the callback is
        # consulted unconditionally and `confidence_threshold` is irrelevant here.
        # See decisions.md D-015.
        self._approval_callback: ApprovalCallback = approval_callback or _deny_approval
        self.hitl = HumanInTheLoop(approval_callback=self._approval_callback)

        # DECISION plan-2026-07-21T191807-bf7ffe24/D-039
        # The driver COMPUTES the leash-cap revert and does not EXECUTE it. Do
        # NOT "finish the job" by shelling out to `git checkout -- . && git
        # clean -fd` from here. Three reasons, and the first two are this
        # package's own already-made decisions:
        #   1. `git` is deliberately NOT in `tools.COMMAND_ALLOWLIST` (D-050 of
        #      the predecessor plan) precisely because it executes repo-local
        #      hooks, aliases and pagers, and `Workspace.run_command` is
        #      disabled by default. A driver that shells out to git would hold a
        #      capability the package withholds from every other actor in it.
        #   2. `git clean -fd` destroys UNTRACKED files with no backup, and the
        #      plan directory is untracked (`plans/` is gitignored) -- so the
        #      one action D-009 forbids is exactly the one an unscoped revert
        #      performs. `RevertDirective.exclude` is what states that scope,
        #      and stating it is the driver's job; running it is not.
        #   3. It is the protocol's only IRREVERSIBLE action, so it belongs
        #      behind the human gate the driver already has. A supplied callback
        #      is consulted only after `_APPROVAL_REVERT` is granted, and the
        #      default approval callback DENIES.
        # `None` is therefore not a degraded mode: the directive is computed,
        # scoped and reported in PC-EXECUTE-LEASH either way. See decisions.md
        # D-039.
        self._revert_callback = revert_callback

        self._api: API | None = None
        self._conversation_id: str | None = None
        self._stall_turns = 0
        self._stall_signature: tuple[Any, ...] | None = None
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-029
        # The re-dispatch bound lives HERE -- driver run state, beside
        # `_stall_turns`, the other driver-only counter of exactly this shape --
        # and NOT in the FSM context. Two properties are load-bearing:
        #   1. UNREACHABLE. A worker receives a `RoleRequest` (a context
        #      SNAPSHOT plus counters) and returns an `AgentResult`; it holds no
        #      reference to the driver, and `run`/`api`/`conversation_id` all
        #      raise `HarnessReentrancyError` from inside a dispatch. A context
        #      key would instead sit on the exact surface `_WORKER_WRITABLE` and
        #      `_reassert_driver_owned` exist to police, and would be rendered
        #      into both LLM prompts every turn (D-020's cap). Nothing reads
        #      this number but the bound, no transition condition names it, and
        #      no worker writes it -- so it has no business in gate context.
        #   2. NEVER RESET except by `_run_once`, per RUN, not per EXPLORE
        #      entry. A per-entry reset would be refillable BY A WORKER: the
        #      verifier may set `needs_explore` (it is in `_WORKER_WRITABLE`),
        #      which routes REFLECT -> EXPLORE and would hand back a full budget
        #      every time. `plans/LESSONS.md` [I:5] records that exact shape --
        #      "a safety cap the caller can reset from inside its own callback
        #      is not a cap" -- and the predecessor's 7c escape (90 dispatches
        #      from an approving callback zeroing its own counter) is what it
        #      was written about. See decisions.md D-029.
        self._explore_redispatches = 0
        # DECISION plan-2026-07-22T184813-6549c7cb/D-001
        #: The bounded PLAN re-dispatch budget spent this run.  Driver run
        #: state for exactly the reasons ``_explore_redispatches`` above is
        #: (D-029): no worker can reach it, the approval callback cannot reset
        #: it, and only ``_run_once`` resets it.  Do NOT fold it into
        #: ``fix_attempts``/the leash: a PLAN failure is not an EXECUTE fix
        #: attempt and must not share budget or reset semantics with one.
        #: See decisions.md D-001 (plan-2026-07-22T184813-6549c7cb).
        self._plan_redispatches = 0
        # DECISION plan-2026-07-23T173454-2c22e5f6/D-003
        #: The bounded REFLECT re-dispatch budget spent this run.  Driver run
        #: state for exactly the reasons ``_explore_redispatches`` above is
        #: (D-029): a worker receives a context SNAPSHOT and returns an
        #: ``AgentResult`` -- it holds no reference to the driver -- and the
        #: approval callback is handed a COPY of the context and returns a
        #: bool, so neither can reach or reset this counter; only ``_run_once``
        #: resets it.  Do NOT make it a context key (the [I:5]
        #: cap-reachable-from-callback shape), do NOT fold it into
        #: ``fix_attempts``/the leash (an unroutable VERDICT is not an EXECUTE
        #: fix attempt), and do NOT let a REFLECT-stuck run fall through to the
        #: stall detector's ``slug=None`` raise -- that is the S4b slugless
        #: stall this counter exists to close.  See decisions.md D-003
        #: (plan-2026-07-23T173454-2c22e5f6).
        self._reflect_redispatches = 0
        # DECISION plan-2026-07-24T032539-032ae337/D-001
        #: The bounded close-denial budget spent this run.  Driver run state
        #: for exactly the reasons ``_explore_redispatches`` above is
        #: (D-029): a worker receives a context SNAPSHOT and returns an
        #: ``AgentResult`` -- it holds no reference to the driver -- and the
        #: approval callback is handed a ``dict(context)`` COPY and returns a
        #: bool, so neither can reach or reset this counter; only
        #: ``_run_once`` resets it.  Do NOT make it a context key (the [I:5]
        #: cap-reachable-from-callback shape), do NOT fold it into
        #: ``_reflect_redispatches`` (a denied human CLOSE gate is not an
        #: unroutable verifier verdict -- the two budgets cover DISJOINT
        #: mechanisms and ``halt_slug`` must attribute which one fired), and
        #: do NOT let a denied-CLOSE run fall through to the stall detector's
        #: ``slug=None`` raise -- that is the residual-β slugless stall
        #: (L6 B7 run 3) this counter exists to close.  See decisions.md
        #: D-001 (plan-2026-07-24T032539-032ae337).
        self._close_denials = 0
        #: Slugs handed out to EXPLORE dispatches this run, in order.  Driver
        #: run state for exactly the reasons ``_explore_redispatches`` above is
        #: (D-029): a worker cannot reach it, and only ``_run_once`` resets it.
        #: Read by :meth:`_assign_explore_topic`; nothing else.
        self._assigned_topics: list[str] = []
        # DECISION plan-2026-07-21T125237-191b2eb2/D-055
        # ONE run per instance at a time, enforced -- not documented and hoped
        # for. `_api`, `_conversation_id`, `_stall_turns`, `_stall_signature`
        # and `_driver_owned` above are plain instance attributes rewritten by
        # every run, so two concurrent `run()` calls on ONE instance silently
        # corrupt each other's stall detector and driver-owned table (review
        # W7). D-014's independence claim covers the module-level
        # `threading.local` re-entrancy FLAG, not this run state. Do NOT
        # "simplify" this away as belt-and-braces on top of D-014's guard: that
        # guard is thread-scoped and would not see a second run on a second
        # thread at all, which is precisely the corrupting case. Two harness
        # runs at once need two HarnessAgent instances.
        self._run_lock = threading.Lock()
        #: The driver's authoritative value for every driver-owned context key
        #: (``constants.DRIVER_OWNED_SEEDS`` + ``DRIVER_OWNED_UNSET``).  ``None``
        #: means "this key should be ABSENT from context".  Written only by
        #: :meth:`run` and :meth:`_apply`; read only by
        #: :meth:`_reassert_driver_owned`.  See decisions.md D-044.
        self._driver_owned: dict[str, Any] = {}
        #: Every :class:`Presentation` emitted this run, in order.  Driver run
        #: state for D-029's two reasons: no worker can reach it, and nothing
        #: gates on it -- and for a third, D-020's: a contract block is hundreds
        #: of characters of verbatim artifact text, and the FSM context is
        #: rendered into BOTH prompts on EVERY turn.
        self._presentations: list[Presentation] = []
        #: Every :class:`RevertDirective` the leash-cap action computed.
        self._reverts: list[RevertDirective] = []
        #: The CLOSE audit's findings, or ``None`` if CLOSE was never reached.
        self._audit_issues: list[Issue] | None = None
        #: A halt decided inside a HANDLER, raised by the next loop turn.
        #: `fsm_llm.handlers.execute_handlers` CATCHES every handler exception,
        #: so a halt raised from a state-entry handler is swallowed AND takes
        #: that handler's whole delta (including its ledger entry) with it.
        #: `_on_loop_iteration` is the only place the driver may raise from.
        self._halt_request: _HarnessHalt | None = None
        #: The state the FSM starts in.  Normally ``HarnessStates.INITIAL``;
        #: a resumed run starts where its ``state.md`` left off (D-038).
        self._initial_state = HarnessStates.INITIAL

        logger.info(
            f"HarnessAgent started with model={self.config.model}, "
            f"worker_factory={'set' if worker_factory else 'none'}, "
            f"findings_threshold={findings_threshold}, "
            f"max_fix_attempts={max_fix_attempts}, "
            f"iteration_hard_cap={iteration_hard_cap}"
        )

    @staticmethod
    def _default_config() -> AgentConfig:
        """Build the harness's ``AgentConfig`` profile from ``Defaults``."""
        return AgentConfig(
            model=Defaults.MODEL,
            temperature=Defaults.TEMPERATURE,
            max_tokens=Defaults.MAX_TOKENS,
            max_iterations=Defaults.MAX_TURNS,
            timeout_seconds=Defaults.TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # Guarded accessors (invariant I5)
    # ------------------------------------------------------------------

    @property
    def api(self) -> API | None:
        """The live ``API``, or ``None`` outside a run.

        Raises:
            HarnessReentrancyError: If read from inside a dispatched worker.
        """
        _reject_reentry("api")
        return self._api

    @property
    def conversation_id(self) -> str | None:
        """The live conversation id, or ``None`` outside a run.

        Raises:
            HarnessReentrancyError: If read from inside a dispatched worker.
        """
        _reject_reentry("conversation_id")
        return self._conversation_id

    @property
    def presentations(self) -> tuple[Presentation, ...]:
        """Every user-facing contract block emitted by the last run.

        Raises:
            HarnessReentrancyError: If read from inside a dispatched worker.
        """
        _reject_reentry("presentations")
        return tuple(self._presentations)

    @property
    def reverts(self) -> tuple[RevertDirective, ...]:
        """Every leash-cap revert directive computed by the last run.

        Raises:
            HarnessReentrancyError: If read from inside a dispatched worker.
        """
        _reject_reentry("reverts")
        return tuple(self._reverts)

    @property
    def audit_issues(self) -> tuple[Issue, ...] | None:
        """The CLOSE audit's findings, or ``None`` if CLOSE was never reached.

        Raises:
            HarnessReentrancyError: If read from inside a dispatched worker.
        """
        _reject_reentry("audit_issues")
        return None if self._audit_issues is None else tuple(self._audit_issues)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Drive the protocol until CLOSE, a halt, or a budget limit.

        Args:
            task: The run's goal.
            initial_context: Optional seed context.  Protocol counters set
                below always win.

        Returns:
            An ``AgentResult``.  A halted run reports ``success=False`` with
            the halt reason as the answer and the halt slug in
            ``final_context[ContextKeys.LAST_GATE_SLUG]``.

        Raises:
            HarnessReentrancyError: If called from inside a dispatched worker.
            HarnessError: If a run is already in flight on this instance.
        """
        _reject_reentry("run")
        if not self._run_lock.acquire(blocking=False):
            raise HarnessError(
                "A HarnessAgent run is already in flight on this instance. "
                "The stall detector, the live API handle and the driver-owned "
                "context table are per-instance run state, so a second "
                "concurrent run would corrupt both. Construct a second "
                "HarnessAgent instead.",
                {"agent_type": _AGENT_TYPE},
            )
        try:
            return self._run_once(task, initial_context)
        finally:
            self._run_lock.release()

    def _run_once(
        self,
        task: str,
        initial_context: dict[str, Any] | None,
    ) -> AgentResult:
        """One protocol run, with the single-run lock already held."""
        self._stall_turns = 0
        self._stall_signature = None
        self._explore_redispatches = 0
        self._plan_redispatches = 0
        self._reflect_redispatches = 0
        self._close_denials = 0
        self._assigned_topics = []
        self._presentations = []
        self._reverts = []
        self._audit_issues = None
        self._halt_request = None

        fsm_def = build_harness_fsm(
            task,
            findings_threshold=self.findings_threshold,
            max_fix_attempts=self.max_fix_attempts,
            iteration_hard_cap=self.iteration_hard_cap,
        )

        # DECISION plan-2026-07-21T125237-191b2eb2/D-016 (REVERSED IN PART by D-044)
        # Every counter below is seeded EAGERLY, before turn 1. Do NOT make them
        # lazy ("set it when the state first needs it"): `iteration` is a
        # `requires_context_keys` term on the PLAN -> EXECUTE gate, and
        # `TransitionEvaluator._evaluate_condition` fails a condition whose
        # required key is ABSENT before it ever evaluates the logic
        # (transition_evaluator.py:351-362). An absent `iteration` therefore
        # BLOCKS PLAN -> EXECUTE permanently, no matter what the plan-writer
        # returns.
        #
        # D-016 originally seeded ONLY the counters and deliberately left every
        # gate FLAG absent, so the FSM's own Pass-1 extraction could still write
        # them -- "the entire degrade path when `worker_factory` is None".
        # MEASURED CONSEQUENCE (review C1): that degrade path IS the fabrication
        # engine. An LLM emitting {"plan_approved": true, "close_confirmed":
        # true} reached the terminal state with every worker dispatch failing and
        # a DENYING approval callback never consulted once. The degrade path and
        # invariants I6/I8 are mutually exclusive; I6 wins. Do NOT re-narrow this
        # seed set to "just the counters" to bring the degrade path back -- a
        # worker-less run is now expected to BLOCK at the first gate and halt
        # legibly via `_check_stall`, which is the honest outcome when no
        # evidence-producing worker exists. See decisions.md D-044 and D-045.
        # DECISION plan-2026-07-21T125237-191b2eb2/D-049
        # The two filesystem roots are read from the CALLER's context here and
        # then become driver-owned for the rest of the run. Do NOT let a role
        # supply or amend them: `plan_dir` is the directory `PlanMemory`
        # confines a role's write tools to, so an LLM-invented value re-points
        # the protocol's own memory. They are in `DRIVER_OWNED_UNSET` rather
        # than `DRIVER_OWNED_SEEDS` because a seed needs a fixed value and a
        # path does not have one -- absent is the correct default, and it
        # degrades to "no plan-directory tools", not to "any directory".
        # See decisions.md D-049.
        supplied = initial_context or {}
        roots: dict[str, Any] = {
            key: str(supplied[key])
            for key in (ContextKeys.PLAN_DIR, ContextKeys.WORKSPACE_ROOT)
            if _as_optional_str(supplied.get(key)) is not None
        }
        resumed_state, resumed_counters = self._resume(roots.get(ContextKeys.PLAN_DIR))
        self._initial_state = resumed_state or HarnessStates.INITIAL
        if self._initial_state != fsm_def["initial_state"]:
            fsm_def["initial_state"] = self._initial_state
        seeds: dict[str, Any] = {
            ContextKeys.GOAL: task,
            **DRIVER_OWNED_SEEDS,
            **resumed_counters,
            **roots,
        }
        self._driver_owned = {
            **dict.fromkeys(DRIVER_OWNED_UNSET),
            **seeds,
        }
        context = self._init_context(
            task,
            initial_context,
            extra={
                **seeds,
                ContextKeys.DISPATCH_LEDGER: [],
                ContextKeys.ROLE_RESULTS: [],
            },
        )

        try:
            return self._standard_run(
                task,
                fsm_def,
                context,
                _AGENT_TYPE,
                extra_answer_keys=[ContextKeys.HALT_REASON],
                execution_evidence_keys=[ContextKeys.ROLE_RESULTS],
            )
        except AgentError as exc:
            # A broken I5 invariant must reach the caller as itself, not as a
            # generic "Harness execution failed"; a halt is normal control flow
            # and becomes an ordinary reported result.
            reentry = _find_cause(exc, HarnessReentrancyError)
            if reentry is not None:
                # `from None`, never `from exc`: `reentry` is ALREADY inside
                # `exc.__cause__`'s chain, so `from exc` would close that chain
                # into a cycle and hang any naive `while e.__cause__` walker.
                raise reentry from None
            halt = _find_cause(exc, _HarnessHalt)
            if halt is None:
                raise
            return self._halt_result(halt)

    def _halt_result(self, halt: _HarnessHalt) -> AgentResult:
        """Turn a controlled halt into a reported ``AgentResult``."""
        logger.warning(f"Harness halted: {halt.reason} (slug={halt.slug})")
        final_context = dict(halt.context)
        final_context[ContextKeys.HALT_REASON] = halt.reason
        if halt.slug is not None:
            final_context[ContextKeys.LAST_GATE_SLUG] = halt.slug
        return AgentResult(
            answer=halt.reason,
            success=False,
            trace=AgentTrace(tool_calls=[], total_iterations=0),
            final_context=self._filter_context(final_context),
        )

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self, api: API) -> None:
        """Register the extraction guard, plus the per-state dispatch handlers."""
        self._api = api

        # DECISION plan-2026-07-21T125237-191b2eb2/D-044
        # BOTH timings are required, and the PRE_PROCESSING one is the
        # load-bearing half. Do NOT drop it as "redundant with CONTEXT_UPDATE":
        # MEASURED 2026-07-21 -- `MessagePipeline` hands the transition evaluator
        # a SECOND copy of the extraction payload
        # (`evaluate_transitions(state, context, extraction_response.extracted_data)`,
        # pipeline.py:1388), and `_prepare_working_context` merges that dict OVER
        # the live context (transition_evaluator.py:146,163). A CONTEXT_UPDATE
        # handler cleans `instance.context` only, so a value extracted on THIS
        # turn still reaches the gate through the payload copy -- reproduced: the
        # guard deleted `plan_approved` and PLAN -> EXECUTE fired anyway. The only
        # place to stop that is BEFORE extraction runs: PRE_PROCESSING fires at
        # pipeline.py:297, immediately before the extraction pass, and a
        # driver-owned key that is present and non-None there is SKIPPED by
        # extraction entirely (pipeline.py:953-956) -- so it never enters the
        # payload in the first place.
        # CONTEXT_UPDATE (pipeline.py:661-668, after the commit, before Step 3's
        # transition evaluation) is still registered: it keeps a fabricated value
        # from persisting into later turns, later prompts and `final_context`.
        # Priority 5 puts the guard ahead of every other harness handler.
        # See decisions.md D-044.
        api.register_handler(
            api.create_handler(HandlerNames.EXTRACTION_GUARD)
            .with_priority(HandlerPriorities.EXTRACTION_GUARD)
            .at(HandlerTiming.PRE_PROCESSING, HandlerTiming.CONTEXT_UPDATE)
            .do(self._reassert_driver_owned)
        )

        api.register_handler(
            api.create_handler(HandlerNames.START_DISPATCH)
            .with_priority(HandlerPriorities.START_DISPATCH)
            .at(HandlerTiming.START_CONVERSATION)
            # `self._initial_state`, not the `HarnessStates.INITIAL` literal:
            # a resumed run starts where its `state.md` left off (D-038), and a
            # START handler naming EXPLORE there would dispatch an EXPLORER
            # while the FSM sits in EXECUTE.
            .do(self._make_entry_handler(self._initial_state))
        )

        for state, handler_name in HandlerNames.BY_STATE.items():
            api.register_handler(
                api.create_handler(handler_name)
                .with_priority(HandlerPriorities.STATE_DISPATCH)
                .on_state_entry(state)
                .do(self._make_entry_handler(state))
            )

    def _reassert_driver_owned(self, context: dict[str, Any]) -> dict[str, Any]:
        """Restore the driver's own value for every driver-owned context key.

        Registered at two timings (see :meth:`_register_handlers`):

        * ``PRE_PROCESSING`` -- before Pass-1 extraction runs.  Re-seeding a key
          here means extraction skips it (``pipeline.py:953-956``), so it never
          enters ``DataExtractionResponse.extracted_data`` and therefore never
          reaches the transition evaluator.  This is the enforcement.
        * ``CONTEXT_UPDATE`` -- after Pass-1's data is committed.  This is the
          cleanup: it keeps a fabricated value out of later turns' prompts and
          out of ``final_context``.

        The driver never writes context through either path -- its own writes
        are POST_TRANSITION handler deltas and ``API.update_context`` calls,
        both of which update :attr:`_driver_owned` through :meth:`_apply` first.
        So any disagreement between context and :attr:`_driver_owned` seen here
        was authored by the LLM, and the driver's value wins.

        Args:
            context: The current context snapshot.

        Returns:
            A handler delta restoring the driver's values.  Empty on the normal
            path, where the LLM wrote nothing the driver owns.
        """
        delta: dict[str, Any] = {}
        for key, owned in self._driver_owned.items():
            if owned is None:
                if key in context:
                    delta[key] = None
            elif not _exactly(context.get(key), owned):
                delta[key] = owned

        if delta:
            logger.warning(
                f"Restoring {len(delta)} driver-owned key(s) {sorted(delta)} "
                "the FSM's own extraction wrote over. A gate flag records "
                "driver or worker evidence and is never read out of model "
                "prose (invariant I6)."
            )
        return delta

    def _make_entry_handler(
        self, state: str
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Build the state-entry handler for *state*."""

        def handler(context: dict[str, Any]) -> dict[str, Any]:
            return self._on_state_entry(state, context)

        return handler

    # ------------------------------------------------------------------
    # Ledger (invariant I4)
    # ------------------------------------------------------------------

    # DECISION plan-2026-07-21T125237-191b2eb2/D-017
    # The ledger holds TWO entry kinds in one list, and the dispatch key is a
    # 3-tuple `(state, iteration, step_number)` exactly as invariant I4 states.
    # Do NOT "fix" the re-dispatch of a failed step by widening the key with
    # `fix_attempts` (or by relaxing the guard to `count < N`): the key is what
    # the on-disk protocol will be audited against in a later step, and a wider
    # key silently authorises an unbounded number of attempts. A second attempt
    # is authorised EXPLICITLY, by the state-entry handler removing the key --
    # and that removal is itself made idempotent by the `entry:` marker, which
    # carries the pipeline's per-transition `_transition_timestamp`. Together:
    # a duplicate handler fire is a no-op, a genuine re-entry re-dispatches
    # exactly once. See decisions.md D-017.
    @staticmethod
    def _read_ledger(context: Mapping[str, Any]) -> list[str]:
        """Return the dispatch ledger, or an empty one if it is unusable."""
        ledger = context.get(ContextKeys.DISPATCH_LEDGER)
        if not isinstance(ledger, list):
            return []
        return [entry for entry in ledger if isinstance(entry, str)]

    @staticmethod
    def _ledger_delta(entries: list[str]) -> dict[str, Any]:
        """Context delta writing *entries* back, evicting oldest past the cap."""
        return {ContextKeys.DISPATCH_LEDGER: entries[-_MAX_LEDGER_ENTRIES:]}

    @staticmethod
    def _dispatch_key(context: Mapping[str, Any], state: str) -> str:
        """Ledger key for dispatching *state* at the context's counters."""
        iteration = as_int(context.get(ContextKeys.ITERATION), 0)
        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        return f"{_LEDGER_DISPATCH}:{state}:{iteration}:{step}"

    @staticmethod
    def _entry_marker(context: Mapping[str, Any], state: str) -> str:
        """Ledger marker identifying one state-entry handler fire.

        Keyed on ``_transition_timestamp``, which the pipeline rewrites on
        every transition (``pipeline.py:1723-1729``), so two fires for the same
        transition share a marker and the second is a no-op.  At conversation
        start there is no transition yet, hence the ``start`` literal.
        """
        token = context.get("_transition_timestamp", "start")
        return f"{_LEDGER_ENTRY}:{state}:{token!r}"

    # ------------------------------------------------------------------
    # State entry
    # ------------------------------------------------------------------

    # DECISION plan-2026-07-21T125237-191b2eb2/D-021
    # Dispatch-once is per state OCCUPANCY, not per run. A completion-fix run
    # legitimately dispatches BOTH `execute:1:1` and `reflect:1:1` twice, and
    # both of those second dispatches are required by the protocol (the leash
    # explicitly permits the retry; the second REFLECT verifies it). Do NOT
    # "restore" a literal once-per-run reading by bumping `step_number` or
    # `iteration` to make the key differ: that corrupts the EXECUTE plan-step
    # cursor and the iteration budget (D-018) purely to satisfy bookkeeping.
    # See decisions.md D-021.
    def _on_state_entry(self, state: str, context: dict[str, Any]) -> dict[str, Any]:
        """Handle one entry into *state*: bookkeeping, then dispatch.

        Args:
            state: The state just entered.
            context: FSM context snapshot handed to the handler.

        Returns:
            A context delta.  Empty when this is a duplicate fire.
        """
        ledger = self._read_ledger(context)
        marker = self._entry_marker(context, state)
        if marker in ledger:
            logger.debug(f"Duplicate state-entry fire for '{state}'; no-op")
            return {}

        working = dict(context)
        updates: dict[str, Any] = {}
        self._apply(updates, working, self._ledger_delta([*ledger, marker]))
        self._apply(updates, working, self._entry_bookkeeping(state, context))

        # The transition record, written BEFORE the dispatch so a worker reads
        # the position it is being dispatched into -- and outside
        # `_dispatch_if_needed`, so the pre-step gate below never reads a
        # document written inside its own call (D-038).
        self._sync_state_doc(state, working)

        # A genuine entry authorises exactly one dispatch for this state's key,
        # even when the counters did not move (a completion-fix retry of the
        # same step, or a loop-back into an already-visited state).
        key = self._dispatch_key(working, state)
        self._apply(
            updates,
            working,
            self._ledger_delta(
                [entry for entry in self._read_ledger(working) if entry != key]
            ),
        )

        self._apply(updates, working, self._dispatch_if_needed(state, working))
        return updates

    def _entry_bookkeeping(
        self, state: str, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Counter updates owed on entry to *state*.

        ``_previous_state`` is seeded by the pipeline immediately before the
        POST_TRANSITION handlers run, which is what lets a single EXECUTE entry
        handler tell a fresh iteration (PLAN -> EXECUTE) from a same-iteration
        completion fix (REFLECT -> EXECUTE).
        """
        previous = context.get("_previous_state")

        # DECISION plan-2026-07-21T125237-191b2eb2/D-044
        # Every clear below writes `False`, never `None`. Do NOT "restore" the
        # `None` (= delete) form: an ABSENT gate flag is exactly what makes it
        # extractable again (Pass-1 skips only fields already non-None,
        # pipeline.py:953-956), so deleting `plan_approved` on PLAN entry --
        # which this method used to do -- handed the next turn's LLM a second
        # chance to grant its own approval. `False` blocks the gate identically
        # (`TransitionCondition.logic` tests `== True` on all of these) while
        # keeping the key present and therefore unextractable.
        # ROUTING FLAGS ARE CLEARED WHERE THEY ARE CONSUMED (review W4): the
        # flag that routed us into a state is spent on arrival, so entering
        # EXPLORE clears `needs_explore` and entering PLAN clears
        # `pivot_resolved`. Without that, one `needs_explore=True` survived for
        # the rest of the run and silently won REFLECT routing at priority 600
        # on every later visit.
        # See decisions.md D-044.
        if state == HarnessStates.EXECUTE:
            return self._enter_execute(previous, context)

        if state == HarnessStates.EXPLORE:
            # The exploration request is honoured by being here.
            return {ContextKeys.NEEDS_EXPLORE: False}

        if state == HarnessStates.PLAN:
            # A plan must be approved again after any revision; the approval
            # records a human decision and never survives a re-entry.  The
            # pivot that sent us here (if any) is likewise spent.
            return {
                ContextKeys.PLAN_APPROVED: False,
                ContextKeys.PIVOT_RESOLVED: False,
            }

        if state == HarnessStates.PIVOT:
            # The leash never carries across a pivot: the next EXECUTE is a
            # different approach, not a third attempt at the same one -- so
            # BOTH halves of the per-step leash budget reset here (D-052).
            return {
                **_LEASH_RESET,
                ContextKeys.COMPLETION_FIX: False,
                ContextKeys.NEEDS_PIVOT: False,
            }

        if state == HarnessStates.REFLECT:
            return {ContextKeys.EXECUTE_COMPLETE: False}

        return {}

    def _enter_execute(
        self, previous: Any, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Counter updates owed on entry to EXECUTE."""
        updates: dict[str, Any] = {
            # Cleared so the EXECUTE -> REFLECT edge cannot fire on a stale flag.
            # `False`, not `None` -- see `_entry_bookkeeping`'s D-044 note.
            ContextKeys.EXECUTE_COMPLETE: False,
            ContextKeys.COMPLETION_FIX: False,
        }

        # DECISION plan-2026-07-21T125237-191b2eb2/D-018
        # `iteration` increments HERE and ONLY here -- on a PLAN -> EXECUTE entry.
        # Do NOT move it to "every EXECUTE entry" or "every REFLECT exit": a
        # REFLECT -> EXECUTE completion fix is the SAME iteration by definition
        # (plan.md's edge-case list), and counting it would burn the iteration
        # budget on remediation and make `iteration-cap` fire on a run that never
        # re-planned. `fix_attempts` resets on the same edge because a re-planned
        # iteration is user direction, not a third attempt at the old step.
        # See decisions.md D-018.
        if previous == HarnessStates.PLAN:
            updates[ContextKeys.ITERATION] = (
                as_int(context.get(ContextKeys.ITERATION), 0) + 1
            )
            updates[ContextKeys.STEP_NUMBER] = 1
            updates.update(_LEASH_RESET)
            updates[ContextKeys.LAST_GATE_SLUG] = None
            updates[ContextKeys.HALT_REASON] = None

        return updates

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch_if_needed(
        self, state: str, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Dispatch *state*'s worker unless its key is already in the ledger.

        Args:
            state: The state whose worker should run.
            context: Context including any updates applied so far this turn.

        Returns:
            A context delta; empty when the key was already dispatched.
        """
        ledger = self._read_ledger(context)
        key = self._dispatch_key(context, state)
        if key in ledger:
            return {}

        working = dict(context)
        updates: dict[str, Any] = {}
        self._apply(updates, working, self._ledger_delta([*ledger, key]))

        if state == HarnessStates.EXECUTE:
            blocked = self._pre_step_gate(working)
            if blocked is not None:
                self._apply(updates, working, blocked)
                return updates

        role = ROLE_BY_STATE[state]
        self._apply(updates, working, {ContextKeys.CURRENT_ROLE: role})

        result, error = self._run_worker(role, state, working)
        self._apply(
            updates,
            working,
            self._record_role_result(role, state, working, result, error),
        )
        role_delta: dict[str, Any] = {}
        if result is not None and result.success:
            role_delta = self._apply_role_result(state, result)
            self._apply(updates, working, role_delta)
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-032
        # This runs on EVERY attempted dispatch -- successful or FAILED -- and
        # that asymmetry with the line above is the entire point. Invariant I8
        # discards a failed dispatch's gate keys because a worker that did not
        # finish has not earned the right to be believed; that is TESTIMONY.
        # A count the driver reads off the filesystem itself is EVIDENCE: the
        # files exist whether or not the dispatch that wrote them reported
        # success. Measured (step 23, live `:4b`): one run in five ended holding
        # a real `findings/*.md` file and a gate value of 0, because all six of
        # its dispatches were recorded FAILED.
        # Do NOT "simplify" this by moving it inside the `result.success`
        # branch (that restores the bug) and, far more importantly, do NOT
        # generalise it to worker-CLAIMED keys by reading `_apply_role_result`
        # unconditionally. That would re-open exactly the fail-open gate step 20
        # closed: review C1 reproduced an EXPLORE dispatch that wrote ZERO bytes,
        # claimed `findings_count: 3` and SATISFIED the EXPLORE -> PLAN gate.
        # The rule is narrow and must stay narrow: values the DRIVER derives
        # from the filesystem are read regardless of dispatch success; values a
        # WORKER reports are read only from a successful dispatch, through
        # `_WORKER_WRITABLE`'s exact-type allowlist. A failed dispatch is still
        # recorded as failed, still spends whatever counter it spends and still
        # meets the leash -- this changes what the driver KNOWS, not what it
        # counts as work done.
        # The one case that derives NOTHING is `(None, None)` -- no worker
        # configured at all (D-045). Nothing was attempted, so the diagnostic
        # mode stays byte-identical: a directory left behind by an earlier run
        # must not open a gate in a run that dispatched no one.
        # See decisions.md D-032.
        if result is not None or error is not None:
            self._apply(updates, working, self._derive_gate_counts(state, working))
        if state == HarnessStates.REFLECT:
            # Runs on EVERY reflect dispatch, over the MERGED view -- see D-044.
            self._apply(
                updates,
                working,
                self._enforce_routing_exclusivity(role_delta, working),
            )
        self._apply(
            updates, working, self._post_dispatch(state, result, error, working)
        )
        return updates

    def _run_worker(
        self, role: str, state: str, context: Mapping[str, Any]
    ) -> tuple[AgentResult | None, Exception | None]:
        """Run one worker under the re-entrancy guard.

        Mirrors ``OrchestratorAgent._delegate_to_workers``
        (``orchestrator.py:155-182``): a raising worker is caught and reported,
        and a missing factory degrades instead of failing.  The single
        exception is ``HarnessReentrancyError`` -- swallowing it would hide the
        very invariant the guard exists to enforce.

        Returns:
            ``(result, error)``; exactly one of the two is not ``None``.
        """
        # DECISION plan-2026-07-21T125237-191b2eb2/D-045
        # `(None, None)` here is the DIAGNOSTIC mode, and it must stay unable to
        # advance anything. Do NOT "improve" a worker-less run by letting the
        # FSM's own Pass-1/Pass-2 calls fill the gate flags in -- that was
        # D-016's degrade path, and review-iter-1.md C1 measured what it
        # actually does: a full EXPLORE -> CLOSE traverse on fabricated
        # `plan_approved`/`close_confirmed` while a DENYING approval callback
        # was never consulted once. The degrade path and invariants I6/I8 are
        # mutually exclusive; this returns "nothing was attempted" so every gate
        # stays shut and the stall halt reports which one. The discriminator
        # matters downstream: `(None, None)` spends no leash attempt, whereas
        # `(None, exc)` -- a worker that RAISED -- does (D-051).
        # See decisions.md D-045.
        if self.worker_factory is None:
            return None, None

        execute_target, execute_target_reason = (
            self._assign_execute_target(context)
            if state == HarnessStates.EXECUTE
            else (None, None)
        )
        request = RoleRequest(
            role=role,
            state=state,
            goal=str(context.get(ContextKeys.GOAL, "")),
            operative_rules=get_rules(state).operative_rules,
            gate_summary=get_rules(state).gate_summary,
            iteration=as_int(context.get(ContextKeys.ITERATION), 0),
            step_number=as_int(context.get(ContextKeys.STEP_NUMBER), 0),
            total_steps=as_int(context.get(ContextKeys.TOTAL_STEPS), 1),
            fix_attempts=as_int(context.get(ContextKeys.FIX_ATTEMPTS), 0),
            context=MappingProxyType(dict(context)),
            plan_dir=_as_optional_str(context.get(ContextKeys.PLAN_DIR)),
            workspace_root=_as_optional_str(context.get(ContextKeys.WORKSPACE_ROOT)),
            # Assigned only for EXPLORE, and only AFTER the `worker_factory is
            # None` return above: the diagnostic mode attempts no dispatch, so
            # it must not consume a topic either (D-045).
            assigned_topic=(
                self._assign_explore_topic(context)
                if state == HarnessStates.EXPLORE
                else None
            ),
            assigned_write_target=execute_target,
            execute_target_reason=execute_target_reason,
        )

        try:
            with _dispatch_in_progress(role):
                result = self.worker_factory(request)
        except HarnessReentrancyError:
            raise
        except Exception as exc:  # deliberate: the turn must survive
            logger.warning(f"Worker for role '{role}' failed: {exc}", exc_info=True)
            return None, exc

        if not isinstance(result, AgentResult):
            logger.warning(
                f"Worker for role '{role}' returned {type(result).__name__}, "
                "expected AgentResult; recording as a failed dispatch"
            )
            return None, TypeError(f"worker returned {type(result).__name__}")

        return result, None

    def _record_role_result(
        self,
        role: str,
        state: str,
        context: Mapping[str, Any],
        result: AgentResult | None,
        error: Exception | None,
    ) -> dict[str, Any]:
        """Append this dispatch to ``role_results`` and set ``current_role_result``."""
        if result is not None:
            answer = result.answer
            success = result.success
        elif error is not None:
            answer = f"Worker error: {error}"
            success = False
        else:
            answer = f"[No worker configured for role '{role}']"
            success = False

        entry = {
            "role": role,
            "state": state,
            "iteration": as_int(context.get(ContextKeys.ITERATION), 0),
            "step_number": as_int(context.get(ContextKeys.STEP_NUMBER), 0),
            "answer": answer[:_MAX_ANSWER_CHARS],
            "success": success,
        }

        history = context.get(ContextKeys.ROLE_RESULTS)
        if not isinstance(history, list):
            history = []

        return {
            ContextKeys.ROLE_RESULTS: [*history, entry][-_MAX_ROLE_RESULTS:],
            ContextKeys.CURRENT_ROLE_RESULT: entry,
        }

    def _apply_role_result(self, state: str, result: AgentResult) -> dict[str, Any]:
        """Read gate flags out of a successful worker result.

        Only keys in :data:`_WORKER_WRITABLE` for *state* are read, and only
        when their runtime type matches exactly.  Anything else is dropped,
        which leaves the gate BLOCKED (invariant I8).

        The TABLE (``_WORKER_WRITABLE``) is this module's; the ALGORITHM is
        ``hardening.coerce_worker_output``'s, shared with ``roles.py``'s
        worker factory.  That split is D-028's, and step 7e is where the
        driver actually stopped keeping its own copy of it (D-059).
        """
        allowed = _WORKER_WRITABLE[state]
        if not allowed:
            return {}

        payload: dict[str, Any] = {}
        structured = getattr(result, "structured_output", None)
        if structured is not None and hasattr(structured, "model_dump"):
            dumped = structured.model_dump()
            if isinstance(dumped, dict):
                payload.update(dumped)
        if isinstance(result.final_context, dict):
            payload.update(result.final_context)

        return coerce_worker_output(
            payload, allowed, where=f"Worker for state '{state}'"
        )

    def _derive_gate_counts(
        self, state: str, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Count *state*'s disk-derived gate keys off the filesystem.

        The driver's own read, over the run's plan directory, through the same
        ``tools.derive_disk_counts`` the worker factory and the write tools use
        -- one derivation, so the gate value, the number reported back to the
        model and the number the re-dispatch loop tests can never disagree.

        Args:
            state: The state whose ``_WORKER_WRITABLE`` keys are considered.
            context: The merged view; supplies ``plan_dir``.

        Returns:
            ``{key: count}`` for every disk-derived gate key *state* owns.
            ``{}`` when there is no plan directory, when the state owns no such
            key, or when the directory cannot be read -- an unreadable
            directory is not evidence of anything and must leave the gate
            exactly as it was.
        """
        allowed = _WORKER_WRITABLE[state]
        if not any(key in allowed for key in DISK_DERIVED_COUNTS):
            return {}
        memory = self._read_only_memory(state, context)
        if memory is None:
            return {}
        try:
            return dict(derive_disk_counts(memory, allowed))
        except Exception as exc:  # unreadable root: no evidence, not zero
            logger.warning(
                f"Could not derive gate counts for state '{state}': {exc}. "
                "Leaving the gate value unchanged."
            )
            return {}

    @staticmethod
    def _existing_plan_dir(context: Mapping[str, Any]) -> Path | None:
        """The run's plan directory, but ONLY if it is already a directory.

        Interface contract (shared guard, 4 call sites: the two accessors
        below, the disk gate and the resume path):
            - Returns ``None`` when there is no ``plan_dir`` or the path is not
              an existing directory.  ``None`` means "nothing to observe",
              never "observed nothing".
            - Never raises and never creates anything.

        The existence precheck is the whole point.  ``PlanMemory`` (and through
        it ``PlanDirectory``) CREATES its plan directory on construction, and
        the driver has no business bringing protocol memory into existence as a
        side effect of looking at it -- a directory it created itself would
        also be a directory whose emptiness it then mistook for evidence, and
        whose absence the ``no-plan`` gate slug exists to report.  Every caller
        goes through here so that guarantee is one line in one place rather
        than a rule four methods have to remember.
        """
        plan_dir = _as_optional_str(context.get(ContextKeys.PLAN_DIR))
        if plan_dir is None:
            return None
        try:
            path = Path(plan_dir).expanduser()
            return path if path.is_dir() else None
        except (OSError, ValueError):  # unusable path: no observation
            return None

    @classmethod
    def _read_only_memory(
        cls, state: str, context: Mapping[str, Any]
    ) -> PlanMemory | None:
        """A role-scoped :class:`PlanMemory` for READING the plan directory.

        Interface contract (shared helper, 2 call sites:
        :meth:`_derive_gate_counts` and :meth:`_assign_explore_topic` -- the two
        places the driver reads the plan directory AS A ROLE would):
            - Returns ``None`` when :meth:`_existing_plan_dir` does, or when
              the memory cannot be constructed.
            - Never raises.
        """
        path = cls._existing_plan_dir(context)
        if path is None:
            return None
        try:
            return PlanMemory(path, role=ROLE_BY_STATE[state])
        except Exception as exc:  # unreadable / unusable root: no observation
            logger.warning(
                f"Could not open plan directory '{path}' for state "
                f"'{state}': {exc}. Treating it as unobserved."
            )
            return None

    @classmethod
    def _plan_directory(cls, context: Mapping[str, Any]) -> PlanDirectory | None:
        """The DRIVER's own accessor for the plan directory.

        Interface contract (shared helper, 5+ call sites: the state.md sync,
        the resume path, the four contract builders and the CLOSE audit):
            - Scoped to ``Role.ORCHESTRATOR``, so a write to an artifact the
              driver does not own raises ``HarnessOwnershipError`` from
              ``PlanMemory.authorise`` rather than being policed here.  The
              ownership table stays the single model (rules.py D-048).
            - Reads go through ``PlanDirectory``, which is uncapped (storage.py
              D-037); writes go through it too, which makes them ATOMIC
              (storage.py D-019).
            - Returns ``None`` exactly when :meth:`_existing_plan_dir` does.
              Never raises, never creates the directory.
        """
        path = cls._existing_plan_dir(context)
        if path is None:
            return None
        try:
            return PlanDirectory(path, role=Role.ORCHESTRATOR)
        except Exception as exc:  # unusable root: no protocol memory
            logger.warning(
                f"Could not open plan directory '{path}': {exc}. "
                "The driver will keep protocol state in context only."
            )
            return None

    @staticmethod
    def _artifact(
        directory: PlanDirectory, name: str, model: type[Artifact]
    ) -> Any | None:
        """Read and parse one owned artifact, or ``None`` if it is unusable.

        Interface contract (shared reader, 5 call sites: the four contract
        builders and the resume path):
            - Returns an instance of *model*, or ``None`` when the artifact is
              absent, unreadable, oversized or malformed.
            - Never raises.  A contract block is a REPORT: an unreadable
              artifact must leave its field empty (and therefore show up in
              ``Presentation.missing_floor``), not abort the protocol turn --
              handler exceptions are swallowed whole by the core handler
              system, taking the turn's ledger entry with them.
        """
        try:
            if not directory.exists(name):
                return None
            doc = directory.read_artifact(name)
        except (HarnessError, OSError) as exc:
            logger.warning(f"Could not read '{name}': {exc}")
            return None
        return doc if isinstance(doc, model) else None

    # ------------------------------------------------------------------
    # state.md -- the driver's one owned WRITE (invariant I7)
    # ------------------------------------------------------------------

    # DECISION plan-2026-07-23T124347-09045e6e/D-001
    def _render_plan_from_structured(
        self, result: AgentResult | None, context: Mapping[str, Any]
    ) -> str | None:
        # The plan-writer authors the 11 PlanSchema.SECTIONS as response_format
        # STRING fields (via the existing D-002 repair turn; roles.py Step 1 set
        # PLAN's output_schema to the 11 slug fields).  This renderer maps those
        # fields, IN SECTIONS ORDER, into a PlanDoc and writes plan.md through the
        # driver's own atomic write.  Load-bearing restrictions -- do NOT relax:
        #   1. DISTRIBUTION BY CONSTRUCTION.  Each field is placed under ITS
        #      heading here.  Do NOT go back to APPENDING model output to plan.md:
        #      `append_plan_file` lands at the file END and cannot distribute into
        #      the 11 ordered sections -- the iter-6 scaffold+append mechanism,
        #      REFUTED live across B0-B3 (floor 0/3; content concentrated in one
        #      section or duplicated headers).  The model authors fields; the
        #      driver only formats them.
        #   2. NEVER INVENT FILLER.  A section body is EXACTLY the model's field
        #      value (only heading-injection is escaped, `_demote_heading_lines`).
        #      An empty/whitespace field renders an EMPTY section body, which the
        #      UNCHANGED `_plan_is_approvable` (read off disk by `_plan_has_content`
        #      next) treats as a placeholder and DENIES -> honest `plan-cap` halt.
        #      Do NOT substitute a default, a heading echo, or any placeholder.
        #      This is the ethos safeguard (plan invariant 3).
        #   3. DISK STAYS THE GATE TRUTH.  This only writes plan.md; the payload
        #      itself never opens the gate.  `_plan_has_content` re-reads the
        #      rendered file and gates on it (plan invariant 2).
        #   4. FAIL-CLOSED.  A missing/partial `structured_output` (repair turn
        #      did not fire) writes NOTHING substantive, so the unchanged
        #      `_plan_has_content` check finds no approvable plan and the existing
        #      redispatch budget fires -- no behaviour change on the failure path.
        #      No plan directory (degrade path) -> None.
        # See decisions.md D-001.
        # DECISION plan-2026-07-23T124347-09045e6e/D-002
        # RENDER ON `structured_output` PRESENCE, NOT `result.success`.  Do NOT
        # re-add a `not result.success` guard here.  Under response_format the
        # PLAN model authors every field but calls NO write tool (the driver
        # renders), so the D-016 write-obligation (roles.py:1360-1363 -- a role
        # HOLDING a write tool must show a verified byte-write) fires and forces
        # `result.success=False` even on a perfectly VALID 14-field reply.  That
        # obligation is correct for tool-writing roles (EXPLORE/EXECUTE) but WRONG
        # for PLAN, whose deliverable is the DRIVER-rendered plan.md.  Gating the
        # render on `result.success` DISCARDED the valid plan -> plan.md stayed
        # 0 bytes -> plan-cap (the L6 B4 "config could-not-succeed" defect,
        # measured).  The ethos is untouched: the model still authors every field
        # (schema-validated structured_output, unfakeable), the driver only
        # formats, and an empty/placeholder field still renders an empty section
        # the UNCHANGED `_plan_is_approvable` denies.  Disk stays the gate truth.
        # See decisions.md D-002.
        if result is None:
            return None
        structured = getattr(result, "structured_output", None)
        if structured is None or not hasattr(structured, "model_dump"):
            return None
        dumped = structured.model_dump()
        if not isinstance(dumped, dict):
            return None
        directory = self._plan_directory(context)
        if directory is None:
            return None
        sections = [
            Section(
                name=title,
                body=_demote_heading_lines(str(dumped.get(slug, "") or "")),
            )
            for title, slug in PlanSchema.SLUG_BY_SECTION.items()
        ]
        try:
            plan = PlanDoc(title="Plan", sections=sections)
            return directory.write_artifact(ArtifactNames.PLAN, plan)
        except (HarnessError, OSError, ValueError) as exc:
            logger.warning(
                f"Could not render {ArtifactNames.PLAN} from the plan-writer's "
                f"structured reply: {exc}. The disk-read PLAN gate will find no "
                "approvable plan and the redispatch budget will fire."
            )
            return None

    def _sync_state_doc(self, state: str, context: Mapping[str, Any]) -> str | None:
        """Record the driver's position in ``state.md``, atomically.

        Called on every state ENTRY and after every loop-turn dispatch -- the
        two events at which the driver's position actually changes -- and never
        from inside :meth:`_dispatch_if_needed`, so the pre-step gate always
        reads a document written by an EARLIER event than itself.

        Returns:
            The path written, or ``None`` when there is no plan directory or
            the write failed.  A failed write is logged and swallowed: protocol
            memory is a record of the run, not a precondition for it, and
            raising here would lose the whole handler delta.
        """
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-038
        # `state.md` is the ONE artifact the driver writes, and it writes
        # exactly the fields it is the authority for. Two rules are
        # load-bearing:
        #   1. `## Transition History:` is PRESERVED, never appended to. Do NOT
        #      "improve" this by logging one line per FSM entry: the protocol's
        #      own auditor DERIVES an iteration count from the `EXECUTE ->
        #      REFLECT` arrows in that section and takes the maximum against the
        #      declared `## Iteration:` (`plan_validator._iteration_of`). A
        #      completion-fix cycle legitimately re-enters REFLECT several times
        #      inside ONE iteration (D-018), so a line per entry would make the
        #      derived count outrun the real one and fire `iteration-cap` on a
        #      run that never re-planned. The history is a narrative the
        #      orchestrator and the human write in phase-sized units; the driver
        #      owns `## Iteration:`, which is a counter it actually holds.
        #   2. The write is `PlanDirectory.write_artifact`, i.e. ATOMIC
        #      (storage.py D-019). Do NOT route it through `PlanMemory.write_text`
        #      to "reuse the tool the roles use": that is a plain
        #      `Path.write_text`, and a crash mid-write leaves a truncated
        #      state.md that still PARSES -- which the pre-step gate would then
        #      read as a smaller `fix_attempt_count`, i.e. a leash that resets
        #      itself on a torn write.
        # See decisions.md D-038.
        directory = self._plan_directory(context)
        if directory is None:
            return None
        try:
            return directory.write_artifact(
                ArtifactNames.STATE, self._state_doc(directory, state, context)
            )
        except (HarnessError, OSError) as exc:
            logger.warning(
                f"Could not write {ArtifactNames.STATE} for state '{state}': "
                f"{exc}. The run continues on context alone; the pre-step gate "
                "will report the disagreement."
            )
            return None

    def _state_doc(
        self, directory: PlanDirectory, state: str, context: Mapping[str, Any]
    ) -> StateDoc:
        """Build the ``state.md`` document for the driver's current position."""
        existing = self._artifact(directory, ArtifactNames.STATE, StateDoc)
        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        total = max(1, as_int(context.get(ContextKeys.TOTAL_STEPS), 1))
        attempts = as_int(context.get(ContextKeys.FIX_ATTEMPTS), 0)
        return StateDoc(
            state=state,
            skill_version=existing.skill_version if existing else None,
            iteration=as_int(context.get(ContextKeys.ITERATION), 0),
            current_step=f"{step} of {total}",
            checklist=list(existing.checklist) if existing else [],
            # The protocol's own Fix-Attempts grammar, which
            # `StateDoc.fix_attempt_count` -- and therefore the `leash-cap` gate
            # -- counts. One line per attempt already spent on THIS step.
            fix_attempts=[
                f"Step {step}, attempt {index}" for index in range(1, attempts + 1)
            ],
            change_manifest=list(existing.change_manifest) if existing else [],
            last_transition=f"{context.get('_previous_state') or 'INIT'} → {state.upper()}",
            transition_history=list(existing.transition_history) if existing else [],
        )

    def _resume(self, plan_dir: Any) -> tuple[str | None, dict[str, Any]]:
        """Read a previous run's position out of ``state.md``.

        Interface contract (1 call site: :meth:`_run_once`, before the FSM is
        built):
            - Returns ``(initial_state, counter_seeds)``.  ``(None, {})`` means
              there is nothing to resume: no plan directory, no ``state.md``,
              or one that could not be parsed.
            - Reads only.  Never raises, never creates the directory.
        """
        context = {ContextKeys.PLAN_DIR: plan_dir}
        directory = self._plan_directory(context)
        if directory is None:
            return None, {}
        doc = self._artifact(directory, ArtifactNames.STATE, StateDoc)
        if doc is None:
            return None, {}

        seeds: dict[str, Any] = {
            ContextKeys.ITERATION: doc.iteration,
            ContextKeys.FIX_ATTEMPTS: doc.fix_attempt_count,
        }
        cursor = _STEP_CURSOR_RE.match(doc.current_step.strip())
        if cursor is not None:
            seeds[ContextKeys.STEP_NUMBER] = int(cursor.group(1))
        plan = self._artifact(directory, ArtifactNames.PLAN, PlanDoc)
        if plan is not None and plan.steps():
            seeds[ContextKeys.TOTAL_STEPS] = len(plan.steps())
        logger.info(
            f"Resuming plan '{directory.plan_id}' from {ArtifactNames.STATE}: "
            f"state={doc.state.upper()}, iteration={seeds[ContextKeys.ITERATION]}, "
            f"step={seeds.get(ContextKeys.STEP_NUMBER, '?')}, "
            f"fix_attempts={seeds[ContextKeys.FIX_ATTEMPTS]}."
        )
        if doc.state == HarnessStates.TERMINAL:
            # A CLOSED plan is not resumable, and the reason is structural
            # rather than a policy call: `CLOSE` has no outbound transitions, so
            # making it the initial state leaves every other state ORPHANED and
            # `FSMDefinition` refuses the whole definition. The counters still
            # resume; the run starts a fresh EXPLORE over the same directory,
            # which is what "run again against a closed plan" has to mean.
            logger.warning(
                f"Plan '{directory.plan_id}' is already CLOSED. Its counters are "
                f"resumed, but the run starts a fresh {HarnessStates.INITIAL.upper()}"
                ": a closed plan has nowhere to resume TO."
            )
            return None, seeds
        return doc.state, seeds

    def _assign_explore_topic(self, context: Mapping[str, Any]) -> str | None:
        """Pick the ONE ``findings/<slug>.md`` topic this dispatch is to write.

        Returns:
            A kebab-case slug, or ``None`` when there is no plan directory to
            check or every topic already has a non-empty file on disk.
        """
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-035
        # The DRIVER assigns the topic; the model is never asked to invent one.
        # `agents/ip-orchestrator.md`'s EXPLORE dispatch step 4 is literally
        # "assign each topic a distinct kebab-case `findings/{topic-slug}.md`
        # slug and name it in the spawn prompt; first check `findings/` for an
        # existing file with that name -- no two live explorers may share a
        # slug", and this harness had never done it: it asked one dispatch to
        # choose its own topics and persist toward a COUNT. Four mechanisms
        # were measured against that shape on `ollama_chat/qwen3.5:4b` and all
        # four missed the bar (turn budget 0/5, observational feedback 0/5,
        # bounded re-dispatch 1/5, bound re-sizing 4/10). Step 24's falsifier
        # showed why none of them could work: there is a HORIZON, not a rate --
        # ~60 measured dispatches past the 9th added ZERO files -- so no budget
        # is the binding constraint. Do NOT replace this with a fifth attempt to
        # make one dispatch produce three topics.
        #
        # Three properties are load-bearing:
        #   1. The on-disk set comes from `tools.gate_files`, the SAME
        #      derivation the gate reads (D-015/D-027/D-032). A second listing
        #      here would be an assignment and a gate that can disagree, and the
        #      collision rule would then hand out a slug the gate already counts.
        #   2. Assigning a slug WRITES NOTHING. `_read_only_memory` refuses to
        #      construct memory for a directory that is not there, and nothing
        #      below touches the filesystem. A driver that pre-created empty
        #      files to "reserve" its topics would be review C1's fail-open
        #      wearing a different hat: the count would move without a role ever
        #      producing bytes. `gate_files` ignores empty files, so even a
        #      reserved-and-unwritten file could not open the gate -- but the
        #      driver must not create one at all, and a test pins that.
        #   3. `min(..., key=count)` is BOTH rules at once: prefer a topic never
        #      assigned this run (count 0, first in table order), and once every
        #      remaining topic has been tried, round-robin the least-tried one.
        #      Do NOT "simplify" it to `free[0]`: a topic the model keeps
        #      failing to write would then be re-assigned forever and the other
        #      topics would never be reached at all.
        # See decisions.md D-035.
        memory = self._read_only_memory(HarnessStates.EXPLORE, context)
        if memory is None:
            # No directory to check for collisions, and no plan-file tool for
            # the role either. Fail CLOSED: no assignment, no fabricated topic.
            return None
        on_disk = {
            name.removesuffix(".md")
            for name in gate_files(memory, ArtifactNames.FINDINGS_DIR)
        }
        free = [
            topic
            for topic in explore_topics(self.findings_threshold)
            if topic.slug not in on_disk
        ]
        if not free:
            return None
        chosen = min(free, key=lambda topic: self._assigned_topics.count(topic.slug))
        self._assigned_topics.append(chosen.slug)
        logger.info(
            f"EXPLORE dispatch assigned topic '{chosen.slug}' "
            f"({ArtifactNames.FINDINGS_DIR}/{chosen.slug}.md); "
            f"{len(on_disk)} of {self.findings_threshold} already on disk."
        )
        return chosen.slug

    @classmethod
    def _assign_execute_target(
        cls, context: Mapping[str, Any]
    ) -> tuple[str | None, str]:
        """Plan.md's target for this EXECUTE step, and WHY: ``(target, reason)``.

        A ``None`` target -> the prompt stays byte-identical (fail-open, never
        a guess -- D-010).  The reason is one of the ``EXECUTE_TARGET_*``
        literals, DIAGNOSTIC only.  A READ."""
        # DECISION plan-2026-07-22T184813-6549c7cb/D-005
        # Three distinct no-assignment causes used to collapse into one silent
        # None, so a live run whose plan-writer DID write a Files-To-Modify
        # table in a shape _TARGET_RE rejects was indistinguishable from a run
        # with no plan.md at all (L6 B0, reviewer W2).  Do NOT merge these
        # early returns back into a bare None, and do NOT let any prompt
        # builder read the reason -- it is observability for the bench rubric,
        # not dispatch content.  See decisions.md D-005.
        directory = cls._plan_directory(context)
        if directory is None:
            return None, EXECUTE_TARGET_NO_PLAN_DIR
        plan = cls._artifact(directory, ArtifactNames.PLAN, PlanDoc)
        if plan is None:
            return None, EXECUTE_TARGET_NO_PLAN_DOC
        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        target = derive_execute_target(plan, step)
        if target is None:
            # DECISION plan-2026-07-23T155204-fdc2d181/D-001
            # Additive S4a fallback: a real 4b plan names its target in PROSE
            # (no backticks), so the strict `derive_execute_target` finds no
            # token.  Try the existence-gated prose derivation before conceding
            # `no-target-token`.  This is the ONLY caller of `_derive_prose_target`
            # and the existence set comes from the workspace root, so a token
            # that names no real file cannot be assigned (D-010 fail-open holds).
            prose = _derive_prose_target(plan, step, cls._workspace_files(context))
            if prose is not None:
                return prose, EXECUTE_TARGET_ASSIGNED_PROSE
            return None, EXECUTE_TARGET_NO_TOKEN
        return target, EXECUTE_TARGET_ASSIGNED

    @staticmethod
    def _workspace_files(context: Mapping[str, Any]) -> frozenset[str]:
        """Top-level file names under the workspace root, for D-001's prose
        target existence gate.  Empty (fail-open) when the root is unset or
        unreadable -- never raises.  A READ."""
        root = _as_optional_str(context.get(ContextKeys.WORKSPACE_ROOT))
        if not root:
            return frozenset()
        try:
            return frozenset(p.name for p in Path(root).iterdir() if p.is_file())
        except OSError:
            return frozenset()

    @staticmethod
    def _enforce_routing_exclusivity(
        written: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Keep at most one REFLECT routing flag true.

        The flags gate three different outbound edges at three priorities, so
        two true flags would silently resolve by priority -- a completion fix
        would swallow a pivot the reviewer explicitly asked for.

        Args:
            written: The routing flags THIS dispatch's worker set (its filtered
                delta).  A fresh verdict outranks anything already in context.
            context: The merged view -- context plus every delta applied so far
                this turn.

        Returns:
            A delta clearing every loser; empty when no flag is true anywhere.
        """
        # DECISION plan-2026-07-21T125237-191b2eb2/D-044
        # TWO precedence layers, and both are load-bearing:
        #   1. A flag this dispatch's verifier wrote beats a flag that was
        #      merely lying in context. Do NOT collapse this to a single
        #      merged-view scan ranked by _REFLECT_ROUTING_FLAGS order: the
        #      executor sets `completion_fix` on its way OUT of a failed step,
        #      so at every REFLECT entry after a failure `completion_fix` is
        #      already true -- and completion_fix outranks needs_pivot. A merged
        #      scan therefore makes a verifier's explicit PIVOT verdict
        #      unreachable, which is the exact swallowing this function exists
        #      to prevent (pinned by test_reset_on_pivot).
        #   2. Within one layer, _REFLECT_ROUTING_FLAGS order decides.
        # The function also runs on EVERY reflect dispatch now, not only on a
        # SUCCESSFUL worker reply, and it reads `context` rather than only the
        # worker delta -- that is what stops a stale flag (review W4) from
        # silently winning REFLECT routing when the current worker sets none.
        # Losers are cleared to False, never to None: an absent flag is an
        # extractable flag (see `_entry_bookkeeping`). See decisions.md D-044.
        fresh = [key for key in _REFLECT_ROUTING_FLAGS if written.get(key) is True]
        stale = [key for key in _REFLECT_ROUTING_FLAGS if context.get(key) is True]
        chosen = fresh or stale
        if not chosen:
            return {}
        winner = chosen[0]
        return {key: False for key in _REFLECT_ROUTING_FLAGS if key != winner}

    def _post_dispatch(
        self,
        state: str,
        result: AgentResult | None,
        error: Exception | None,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Driver-owned bookkeeping that runs after every dispatch.

        *result* and *error* are :meth:`_run_worker`'s pair, and both are
        needed: ``result is None`` alone cannot tell a worker that RAISED
        (a spent attempt) from no worker being configured at all (nothing
        attempted).  See :meth:`_after_execute_dispatch`.
        """
        if state == HarnessStates.EXECUTE:
            return self._after_execute_dispatch(result, error, context)
        if state == HarnessStates.EXPLORE:
            return self._after_explore_dispatch(result, error, context)
        if state == HarnessStates.PLAN:
            return self._after_plan_dispatch(result, error, context)
        if state == HarnessStates.REFLECT:
            return self._after_reflect_dispatch(result, error, context)
        if state == HarnessStates.CLOSE:
            return self._after_close_dispatch(context)
        return {}

    def _after_close_dispatch(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Audit the finished plan directory.  Reports; never blocks the close.

        The CLOSE gate is ``close_confirmed``, a HUMAN decision (invariant I6).
        An audit finding is advisory by construction -- ``audit()`` returns
        typed issues and raises for none of them -- so it is recorded and
        logged, and the human who confirms the close is the one it is for.
        """
        path = self._existing_plan_dir(context)
        if path is None:
            return {}
        workspace = _as_optional_str(context.get(ContextKeys.WORKSPACE_ROOT))
        try:
            issues = audit(path, workspace_root=workspace)
        except Exception as exc:  # audit's contract says it cannot; log, do not crash
            logger.warning(f"CLOSE audit raised ({exc}); no findings recorded.")
            return {}
        self._audit_issues = issues
        errors = sum(1 for issue in issues if issue.severity == Severity.ERROR)
        logger.info(
            f"CLOSE audit of '{path.name}': {len(issues)} issue(s), "
            f"{errors} of them errors."
        )
        for issue in issues:
            logger.log(
                "ERROR" if issue.severity == Severity.ERROR else "WARNING", str(issue)
            )
        return {}

    # ------------------------------------------------------------------
    # Per-state post-dispatch
    # ------------------------------------------------------------------

    def _after_execute_dispatch(
        self,
        result: AgentResult | None,
        error: Exception | None,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Advance the step cursor, or spend a fix attempt.

        The worker never writes these counters itself: ``rules.py`` warns the
        executor that understating its attempt count bypasses the leash, and
        the way to make that impossible is for the driver to derive the count
        from ``AgentResult.success`` rather than to read it back.
        """
        # DECISION plan-2026-07-21T125237-191b2eb2/D-051
        # "The worker RAISED" and "there is no worker" are DIFFERENT events and
        # must not be collapsed into `result is None`. Do NOT re-simplify this
        # to a single `if result is None: return {}`: measured 2026-07-21
        # (findings/review-iter-1.md C3a) an executor that raised on every
        # dispatch spent ZERO attempts -- `fix_attempts` stayed 0, no
        # `leash-cap` slug was ever emitted, and EXECUTE held until the stall
        # halt, i.e. the autonomy leash did not engage AT ALL on the loudest
        # possible failure. `_record_role_result` had always recorded the same
        # event as `success=False` (as do plan.md's Failure Modes table and
        # Edge Cases), so the two writers read one condition in opposite
        # directions. A raising worker is a genuine failed attempt and spends
        # leash budget; `worker_factory=None` is the D-045 diagnostic mode and
        # must advance nothing. `_run_worker` already returns exactly one of
        # the two as non-None -- that is the discriminator.
        # See decisions.md D-051.
        if result is None and error is None:
            # No worker configured (D-045). Nothing was attempted, so nothing
            # is spent: every gate stays shut and EXECUTE holds until the stall
            # halt names the gate it is sitting behind.
            return {}

        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        total = max(1, as_int(context.get(ContextKeys.TOTAL_STEPS), 1))

        if result is not None and result.success:
            self._emit_execute_step(context)
            if step < total:
                # More steps remain: hold EXECUTE and let the loop-iteration
                # dispatch pick up the new (state, iteration, step) key.  A new
                # step is a new leash budget, both halves of it (D-052).
                return {
                    ContextKeys.STEP_NUMBER: step + 1,
                    **_LEASH_RESET,
                }
            return {ContextKeys.EXECUTE_COMPLETE: True}

        attempts = as_int(context.get(ContextKeys.FIX_ATTEMPTS), 0) + 1
        updates: dict[str, Any] = {
            ContextKeys.FIX_ATTEMPTS: attempts,
            ContextKeys.EXECUTE_COMPLETE: True,
        }
        if attempts >= self.max_fix_attempts:
            updates[ContextKeys.LAST_GATE_SLUG] = GateSlug.LEASH_CAP
            updates[ContextKeys.HALT_REASON] = (
                f"Autonomy leash: step {step} failed {attempts} times "
                f"(cap {self.max_fix_attempts}); reporting instead of retrying."
            )
            # The SAME action the pre-step gate's `leash-cap` slug runs: this is
            # where a healthy protocol actually hits the leash, and the block
            # must not depend on which of the two paths got there (D-040).
            self.on_leash_cap(context, step=step, attempts=attempts)
        else:
            updates[ContextKeys.COMPLETION_FIX] = True
        return updates

    def _after_explore_dispatch(
        self,
        result: AgentResult | None,
        error: Exception | None,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Re-authorise ONE more explorer while the findings gate is BLOCKED.

        Returns:
            A delta re-opening the EXPLORE dispatch key, or one recording the
            ``explore-cap`` halt, or ``{}`` when the gate is satisfied (or when
            there is no worker at all).
        """
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-028
        # The protocol asks ONE explorer for ONE topic and re-dispatches while
        # its gate fails -- `agents/ip-orchestrator.md`'s EXPLORE step 7 is
        # literally "If gate fails: spawn additional explorers for gaps". This
        # harness used to ask ONE dispatch for THREE findings, which is a bar the
        # source protocol never sets. Do NOT "simplify" this back to a single
        # dispatch: four mechanisms were measured against that shape on
        # `ollama_chat/qwen3.5:4b` and ALL FOUR failed at 0 runs in 10 reaching
        # 3 distinct findings files -- tool count (falsified in the OPPOSITE
        # direction: fewer tools wrote LESS), plan-directory seeding, prompt
        # wording (three independent refutations), and observational feedback
        # (step 22 proved the model READS the derived count back accurately and
        # still stops at 1). See decisions.md D-022, D-027, D-028.
        #
        # Three properties keep this from being a livelock, and all three are
        # load-bearing:
        #   1. The condition is the GATE's OWN value -- `findings_count` against
        #      `findings_threshold`, the same variable the EXPLORE -> PLAN
        #      JsonLogic term reads, which since D-015 is derived from the
        #      `findings/*.md` files really on disk. Do NOT re-count the
        #      directory here: a second count is a gate and a loop condition
        #      that can disagree, and this loop would then either spin past a
        #      satisfied gate or stop short of one. It also means a worker
        #      cannot stop the re-dispatch without ALSO opening the gate, so the
        #      loop grants no authority the gate did not already grant.
        #   2. The BOUND is `self._explore_redispatches`, which no worker can
        #      reach and which only `_run_once` resets (see its D-029 block).
        #   3. Spending the bound HALTS -- a distinct slug, an honest reason,
        #      and the gate left exactly as shut as it was. Do NOT "finish the
        #      job" here by writing `findings_count` or `needs_explore`: an
        #      exploration that cannot reach its gate is a real failure and the
        #      run must report it (invariant I8).
        if result is None and error is None:
            # No worker configured (D-045): nothing was attempted, so there is
            # nothing to re-attempt. The diagnostic mode's stall halt is
            # unchanged, and it must not be relabelled as an exploration cap.
            return {}

        found = as_int(context.get(ContextKeys.FINDINGS_COUNT), 0)
        if found >= self.findings_threshold:
            # The EXPLORE -> PLAN handoff: the gate is satisfied, so this is the
            # last EXPLORE dispatch and the index is final.
            self._emit_explore(context)
            return {}

        if self._explore_redispatches >= self.max_explore_redispatches:
            logger.warning(
                f"Exploration cap [{GateSlug.EXPLORE_CAP}]: "
                f"{self._explore_redispatches} re-dispatch(es) spent (cap "
                f"{self.max_explore_redispatches}) and findings_count is still "
                f"{found} of {self.findings_threshold}; not dispatching again."
            )
            return {
                ContextKeys.LAST_GATE_SLUG: GateSlug.EXPLORE_CAP,
                ContextKeys.HALT_REASON: (
                    f"Exploration cap: {self._explore_redispatches} extra "
                    f"explorer(s) dispatched (cap "
                    f"{self.max_explore_redispatches}) and findings/ still "
                    f"holds {found} of the {self.findings_threshold} findings "
                    "the EXPLORE -> PLAN gate requires. The gate is NOT "
                    "satisfied; re-scope the exploration."
                ),
            }

        self._explore_redispatches += 1
        logger.info(
            f"EXPLORE gate BLOCKED at findings_count={found} of "
            f"{self.findings_threshold}; re-dispatching "
            f"({self._explore_redispatches}/{self.max_explore_redispatches})."
        )
        key = self._dispatch_key(context, HarnessStates.EXPLORE)
        return self._ledger_delta(
            [entry for entry in self._read_ledger(context) if entry != key]
        )

    def _plan_has_content(self, context: Mapping[str, Any]) -> bool:
        # DECISION plan-2026-07-23T095051-a6dcb40d/D-001
        # Whether a CONFIGURED plan directory carries an APPROVABLE plan.md, read
        # OFF DISK. Used by `_after_plan_dispatch` to catch the empty-plan stall
        # (predecessor D-005) AND the seeded-scaffold / invalid-plan /
        # PARTIALLY-filled shapes. "Approvable" = plan.md parses as a valid
        # `PlanDoc` (all 11 sections, in order) AND EVERY section body is
        # NON-placeholder -- the EXACT SAME bar `DiskEvidenceApprovals` (the
        # approval gate) uses, via the shared `_plan_is_approvable` predicate.
        #   (a) ALL-non-placeholder (NOT not-all-placeholder) is the bar, and the
        #       ALIGNMENT with the approval gate is load-bearing: the harness
        #       does NOT redispatch on approval DENIAL, so a plan this BUDGET gate
        #       PASSED but the approval gate then DENIED would fall through to
        #       `_check_stall` and stall slug=None -- the slugless stall this
        #       iteration closes. The budget gate MUST therefore match the
        #       approval gate exactly (both call `_plan_is_approvable`); a LOOSER
        #       bar here reopens the gap. So an un-filled scaffold (all
        #       placeholder), a PARTIALLY-filled plan (some placeholder), OR an
        #       invalid plan -> not approvable -> the redispatch budget fires ->
        #       the model is redispatched to fill MORE -> exhaustion halts on the
        #       honest GateSlug.PLAN_CAP; NO generic STALL slug is minted
        #       (predecessor D-003 respected).
        #   (b) The seeded scaffold is ALWAYS `PlanDoc`-valid, so validity alone
        #       never gates it out -- the placeholder check does. A NON-empty but
        #       SCHEMA-INVALID plan.md (bad/missing headers) fails `PlanDoc`
        #       parse, `_artifact` returns None -> False (closing predecessor
        #       S2's non-empty-but-invalid slugless stall).
        #   (c) Reads through the DRIVER's OWN `PlanDirectory` (uncapped,
        #       storage.py D-037) via the existing `_artifact` parse-or-None
        #       helper; the content bar is the shared `_plan_is_approvable` ->
        #       `plan_validator._is_placeholder` (I1/DRY): ONE predicate, TWO
        #       gates, no drift. No worker-claimed flag is trusted (I1).
        #   (d) PRESERVE the no-plan-directory degrade path (predecessor D-005):
        #       when `_plan_directory` is None (PLAN_DIR absent) there is no disk
        #       to read, so return True so the OR-ed condition defers to
        #       `result.success` exactly as before. Returning False here would
        #       reclassify every successful degrade-path plan reply as a failure
        #       and redden ~59 traverse tests (D-005's measured trap). The real
        #       PLAN shape always has a plan directory, so the override still
        #       fires for every production/L6 case.
        # See decisions.md D-001.
        directory = self._plan_directory(context)
        if directory is None:
            return True
        plan = self._artifact(directory, ArtifactNames.PLAN, PlanDoc)
        if plan is None:
            return False
        return _plan_is_approvable(plan)

    def _after_plan_dispatch(
        self,
        result: AgentResult | None,
        error: Exception | None,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Retry a FAILED plan-writer under a bounded budget, or seek approval.

        Returns:
            A delta re-opening the PLAN dispatch key, or one recording the
            ``plan-cap`` halt, or the approval outcome on a successful reply
            (or ``{}`` when there is no worker at all).
        """
        # DECISION plan-2026-07-22T184813-6549c7cb/D-001
        # PLAN previously dispatched exactly ONCE per (iteration, step): this
        # handler returned {} on a failed reply, nothing re-opened the dispatch
        # key, and the eventual halt was the 3-turn stall detector's -- which
        # always raises slug=None. MEASURED (L6 B0 run 3,
        # scripts/bench_data/l6-e2e/rows.jsonl, `:4b`): one empty plan-writer
        # reply was terminal; the run recorded plan_md_bytes 0 and halt_slug
        # null -- the slugless stall shape the L6 floor exists to catch. The
        # fix mirrors `_after_explore_dispatch` (D-028/D-029), and the same
        # properties are load-bearing:
        #   1. A retry is authorised EXPLICITLY, by removing the SAME
        #      (state, iteration, step) key from the ledger. Do NOT widen
        #      `_dispatch_key` with a retry counter instead: D-017 forbids
        #      exactly that "fix" -- a wider key silently authorises an
        #      unbounded number of attempts.
        #   2. The BOUND is `self._plan_redispatches` -- driver run state on
        #      `self` (see its D-001 block in `__init__`), never a context
        #      key, reset only by `_run_once`, unreachable from any worker
        #      and from `_request_approval`'s callback. A PLAN failure is NOT
        #      an EXECUTE fix attempt: it touches neither `fix_attempts` nor
        #      the leash, and `_check_iteration_cap` (D-019, a distinct halt
        #      on genuine PLAN re-entry) is untouched.
        #   3. Spending the bound HALTS honestly: LAST_GATE_SLUG/HALT_REASON
        #      are pre-written HERE because `_check_stall` always raises with
        #      slug=None and `_halt_result` only preserves a slug already in
        #      context -- the same mechanism EXPLORE's cap relies on. Do NOT
        #      "finish the job" by writing `plan_approved` or `total_steps`:
        #      a plan that was never written must not be approved (I8).
        # The default budget (Defaults.MAX_PLAN_REDISPATCHES = 3) is an
        # UNMEASURED placeholder -- see the constant's own block. See
        # decisions.md D-001 (plan-2026-07-22T184813-6549c7cb).
        if result is None and error is None:
            # No worker configured (D-045): nothing was attempted, so there is
            # nothing to re-attempt and no cap to spend.
            return {}
        # DECISION plan-2026-07-23T124347-09045e6e/D-001
        # Render the plan-writer's structured reply into plan.md BEFORE the
        # disk-read gate below. This REPLACES appending model output to the file:
        # append lands at the file END and cannot distribute into the 11 ordered
        # sections (iter-6 scaffold+append, REFUTED live B0-B3). The renderer is
        # fail-closed and invents no filler, so on a partial/failed reply it
        # writes nothing substantive and the unchanged `_plan_has_content` check
        # below redispatches exactly as before. See decisions.md D-001.
        self._render_plan_from_structured(result, context)
        # DECISION plan-2026-07-22T212329-16de43da/D-005
        # `_plan_has_content` (disk-derived) is OR-ed into the retry condition
        # so a `success=True` reply that left plan.md EMPTY consumes the budget
        # exactly like a worker failure and halts on PLAN_CAP, not a slugless
        # stall. A genuine success WITH a non-empty plan.md still falls through
        # to `_emit_plan`/`_request_approval` unchanged -- only the empty case
        # is newly caught. See _plan_has_content and decisions.md D-005.
        # DECISION plan-2026-07-23T124347-09045e6e/D-002
        # KEY THIS GATE ON `_plan_has_content` (DISK), NOT `result.success`, WHEN
        # A PLAN DIRECTORY EXISTS.  Do NOT re-add `not result.success` to the
        # plan-directory branch.  The render above now writes plan.md from
        # `structured_output` regardless of success, so the authority for "is
        # there an approvable plan?" is the disk read, not the write-obligation-
        # polluted `result.success` (see the renderer's D-002 block: response_
        # format PLAN replies are success=False by construction because the model
        # calls no write tool).  `_plan_has_content` -- via the shared, byte-
        # unchanged `_plan_is_approvable` -- is True IFF plan.md parses as a valid
        # PlanDoc with every section non-placeholder.  A dispatch whose
        # structured_output was None/partial renders nothing substantive ->
        # `_plan_has_content` False -> redispatch, exactly as before: dropping
        # `not result.success` does NOT let a failed dispatch through, the disk
        # gate catches it.  DEGRADE PATH PRESERVED (plan invariant edge-case d,
        # predecessor D-005): when there is NO plan directory `_plan_has_content`
        # cannot read disk, so this defers to `result.success` exactly as today
        # -- the real PLAN/L6 shape ALWAYS has a plan directory, so the disk
        # authority fires for every production case.  The `result is None` guard
        # stays (no worker / no reply).  See decisions.md D-002.
        directory = self._plan_directory(context)
        plan_approvable = (
            self._plan_has_content(context)
            if directory is not None
            else (result is not None and bool(result.success))
        )
        if result is None or not plan_approvable:
            if self._plan_redispatches >= self.max_plan_redispatches:
                logger.warning(
                    f"Plan cap [{GateSlug.PLAN_CAP}]: "
                    f"{self._plan_redispatches} re-dispatch(es) spent (cap "
                    f"{self.max_plan_redispatches}) and the plan-writer has "
                    "still not returned a successful reply; not dispatching "
                    "again."
                )
                return {
                    ContextKeys.LAST_GATE_SLUG: GateSlug.PLAN_CAP,
                    ContextKeys.HALT_REASON: (
                        f"Plan cap: {self._plan_redispatches} extra "
                        f"plan-writer dispatch(es) spent (cap "
                        f"{self.max_plan_redispatches}) and none returned a "
                        "successful reply, so there is no plan to approve. "
                        "The PLAN -> EXECUTE gate is NOT satisfied; "
                        "characterize the plan-writer failure before raising "
                        "the budget."
                    ),
                }
            self._plan_redispatches += 1
            logger.info(
                f"PLAN dispatch failed; re-dispatching the plan-writer "
                f"({self._plan_redispatches}/{self.max_plan_redispatches})."
            )
            key = self._dispatch_key(context, HarnessStates.PLAN)
            return self._ledger_delta(
                [entry for entry in self._read_ledger(context) if entry != key]
            )
        if context.get(ContextKeys.NEEDS_EXPLORE) is True:
            # The plan-writer bounced back to EXPLORE; there is no plan to approve.
            return {}
        # Emitted BEFORE the approval is requested, which is the contract's own
        # "when emitted": a human cannot approve a plan they have not been shown.
        self._emit_plan(context)
        approved = self._request_approval(
            _APPROVAL_PLAN,
            context,
            reasoning=(
                f"Approve the plan for iteration "
                f"{as_int(context.get(ContextKeys.ITERATION), 0) + 1}?"
            ),
        )
        return {ContextKeys.PLAN_APPROVED: approved}

    def _after_reflect_dispatch(
        self,
        result: AgentResult | None,
        error: Exception | None,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Ask the human to confirm a close, or to continue past the leash --
        or retry an UNROUTABLE verifier, or a DENIED close, each under its own
        bounded budget.

        Returns:
            The close-approval or leash outcome, ``{}`` when a routing flag
            lets an outbound edge fire, a delta re-opening the REFLECT dispatch
            key, or one recording the ``reflect-cap`` or ``close-cap`` halt.
        """
        # Emitted before the routing decision, which is this contract's own
        # "when emitted" -- the verdict is what the routing is read from.
        self._emit_reflect(context)
        if context.get(ContextKeys.LAST_GATE_SLUG) == GateSlug.LEASH_CAP:
            return self._offer_leash_continue(context)

        if context.get(ContextKeys.ALL_CRITERIA_PASS) is True:
            confirmed = self._request_approval(
                _APPROVAL_CLOSE,
                context,
                reasoning="Every success criterion is verified PASS. Close the plan?",
            )
            if confirmed is True:
                return {ContextKeys.CLOSE_CONFIRMED: confirmed}
            # DECISION plan-2026-07-24T032539-032ae337/D-001
            # A DENIED close approval must not end the run slugless. This arm
            # previously returned unconditionally, so a denial left every
            # REFLECT edge BLOCKED (`close_confirmed` False shuts REFLECT ->
            # CLOSE), nothing re-opened the dispatch key, and the eventual
            # halt was the stall detector's -- which always raises slug=None.
            # MEASURED (L6 B7 run 3, scripts/bench_data/l6-e2e/B7/rows.jsonl,
            # `:4b`): exactly ONE verifier dispatch, 4/4 criteria PASS, ONE
            # denial on "verification.md is absent or empty", then halt_slug
            # null off "Stalled in REFLECT for 3 turns" -- the residual-β
            # slugless stall. The fix mirrors `_after_plan_dispatch`'s budget
            # exactly.
            # DECISION plan-2026-07-24T032539-032ae337/D-002
            # CORRECTED post-review (B8): the original rationale here -- that
            # the re-dispatched verifier "holds the write tool for
            # verification.md" and can genuinely repair the denial -- was
            # FALSE. The redispatch gives the protocol a bounded chance to
            # produce fresh evidence, but on the CURRENT configuration the
            # verifier holds READ_ONLY + SHELL tools only (roles.py), owns no
            # artifact (rules.py OWNERSHIP), and no driver path writes
            # verification.md, so the retry CANNOT repair an "absent or empty
            # verification.md" denial -- MEASURED (L6 B8 run 1): a PARSEABLE
            # success verdict plus 3 funded retries left the file at 0 bytes.
            # The budget's value here is the bounded HONEST halt (close-cap),
            # not repair. Do NOT re-justify a larger cap on repair grounds.
            # The same properties are load-bearing:
            #   1. A retry is authorised EXPLICITLY, by removing the SAME
            #      (state, iteration, step) key from the ledger -- never a
            #      widened key (D-017).
            #   2. The BOUND is `self._close_denials` -- driver run state on
            #      `self` (see its D-001 block in `__init__`), DISJOINT from
            #      `_reflect_redispatches`: a denied human gate and an
            #      unroutable verdict are different mechanisms, they spend
            #      independent budgets, and `halt_slug` must attribute which
            #      one fired (LESSONS [I:5]).
            #   3. Spending the bound HALTS honestly on `close-cap`:
            #      LAST_GATE_SLUG/HALT_REASON are pre-written HERE because
            #      `_check_stall` always raises with slug=None. Do NOT
            #      "finish the job" by writing `close_confirmed` True or
            #      touching `all_criteria_pass`: a close the human refused
            #      must not close (I8). A ROUTABLE fresh verifier reply may
            #      re-derive the verdict; an empty or unparseable retry
            #      leaves the prior flag standing (measured, B8 run 1: all 3
            #      budgeted retries empty-replied and the loop ran on the
            #      stale all_criteria_pass=True from the first verdict).
            # See decisions.md D-001 and D-002
            # (plan-2026-07-24T032539-032ae337).
            if self._close_denials >= self.max_close_denials:
                logger.warning(
                    f"Close cap [{GateSlug.CLOSE_CAP}]: "
                    f"{self._close_denials} re-dispatch(es) spent (cap "
                    f"{self.max_close_denials}) and the human CLOSE approval "
                    "is still denied; not dispatching again."
                )
                return {
                    ContextKeys.LAST_GATE_SLUG: GateSlug.CLOSE_CAP,
                    ContextKeys.HALT_REASON: (
                        f"Close cap: {self._close_denials} extra verifier "
                        f"dispatch(es) spent (cap {self.max_close_denials}) "
                        "and the human CLOSE approval is still denied on an "
                        "all-criteria-pass verdict, so the REFLECT -> CLOSE "
                        "gate is NOT satisfied. Characterize the denial "
                        "evidence before raising the budget."
                    ),
                    ContextKeys.CLOSE_CONFIRMED: False,
                }
            self._close_denials += 1
            logger.info(
                f"CLOSE approval denied; re-dispatching the verifier "
                f"({self._close_denials}/{self.max_close_denials})."
            )
            key = self._dispatch_key(context, HarnessStates.REFLECT)
            delta = self._ledger_delta(
                [entry for entry in self._read_ledger(context) if entry != key]
            )
            delta[ContextKeys.CLOSE_CONFIRMED] = False
            return delta

        # DECISION plan-2026-07-23T173454-2c22e5f6/D-003
        # A REFLECT whose verdict routes NOWHERE must not end the run slugless.
        # REFLECT previously dispatched exactly ONCE per (iteration, step):
        # this handler returned {} on an unroutable verdict, nothing re-opened
        # the dispatch key, all four REFLECT edges stayed BLOCKED, and the
        # eventual halt was the 3-turn stall detector's -- which always raises
        # slug=None. MEASURED (S5 probe run 1, scripts/bench_data/l6-e2e/
        # probe-s5-mechanism/, `:4b`; same shape in L6 B5 run 1 and B6 run 2):
        # a success=True verifier reply (526 answer chars) whose coerced delta
        # set NO routing key was terminal -- halt_slug null, honest_halt false,
        # the S4b slugless stall. The fix mirrors `_after_plan_dispatch`
        # (D-001 of plan-2026-07-22T184813-6549c7cb) exactly, and the same
        # properties are load-bearing:
        #   1. A retry is authorised EXPLICITLY, by removing the SAME
        #      (state, iteration, step) key from the ledger -- never a widened
        #      key (D-017).
        #   2. The BOUND is `self._reflect_redispatches` -- driver run state on
        #      `self` (see its D-003 block in `__init__`), never a context key,
        #      reset only by `_run_once`, unreachable from any worker and from
        #      `_request_approval`'s callback.
        #   3. Spending the bound HALTS honestly: LAST_GATE_SLUG/HALT_REASON
        #      are pre-written HERE because `_check_stall` always raises with
        #      slug=None and `_halt_result` only preserves a slug already in
        #      context. Do NOT "finish the job" by writing `all_criteria_pass`
        #      or a routing flag: a verdict the verifier never produced must
        #      not route the run (I8).
        # The condition is ROUTABILITY, not `result.success`: the measured
        # stall variant IS a success=True reply, so gating on success would
        # miss it (the same lesson as _after_plan_dispatch's D-002 disk-keyed
        # gate). A denied CLOSE approval (all_criteria_pass True, human said
        # no) still does NOT consume THIS budget -- approval denial is a shut
        # human gate, not a retryable verifier failure -- but it is NO LONGER
        # an out-of-scope slugless shape: since the aligned-gates iteration
        # this comment pre-named, the denial path is bounded by its OWN
        # disjoint budget (`self._close_denials` -> `close-cap`) inside the
        # all-criteria-pass arm above.
        # DECISION plan-2026-07-24T032539-032ae337/D-001
        if result is None and error is None:
            # No worker configured (D-045): nothing was attempted, so there is
            # nothing to re-attempt and no cap to spend.
            return {}
        if any(context.get(flag) is True for flag in _REFLECT_ROUTING_FLAGS):
            # An outbound edge can fire (completion fix / pivot / explore
            # loop-back): the verdict is routable, no budget is consumed.
            return {}

        if self._reflect_redispatches >= self.max_reflect_redispatches:
            logger.warning(
                f"Reflect cap [{GateSlug.REFLECT_CAP}]: "
                f"{self._reflect_redispatches} re-dispatch(es) spent (cap "
                f"{self.max_reflect_redispatches}) and the verifier has still "
                "not produced a routable verdict; not dispatching again."
            )
            return {
                ContextKeys.LAST_GATE_SLUG: GateSlug.REFLECT_CAP,
                ContextKeys.HALT_REASON: (
                    f"Reflect cap: {self._reflect_redispatches} extra "
                    f"verifier dispatch(es) spent (cap "
                    f"{self.max_reflect_redispatches}) and none produced a "
                    "routable verdict (no criteria verdict, no routing flag), "
                    "so no REFLECT edge can fire. Characterize the verifier "
                    "failure before raising the budget."
                ),
            }
        self._reflect_redispatches += 1
        logger.info(
            f"REFLECT verdict unroutable; re-dispatching the verifier "
            f"({self._reflect_redispatches}/{self.max_reflect_redispatches})."
        )
        key = self._dispatch_key(context, HarnessStates.REFLECT)
        return self._ledger_delta(
            [entry for entry in self._read_ledger(context) if entry != key]
        )

    def _offer_leash_continue(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Ask the human to continue past an exhausted leash, at most N times.

        Returns:
            A delta granting one more pair of attempts, or one recording that
            the grant budget is spent, or ``{}`` on a denial.
        """
        # DECISION plan-2026-07-21T125237-191b2eb2/D-052
        # `leash_grants` is what makes the leash a LEASH. Do NOT restore the
        # unconditional `fix_attempts: 0` reset that used to live here: measured
        # 2026-07-21 (findings/review-iter-1.md C3b), with an always-approving
        # callback -- which is the test suite's own default -- and an
        # always-failing executor, resetting the attempt counter on every grant
        # cycled EXECUTE <-> REFLECT until `BudgetExhaustedError: iterations
        # limit (60)`. The cap the callback saw was infinite. The counter lives
        # in DRIVER-OWNED context, so the callback (which is handed a COPY of
        # the context and returns a bool) cannot reach it, and the FSM's own
        # extraction cannot invent it either (D-044). It resets only where a
        # genuinely NEW step begins (_LEASH_RESET's three sites), never on a
        # grant. Property: executor dispatches on ONE plan step are bounded by
        # max_fix_attempts * (1 + max_leash_grants) for ANY approval sequence.
        # Once the budget is spent the driver stops ASKING -- an approval whose
        # answer would be discarded is theatre. See decisions.md D-052.
        grants = as_int(context.get(ContextKeys.LEASH_GRANTS), 0)
        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        if grants >= self.max_leash_grants:
            logger.warning(
                f"Leash grant budget spent on step {step}: {grants} "
                f"continuation(s) already granted (cap {self.max_leash_grants}); "
                "not asking again."
            )
            return {
                ContextKeys.HALT_REASON: (
                    f"Autonomy leash: step {step} has already been continued "
                    f"{grants} time(s) by hand (cap {self.max_leash_grants}) and "
                    f"still fails. Re-plan or re-scope the step."
                )
            }

        granted = self._request_approval(
            _APPROVAL_LEASH,
            context,
            reasoning=(
                f"The {self.max_fix_attempts}-attempt autonomy leash is "
                f"exhausted ({grants}/{self.max_leash_grants} continuations "
                "used). Continue with user direction, or stop here?"
            ),
        )
        if not granted:
            return {}
        # Explicit user direction buys one more pair of attempts; it is not a
        # third unattended attempt.  `leash_grants` is NOT reset here -- that
        # is the whole point.
        return {
            ContextKeys.FIX_ATTEMPTS: 0,
            ContextKeys.LEASH_GRANTS: grants + 1,
            ContextKeys.LAST_GATE_SLUG: None,
            ContextKeys.HALT_REASON: None,
            ContextKeys.COMPLETION_FIX: True,
        }

    # ------------------------------------------------------------------
    # Approvals (invariant I6)
    # ------------------------------------------------------------------

    def _request_approval(
        self, gate: str, context: Mapping[str, Any], *, reasoning: str
    ) -> bool:
        """Consult the human callback for one protocol gate.

        Returns ``False`` -- never ``True`` -- when the callback itself fails,
        so a broken approval path denies rather than opens the gate.
        """
        call = ToolCall(
            tool_name=gate,
            parameters={
                ContextKeys.ITERATION: as_int(context.get(ContextKeys.ITERATION), 0),
                ContextKeys.STEP_NUMBER: as_int(
                    context.get(ContextKeys.STEP_NUMBER), 0
                ),
                ContextKeys.FIX_ATTEMPTS: as_int(
                    context.get(ContextKeys.FIX_ATTEMPTS), 0
                ),
            },
            reasoning=reasoning,
        )
        try:
            return bool(self.hitl.request_approval(call, dict(context)))
        except Exception as exc:  # fail closed, never open
            logger.warning(
                f"Approval gate '{gate}' errored ({exc}); treating as DENIED"
            )
            return False

    # ------------------------------------------------------------------
    # Gates and halts
    # ------------------------------------------------------------------

    def _pre_step_gate(self, context: Mapping[str, Any]) -> dict[str, Any] | None:
        """Decide whether this EXECUTE step may be dispatched, and act on it.

        Two channels are consulted and BOTH must pass.  ``plan_validator``'s
        on-disk gate answers all four slugs from ``state.md``; the in-memory
        leash answers ``leash-cap`` from the driver's own counter.  Each slug
        then routes to its OWN action -- they are four different events, not
        four names for one halt.

        Returns:
            A context delta when the step must not be dispatched, else ``None``.
        """
        result = self._disk_gate(context)
        if result.passed:
            result = self._memory_leash(context)
        if result.passed:
            return None
        assert result.slug is not None  # GateResult's own model validator
        return _GATE_ACTIONS[result.slug](self, result, context)

    def _disk_gate(self, context: Mapping[str, Any]) -> GateResult:
        """Run ``plan_validator.pre_step_gate`` over this run's plan directory.

        Returns ``GateResult(passed=True)`` when there is NO plan directory:
        the four slugs are statements about an on-disk protocol, and a run
        without one is not failing them -- it simply has none, and the
        in-memory leash below is then the whole gate.
        """
        path = self._existing_plan_dir(context)
        if path is None:
            return GateResult(passed=True)
        try:
            return pre_step_gate(
                path,
                expected_state=HarnessStates.EXECUTE,
                max_fix_attempts=self.max_fix_attempts,
                iteration_cap=self.iteration_hard_cap,
            )
        except Exception as exc:  # contract says it cannot; fail CLOSED anyway
            logger.warning(f"Pre-step gate raised ({exc}); treating as no-plan.")
            return GateResult(
                passed=False, slug=GateSlug.NO_PLAN, detail=f"gate raised: {exc}"
            )

    def _memory_leash(self, context: Mapping[str, Any]) -> GateResult:
        """The driver's OWN leash check, over its own counter.

        Kept alongside the on-disk gate rather than replaced by it, and the
        redundancy is the point: ``fix_attempts`` in context is derived by
        :meth:`_after_execute_dispatch` from ``AgentResult.success``, which no
        worker can understate, whereas ``state.md``'s attempt lines are a file
        that a failed write, a truncation or a hand edit can leave behind.  If
        the two ever disagree the SHUT one wins.
        """
        attempts = as_int(context.get(ContextKeys.FIX_ATTEMPTS), 0)
        if attempts < self.max_fix_attempts:
            return GateResult(passed=True)
        return GateResult(
            passed=False,
            slug=GateSlug.LEASH_CAP,
            detail=f"attempts={attempts} cap={self.max_fix_attempts} (in-memory)",
        )

    # -- the four slug actions ------------------------------------------
    #
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-040
    # The four pre-step-gate slugs map to four DIFFERENT actions. Do NOT
    # "simplify" this table into one halt path parameterised by a slug string,
    # and in particular do NOT emit the leash block from more than one of them:
    #   * `leash-cap` is the only genuine LEASH hit. It is the only one that
    #     computes a revert directive, the only one that emits PC-EXECUTE-LEASH,
    #     and the only one that sets `execute_complete` -- because it is the
    #     only one that CONTINUES, routing EXECUTE -> REFLECT so the failure is
    #     reviewed rather than ending the run.
    #   * `iteration-cap` ends the run and emits NO leash block. A run can hit
    #     the iteration cap with ZERO recorded attempts, so PC-EXECUTE-LEASH's
    #     "attempts" field could not be filled with anything but a fiction, and
    #     the block would point the user at a failing step that does not exist
    #     (the predecessor's D-019 made the same call for the PLAN-edge cap).
    #   * `wrong-state` and `no-plan` are RECOVERY paths and neither writes
    #     `state.md`. That is not an oversight: `no-plan` fires precisely
    #     because state.md is unreadable or absent, and "fixing" it by writing
    #     over it would manufacture the agreement the gate exists to check --
    #     the driver would silence its own alarm. `wrong-state` is the same
    #     defect one step further on: something other than the driver moved the
    #     protocol, and overwriting it discards the evidence of what.
    # See decisions.md D-040.

    def _act_leash_cap(
        self, result: GateResult, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The autonomy leash: revert, report, and route to REFLECT."""
        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        attempts = as_int(context.get(ContextKeys.FIX_ATTEMPTS), 0)
        logger.warning(
            f"Pre-step gate [{GateSlug.LEASH_CAP}]: step {step} has "
            f"{result.detail}; refusing to dispatch."
        )
        self.on_leash_cap(context, step=step, attempts=attempts)
        return {
            ContextKeys.EXECUTE_COMPLETE: True,
            ContextKeys.LAST_GATE_SLUG: GateSlug.LEASH_CAP,
            ContextKeys.HALT_REASON: (
                f"Autonomy leash: step {step} already used {attempts} fix "
                f"attempts (cap {self.max_fix_attempts})."
            ),
        }

    def _act_iteration_cap(
        self, result: GateResult, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The iteration hard cap: end the run, with NO leash block."""
        reason = (
            f"Iteration cap: {result.detail}; this plan may not spend another "
            "EXECUTE step. Decompose the goal into a fresh plan."
        )
        logger.warning(f"Pre-step gate [{GateSlug.ITERATION_CAP}]: {result.detail}")
        self._halt_request = _HarnessHalt(
            reason=reason, slug=GateSlug.ITERATION_CAP, context=dict(context)
        )
        return {
            ContextKeys.LAST_GATE_SLUG: GateSlug.ITERATION_CAP,
            ContextKeys.HALT_REASON: reason,
        }

    def _act_wrong_state(
        self, result: GateResult, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Recovery: something else moved the protocol.  Report, write nothing."""
        return self._recovery(
            GateSlug.WRONG_STATE,
            result,
            f"Recovery [{GateSlug.WRONG_STATE}]: {ArtifactNames.STATE} disagrees "
            f"with the driver ({result.detail}). Something other than the driver "
            "moved the protocol; reconcile it by hand. The driver will NOT "
            "overwrite state.md to make itself right.",
        )

    def _act_no_plan(
        self, result: GateResult, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Recovery: there is no readable protocol memory.  Write nothing."""
        return self._recovery(
            GateSlug.NO_PLAN,
            result,
            f"Recovery [{GateSlug.NO_PLAN}]: {result.detail}. Restore or "
            f"re-bootstrap the plan directory. The driver will NOT create "
            f"{ArtifactNames.STATE} to satisfy its own gate.",
        )

    @staticmethod
    def _recovery(slug: str, result: GateResult, reason: str) -> dict[str, Any]:
        """Both recovery actions: record the slug and the reason, nothing else.

        No ``execute_complete`` (the run does not route onward on a directory
        it cannot trust) and no artifact write of any kind.  EXECUTE holds and
        ``_check_stall`` ends the run with this reason prepended.
        """
        logger.warning(reason)
        return {
            ContextKeys.LAST_GATE_SLUG: slug,
            ContextKeys.HALT_REASON: reason,
        }

    # ------------------------------------------------------------------
    # The leash-cap recovery (D-009 scope, D-039 execution)
    # ------------------------------------------------------------------

    def on_leash_cap(
        self, context: Mapping[str, Any], *, step: int, attempts: int
    ) -> RevertDirective | None:
        """The ``leash-cap`` action itself: revert, then report.

        Interface contract (shared action, 2 call sites -- and the sharing is
        the point):
            - ``_act_leash_cap``, the pre-step gate, which refuses a dispatch
              that has already spent the leash;
            - ``_after_execute_dispatch``, which is where the leash ACTUALLY
              bites in a healthy protocol -- the FSM's REFLECT -> EXECUTE edge
              carries ``fix_attempts < cap``, so the gate above is defence in
              depth and fires only if that edge was bypassed.
            - *step* and *attempts* are passed EXPLICITLY, not read from
              *context*: the post-dispatch caller has not applied its own delta
              yet, so context still holds the pre-increment count.
            - Never raises; both callers run inside an FSM handler.

        Emitting the leash block from one of the two sites only would make the
        block's presence depend on which path the leash happened to take, which
        is exactly the kind of "reported sometimes" the contract exists to rule
        out.
        """
        directive = self._revert_uncommitted(context)
        self._emit_execute_leash(context, directive, step=step, attempts=attempts)
        return directive

    def _revert_uncommitted(self, context: Mapping[str, Any]) -> RevertDirective | None:
        """Compute -- and only then, if permitted, execute -- the revert.

        Returns:
            The directive, or ``None`` when there is no ``workspace_root`` to
            revert.  A returned directive says what SHOULD be reverted; whether
            it WAS is the callback's business and is logged either way.
        """
        root = _as_optional_str(context.get(ContextKeys.WORKSPACE_ROOT))
        if root is None:
            logger.info(
                "Leash revert skipped: the run has no workspace_root, so there "
                "is no code tree to revert."
            )
            return None
        directive = RevertDirective(
            root=root,
            exclude=self._spared_paths(root, context),
            commands=_REVERT_COMMANDS,
        )
        self._reverts.append(directive)

        if self._revert_callback is None:
            logger.warning(
                f"Leash revert NOT executed (no revert_callback): {directive}. "
                "It is reported in the leash block for a human to run."
            )
            return directive
        if not self._request_approval(
            _APPROVAL_REVERT,
            context,
            reasoning=f"Revert uncommitted work? {directive}",
        ):
            logger.warning(f"Leash revert DENIED at the approval gate: {directive}")
            return directive
        try:
            done = bool(self._revert_callback(directive))
        except Exception as exc:  # a failed revert must not crash the turn
            logger.warning(f"Leash revert callback failed ({exc}): {directive}")
            return directive
        logger.info(f"Leash revert {'executed' if done else 'declined'}: {directive}")
        return directive

    @staticmethod
    def _spared_paths(root: str, context: Mapping[str, Any]) -> tuple[str, ...]:
        """Paths under *root* the revert must NOT touch.

        The plan directory, when it lives inside the workspace -- which is the
        usual layout (``<repo>/plans/<plan-id>``) and exactly the case D-009 is
        about.  ``plans/`` is gitignored, so it survives ``git checkout -- .``
        for free but NOT ``git clean -fd``: an unscoped revert deletes the run's
        own memory, including the record of the failure being reported.
        """
        plan_dir = _as_optional_str(context.get(ContextKeys.PLAN_DIR))
        if plan_dir is None:
            return ()
        try:
            relative = (
                Path(plan_dir)
                .expanduser()
                .resolve()
                .relative_to(Path(root).expanduser().resolve())
            )
        except (OSError, ValueError):
            return ()  # outside the workspace: nothing to spare, D-009 holds
        return (str(relative),)

    # ------------------------------------------------------------------
    # Presentation contracts
    # ------------------------------------------------------------------

    def _emit_contract(self, name: str, fields: Mapping[str, str]) -> Presentation:
        """Render one user-facing block and record what it could not fill.

        Interface contract (shared emitter, 5 call sites -- one per contract
        the driver has real data for):
            - *fields* is keyed by the contract's own field names; unsupplied
              and blank fields are rendered as :data:`_CONTRACT_ABSENT` and are
              NOT counted as supplied, so a placeholder can never satisfy a
              floor field.
            - Never raises for missing data: a report that aborts the turn is
              worse than an incomplete report, and handler exceptions are
              swallowed whole by the core handler system.  A missing FLOOR
              field is logged at ERROR and recorded in
              ``Presentation.missing_floor``, which is what a test asserts on.

        Raises:
            KeyError: If *name* is not a known contract -- a typo in the
                driver's own source, not a runtime condition.
        """
        missing = missing_floor_fields(
            name, [key for key, value in fields.items() if str(value).strip()]
        )
        ordered = PRESENTATION_CONTRACTS[name].required
        rows = [
            f"- **{field}**: {str(fields.get(field, '')).strip() or _CONTRACT_ABSENT}"
            for field in ordered
        ]
        presentation = Presentation(
            name=name,
            fields=MappingProxyType({key: str(value) for key, value in fields.items()}),
            missing_floor=missing,
            block="\n".join([f"### {name}", *rows]),
        )
        self._presentations.append(presentation)
        if missing:
            logger.error(
                f"{name} emitted WITHOUT its floor field(s) {list(missing)}: the "
                "artifacts they are read from were absent or unreadable."
            )
        else:
            logger.info(
                f"{name} emitted with all "
                f"{len(PRESENTATION_CONTRACTS[name].floor)} of its floor fields."
            )
        return presentation

    def _emit_explore(self, context: Mapping[str, Any]) -> None:
        """PC-EXPLORE: the findings index and constraints, verbatim from disk."""
        directory = self._plan_directory(context)
        index = (
            None
            if directory is None
            else self._artifact(
                directory, ArtifactNames.FINDINGS_INDEX, FindingsIndexDoc
            )
        )
        found = as_int(context.get(ContextKeys.FINDINGS_COUNT), 0)
        self._emit_contract(
            "PC-EXPLORE",
            {
                "findings-index": "" if index is None else index.body_of("Index"),
                "key-constraints": (
                    "" if index is None else index.body_of("Key Constraints")
                ),
                # Not floor fields: the confidence line and the synthesis are
                # the explorer's prose, and the driver has neither. Reported
                # honestly as absent rather than invented.
                "exploration-confidence": "",
                "synthesis": f"{found} of {self.findings_threshold} findings on disk.",
            },
        )

    def _emit_plan(self, context: Mapping[str, Any]) -> None:
        """PC-PLAN: the 11 plan sections, verbatim, plus the approval prompt."""
        directory = self._plan_directory(context)
        plan = (
            None
            if directory is None
            else self._artifact(directory, ArtifactNames.PLAN, PlanDoc)
        )
        fields = {
            _contract_field(section): (
                "" if plan is None else _excerpt(plan.body_of(section))
            )
            for section in PlanSchema.SECTIONS
        }
        fields["approval-prompt"] = (
            f"Approve this plan for iteration "
            f"{as_int(context.get(ContextKeys.ITERATION), 0) + 1}?"
        )
        self._emit_contract("PC-PLAN", fields)

    def _emit_execute_step(self, context: Mapping[str, Any]) -> None:
        """PC-EXECUTE-STEP: what the finished step actually changed, from disk."""
        directory = self._plan_directory(context)
        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        iteration = as_int(context.get(ContextKeys.ITERATION), 0)
        files, commits = self._changelog_digest(directory, iteration, step)
        self._emit_contract(
            "PC-EXECUTE-STEP",
            {
                "step": f"iter-{iteration}/step-{step}: {self._step_intent(directory, step)}",
                "files": ", ".join(files),
                "commit": ", ".join(commits),
                "surprises": self._last_answer(context),
                "next-preview": self._step_intent(directory, step + 1),
            },
        )

    def _emit_execute_leash(
        self,
        context: Mapping[str, Any],
        directive: RevertDirective | None,
        *,
        step: int,
        attempts: int,
    ) -> None:
        """PC-EXECUTE-LEASH: the block that is emitted for this slug and no other."""
        directory = self._plan_directory(context)
        revert = "no workspace_root to revert" if directive is None else str(directive)
        self._emit_contract(
            "PC-EXECUTE-LEASH",
            {
                "step-intent": self._step_intent(directory, step),
                "attempts": self._attempt_digest(context, step, attempts),
                # "no worker answer was recorded" IS the root-cause information
                # available, so it fills the field rather than leaving a gap.
                "root-cause-guess": (
                    self._last_answer(context) or "no worker answer recorded"
                ),
                "checkpoints": self._checkpoint_digest(directory),
                "prompt": (
                    f"The {self.max_fix_attempts}-attempt autonomy leash is spent "
                    f"on step {step}. Revert directive: {revert}. Continue with "
                    "direction, pivot to a different approach, or stop here?"
                ),
            },
        )

    def _emit_reflect(self, context: Mapping[str, Any]) -> None:
        """PC-REFLECT: progress and verification, verbatim from the two artifacts."""
        directory = self._plan_directory(context)
        progress = (
            None
            if directory is None
            else self._artifact(directory, ArtifactNames.PROGRESS, ProgressDoc)
        )
        verification = (
            None
            if directory is None
            else self._artifact(directory, ArtifactNames.VERIFICATION, VerificationDoc)
        )
        recommendation = (
            ""
            if verification is None
            else dict(verification.verdict_bullets()).get("Recommendation", "")
        )
        self._emit_contract(
            "PC-REFLECT",
            {
                "completed": (
                    "" if progress is None else _excerpt(progress.body_of("Completed"))
                ),
                "remaining": (
                    "" if progress is None else _excerpt(progress.body_of("Remaining"))
                ),
                "verification-results": (
                    ""
                    if verification is None
                    else _excerpt(verification.body_of("Criteria Verification"))
                ),
                "issues": (
                    ""
                    if verification is None
                    else _excerpt(verification.body_of("Not Verified"))
                ),
                "recommendation": recommendation
                or (
                    "PASS"
                    if context.get(ContextKeys.ALL_CRITERIA_PASS) is True
                    else "CONTINUE"
                ),
            },
        )

    # -- contract field sources -----------------------------------------

    def _step_intent(self, directory: PlanDirectory | None, step: int) -> str:
        """``plan.md``'s own text for *step*, verbatim, or an honest fallback."""
        plan = (
            None
            if directory is None
            else self._artifact(directory, ArtifactNames.PLAN, PlanDoc)
        )
        if plan is not None:
            for parsed in plan.steps():
                if parsed.number == str(step):
                    return _excerpt(parsed.text)
        return f"plan step {step} (no readable {ArtifactNames.PLAN} entry)"

    def _changelog_digest(
        self, directory: PlanDirectory | None, iteration: int, step: int
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The files and commits ``changelog.md`` records for one step.

        Read off the EXECUTOR's own ledger rather than asked of the worker:
        the same evidence-over-testimony split as ``_derive_gate_counts``
        (D-032).  A worker that reports a commit it did not make cannot make
        this field say so.
        """
        wanted = f"iter-{iteration}/step-{step}"
        files: list[str] = []
        commits: list[str] = []
        if directory is None:
            return (), ()
        try:
            text = directory.read_text(ArtifactNames.CHANGELOG)
        except (HarnessError, OSError):
            return (), ()
        for line in text.splitlines():
            try:
                entry = parse_changelog_line(line)
            except HarnessError:
                continue  # header, prose, or a malformed row: audit's business
            if entry.step != wanted:
                continue
            if entry.path not in files:
                files.append(entry.path)
            if entry.commit not in commits and entry.commit != "uncommitted":
                commits.append(entry.commit)
        return tuple(files), tuple(commits)

    def _checkpoint_digest(self, directory: PlanDirectory | None) -> str:
        """Every ``checkpoints/cp-NNN-iterN.md`` on disk, as one line.

        "The directory is readable and holds none" is an ANSWER and fills the
        field; "there is no directory to read" is a gap and leaves it empty, so
        it shows up in ``Presentation.missing_floor`` instead of being papered
        over with the same words.
        """
        if directory is None:
            return ""
        try:
            names = [
                name
                for name in directory.list_dir(ArtifactNames.CHECKPOINTS_DIR)
                if name.endswith(".md")
            ]
        except (HarnessError, OSError):
            return ""
        return ", ".join(names) if names else "none on disk"

    @staticmethod
    def _attempt_digest(context: Mapping[str, Any], step: int, attempts: int) -> str:
        """Both failed attempts on *step*, from the driver's own role ledger."""
        history = context.get(ContextKeys.ROLE_RESULTS)
        rows = [
            entry
            for entry in (history if isinstance(history, list) else [])
            if isinstance(entry, dict)
            and entry.get("state") == HarnessStates.EXECUTE
            and as_int(entry.get("step_number"), -1) == step
        ]
        if not rows:
            return f"{attempts} attempt(s) spent; no dispatch recorded on this step."
        return "; ".join(
            f"{index}) {'PASS' if row.get('success') else 'FAIL'}: "
            f"{_excerpt(str(row.get('answer', '')), 200)}"
            for index, row in enumerate(rows, start=1)
        )

    @staticmethod
    def _last_answer(context: Mapping[str, Any]) -> str:
        """The most recent dispatch's answer -- the driver's only root-cause data."""
        entry = context.get(ContextKeys.CURRENT_ROLE_RESULT)
        if not isinstance(entry, dict):
            return ""
        return _excerpt(str(entry.get("answer", "")))

    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        """Dispatch for a HELD state, and enforce the run-ending halts.

        ``on_state_entry`` never fires while the FSM holds a state (measured;
        see the module docstring), so without this hook a multi-step EXECUTE
        would dispatch step 1 and then sit there.  It shares the ledger with
        the entry handlers, so a state that was just entered is a no-op here.
        """
        self._api = api
        self._conversation_id = conv_id

        state = api.get_current_state(conv_id)
        context = api.get_data(conv_id)

        self._check_iteration_cap(state, context)

        updates = self._dispatch_if_needed(state, context)
        if updates:
            api.update_context(conv_id, self._writable_updates(updates))
            context = api.get_data(conv_id)
            # The step cursor and the leash counter move on this path without a
            # transition, so state.md is re-synced here -- AFTER the gate that
            # ran inside the dispatch, never before it (D-038).
            self._sync_state_doc(state, context)

        self._raise_pending_halt()
        self._check_stall(state, context, bool(updates))

    def _raise_pending_halt(self) -> None:
        """Raise a halt a HANDLER asked for, now that we are outside one.

        Raises:
            _HarnessHalt: If a handler recorded one (currently only the
                ``iteration-cap`` pre-step-gate action).
        """
        halt, self._halt_request = self._halt_request, None
        if halt is not None:
            raise halt

    def _check_iteration_cap(self, state: str, context: Mapping[str, Any]) -> None:
        """Halt when PLAN can no longer reach EXECUTE.

        Raises:
            _HarnessHalt: With the ``iteration-cap`` slug.
        """
        if state != HarnessStates.PLAN:
            return
        iteration = as_int(context.get(ContextKeys.ITERATION), 0)
        if iteration < self.iteration_hard_cap:
            return
        # DECISION plan-2026-07-21T125237-191b2eb2/D-019
        # An iteration-cap halt emits NO leash presentation and NO leash slug.
        # Do NOT merge this path with the leash halt "because both stop the run":
        # a run can hit the iteration cap with ZERO fix attempts, and reporting a
        # leash there would tell the user to look at a failing step that does not
        # exist. The two halts differ in slug, in whether a leash block is shown,
        # and in whether the run continues (leash routes to REFLECT; the iteration
        # cap ends the run). See decisions.md D-019.
        raise _HarnessHalt(
            reason=(
                f"Iteration cap: {iteration} iterations used (cap "
                f"{self.iteration_hard_cap}); PLAN cannot re-enter EXECUTE. "
                "Decompose the goal into a fresh plan."
            ),
            slug=GateSlug.ITERATION_CAP,
            context=dict(context),
        )

    def _check_stall(
        self, state: str, context: Mapping[str, Any], dispatched: bool
    ) -> None:
        """Halt when the protocol has made no progress for too many turns.

        Raises:
            _HarnessHalt: With no slug -- a stall is not one of the four
                pre-step-gate failures.
        """
        signature = (
            state,
            as_int(context.get(ContextKeys.ITERATION), 0),
            as_int(context.get(ContextKeys.STEP_NUMBER), 0),
            len(self._read_ledger(context)),
        )
        if dispatched or signature != self._stall_signature:
            self._stall_signature = signature
            self._stall_turns = 0
            return

        self._stall_turns += 1
        if self._stall_turns < self.max_stall_turns:
            return

        stalled = (
            f"Stalled in {state.upper()} for {self._stall_turns} turns with no "
            f"progress. Gate: {get_rules(state).gate_summary}"
        )
        # A halt reason already in context is the REASON the gate is shut (most
        # often the leash cap). Prepend it rather than overwrite: the stall is
        # the symptom, the recorded reason is the diagnosis.
        prior = context.get(ContextKeys.HALT_REASON)
        reason = f"{prior} {stalled}" if isinstance(prior, str) and prior else stalled

        raise _HarnessHalt(reason=reason, slug=None, context=dict(context))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    # DECISION plan-2026-07-21T125237-191b2eb2/D-053
    # ONE deletion convention, realised the same way by both call sites of
    # `_dispatch_if_needed`. A `None` in a driver delta means "this key returns
    # to its DRIVER-OWNED DEFAULT", never "write a literal None" and never
    # "silently do nothing":
    #   * for a `DRIVER_OWNED_SEEDS` key, `_apply` coerces it to the seed
    #     BEFORE it reaches either call site, so no `None` for a seeded key
    #     ever gets here (D-044);
    #   * for a `DRIVER_OWNED_UNSET` key (`halt_reason`, `last_gate_slug`,
    #     `pivot_reason`, the two roots), absent IS the default. `_apply`
    #     records it as absent in `_driver_owned`, so the entry-handler path
    #     deletes it immediately (a handler delta's `None` means delete) and
    #     this path -- `API.update_context`, which has no delete -- has the
    #     PRE_PROCESSING half of `_reassert_driver_owned` delete it before the
    #     next extraction, i.e. strictly before any transition can read it.
    # Do NOT "restore" the old bare `_drop_deletions` filter: it discarded a
    # `None` for ANY key with no record that it had done so, which made a
    # "clear this flag" instruction a silent no-op on the loop path only
    # (review W3) -- and the only reason that was latent rather than live is
    # that D-044 had just converted every gate-flag clear to `False`.
    def _writable_updates(self, updates: Mapping[str, Any]) -> dict[str, Any]:
        """Project a driver delta onto what ``API.update_context`` can express.

        ``update_context`` merges; it cannot delete, and it writes ``None``
        verbatim -- and a key present with value ``None`` still satisfies
        ``TransitionCondition.requires_context_keys``, so passing one through
        would turn a cleared flag into a present-but-null one and change what
        the gate sees.

        Args:
            updates: A delta that has already been through :meth:`_apply`.

        Returns:
            The same delta without its deletions.
        """
        writable: dict[str, Any] = {}
        for key, value in updates.items():
            if value is not None:
                writable[key] = value
            elif key not in self._driver_owned:
                logger.warning(
                    f"Delta clears non-driver-owned key '{key}', which this "
                    "path cannot delete; leaving the stale value in place."
                )
        return writable

    def _apply(
        self,
        updates: dict[str, Any],
        working: dict[str, Any],
        delta: Mapping[str, Any],
    ) -> None:
        """Record *delta* in the returned update set and in the working view.

        A ``None`` value means "delete this key" in a handler delta
        (``pipeline.execute_handlers``'s merge contract), so the working view
        drops it rather than storing a literal ``None``.

        This is also the single choke point where the driver records what it
        believes a driver-owned key holds (:attr:`_driver_owned`), which is what
        lets :meth:`_reassert_driver_owned` tell a driver write from an LLM one.
        Every delta the driver produces -- entry bookkeeping, the worker
        allowlist, post-dispatch bookkeeping, the pre-step gate -- passes
        through here.  A new delta site that bypasses ``_apply`` would have its
        gate-flag writes silently reverted on the next extraction turn.
        """
        for key, raw in delta.items():
            # DECISION plan-2026-07-21T125237-191b2eb2/D-044
            # A seeded driver-owned key can never be DELETED, only reset to its
            # falsy seed. Do NOT remove this coercion as "defensive": a `None`
            # delta means "delete" to the pipeline's merge contract, an ABSENT
            # key is one the FSM's Pass-1 extraction is asked to invent, and an
            # invented value reaches the transition evaluator through
            # `extraction_response.extracted_data` -- a channel no handler at any
            # timing can clean (transition_evaluator.py:146). Reproduced: with
            # the pre-7a `plan_approved: None` clear on PLAN entry restored, a
            # fabricating LLM opened PLAN -> EXECUTE even with the guard handler
            # registered. This turns "never clear a gate flag with None" from a
            # rule a future edit can forget into a mechanism.
            # See decisions.md D-044.
            value = (
                DRIVER_OWNED_SEEDS[key]
                if raw is None and key in DRIVER_OWNED_SEEDS
                else raw
            )
            updates[key] = value
            if key in self._driver_owned:
                self._driver_owned[key] = value
            if value is None:
                working.pop(key, None)
            else:
                working[key] = value


#: One pre-step-gate slug's action: it may write context, emit a presentation,
#: compute a revert and record a halt -- but it never dispatches and never
#: raises, because it runs inside an FSM handler.
_GateAction = Callable[[HarnessAgent, GateResult, Mapping[str, Any]], "dict[str, Any]"]

#: Slug -> action.  Complete over ``GateSlug.ORDER`` and injective: four slugs,
#: four distinct functions (D-040).  A test asserts both.
_GATE_ACTIONS: Mapping[str, _GateAction] = MappingProxyType(
    {
        GateSlug.NO_PLAN: HarnessAgent._act_no_plan,
        GateSlug.WRONG_STATE: HarnessAgent._act_wrong_state,
        GateSlug.LEASH_CAP: HarnessAgent._act_leash_cap,
        GateSlug.ITERATION_CAP: HarnessAgent._act_iteration_cap,
    }
)


def _contract_field(section: str) -> str:
    """``PlanSchema`` heading -> the PC-PLAN field name for it.

    ``"Pre-Mortem & Falsification Signals"`` -> ``"pre-mortem"``: the contract
    names the field by its first word, the plan names the section in full.
    Derived rather than tabulated so the 11 sections stay defined once.
    """
    head = section.split("&")[0].strip().lower()
    return "-".join(head.replace("-", " ").split())


def _excerpt(text: str, limit: int = _CONTRACT_FIELD_CHARS) -> str:
    """One artifact excerpt, whitespace-folded and bounded."""
    folded = " ".join(str(text).split())
    return folded if len(folded) <= limit else f"{folded[: limit - 1]}…"


def _deny_approval(request: Any) -> bool:
    """Default approval callback: deny everything."""
    logger.info(
        f"No approval callback configured; DENYING "
        f"{getattr(request, 'tool_name', 'approval')}"
    )
    return False


def _as_optional_str(value: Any) -> str | None:
    """Return a non-empty ``str`` for a filesystem root, else ``None``.

    Only a real ``str`` is accepted.  A root is a path a role's write tools get
    confined to, so "something that stringifies" is not good enough: ``None``
    means "no plan-directory tools at all", which is the safe reading.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _exactly(current: Any, owned: Any) -> bool:
    """True when *current* is *owned*'s exact type AND equal to it.

    ``type(...) is type(...)`` rather than ``==`` alone because Python's numeric
    tower makes ``False == 0`` and ``True == 1``: a fabricated
    ``findings_count = True`` would compare equal to the driver's seeded ``0``
    and slip past the revert.

    Deliberately NOT moved to ``hardening.py`` alongside
    :func:`~fsm_llm_harness.hardening.type_matches` (D-059): it is a
    two-operand EQUALITY predicate for the driver-ownership revert, not a
    one-operand type predicate for a worker reply, it has one call site, and
    its ``type(x) is type(y)`` rule is deliberately stricter than
    ``type_matches``'s ``isinstance`` (which accepts subclasses).  Expressing
    it as ``type_matches(current, type(owned)) and current == owned`` would
    silently loosen it for any subclass.
    """
    return type(current) is type(owned) and bool(current == owned)
