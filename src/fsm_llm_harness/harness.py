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

**No artifact I/O, with ONE read-only exception.**  This module keeps all
protocol memory in the FSM context, and it never WRITES a plan directory;
``storage.py`` / ``plan_validator.py`` are wired in by a later step.  The
exception is :meth:`HarnessAgent._derive_gate_counts`, which COUNTS the files
behind a disk-derived gate key after a dispatch (D-032).  A gate value the
driver reads off the filesystem itself is evidence; one a worker reports is
testimony, and the two are handled differently on purpose.

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

from .constants import (
    DRIVER_OWNED_SEEDS,
    DRIVER_OWNED_UNSET,
    ContextKeys,
    Defaults,
    GateSlug,
    HandlerNames,
    HandlerPriorities,
    HarnessStates,
)
from .exceptions import HarnessError, HarnessReentrancyError
from .fsm_definition import build_harness_fsm
from .hardening import as_int, coerce_worker_output
from .rules import ROLE_BY_STATE, get_rules
from .tools import DISK_DERIVED_COUNTS, PlanMemory, derive_disk_counts

__all__ = [
    "HarnessAgent",
    "RoleRequest",
    "WorkerFactory",
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


#: A role worker: one :class:`RoleRequest` in, one ``AgentResult`` out.
#:
#: Deliberately WIDER than ``OrchestratorAgent``'s ``Callable[[str],
#: AgentResult]`` (``orchestrator.py:54``): that seam dispatches homogeneous
#: subtasks, while every harness dispatch is role- and state-specific and needs
#: the protocol counters to do its job at all (an executor cannot know which
#: step to run, or that it is on its second attempt, from a string).
WorkerFactory = Callable[[RoleRequest], AgentResult]


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
        config: AgentConfig | None = None,
        findings_threshold: int = Defaults.FINDINGS_THRESHOLD,
        max_fix_attempts: int = Defaults.MAX_FIX_ATTEMPTS,
        max_leash_grants: int = Defaults.MAX_LEASH_GRANTS,
        iteration_hard_cap: int = Defaults.ITERATION_HARD_CAP,
        max_explore_redispatches: int = Defaults.MAX_EXPLORE_REDISPATCHES,
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
        seeds: dict[str, Any] = {
            ContextKeys.GOAL: task,
            **DRIVER_OWNED_SEEDS,
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
            .do(self._make_entry_handler(HarnessStates.INITIAL))
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
        plan_dir = _as_optional_str(context.get(ContextKeys.PLAN_DIR))
        try:
            # A directory that is not there yet is not evidence, and this check
            # is also what keeps the read a READ: `PlanMemory.__init__` CREATES
            # its plan directory, and the driver has no business creating
            # protocol memory as a side effect of counting it.
            if plan_dir is None or not Path(plan_dir).expanduser().is_dir():
                return {}
            memory = PlanMemory(plan_dir, role=ROLE_BY_STATE[state])
            return dict(derive_disk_counts(memory, allowed))
        except Exception as exc:  # unreadable root: no evidence, not zero
            logger.warning(
                f"Could not derive gate counts for state '{state}' from "
                f"'{plan_dir}': {exc}. Leaving the gate value unchanged."
            )
            return {}

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
            return self._after_plan_dispatch(result, context)
        if state == HarnessStates.REFLECT:
            return self._after_reflect_dispatch(context)
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

    def _after_plan_dispatch(
        self, result: AgentResult | None, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Ask the human to approve the plan."""
        if result is None or not result.success:
            return {}
        if context.get(ContextKeys.NEEDS_EXPLORE) is True:
            # The plan-writer bounced back to EXPLORE; there is no plan to approve.
            return {}
        approved = self._request_approval(
            _APPROVAL_PLAN,
            context,
            reasoning=(
                f"Approve the plan for iteration "
                f"{as_int(context.get(ContextKeys.ITERATION), 0) + 1}?"
            ),
        )
        return {ContextKeys.PLAN_APPROVED: approved}

    def _after_reflect_dispatch(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Ask the human to confirm a close, or to continue past the leash."""
        if context.get(ContextKeys.LAST_GATE_SLUG) == GateSlug.LEASH_CAP:
            return self._offer_leash_continue(context)

        if context.get(ContextKeys.ALL_CRITERIA_PASS) is not True:
            return {}

        confirmed = self._request_approval(
            _APPROVAL_CLOSE,
            context,
            reasoning="Every success criterion is verified PASS. Close the plan?",
        )
        return {ContextKeys.CLOSE_CONFIRMED: confirmed}

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
        """In-memory analogue of ``plan_validator.pre_step_gate()``.

        Only the ``leash-cap`` slug is decidable without a plan directory; the
        other three (``no-plan`` / ``wrong-state`` / ``iteration-cap``) need
        on-disk state or belong to a different edge, and a later step adds
        them.  Returning a delta rather than raising keeps the halt *inside*
        the protocol: the run routes to REFLECT and reports, it does not crash.

        Returns:
            A context delta when the step must not be dispatched, else ``None``.
        """
        attempts = as_int(context.get(ContextKeys.FIX_ATTEMPTS), 0)
        if attempts < self.max_fix_attempts:
            return None
        step = as_int(context.get(ContextKeys.STEP_NUMBER), 0)
        logger.warning(
            f"Pre-step gate [{GateSlug.LEASH_CAP}]: step {step} has {attempts} "
            f"fix attempts (cap {self.max_fix_attempts}); refusing to dispatch."
        )
        return {
            ContextKeys.EXECUTE_COMPLETE: True,
            ContextKeys.LAST_GATE_SLUG: GateSlug.LEASH_CAP,
            ContextKeys.HALT_REASON: (
                f"Autonomy leash: step {step} already used {attempts} fix "
                f"attempts (cap {self.max_fix_attempts})."
            ),
        }

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

        self._check_stall(state, context, bool(updates))

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
