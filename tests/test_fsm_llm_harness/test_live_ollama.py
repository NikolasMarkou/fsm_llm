"""LIVE end-to-end tests for ``fsm_llm_harness`` against a real small model.

These are the shipped form of plan.md's five live success criteria.  They are
DOUBLE-GATED and skip by default:

* ``FSM_LLM_HARNESS_LIVE=1`` must be set (``Defaults.ENV_LIVE_TESTS``), and
* Ollama must be up with ``qwen3.5:4b`` pulled.

The env gate is checked FIRST and the two are combined with ``or`` so the
network probe never runs in the default ``make test`` path -- a live suite that
costs a 3-second socket timeout per collection is a live suite people turn off.

What each test drives, and why
------------------------------
The five criteria are not the same KIND of claim, so they are not measured the
same way, and the split is deliberate rather than convenient:

* **L1 / L2 / L3** are claims about the HARNESS -- an audit verdict, a counter,
  an FSM edge.  The FSM itself runs live (every protocol turn is a real
  ``:4b`` completion through the real ``MessagePipeline``), but the role workers
  are SCRIPTED and cost zero LLM calls.  A scripted worker makes the claim
  falsifiable: "the leash stopped at exactly 2" is only meaningful when the
  executor is guaranteed to fail, and "the plan directory audits clean" is only
  meaningful when the artifacts on disk are real protocol documents.  Model
  unpredictability would add noise to a measurement that is not about the model.
* **L4 / L5** are claims about the MODEL -- will ``:4b`` call a write tool and
  produce the content it was asked for, and will a real explorer dispatch put
  three distinct findings files on disk.  Those run the real
  ``build_default_worker_factory`` and assert the EXISTENTIAL form the criteria
  are written in ("*a* role dispatch that ...", "*a* genuine findings_count").
  Each also reports its raw k/n in the failure message, so a partial result is
  legible instead of being flattened to red/green.

Nothing here weakens a gate to pass.  The disk-derived findings count, the
write-evidence check, ``success``, confinement and ownership are all the
shipped ones; the scripted workers write THROUGH ``PlanMemory``, so every
artifact they produce was authorised by ``rules.OWNERSHIP`` exactly as a live
role's would be.

The audit-clean plan-directory corpus is IMPORTED from
``test_plan_validator.py`` rather than re-typed.  That module's base fixture is
pinned to ``audit() == []`` by its own test, so importing it means this suite
cannot drift into auditing a document shape the validator suite has stopped
agreeing with -- and a corpus copied here would be exactly the hand-kept
duplicate ``plans/LESSONS.md`` warns about.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_llm, pytest.mark.slow]

from fsm_llm_agents.definitions import AgentResult, ApprovalRequest
from fsm_llm_agents.tools import ToolRegistry
from fsm_llm_harness.artifacts import PlanDoc, StateDoc
from fsm_llm_harness.constants import (
    ArtifactNames,
    ContextKeys,
    Defaults,
    GateSlug,
    HarnessStates,
    Severity,
)
from fsm_llm_harness.exceptions import HarnessArtifactError
from fsm_llm_harness.harness import (
    EXECUTE_TARGET_ASSIGNED,
    HarnessAgent,
    RoleRequest,
    derive_execute_target,
)
from fsm_llm_harness.plan_validator import Issue, audit
from fsm_llm_harness.roles import (
    build_default_worker_factory,
    build_role_system_prompt,
    build_role_task_prompt,
    get_role_spec,
    held_tools,
)
from fsm_llm_harness.rules import get_rules
from fsm_llm_harness.tools import PlanMemory, Workspace, gate_files
from tests.conftest import ollama_available
from tests.test_fsm_llm_harness.test_plan_validator import (
    PLAN_MD,
    STATE_MD,
    VERIFICATION_MD,
    make_plan_dir,
)

# ---------------------------------------------------------------------------
# Configuration and the double gate
# ---------------------------------------------------------------------------

MODEL = "ollama_chat/qwen3.5:4b"
MODEL_TAG = "qwen3.5:4b"

#: The env var that arms this file.  Read from ``Defaults`` rather than typed
#: as a literal so the package and its live suite name the same switch.
LIVE_ENV = Defaults.ENV_LIVE_TESTS


def _live_requested() -> bool:
    """Whether the caller explicitly armed the live suite."""
    return os.environ.get(LIVE_ENV) == "1"


def _ollama_available() -> bool:
    """Whether Ollama is up with the model these tests are written against."""
    return ollama_available(MODEL_TAG)


def _live_gate_closed() -> bool:
    """Whether the live suite is switched OFF right now.

    The env term is read FIRST and the two are combined with ``or``, which
    SHORT-CIRCUITS: with the env gate off, ``_ollama_available()`` is never
    called, so a default ``pytest`` run pays no socket timeout for this file.

    Named rather than inlined into the ``skipif`` so
    :class:`TestTheGateOpensAndClosesInBothDirections` can exercise the very
    expression the mark was built from instead of a re-typed copy of it.
    """
    return not _live_requested() or not _ollama_available()


requires_live = pytest.mark.skipif(
    _live_gate_closed(),
    reason=(
        f"live harness tests are opt-in: set {LIVE_ENV}=1 and run Ollama "
        f"with {MODEL_TAG} pulled"
    ),
)

#: Runs per HARNESS criterion (L1, L2, L3).  plan.md's Verification Strategy
#: asks for n>=3 on criteria 3 and 5 and states no n for criterion 4; a live
#: traverse costs 20-40 s, so this is the plan's number, not a shortcut.
RUNS = 3

#: Runs per MODEL criterion (L4, L5).  plan.md asks for **n>=5** on criteria 1
#: and 2 -- both rows read "Live, n>=5" -- so 3 would have under-sampled the two
#: measurements the plan says the most about.  A real explorer arm costs
#: 75-215 s, which is what makes this the expensive half of the file.
RUNS_MODEL = 5

#: The pass condition plan.md's Verification Strategy states for criteria 1 and
#: 2, transcribed rather than invented: ">=4/5 runs".  Kept as data next to
#: :data:`RUNS_MODEL` so the two cannot drift apart silently.
MODEL_BAR = 4

GOAL = "add a retry with exponential backoff to the uploader"

#: The workspace the live EXECUTE dispatch is pointed at.  Small, real, and
#: missing exactly the thing the goal asks for, so "did the model write what it
#: was asked to write" is decidable from the file's own text.
SEED_FILES: Mapping[str, str] = {
    "uploader.py": (
        "import requests\n\n\n"
        "def upload(path, url):\n"
        "    with open(path, 'rb') as fh:\n"
        "        return requests.post(url, data=fh.read())\n"
    ),
    "config.py": "TIMEOUT = 30\nRETRIES = 0\nENDPOINT = 'https://example.com/upload'\n",
    "README.md": "# uploader\n\nUploads files. No retry logic yet.\n",
}

#: Every tool name that puts bytes on disk, in either root.
WRITE_TOOLS = frozenset(
    {"write_file", "append_file", "write_plan_file", "append_plan_file"}
)

#: Tokens that make an edit to ``uploader.py`` recognisably THE asked-for edit.
#: Content, never ``stat``: two runs of the plan's step-5 bench "wrote" the file
#: by echoing its original text back, which a size comparison scores as done.
RETRY_TOKENS = ("retry", "retries", "backoff")

#: The plan L4's executor is dispatched against.  The imported corpus is a plan
#: for THIS repository, and an executor handed it is told to run a step that has
#: nothing to do with the seeded workspace -- which is not a measurement of the
#: model, it is a measurement of a broken bench (the same defect the plan's own
#: step-4b arm hit, recorded in state.md).  Same 11 sections, in the validator's
#: order; only the subject differs.
EXECUTE_PLAN_MD = """# Plan v1: add a retry with backoff to the uploader

## Goal
Make `upload()` in `uploader.py` survive a transient network failure.

## Problem Statement
`upload()` issues exactly one POST. Any transient failure loses the upload.

## Context
`config.py` already carries `TIMEOUT`, `RETRIES` and `ENDPOINT`. `README.md`
records that there is no retry logic yet.

## Files To Modify
| File | Change | Reason |
|---|---|---|
| `uploader.py` | wrap the POST in a retry loop | the goal |

## Steps
1. [ ] **Add a retry with exponential backoff to `upload()` in `uploader.py`.**
   Wrap the `requests.post` call in a loop of up to 3 attempts, sleeping 0.5s,
   1s then 2s between them, and re-raise the last error if every attempt fails.
   Write the edited file back with `write_file`. [RISK: low] [deps: none]

## Assumptions
- **A1.** `requests` raises on a transport failure rather than returning `None`.

## Failure Modes
| Dependency | Slow | Bad data | Down | Blast radius |
|---|---|---|---|---|
| the upload endpoint | retried | non-2xx body | every upload fails | `upload()` only |

## Pre-Mortem & Falsification Signals
1. **The backoff busy-waits.** → **STOP IF** the loop sleeps zero seconds.

## Success Criteria
1. `uploader.py` retries a failed POST up to 3 times with growing backoff.

## Verification Strategy
| # | Criterion | Method | Command | Pass condition |
|---|---|---|---|---|
| 1 | Retry present | Manual | read `uploader.py` | a bounded loop with sleeps |

## Complexity Budget
| Metric | Budget | Notes |
|---|---|---|
| Files added | 0/0 | the change is in place |
"""

#: ``state.md`` for the same dispatch, DERIVED from the imported corpus rather
#: than re-typed, so the protocol grammar stays owned by one module.  Only the
#: step line moves: the corpus points at step 9 of a plan for THIS repository,
#: and :data:`EXECUTE_PLAN_MD` has exactly one step.
#:
#: This file's first live L4 run is why it exists.  With ``state.md`` DELETED --
#: which is right for the traverse tests, where the driver must write it -- the
#: executor spent all 17 of its tool calls re-reading a ``state.md`` that could
#: not exist (measured 3/3 native runs, 0/3 write tools issued, 0/3 bytes).  The
#: executor's own Pre-Step Checklist tells it to read that file FIRST, so a
#: bench that withholds it measures the bench, not the model -- the identical
#: defect state.md records against the plan's step-4b EXECUTE arm, one artifact
#: over.  Supplying it is not a weakened assertion: every assertion below is
#: unchanged, and the ENOENT-loop shape stays measurable because
#: ``test_the_L4_bench_hands_the_executor_a_readable_plan_directory`` pins the
#: four artifacts the checklist names as present on disk.
# DECISION plan-2026-07-21T191807-bf7ffe24/D-048
# Do NOT reach for `_fresh_plan_dir` here, and do NOT delete `state.md` to
# "match" the traverse tests: measured 0/3 write tools issued when it was
# absent, 3/5 when it was present.
EXECUTE_STATE_MD = STATE_MD.replace(
    "## Current Plan Step: 9 (plan_validator.py)",
    "## Current Plan Step: 1 (add a retry with backoff to upload())",
)

#: The Pre-Step Checklist artifacts an executor is told to read before editing.
PRE_STEP_READS = (
    ArtifactNames.STATE,
    ArtifactNames.PLAN,
    ArtifactNames.PROGRESS,
    ArtifactNames.DECISIONS,
)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _digests(root: Path) -> dict[str, str]:
    """``relative path -> sha256`` for every file under *root*."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _seed_workspace(tmp_path: Path) -> Path:
    """A workspace root holding :data:`SEED_FILES`."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, body in SEED_FILES.items():
        (workspace / name).write_text(body, encoding="utf-8")
    return workspace


def _findings_on_disk(plan_dir: Path) -> tuple[str, ...]:
    """The DISTINCT non-empty ``findings/*.md`` files, counted as the gate does.

    Read through ``tools.gate_files`` -- the shipped derivation -- rather than a
    local ``glob``.  A second count here would be a test and a gate that can
    disagree, which is the fail-open shape decisions.md D-015 closed.
    """
    memory = PlanMemory(plan_dir, role=ROLE_FOR_READING)
    return gate_files(memory, ArtifactNames.FINDINGS_DIR)


#: Any role may READ the plan directory; ownership only constrains writes.  The
#: explorer is used because it is the role that owns ``findings/``.
ROLE_FOR_READING = "ip-explorer"


# ---------------------------------------------------------------------------
# The scripted, zero-LLM worker used by L1 / L2 / L3
# ---------------------------------------------------------------------------


def _topic_body(slug: str) -> str:
    """A findings topic file in the protocol's own 5-section schema."""
    return (
        f"# Finding: {slug.replace('-', ' ')}\n\n"
        "## Summary\n"
        f"What `{GOAL}` requires, read off the seeded workspace.\n\n"
        "## Key Findings\n"
        "- `uploader.py` is the only module the goal names.\n"
        "- `config.py` already carries a `RETRIES` constant, currently `0`.\n\n"
        "## Constraints\n"
        "- **HARD** every write is confined to the workspace root.\n\n"
        "## Code Patterns\n"
        "- `[REUSE]` the endpoint and timeout live in `config.py`; do not inline them.\n\n"
        "## Risks & Unknowns\n"
        "- A small model may answer without calling a write tool at all.\n"
    )


def _changelog_line(request: RoleRequest) -> str:
    """One append-only ledger line in the protocol's 8-field pipe format."""
    return (
        "2026-07-22T12:00:00Z | "
        f"iter-{request.iteration}/step-{request.step_number} | uncommitted | "
        "uploader.py | EDIT(+12,-2) | radius:LOW(1) | - | add retry with backoff\n"
    )


class ScriptedRoles:
    """A ``WorkerFactory`` that writes REAL artifacts and costs zero LLM calls.

    Interface contract (3 call sites -- L1, L2 and L3):

    * ``execute_succeeds`` -- ``False`` makes every executor dispatch fail, which
      is what the autonomy leash is measured against.
    * ``pivot_first_reflect`` -- ``True`` makes the FIRST verifier verdict a
      PIVOT and every later one a pass, which is what the loop-back is measured
      against.
    * Every plan-directory write goes through a role-scoped
      :class:`~fsm_llm_harness.tools.PlanMemory`, so ``rules.OWNERSHIP`` refuses
      a write this class has no business making -- the artifacts are produced
      under the shipped authorisation path, not around it.
    * ``requests`` holds every :class:`RoleRequest` in dispatch order;
      :attr:`fix_attempts_for` projects one state's attempt counters, which is
      how "exactly 2, never 3" is asserted.
    """

    def __init__(
        self,
        *,
        execute_succeeds: bool = True,
        pivot_first_reflect: bool = False,
        total_steps: int = 2,
    ) -> None:
        self.execute_succeeds = execute_succeeds
        self.pivot_first_reflect = pivot_first_reflect
        self.total_steps = total_steps
        self.requests: list[RoleRequest] = []

    # -- projections -----------------------------------------------------

    @property
    def states(self) -> list[str]:
        return [request.state for request in self.requests]

    def count_for(self, state: str) -> int:
        return self.states.count(state)

    def fix_attempts_for(self, state: str) -> list[int]:
        return [r.fix_attempts for r in self.requests if r.state == state]

    # -- the dispatch ----------------------------------------------------

    def __call__(self, request: RoleRequest) -> AgentResult:
        self.requests.append(request)
        memory = (
            None
            if request.plan_dir is None
            else PlanMemory(request.plan_dir, role=request.role)
        )
        context: dict[str, Any] = {}
        success = True

        if request.state == HarnessStates.EXPLORE:
            if memory is not None and request.assigned_topic:
                memory.write_text(
                    f"{ArtifactNames.FINDINGS_DIR}/{request.assigned_topic}.md",
                    _topic_body(request.assigned_topic),
                )
            # Nothing is CLAIMED: `findings_count` is derived from the files the
            # write above really produced (D-015), and this worker deliberately
            # exercises that path rather than reporting an integer.
        elif request.state == HarnessStates.PLAN:
            if memory is not None:
                memory.write_text(ArtifactNames.PLAN, PLAN_MD)
                memory.write_text(ArtifactNames.VERIFICATION, VERIFICATION_MD)
            context = {ContextKeys.TOTAL_STEPS: self.total_steps}
        elif request.state == HarnessStates.EXECUTE:
            success = self.execute_succeeds
            if memory is not None and success:
                memory.append_text(ArtifactNames.CHANGELOG, _changelog_line(request))
        elif request.state == HarnessStates.REFLECT:
            first = self.count_for(HarnessStates.REFLECT) == 1
            if self.pivot_first_reflect and first:
                context = {ContextKeys.NEEDS_PIVOT: True}
            elif not self.execute_succeeds:
                # Write NO routing flag.  The driver's own bookkeeping is what
                # must choose between a completion fix and a leash halt, and a
                # verifier that volunteered `all_criteria_pass` here would route
                # a failing run straight to CLOSE -- i.e. it would decide the
                # thing the leash test exists to measure.
                context = {}
            else:
                context = {
                    ContextKeys.ALL_CRITERIA_PASS: True,
                    ContextKeys.CRITERIA_PASS_COUNT: 2,
                    ContextKeys.CRITERIA_TOTAL: 2,
                }
        elif request.state == HarnessStates.PIVOT:
            context = {
                ContextKeys.PIVOT_RESOLVED: True,
                ContextKeys.PIVOT_REASON: "direction B: finish the remaining tier",
            }
        elif request.state == HarnessStates.CLOSE:
            context = {ContextKeys.HALT_REASON: "plan closed"}

        return AgentResult(
            answer=f"{request.role} completed {request.state}",
            success=success,
            final_context=context,
        )


class Approvals:
    """An approval callback with a per-gate verdict table and a record."""

    def __init__(self, verdicts: Mapping[str, bool] | None = None) -> None:
        self.verdicts = dict(verdicts or {})
        self.requests: list[ApprovalRequest] = []

    def count(self, gate: str) -> int:
        return sum(1 for r in self.requests if r.tool_name == gate)

    def __call__(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return bool(self.verdicts.get(request.tool_name, True))


def _live_agent(
    worker: Any,
    *,
    approvals: Any,
    **kwargs: Any,
) -> HarnessAgent:
    """A ``HarnessAgent`` whose FSM turns are real ``:4b`` completions.

    The model is NOT passed through here.  ``BaseAgent._create_api`` supplies
    ``model``/``temperature``/``max_tokens`` from ``self.config``, so passing
    them as ``api_kwargs`` raises ``TypeError`` for a duplicate keyword; the
    driver's own default profile already names :data:`MODEL`, and
    ``test_the_live_model_is_the_package_default`` pins that so this comment
    cannot go stale silently.
    """
    return HarnessAgent(
        worker_factory=worker,
        approval_callback=approvals,
        **kwargs,
    )


def _fresh_plan_dir(tmp_path: Path, run: int) -> Path:
    """An audit-clean plan directory with NO ``state.md``.

    Deleting ``state.md`` is what makes the traverse a traverse: the driver
    RESUMES from that file when it is present, and the imported corpus is
    pinned at ``# Current State: EXECUTE``, so keeping it would start the run
    three states in.  The driver writes its own on the first state entry.
    ``findings/`` is emptied for the same reason -- a topic file already on disk
    counts toward the gate, and the explorer must be the thing that opens it.
    """
    root = tmp_path / f"run-{run}"
    return make_plan_dir(root, state_md=None, findings_topic_md=None)


def _roots(plan_dir: Path, workspace: Path) -> dict[str, str]:
    return {
        ContextKeys.PLAN_DIR: str(plan_dir),
        ContextKeys.WORKSPACE_ROOT: str(workspace),
    }


def _errors(issues: tuple[Issue, ...] | list[Issue] | None) -> list[Issue]:
    return [i for i in (issues or ()) if i.severity == Severity.ERROR]


def _report(label: str, rows: list[dict[str, Any]]) -> None:
    """Print one line per live run, whatever the verdict.

    Interface contract (2 call sites, L4 and L5): writes to stdout, asserts
    nothing, returns nothing.  A criterion whose bar is EXISTENTIAL is only
    honest if its raw k/n is legible on a PASS as well as on a failure, and
    pytest shows this under ``-s``.  It is a report, never a gate.
    """
    print(f"\n[{label}] n={len(rows)}")
    for row in rows:
        print(f"  {row}")


# ---------------------------------------------------------------------------
# The assertions that are NOT gated: the file's own premise, and its gate
# ---------------------------------------------------------------------------


def test_the_live_model_is_the_package_default() -> None:
    """The model this file probes for is the model the driver will actually use.

    Deliberately UNGATED and offline.  ``_live_agent`` cannot pass a model
    through ``api_kwargs``, so the driver runs on ``Defaults.MODEL``; if that
    ever diverged from :data:`MODEL` the gate would probe for one model and the
    tests would exercise another, and every live result would silently be about
    something else.
    """
    assert Defaults.MODEL == MODEL
    assert LIVE_ENV == "FSM_LLM_HARNESS_LIVE"


class TestTheGateOpensAndClosesInBothDirections:
    """A live suite that skips no matter what is worth exactly nothing.

    Deliberately UNGATED and offline.  ``requires_live`` is evaluated ONCE, at
    import, so no test can watch the mark flip -- but the silent-failure mode
    does not live in the mark, it lives in the PREDICATE the mark was built
    from.  A ``skipif`` whose condition became constantly ``True`` (an inverted
    term, a renamed env var, a probe that started returning ``False``) would
    report "14 skipped" forever and read exactly like a healthy default run.

    So the 2x2 truth table is asserted here, with the daemon probe stubbed:
    the gate must OPEN when both terms hold and CLOSE when either does not.
    The one direction a stub cannot prove -- that an OPEN gate really reaches
    Ollama -- is what the k/n rows in L4 and L5 report.
    """

    @pytest.mark.parametrize(
        ("armed", "daemon_up", "expected_closed"),
        [
            pytest.param("1", True, False, id="armed+daemon-up=OPEN"),
            pytest.param("1", False, True, id="armed+daemon-down=closed"),
            pytest.param("0", True, True, id="disarmed+daemon-up=closed"),
            pytest.param("true", True, True, id="not-exactly-1=closed"),
            pytest.param(None, True, True, id="unset+daemon-up=closed"),
        ],
    )
    def test_the_gate_is_open_only_when_both_terms_hold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        armed: str | None,
        daemon_up: bool,
        expected_closed: bool,
    ) -> None:
        if armed is None:
            monkeypatch.delenv(LIVE_ENV, raising=False)
        else:
            monkeypatch.setenv(LIVE_ENV, armed)
        monkeypatch.setattr(f"{__name__}._ollama_available", lambda: daemon_up)
        assert _live_gate_closed() is expected_closed

    def test_the_daemon_is_never_probed_while_the_env_gate_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ordering that keeps `make test` free of a 3-second socket wait."""
        probes: list[int] = []

        def _probe() -> bool:
            probes.append(1)
            return True

        monkeypatch.setattr(f"{__name__}._ollama_available", _probe)
        monkeypatch.delenv(LIVE_ENV, raising=False)

        assert _live_gate_closed() is True
        assert probes == [], "the default `make test` path paid for a socket probe"

    def test_the_probe_is_the_SHARED_helper_asked_for_this_file_s_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No third daemon probe: this file delegates to ``tests/conftest.py``.

        ``tests/test_integration_ollama.py`` and this suite must agree about
        which daemon and which tag "available" means; a hand-rolled copy here
        is the duplication `plans/LESSONS.md` warns about, and it would drift
        silently because both copies would keep returning ``False`` off-line.
        """
        asked: list[str] = []
        monkeypatch.setattr(
            f"{__name__}.ollama_available",
            lambda tag: asked.append(tag) or True,
        )
        assert _ollama_available() is True
        assert asked == [MODEL_TAG]
        assert MODEL.endswith(MODEL_TAG)


# ---------------------------------------------------------------------------
# L1 -- criterion 3: a full traverse whose plan directory audits clean
# ---------------------------------------------------------------------------


@requires_live
class TestL1FullTraverseAuditsClean:
    """EXPLORE -> PLAN -> EXECUTE -> REFLECT -> CLOSE, then ``audit()``."""

    @pytest.mark.parametrize("run", range(1, RUNS + 1))
    def test_the_traverse_completes_and_the_audit_reports_no_errors(
        self, tmp_path: Path, run: int
    ) -> None:
        plan_dir = _fresh_plan_dir(tmp_path, run)
        workspace = _seed_workspace(tmp_path / f"run-{run}")

        # The BEFORE audit, pinned exactly.  It is NOT clean, and that is the
        # anti-vacuity property: `_fresh_plan_dir` removes `state.md`, which is
        # an ERROR the driver has to CLEAR by writing its own during the
        # traverse.  Pinning the whole error list (rather than just its
        # emptiness) also catches the other direction -- a corpus that drifted
        # dirty would make "zero errors after" unreachable and the failure would
        # be blamed on the traverse.
        before = _errors(audit(plan_dir, workspace_root=str(workspace)))
        assert [(i.check, i.artifact) for i in before] == [
            ("state", ArtifactNames.STATE)
        ], f"unexpected baseline errors: {[str(i) for i in before]}"

        worker = ScriptedRoles()
        agent = _live_agent(worker, approvals=Approvals())
        started = time.monotonic()
        result = agent.run(GOAL, initial_context=_roots(plan_dir, workspace))
        elapsed = time.monotonic() - started

        # The traverse itself: every protocol state was entered, in order.
        first_seen = [
            state
            for i, state in enumerate(worker.states)
            if state not in worker.states[:i]
        ]
        assert first_seen == [
            HarnessStates.EXPLORE,
            HarnessStates.PLAN,
            HarnessStates.EXECUTE,
            HarnessStates.REFLECT,
            HarnessStates.CLOSE,
        ], f"run {run} took the path {worker.states} in {elapsed:.0f}s"
        assert result.final_context.get(ContextKeys.CLOSE_CONFIRMED) is True

        # The criterion: the CLOSE audit the DRIVER ran, over the directory the
        # run actually wrote.  `audit_issues` is None when CLOSE was never
        # reached, so this also re-proves the traverse.
        issues = agent.audit_issues
        assert issues is not None, "CLOSE was never reached, so nothing was audited"
        assert _errors(issues) == [], (
            f"run {run}: audit reported "
            f"{len(_errors(issues))} error(s): {[str(i) for i in _errors(issues)]}"
        )

        # And the artifacts really are on disk, written by the roles that own
        # them -- not merely absent-and-therefore-unaudited.
        assert (plan_dir / ArtifactNames.STATE).is_file()
        assert len(_findings_on_disk(plan_dir)) >= Defaults.FINDINGS_THRESHOLD


# ---------------------------------------------------------------------------
# L2 -- criterion 4: the leash halts at EXACTLY 2
# ---------------------------------------------------------------------------


@requires_live
class TestL2LeashHaltsAtExactlyTwo:
    """Not 1, not 3, and not resettable from inside an approving callback."""

    @pytest.mark.parametrize("run", range(1, RUNS + 1))
    def test_a_denied_continue_stops_after_the_second_attempt(
        self, tmp_path: Path, run: int
    ) -> None:
        plan_dir = _fresh_plan_dir(tmp_path, run)
        workspace = _seed_workspace(tmp_path / f"run-{run}")

        worker = ScriptedRoles(execute_succeeds=False, total_steps=1)
        approvals = Approvals({"harness.continue_after_leash": False})
        agent = _live_agent(worker, approvals=approvals)
        result = agent.run(GOAL, initial_context=_roots(plan_dir, workspace))

        attempts = worker.fix_attempts_for(HarnessStates.EXECUTE)
        # EXACTLY two dispatches: one more than a leash of 1 would allow, one
        # fewer than a leash of 3 would.  The counters they SAW are asserted
        # too, so a run that dispatched twice on two different steps cannot
        # masquerade as a leash hit.
        assert attempts == [0, 1], f"run {run}: executor saw fix_attempts={attempts}"
        assert worker.count_for(HarnessStates.EXECUTE) == Defaults.MAX_FIX_ATTEMPTS
        assert result.final_context[ContextKeys.FIX_ATTEMPTS] == 2
        assert result.final_context[ContextKeys.LAST_GATE_SLUG] == GateSlug.LEASH_CAP
        assert approvals.count("harness.continue_after_leash") >= 1
        assert any(p.name == "PC-EXECUTE-LEASH" for p in agent.presentations)

    @pytest.mark.parametrize("run", range(1, RUNS + 1))
    def test_an_approving_callback_cannot_raise_the_cap(
        self, tmp_path: Path, run: int
    ) -> None:
        """The escape shape the predecessor plan shipped, disproved live.

        A callback that grants EVERY leash-continue must still land on the
        bounded product ``max_fix_attempts * (1 + max_leash_grants)``: each
        grant buys one fresh PAIR of attempts and the grant counter is
        driver-owned, so no approval sequence reaches a third unattended
        attempt inside any window.
        """
        plan_dir = _fresh_plan_dir(tmp_path, run)
        workspace = _seed_workspace(tmp_path / f"run-{run}")

        worker = ScriptedRoles(execute_succeeds=False, total_steps=1)
        approvals = Approvals()  # approves everything, including the leash
        agent = _live_agent(worker, approvals=approvals)
        result = agent.run(GOAL, initial_context=_roots(plan_dir, workspace))

        bound = Defaults.MAX_FIX_ATTEMPTS * (1 + Defaults.MAX_LEASH_GRANTS)
        attempts = worker.fix_attempts_for(HarnessStates.EXECUTE)
        assert worker.count_for(HarnessStates.EXECUTE) == bound, (
            f"run {run}: {worker.count_for(HarnessStates.EXECUTE)} executor "
            f"dispatches against a bound of {bound}; saw {attempts}"
        )
        # No window ever reached a THIRD unattended attempt.
        assert max(attempts) == Defaults.MAX_FIX_ATTEMPTS - 1, (
            f"run {run}: the executor saw fix_attempts={attempts}, so some "
            "window ran past the 2-attempt leash"
        )
        assert attempts == [0, 1] * (1 + Defaults.MAX_LEASH_GRANTS)
        # The driver STOPS ASKING once the grant budget is spent.
        assert (
            approvals.count("harness.continue_after_leash") == Defaults.MAX_LEASH_GRANTS
        )
        assert result.final_context[ContextKeys.LEASH_GRANTS] == (
            Defaults.MAX_LEASH_GRANTS
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# L3 -- criterion 5: a PIVOT loop-back completes and the run continues
# ---------------------------------------------------------------------------


@requires_live
class TestL3PivotLoopBack:
    """REFLECT -> PIVOT -> PLAN, and the run keeps going afterwards."""

    @pytest.mark.parametrize("run", range(1, RUNS + 1))
    def test_the_pivot_loops_back_to_plan_and_the_run_continues(
        self, tmp_path: Path, run: int
    ) -> None:
        plan_dir = _fresh_plan_dir(tmp_path, run)
        workspace = _seed_workspace(tmp_path / f"run-{run}")

        worker = ScriptedRoles(pivot_first_reflect=True, total_steps=1)
        agent = _live_agent(worker, approvals=Approvals())
        result = agent.run(GOAL, initial_context=_roots(plan_dir, workspace))

        states = worker.states
        assert HarnessStates.PIVOT in states, f"run {run} never pivoted: {states}"
        pivot_at = states.index(HarnessStates.PIVOT)
        assert states[pivot_at - 1] == HarnessStates.REFLECT
        assert HarnessStates.PLAN in states[pivot_at:], (
            f"run {run}: the pivot did not loop back to PLAN: {states}"
        )
        # The run CONTINUED rather than ending at the pivot.
        assert HarnessStates.CLOSE in states[pivot_at:], f"run {run}: {states}"
        assert result.final_context.get(ContextKeys.CLOSE_CONFIRMED) is True
        # A pivot is a re-plan: the iteration advances on the PLAN edge that
        # follows it, never on the pivot itself.
        assert result.final_context[ContextKeys.ITERATION] == 2
        # The leash is cleared across the pivot.
        assert worker.fix_attempts_for(HarnessStates.PIVOT) == [0]


# ---------------------------------------------------------------------------
# L4 -- criterion 1: a dispatch that calls a tool AND writes the asked-for file
# ---------------------------------------------------------------------------


def _spy_on_tools(sink: list[dict[str, Any]]):
    """Wrap ``ToolRegistry.execute`` so every call is recorded.

    A spy, never a substitution: the real method runs and its real result is
    returned unchanged.  Returns the original for restoration.
    """
    original = ToolRegistry.execute

    def spied(self, tool_call):  # type: ignore[no-untyped-def]
        result = original(self, tool_call)
        sink.append(
            {
                "tool": tool_call.tool_name,
                "ok": bool(result.success),
            }
        )
        return result

    ToolRegistry.execute = spied  # type: ignore[method-assign]
    return original


def _execute_request(plan_dir: Path, workspace: Path) -> RoleRequest:
    """One EXECUTE dispatch, shaped exactly as the driver shapes it.

    ``assigned_write_target`` goes through the driver's OWN derivation
    (D-010), fed the same ``EXECUTE_PLAN_MD`` the plan directory carries --
    from the string, not the disk, so the bench's template render
    (``harness_bench._execute_render``, placeholder paths) sees the same
    request a real dispatch does and B1's prompt hash records the change.
    ``test_the_bench_request_carries_the_driver_assigned_target`` pins the
    two paths against each other offline.
    """
    spec = get_role_spec(HarnessStates.EXECUTE)
    rules = get_rules(HarnessStates.EXECUTE)
    return RoleRequest(
        role=spec.role,
        state=spec.state,
        goal=GOAL,
        operative_rules=rules.operative_rules,
        gate_summary=rules.gate_summary,
        iteration=1,
        step_number=1,
        total_steps=1,
        fix_attempts=0,
        context={},
        plan_dir=str(plan_dir),
        workspace_root=str(workspace),
        assigned_write_target=derive_execute_target(
            PlanDoc.from_markdown(EXECUTE_PLAN_MD), 1
        ),
    )


def _execute_plan_dir(root: Path) -> Path:
    """A COMPLETE plan directory for a SINGLE EXECUTE dispatch.

    Interface contract (2 call sites -- the L4 bench and the offline test that
    pins it): returns a plan directory in which every artifact the executor's
    Pre-Step Checklist names is present and consistent with
    :data:`EXECUTE_PLAN_MD`.

    Deliberately NOT :func:`_fresh_plan_dir`.  That one deletes ``state.md``
    because the traverse tests need the DRIVER to write it; here there is no
    driver and no traverse, just one dispatch, so the same deletion only
    guarantees an ENOENT loop.  See :data:`EXECUTE_STATE_MD`.
    """
    return make_plan_dir(root, plan_md=EXECUTE_PLAN_MD, state_md=EXECUTE_STATE_MD)


def _one_execute_dispatch(
    tmp_path: Path, run: int, *, native: bool, seed: int | None = None
) -> dict[str, Any]:
    """One live EXECUTE dispatch; returns its row, asserts nothing.

    ``seed`` (default ``None`` = exactly the pre-seed call shape) reaches the
    completion call only on the NATIVE arm — ``build_default_worker_factory``
    documents native-arm-only honoring — so the row records both the seed it
    was ASKED to use and ``seed_effective``, which is False for every react
    row even when a seed was requested.
    """
    root = tmp_path / f"{'native' if native else 'react'}-{run}"
    plan_dir = _execute_plan_dir(root)
    workspace = _seed_workspace(root)
    before = _digests(workspace)

    calls: list[dict[str, Any]] = []
    factory = build_default_worker_factory(
        Workspace(str(workspace)),
        model=MODEL,
        timeout_seconds=600,
        retry_attempts=1,
        native_function_calling=native,
        seed=seed,
    )
    original = _spy_on_tools(calls)
    started = time.monotonic()
    try:
        result = factory(_execute_request(plan_dir, workspace))
    finally:
        ToolRegistry.execute = original  # type: ignore[method-assign]

    after = _digests(workspace)
    target = workspace / "uploader.py"
    body = target.read_text(encoding="utf-8") if target.is_file() else ""
    return {
        # The label carries which arm SHIPS, because that moved at D-049 and a
        # reader comparing this table to an older one needs to see it.  The
        # boolean beside it is what the assertions select on, so renaming the
        # label can never silently re-point a bar at the other arm.
        "arm": "native_fc (package default)" if native else "react (opt-in)",
        "native": native,
        "run": run,
        "seed": seed,
        "seed_effective": bool(native and seed is not None),
        "elapsed_s": round(time.monotonic() - started, 1),
        "tool_calls": len(calls),
        "write_tool_issued": any(c["tool"] in WRITE_TOOLS for c in calls),
        "bytes_on_disk": any(
            before.get(name) != digest for name, digest in after.items()
        ),
        # The strict form: the TARGET file's content hash moved AND the new
        # text is recognisably the edit that was requested.
        "content_matched": (
            before.get("uploader.py") != after.get("uploader.py")
            and any(token in body.lower() for token in RETRY_TOKENS)
        ),
        "success": bool(result.success),
    }


def test_the_L4_bench_hands_the_executor_a_readable_plan_directory(
    tmp_path: Path,
) -> None:
    """UNGATED: the bench itself, checked before any model is blamed for it.

    L4's first live run scored 0/3 on "a write tool was issued" and the traces
    said why: ``state.md`` was absent, so the executor looped on the read its
    own Pre-Step Checklist mandates and never reached the edit.  A measurement
    taken through a directory the role cannot read is a measurement of the
    bench.  This pins the repair OFFLINE -- if the imported corpus renames its
    step line, or a future edit reaches for ``_fresh_plan_dir`` here, this goes
    red in ``make test`` instead of silently costing another live 0/3.
    """
    assert EXECUTE_STATE_MD != STATE_MD, (
        "the substitution did not land -- the imported corpus renamed its "
        "`## Current Plan Step:` line, so the bench points at a step "
        "EXECUTE_PLAN_MD does not contain"
    )
    # `HarnessStates` names the FSM state (`execute`); `state.md` writes the
    # protocol PHASE (`EXECUTE`).  Same fact, two casings, so relate them
    # rather than typing the literal twice.
    assert EXECUTE_STATE_MD.startswith(
        f"# Current State: {HarnessStates.EXECUTE.upper()}"
    )
    assert "## Current Plan Step: 1 " in EXECUTE_STATE_MD
    # ...and the plan really does have that step, and only that step.
    assert "\n1. [ ] **Add a retry" in EXECUTE_PLAN_MD
    assert "\n2. [ ]" not in EXECUTE_PLAN_MD

    plan_dir = _execute_plan_dir(tmp_path / "bench")
    missing = [name for name in PRE_STEP_READS if not (plan_dir / name).is_file()]
    assert missing == [], f"the executor is told to read, but cannot: {missing}"
    # And the fresh-traverse fixture is deliberately the OTHER way round, which
    # is what makes the two benches distinguishable rather than a copy-paste.
    assert not (_fresh_plan_dir(tmp_path, 1) / ArtifactNames.STATE).exists()


def test_the_bench_request_carries_the_driver_assigned_target(
    tmp_path: Path,
) -> None:
    """UNGATED: the bench dispatch and the driver derive the SAME target.

    Step 5's B1 block measures the D-010 fix; if ``_execute_request`` drifted
    from what the driver hands a real EXECUTE dispatch, B1 would silently
    measure the pre-fix prompt (B0's, 2/40 content-matched) and burn its live
    budget on nothing.  The string-fed derivation in ``_execute_request`` and
    the driver's disk-fed ``_assign_execute_target`` must agree on the plan
    directory this bench really seeds.
    """
    expected = derive_execute_target(PlanDoc.from_markdown(EXECUTE_PLAN_MD), 1)
    assert expected == "uploader.py", (
        "EXECUTE_PLAN_MD's Files To Modify no longer parses to the file the "
        "content_matched metric hashes -- the bench would aim the model at "
        "one file and score it against another"
    )

    plan_dir = _execute_plan_dir(tmp_path / "bench")
    workspace = tmp_path / "ws"
    assert _execute_request(plan_dir, workspace).assigned_write_target == expected
    # The driver seam returns (target, reason) since D-005; the bench must
    # agree on the target AND see the assignment tagged as such.
    assert HarnessAgent._assign_execute_target(
        {ContextKeys.PLAN_DIR: str(plan_dir), ContextKeys.STEP_NUMBER: 1}
    ) == (expected, EXECUTE_TARGET_ASSIGNED)


@requires_live
class TestL4ToolCallAndFileWrite:
    """A real ``:4b`` EXECUTE dispatch, judged by CONTENT HASH, never by stat.

    BOTH agent arms are measured, and that is the point rather than thoroughness
    for its own sake.  Every live number this plan carries for criteria 1 and 2
    (steps 5, 20, 21, 24, 25) was taken with ``native_function_calling=True``,
    and until D-049 ``build_default_worker_factory`` SHIPPED the other arm, so a
    test measuring one arm would either re-report a number nobody ships or hide
    the arm everybody gets.

    THIS TEST IS WHY THE DEFAULT MOVED, and the direction matters: the native
    arm did NOT clear its bar here.  Two n=5 blocks measured it at 3/5 then 2/5
    (5/10 pooled) on the loose form and 0/20 on the strict one, against >=4/5.
    The then-shipped ReAct arm measured 0/5 and 0/5 -- ONE tool call in twenty
    dispatches, most ending in ``Stall detected: 3 consecutive iterations with
    no tool selected``.  D-049 flipped the default on "5/10 beats 0/10 on the
    executor users get", not on a pass, and this test stays RED until something
    actually clears 4/5.  Keep measuring both arms: the losing one is the
    control that makes a default chosen against 0/10 falsifiable, and if the
    ordering ever inverts, this table is where it shows up.
    """

    def test_a_dispatch_calls_a_write_tool_and_writes_the_asked_for_content(
        self, tmp_path: Path
    ) -> None:
        rows = [
            _one_execute_dispatch(tmp_path, run, native=native)
            for native in (True, False)
            for run in range(1, RUNS_MODEL + 1)
        ]
        _report("L4 tool call + file write", rows)

        native = [r for r in rows if r["native"]]
        issued = sum(1 for r in native if r["write_tool_issued"])
        bytes_written = sum(1 for r in native if r["bytes_on_disk"])
        matched = sum(1 for r in rows if r["content_matched"])

        # plan.md's Verification Strategy row 1, verbatim: ">=4/5 runs:
        # `write_file` in the trace AND the target file exists under the
        # workspace root with the expected content".  The bar is the plan's,
        # transcribed rather than chosen here, and it is NOT relaxed to fit a
        # measurement: `MODEL_BAR` is 4 because the plan says 4.
        assert issued >= MODEL_BAR, (
            f"write tool issued {issued}/{RUNS_MODEL} on the native arm: {rows}"
        )
        assert bytes_written >= MODEL_BAR, (
            f"bytes on disk {bytes_written}/{RUNS_MODEL} on the native arm: {rows}"
        )
        # Content-hash match is the STRICTER form, and Success Criterion 1 is
        # EXISTENTIAL -- "*a* role dispatch that calls a tool AND writes a
        # file".  Counted over every dispatch of both arms; the per-arm k/n is
        # printed either way.
        #
        # The ReAct arm is measured and REPORTED but not asserted, and the
        # reason is not indulgence: asserting a rate that is 0/5 would either
        # freeze a defect into the suite or make it red for a fact that belongs
        # in a decision entry and a default flip, not in this test.  That flip
        # has since happened (D-049); the arm is kept, unasserted, as the
        # control.
        assert matched >= 1, (
            f"content-matched writes {matched}/{2 * RUNS_MODEL} across both arms "
            f"(native issued {issued}/{RUNS_MODEL}, "
            f"bytes {bytes_written}/{RUNS_MODEL}): {rows}"
        )


# ---------------------------------------------------------------------------
# L5 -- criterion 2: a genuine, disk-derived findings_count >= 3
# ---------------------------------------------------------------------------


class _RealExplorerScript:
    """Scripted everywhere EXCEPT EXPLORE, which is a real ``:4b`` dispatch.

    Interface contract (1 call site, L5; a class rather than a closure so the
    two halves are inspectable after the run):
        - EXPLORE dispatches go to the live ``build_default_worker_factory``
          worker, so the findings files are produced by the model.
        - Every other state is :class:`ScriptedRoles`, so the run can leave
          EXPLORE and terminate without spending live calls on states this
          criterion does not measure.
    """

    def __init__(self, live_worker: Any) -> None:
        self.live_worker = live_worker
        self.scripted = ScriptedRoles(total_steps=1)
        self.explore_results: list[AgentResult] = []

    def __call__(self, request: RoleRequest) -> AgentResult:
        if request.state != HarnessStates.EXPLORE:
            return self.scripted(request)
        result = self.live_worker(request)
        self.explore_results.append(result)
        return result


@requires_live
class TestL5GenuineFindingsCount:
    """The count the gate reads comes off the disk a real explorer wrote."""

    def test_a_successful_explorer_dispatch_reaches_three_distinct_findings(
        self, tmp_path: Path
    ) -> None:
        rows: list[dict[str, Any]] = []
        for run in range(1, RUNS_MODEL + 1):
            plan_dir = _fresh_plan_dir(tmp_path, run)
            workspace = _seed_workspace(tmp_path / f"run-{run}")

            live = build_default_worker_factory(
                Workspace(str(workspace)),
                model=MODEL,
                timeout_seconds=600,
                retry_attempts=1,
                # DECISION plan-2026-07-21T191807-bf7ffe24/D-048
                # Do NOT delete this keyword as "redundant now that D-049 made
                # it the default".  It PINS the arm this criterion's history
                # was measured on -- step 25 (`2907252`) reached 10/10 with
                # `native_function_calling=True` -- so the number stays
                # comparable across a future default flip in either direction.
                # Omitting it (this file's first live L5 run did, when ReAct
                # still shipped) silently measured the OTHER arm, which is a
                # DIFFERENT claim: n=5 gave 0/5 findings, 0/5 successful
                # dispatches and `Stall detected: 3 consecutive iterations with
                # no tool selected` in every dispatch -- the D-034 collapse
                # that D-049 later cited as its reason, not a result about
                # criterion 2.  L4 is where BOTH arms are measured.
                native_function_calling=True,
            )
            worker = _RealExplorerScript(live)
            agent = _live_agent(worker, approvals=Approvals())
            started = time.monotonic()
            result = agent.run(GOAL, initial_context=_roots(plan_dir, workspace))

            on_disk = _findings_on_disk(plan_dir)
            rows.append(
                {
                    "run": run,
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "explore_dispatches": worker.scripted.count_for(
                        HarnessStates.EXPLORE
                    )
                    + len(worker.explore_results),
                    "successful_dispatches": sum(
                        1 for r in worker.explore_results if r.success
                    ),
                    "findings_on_disk": list(on_disk),
                    "gate_count": result.final_context.get(ContextKeys.FINDINGS_COUNT),
                    "reached_plan": HarnessStates.PLAN in worker.scripted.states,
                    "halt_slug": result.final_context.get(ContextKeys.LAST_GATE_SLUG),
                }
            )

        _report("L5 disk-derived findings_count", rows)
        threshold = Defaults.FINDINGS_THRESHOLD
        met = [
            r
            for r in rows
            if len(r["findings_on_disk"]) >= threshold
            and (r["gate_count"] or 0) >= threshold
        ]
        # plan.md's Verification Strategy row 2, verbatim: ">=4/5 runs".  The
        # same `MODEL_BAR` L4 uses, and STRICTER than the existential form this
        # test previously carried -- step 25 measured this criterion at 10/10
        # (`2907252`), so a bar of 4/5 is the plan's, not a concession.  The k/n
        # is in the message so a partial result stays legible rather than merely
        # red.
        assert len(met) >= MODEL_BAR, (
            f"{len(met)}/{RUNS_MODEL} runs reached {threshold} distinct "
            f"non-empty findings files: {rows}"
        )
        # Whatever the rate, a run that MET the gate must have met it from the
        # disk AND been let through the EXPLORE -> PLAN edge -- the count is
        # evidence, not testimony.
        for row in met:
            assert row["successful_dispatches"] >= 1, row
            assert row["reached_plan"] is True, row
            assert row["halt_slug"] != GateSlug.EXPLORE_CAP, row


# ---------------------------------------------------------------------------
# L6 -- the central question: full runs on REAL workers, graded, floor-gated
# ---------------------------------------------------------------------------

#: Runs for the e2e criterion.  n=3 GRADED vectors, not n=5 binaries: a full
#: traverse costs 3-10+ minutes, and at this price per observation a per-run
#: rubric vector carries more information than a fifth pass/fail bit
#: (`findings/e2e-real-worker-criterion.md` section 7, decisions.md D-004).
RUNS_E2E = 3

#: Per-run completion seed, base + run - 1 -- same convention as the L4 bench
#: blocks (D-008: honored by Ollama on the native arm at this digest).
E2E_SEED_BASE = 20260722100

#: Test-side wall clock per run.  A run still going at 15 minutes FAILS the
#: floor as a hang instead of hanging pytest; the harness's own caps
#: (iteration_hard_cap, explore redispatch budget, stall detector, MAX_TURNS)
#: are expected to fire far earlier.
E2E_WALL_CLOCK_CEILING_S = 900.0

#: Where the committed rubric vectors live.  TRACKED, like every bench block:
#: `plans/` and scratch are how the predecessor's benches vanished.
BENCH_DATA_DIR = Path(__file__).resolve().parents[2] / "scripts" / "bench_data"
L6_BENCH_ID = "l6-e2e"
L6_BLOCK = "B0"

#: The slugs that make a halt HONEST: the four pre-step-gate slugs plus the
#: EXPLORE re-dispatch cap.  A stall halt carries NO slug (`_check_stall`
#: raises with ``slug=None``) and is deliberately NOT in this set -- "the run
#: wedged and could not say why" is the dishonest shape the floor exists to
#: catch.  Reaching terminal CLOSE is the other honest ending.
HONEST_HALT_SLUGS = frozenset({*GateSlug.ORDER, GateSlug.EXPLORE_CAP})

#: Protocol order for "furthest state reached".  PIVOT and CLOSE share the top
#: rank on purpose: both sit beyond REFLECT and neither dominates the other
#: (a pivoting run is not "further" than a closing one, or vice versa).
E2E_STATE_RANK: Mapping[str, int] = {
    HarnessStates.EXPLORE: 1,
    HarnessStates.PLAN: 2,
    HarnessStates.EXECUTE: 3,
    HarnessStates.REFLECT: 4,
    HarnessStates.PIVOT: 5,
    HarnessStates.CLOSE: 5,
}

#: The driver's three human-gate names.  Literals here (as in L2), but pinned
#: against the driver's own constants by an UNGATED test below, so a renamed
#: gate cannot silently turn the stub into deny-everything.
_GATE_PLAN = "harness.approve_plan"
_GATE_CLOSE = "harness.confirm_close"
_GATE_LEASH = "harness.continue_after_leash"

#: The ``audit()`` checks that judge ``plan.md`` ITSELF.  The plan-approval
#: predicate is scoped to these two so an unrelated audit error (say, a
#: malformed decisions entry) cannot veto a sound plan.
_PLAN_CHECKS = ("plan", "plan-section")


def _bench_module() -> Any:
    """Import ``scripts/harness_bench.py`` for its manifest/jsonl plumbing.

    Interface contract (2 call sites: the L6 live test and the offline
    manifest-schema test): returns the module, putting ``scripts/`` on
    ``sys.path`` exactly as ``tests/test_harness_bench.py`` does.  This is
    pure-IO reuse (manifest fields, ``append_row``, ``_write_json``, the
    digest query), NOT a reversal of D-001: the authoritative measurement
    machinery still flows bench -> tests, and ``harness_bench``'s only import
    of this module is lazy, so no cycle exists at import time.
    """
    scripts_dir = str(BENCH_DATA_DIR.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import harness_bench

    return harness_bench


def _l6_placeholder_request(state: str) -> tuple[RoleRequest, Any]:
    """One state's dispatch request at FIXED placeholder paths, plus its spec.

    Interface contract (3 call sites: the prompt hash, the tool surface, the
    recorder's offline test): mirrors ``harness_bench._execute_render``'s
    placeholder-path idea for ALL SIX states, so the L6 manifest can pin the
    prompt TEMPLATES without a live run.
    """
    spec = get_role_spec(state)
    rules = get_rules(state)
    request = RoleRequest(
        role=spec.role,
        state=state,
        goal=GOAL,
        operative_rules=rules.operative_rules,
        gate_summary=rules.gate_summary,
        iteration=1,
        step_number=1,
        total_steps=1,
        fix_attempts=0,
        context={},
        plan_dir="/plan-dir",
        workspace_root="/workspace",
    )
    return request, spec


def _l6_prompt_hash() -> str:
    """sha256 over all six states' rendered system+task prompt templates.

    The L4 blocks hash ONE state's render because they measure one dispatch;
    an e2e run speaks every role's prompt, so its manifest pins all of them,
    in protocol order.
    """
    parts: list[str] = []
    for state in HarnessStates.ALL:
        request, spec = _l6_placeholder_request(state)
        parts.append(build_role_system_prompt(request, spec))
        parts.append(build_role_task_prompt(request, spec))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _l6_tool_surface() -> dict[str, Any]:
    """The worker-factory kwargs plus per-state held tool names."""
    return {
        "native_function_calling": True,
        "timeout_seconds": 600,
        "retry_attempts": 1,
        "declared_tools_by_state": {
            state: sorted(held_tools(*_l6_placeholder_request(state)))
            for state in HarnessStates.ALL
        },
    }


def _l6_fixture_hash() -> str:
    """sha256 pinning GOAL + SEED_FILES -- the ONLY fixed inputs an L6 run gets.

    Deliberately NOT ``harness_bench._fixture_hash``: that one also pins
    ``EXECUTE_PLAN_MD``/``EXECUTE_STATE_MD``, which in L6 are OUTPUTS -- the
    real plan-writer authors ``plan.md`` and the driver authors ``state.md``,
    and pinning an output as a fixture would misdescribe the measurement.
    """
    seeds = "\x00".join(f"{k}\n{v}" for k, v in sorted(SEED_FILES.items()))
    payload = f"{GOAL}\x00{seeds}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _l6_manifest(hb: Any) -> dict[str, Any]:
    """The L6 pre-registration record; same six comparability fields as L4."""
    return {
        "bench_id": L6_BENCH_ID,
        "block": L6_BLOCK,
        "n_preregistered": RUNS_E2E,
        "seed": {
            "base": E2E_SEED_BASE,
            "per_row": "base+run-1",
            "effective_arm": "native",
        },
        "model": MODEL,
        "created_at": hb._utc_now(),
        "prompt_bytes_sha256": _l6_prompt_hash(),
        "tool_surface": _l6_tool_surface(),
        "fixture_hash": _l6_fixture_hash(),
        "model_digest": hb._model_digest(),
        "arm": {"native": True, "display": "native"},
        "git_commit": hb._git_commit(),
    }


class _PassThroughRecorder:
    """Records every :class:`RoleRequest`, then hands it to the REAL factory.

    Interface contract (1 call site, L6): a shim, never a script.  It makes no
    decision, writes no artifact and edits no result -- the workers under
    measurement are exactly the ones ``build_default_worker_factory`` built.
    ``requests`` is the driver's own dispatch record; ``states_entered``,
    the dispatch counts and the EXECUTE ``assigned_write_target`` cells of the
    rubric vector are all read off it rather than off model prose.
    """

    def __init__(self, factory: Any) -> None:
        self.factory = factory
        self.requests: list[RoleRequest] = []

    def __call__(self, request: RoleRequest) -> AgentResult:
        self.requests.append(request)
        return self.factory(request)


# DECISION plan-2026-07-22T114536-879d04a0/D-013
# The e2e approval stub is DENY-DEFAULT and every YES cites bytes on disk. Do
# NOT replace it with a blanket-True callback to "get the traverse through":
# `plan_approved`/`close_confirmed` record a HUMAN decision, and a stub that
# always says yes makes the plan gate vacuous -- the run would measure a
# harness with its gate integrity removed (D-004). Do NOT widen the plan
# predicate beyond the ("plan", "plan-section") audit checks either: a 0-ERROR
# bar over the WHOLE audit is structurally unreachable for :4b-authored
# decisions prose (findings/e2e-real-worker-criterion.md section 6), so it
# would convert this criterion into a guaranteed-fail, which is as
# uninformative as guaranteed-pass. Leash grants are ALWAYS denied: exact
# leash claims live in L2 under a scripted executor (D-047) and an e2e stub
# that granted them would just buy noise. See decisions.md D-013.
class DiskEvidenceApprovals:
    """DENY-default approvals whose every YES is bound to a disk predicate.

    Interface contract (1 live call site, plus the UNGATED predicate tests):

    * ``harness.approve_plan`` -- approve iff ``plan.md`` exists non-empty AND
      ``audit()`` reports zero ERRORs from the two plan.md checks
      (:data:`_PLAN_CHECKS`).  Scoped on purpose; see the D-013 anchor above.
    * ``harness.confirm_close`` -- approve iff ``verification.md`` exists
      non-empty.
    * ``harness.continue_after_leash`` -- ALWAYS denied.
    * anything else -- denied (the package's own default, kept).
    * ``decisions`` records every consultation in order:
      ``{"gate", "approved", "evidence"}`` -- the rubric vector commits it raw.
    """

    def __init__(self, plan_dir: Path) -> None:
        self.plan_dir = Path(plan_dir)
        self.decisions: list[dict[str, Any]] = []

    def _artifact_bytes(self, name: str) -> int:
        path = self.plan_dir / name
        return path.stat().st_size if path.is_file() else 0

    def _decide(self, gate: str) -> tuple[bool, str]:
        if gate == _GATE_PLAN:
            size = self._artifact_bytes(ArtifactNames.PLAN)
            if size == 0:
                return False, f"{ArtifactNames.PLAN} is absent or empty"
            errors = [
                str(issue)
                for issue in audit(self.plan_dir)
                if issue.is_error and issue.check in _PLAN_CHECKS
            ]
            if errors:
                return False, (
                    f"{ArtifactNames.PLAN} has {len(errors)} "
                    f"plan-check error(s): {errors}"
                )
            return True, (
                f"{ArtifactNames.PLAN} carries {size} bytes and zero "
                f"{'/'.join(_PLAN_CHECKS)} audit errors"
            )
        if gate == _GATE_CLOSE:
            size = self._artifact_bytes(ArtifactNames.VERIFICATION)
            if size == 0:
                return False, f"{ArtifactNames.VERIFICATION} is absent or empty"
            return True, f"{ArtifactNames.VERIFICATION} carries {size} bytes"
        if gate == _GATE_LEASH:
            return False, "leash grants are ALWAYS denied in e2e (D-013/D-047)"
        return False, f"unknown gate '{gate}': denied by default"

    def __call__(self, request: ApprovalRequest) -> bool:
        approved, evidence = self._decide(request.tool_name)
        self.decisions.append(
            {"gate": request.tool_name, "approved": approved, "evidence": evidence}
        )
        return approved


def _run_with_ceiling(
    agent: HarnessAgent, goal: str, roots: dict[str, str], ceiling_s: float
) -> dict[str, Any]:
    """Run the agent under a wall clock; a hung run FAILS instead of hanging.

    Interface contract (1 call site, L6): returns a box holding ``result``
    (an ``AgentResult``) or ``error`` (the exception ``run`` raised), plus
    ``timed_out``.  The worker thread is a daemon: on a ceiling hit it is
    ABANDONED, not killed -- its row is already below the floor, and later
    runs use fresh directories and a fresh agent, so a straggler can slow
    them but cannot corrupt them.
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = agent.run(goal, initial_context=roots)
        except BaseException as exc:  # deliberate: graded, never re-raised
            box["error"] = exc

    thread = threading.Thread(target=_target, name="l6-e2e-run", daemon=True)
    thread.start()
    thread.join(ceiling_s)
    box["timed_out"] = thread.is_alive()
    return box


def _bytes_of(plan_dir: Path, name: str) -> int:
    path = plan_dir / name
    return path.stat().st_size if path.is_file() else 0


def _issue_families(issues: list[Issue], *, errors: bool) -> dict[str, int]:
    """``check -> count`` for the ERROR (or WARNING) half of an audit."""
    families: dict[str, int] = {}
    for issue in issues:
        if issue.is_error is errors:
            families[issue.check] = families.get(issue.check, 0) + 1
    return dict(sorted(families.items()))


def _furthest_state(states: list[str]) -> str | None:
    """The highest-ranked protocol state in *states*, or ``None``."""
    ranked = [state for state in states if state in E2E_STATE_RANK]
    return max(ranked, key=E2E_STATE_RANK.__getitem__, default=None)


def _bench_defect(row: Mapping[str, Any]) -> bool:
    """Pre-Mortem #3's trigger shape: a crash, a hang, or a slugless stall.

    These are the endings that indicate a BENCH/HARNESS defect rather than
    model attainment -- a configuration that could not have succeeded.  An
    honest cap halt (named slug) or a CLOSE is never a defect, however early.
    """
    return bool(
        row["error"] is not None
        or row["timed_out"]
        or (not row["close_reached"] and row["halt_slug"] is None)
    )


def _one_e2e_run(tmp_path: Path, run: int) -> dict[str, Any]:
    """One full real-worker protocol run; returns its rubric vector.

    Asserts nothing.  Every dimension is disk- or driver-derived: transition
    evidence comes from the recorder's dispatch log and the final ``state.md``,
    write evidence from sha256 digests plus the factory's own D-016
    verified-write channel, audit numbers from ``audit()`` over the directory
    the run really wrote.  The plan directory starts EMPTY -- unlike L4/L5
    there is no corpus seed, so every artifact found afterwards was produced
    by the run under measurement.
    """
    root = tmp_path / f"e2e-{run}"
    plan_dir = root / "plan"
    plan_dir.mkdir(parents=True)
    workspace = _seed_workspace(root)
    seed = E2E_SEED_BASE + run - 1
    before_ws = _digests(workspace)

    observations: list[dict[str, Any]] = []
    live = build_default_worker_factory(
        Workspace(str(workspace)),
        model=MODEL,
        timeout_seconds=600,
        retry_attempts=1,
        # Pinned for the reason L5 pins it (D-048): the arm this criterion's
        # history is measured on must survive a future default flip.
        native_function_calling=True,
        seed=seed,
        observer=lambda record: observations.append(dict(record)),
    )
    worker = _PassThroughRecorder(live)
    approvals = DiskEvidenceApprovals(plan_dir)
    agent = _live_agent(worker, approvals=approvals)

    started = time.monotonic()
    box = _run_with_ceiling(
        agent, GOAL, _roots(plan_dir, workspace), E2E_WALL_CLOCK_CEILING_S
    )
    wall = round(time.monotonic() - started, 1)

    result = box.get("result")
    error = box.get("error")
    final: Mapping[str, Any] = result.final_context if result is not None else {}

    # -- write evidence: sha256 first (an echo-back write scores not-written),
    #    cross-checked against the factory's own D-016 verified-write channel.
    after_ws = _digests(workspace)
    ws_changed = sorted(
        name for name, digest in after_ws.items() if before_ws.get(name) != digest
    )
    # The plan dir started EMPTY, so everything under it was produced by the
    # run; `state.md` is the DRIVER's one owned write (invariant I7) and is
    # excluded -- a run whose only bytes are the driver's own bookkeeping has
    # shown no verified WORKER write.
    pd_written = sorted(
        name for name in _digests(plan_dir) if name != ArtifactNames.STATE
    )
    write_dispatches = sum(
        1 for record in observations if (record.get("write_evidence") or 0) >= 1
    )
    verified_write = bool(write_dispatches and (ws_changed or pd_written))

    # -- position evidence: the dispatch log plus the final state.md.
    states_entered: list[str] = []
    for request in worker.requests:
        if request.state not in states_entered:
            states_entered.append(request.state)
    dispatch_counts = {
        state: sum(1 for r in worker.requests if r.state == state)
        for state in states_entered
    }
    final_disk_state: str | None = None
    transition_history: list[str] = []
    state_path = plan_dir / ArtifactNames.STATE
    if state_path.is_file():
        try:
            doc = StateDoc.from_markdown(state_path.read_text(encoding="utf-8"))
            final_disk_state = doc.state
            transition_history = list(doc.transition_history)
        except HarnessArtifactError:
            final_disk_state = None  # a garbled state.md is itself a finding
    furthest = _furthest_state(
        states_entered + ([final_disk_state] if final_disk_state else [])
    )

    # -- halt evidence.
    halt_slug = final.get(ContextKeys.LAST_GATE_SLUG)
    close_reached = (
        final_disk_state == HarnessStates.CLOSE
        or HarnessStates.CLOSE in states_entered
        or final.get(ContextKeys.CLOSE_CONFIRMED) is True
    )
    honest_halt = (
        error is None
        and not box["timed_out"]
        and (close_reached or halt_slug in HONEST_HALT_SLUGS)
    )

    issues = audit(plan_dir, workspace_root=str(workspace))
    return {
        "run": run,
        "seed": seed,
        "arm": "native_fc (package default)",
        "native": True,
        "goal": GOAL,
        "wall_clock_s": wall,
        "timed_out": box["timed_out"],
        "error": repr(error) if error is not None else None,
        "success": bool(result.success) if result is not None else False,
        "states_entered": states_entered,
        "dispatch_counts": dispatch_counts,
        "dispatches_total": len(worker.requests),
        "furthest_state": furthest,
        "state_md_final_state": final_disk_state,
        "state_md_transition_history": transition_history,
        "findings_nonempty": len(_findings_on_disk(plan_dir)),
        "plan_md_bytes": _bytes_of(plan_dir, ArtifactNames.PLAN),
        "decisions_md_bytes": _bytes_of(plan_dir, ArtifactNames.DECISIONS),
        "verification_md_bytes": _bytes_of(plan_dir, ArtifactNames.VERIFICATION),
        "changelog_bytes": _bytes_of(plan_dir, ArtifactNames.CHANGELOG),
        "audit_errors": sum(1 for issue in issues if issue.is_error),
        "audit_warnings": sum(1 for issue in issues if not issue.is_error),
        "audit_error_checks": _issue_families(issues, errors=True),
        "audit_warning_checks": _issue_families(issues, errors=False),
        "execute_assigned_targets": [
            r.assigned_write_target
            for r in worker.requests
            if r.state == HarnessStates.EXECUTE
        ],
        "workspace_files_changed": ws_changed,
        "plan_dir_files_written": pd_written,
        "write_evidence_dispatches": write_dispatches,
        "verified_write": verified_write,
        "halt_slug": halt_slug,
        "halt_reason": final.get(ContextKeys.HALT_REASON),
        "close_reached": close_reached,
        "honest_halt": honest_halt,
        "iteration": final.get(ContextKeys.ITERATION),
        "fix_attempts": final.get(ContextKeys.FIX_ATTEMPTS),
        "criteria_pass_count": final.get(ContextKeys.CRITERIA_PASS_COUNT),
        "criteria_total": final.get(ContextKeys.CRITERIA_TOTAL),
        "approvals": list(approvals.decisions),
    }


def test_the_L6_criterion_is_structurally_closed_to_scripted_workers() -> None:
    """UNGATED: the plan invariant, pinned on SOURCE rather than promised.

    Scripted-worker results must never be presented as model capability, so
    the L6 measurement path must be INCAPABLE of holding one: no
    ``ScriptedRoles`` anywhere in it, the real factory constructed by name,
    and none of the exact-count projections D-047 reserves for scripted
    executors (``fix_attempts_for`` is the L2/L3 leash-count probe).
    """
    import inspect

    sources = "".join(
        inspect.getsource(obj)
        for obj in (
            TestL6EndToEndRealWorkers,
            _one_e2e_run,
            _PassThroughRecorder,
            DiskEvidenceApprovals,
            _run_with_ceiling,
        )
    )
    assert "ScriptedRoles" not in sources
    assert "build_default_worker_factory" in inspect.getsource(_one_e2e_run)
    assert "fix_attempts_for" not in sources


class TestTheL6ApprovalStubIsDenyDefaultAndDiskBound:
    """UNGATED: every YES the stub can give is bound to bytes on disk."""

    def test_an_empty_plan_directory_approves_nothing(self, tmp_path: Path) -> None:
        stub = DiskEvidenceApprovals(tmp_path)
        for gate in (_GATE_PLAN, _GATE_CLOSE, _GATE_LEASH, "anything.else"):
            assert stub(ApprovalRequest(tool_name=gate)) is False, gate
        assert [d["approved"] for d in stub.decisions] == [False] * 4

    def test_a_clean_plan_md_earns_plan_approval(self, tmp_path: Path) -> None:
        plan_dir = make_plan_dir(tmp_path)
        stub = DiskEvidenceApprovals(plan_dir)
        assert stub(ApprovalRequest(tool_name=_GATE_PLAN)) is True
        assert "zero" in stub.decisions[-1]["evidence"]

    def test_a_garbled_plan_md_is_denied_with_the_errors_as_evidence(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir / ArtifactNames.PLAN).write_text(
            "# Plan v1: broken\n\n## Goal\nonly a goal, no other section\n",
            encoding="utf-8",
        )
        stub = DiskEvidenceApprovals(plan_dir)
        assert stub(ApprovalRequest(tool_name=_GATE_PLAN)) is False
        assert "plan-check error" in stub.decisions[-1]["evidence"]

    def test_unrelated_audit_errors_do_not_veto_a_sound_plan(
        self, tmp_path: Path
    ) -> None:
        """The predicate is SCOPED to plan.md's own checks (D-013).

        A garbage ``decisions.md`` raises decisions-schema ERRORs; the plan
        gate must not read them -- a 0-ERROR bar over the whole audit is the
        guaranteed-fail shape the finding's section 6 warns about.
        """
        plan_dir = make_plan_dir(tmp_path)
        (plan_dir / ArtifactNames.DECISIONS).write_text(
            "not a decision log at all\n", encoding="utf-8"
        )
        full_errors = [i for i in audit(plan_dir) if i.is_error]
        assert full_errors, "fixture defect: garbage decisions.md audited clean"
        stub = DiskEvidenceApprovals(plan_dir)
        assert stub(ApprovalRequest(tool_name=_GATE_PLAN)) is True

    def test_the_leash_is_denied_even_when_everything_is_clean(
        self, tmp_path: Path
    ) -> None:
        stub = DiskEvidenceApprovals(make_plan_dir(tmp_path))
        assert stub(ApprovalRequest(tool_name=_GATE_LEASH)) is False

    def test_close_approval_requires_a_nonempty_verification_md(
        self, tmp_path: Path
    ) -> None:
        plan_dir = make_plan_dir(tmp_path)
        stub = DiskEvidenceApprovals(plan_dir)
        assert stub(ApprovalRequest(tool_name=_GATE_CLOSE)) is True
        (plan_dir / ArtifactNames.VERIFICATION).write_text("", encoding="utf-8")
        assert stub(ApprovalRequest(tool_name=_GATE_CLOSE)) is False

    def test_the_gate_names_are_the_drivers_own(self) -> None:
        """A renamed gate must break HERE, not silently deny everything."""
        from fsm_llm_harness import harness as harness_module

        assert _GATE_PLAN == harness_module._APPROVAL_PLAN
        assert _GATE_CLOSE == harness_module._APPROVAL_CLOSE
        assert _GATE_LEASH == harness_module._APPROVAL_LEASH


class TestTheL6RubricPlumbingOffline:
    """UNGATED: the graded rubric's own helpers, checked without a daemon."""

    def test_the_recorder_is_a_shim_not_a_script(self) -> None:
        sentinel = AgentResult(answer="x", success=True, final_context={"k": 1})
        recorder = _PassThroughRecorder(lambda request: sentinel)
        request, _ = _l6_placeholder_request(HarnessStates.EXPLORE)
        assert recorder(request) is sentinel
        assert recorder.requests == [request]

    def test_the_state_ranking_grades_every_protocol_state(self) -> None:
        assert set(E2E_STATE_RANK) == set(HarnessStates.ALL)
        assert _furthest_state([]) is None
        assert _furthest_state(list(HarnessStates.ALL[:2])) == HarnessStates.PLAN
        assert (
            _furthest_state([HarnessStates.REFLECT, HarnessStates.EXECUTE])
            == HarnessStates.REFLECT
        )
        assert (
            E2E_STATE_RANK[HarnessStates.PIVOT] == E2E_STATE_RANK[HarnessStates.CLOSE]
        )

    def test_a_slugless_stall_is_a_bench_defect_but_a_named_slug_is_not(
        self,
    ) -> None:
        base: dict[str, Any] = {
            "error": None,
            "timed_out": False,
            "close_reached": False,
            "halt_slug": None,
        }
        assert _bench_defect(base) is True
        assert _bench_defect({**base, "halt_slug": GateSlug.LEASH_CAP}) is False
        assert _bench_defect({**base, "halt_slug": GateSlug.EXPLORE_CAP}) is False
        assert _bench_defect({**base, "close_reached": True}) is False
        assert _bench_defect({**base, "timed_out": True}) is True
        assert _bench_defect({**base, "error": "RuntimeError('x')"}) is True

    def test_the_L6_manifest_pins_the_same_six_fields_as_the_L4_blocks(self) -> None:
        hb = _bench_module()
        static = {
            "prompt_bytes_sha256": _l6_prompt_hash(),
            "tool_surface": _l6_tool_surface(),
            "fixture_hash": _l6_fixture_hash(),
        }
        live_only = {"model_digest", "arm", "git_commit"}
        assert set(hb.MANIFEST_FIELDS) == set(static) | live_only
        # Deterministic: the pre-registration record cannot depend on when it
        # is rendered.
        assert _l6_prompt_hash() == _l6_prompt_hash()
        assert static["tool_surface"]["declared_tools_by_state"][
            HarnessStates.EXECUTE
        ], "the EXECUTE role holds no tools -- the surface pin is vacuous"
        # And it is NOT the L4 fixture hash: plan.md/state.md are OUTPUTS here.
        assert _l6_fixture_hash() != hb._fixture_hash(sys.modules[__name__])


@requires_live
class TestL6EndToEndRealWorkers:
    """THREE full protocol runs on the REAL worker stack, graded not binary.

    The plan's central question (W1, decisions.md D-004): no test before this
    one has ever driven the protocol end-to-end with
    ``build_default_worker_factory`` doing EVERY state's job -- L1-L3 script
    the workers (harness claims), L4/L5 measure one state each (model claims).
    Here the plan directory starts EMPTY: ``plan.md`` is authored by the real
    plan-writer, the EXECUTE target derivation (D-010) reads THAT plan.md, and
    if a :4b-authored plan never parses a target, the dispatch honestly falls
    back to no-assigned-target -- that is part of what is being measured.

    The FLOOR (Success Criterion 4, exact): all 3 runs reach >= EXECUTE
    (dispatch log + final state.md), show >= 1 verified write (sha256-diffed
    bytes AND the factory's own D-016 verified-write channel), and halt
    honestly (a named ``GateSlug`` or terminal CLOSE; an unhandled exception,
    a hang or a slugless stall is a floor FAIL).  The floor sits at EXECUTE,
    not CLOSE, because no real-worker REFLECT has ever been measured (A3).

    REPORTED but not asserted: the full rubric vector per run -- artifact
    bytes, audit ERROR/WARNING counts by check family, approval decisions with
    their disk evidence -- committed raw to ``scripts/bench_data/l6-e2e/``.
    EXCLUDED on purpose: exact-leash-count and exact-loop-back claims, which
    are unfalsifiable on real workers (D-047) and stay in L2/L3.
    """

    def test_three_full_runs_grade_at_or_above_the_floor(self, tmp_path: Path) -> None:
        hb = _bench_module()
        bench_dir = BENCH_DATA_DIR / L6_BENCH_ID
        rows_path = bench_dir / "rows.jsonl"
        if rows_path.exists():
            pytest.fail(
                f"{rows_path} exists -- an L6 block runs ONCE (D-002); a new "
                "question needs a new pre-registered block and decision entry"
            )
        bench_dir.mkdir(parents=True, exist_ok=True)
        hb._write_json(bench_dir / "manifest.json", _l6_manifest(hb))

        rows: list[dict[str, Any]] = []
        for run in range(1, RUNS_E2E + 1):
            row = _one_e2e_run(tmp_path, run)
            row.update(bench_id=L6_BENCH_ID, block=L6_BLOCK, ts=hb._utc_now())
            rows.append(row)
            hb.append_row(rows_path, row)
            print(
                f"  L6 run {run}: furthest={row['furthest_state']} "
                f"write={row['verified_write']} honest={row['honest_halt']} "
                f"slug={row['halt_slug']} {row['wall_clock_s']}s",
                flush=True,
            )
            # Pre-Mortem #3: two crashes/hangs/slugless stalls in two runs is
            # a bench or harness DEFECT, not model attainment.  Do not burn
            # the third run on a configuration that could not have succeeded;
            # the two rows are already committed evidence.
            if run == 2 and all(_bench_defect(r) for r in rows):
                _report("L6 e2e real workers (HALTED: Pre-Mortem #3)", rows)
                pytest.fail(
                    "Pre-Mortem #3: both of the first two runs ended in a "
                    "crash, hang or slugless stall -- halting sampling before "
                    f"run 3; fix the bench/harness first. Vectors: {rows}"
                )

        _report("L6 e2e real workers", rows)
        assert len(rows) == RUNS_E2E

        floor_misses: list[str] = []
        for row in rows:
            missing = [
                name
                for name, passed in (
                    (
                        "reached>=EXECUTE",
                        E2E_STATE_RANK.get(str(row["furthest_state"]), 0)
                        >= E2E_STATE_RANK[HarnessStates.EXECUTE],
                    ),
                    ("verified_write", bool(row["verified_write"])),
                    ("honest_halt", bool(row["honest_halt"])),
                )
                if not passed
            ]
            if missing:
                floor_misses.append(f"run {row['run']} missed {missing}")
        # The floor is the plan's, transcribed: 3/3 at >= EXECUTE with a
        # verified write and an honest halt.  Any run below it FAILS the
        # criterion; the full vectors are in the message either way.
        assert floor_misses == [], (
            f"L6 floor FAILED: {floor_misses}; full vectors: {rows}"
        )
