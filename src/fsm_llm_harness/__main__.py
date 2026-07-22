"""
Command-line entry point for :mod:`fsm_llm_harness`.

Five subcommands, each a thin composition over machinery that already exists:

======================  ====================================================
``new GOAL``            Mint a plan directory (``storage.PlanDirectory``),
                        seed its ``state.md``, and drive the protocol.
``resume PLAN_DIR``     Re-open an existing plan directory and continue from
                        the ``state.md`` it left behind.
``status PLAN_DIR``     Report where a plan is and whether the pre-step gate
                        would let another EXECUTE step run.
``validate PLAN_DIR``   ``plan_validator.audit()``, printed.
``close PLAN_DIR``      The CLOSE-phase size policies, DRY-RUN by default.
======================  ====================================================

Usage::

    python -m fsm_llm_harness new "add a retry to the uploader"
    python -m fsm_llm_harness resume plans/plan-2026-07-22T101500-1a2b3c4d
    python -m fsm_llm_harness status   plans/plan-2026-07-22T101500-1a2b3c4d
    python -m fsm_llm_harness validate plans/plan-2026-07-22T101500-1a2b3c4d
    python -m fsm_llm_harness close    plans/plan-2026-07-22T101500-1a2b3c4d --apply

**Exit codes.**  Exactly three, and the third is a contract rather than a
convention:

* ``0`` -- pass.
* ``1`` -- this command has a negative answer, or could not produce one: an
  ``audit()`` finding at ERROR severity, a failed run, a missing goal, a broken
  install.
* ``2`` -- a HARD pre-step gate refused (``plan_validator.pre_step_gate``).
  RESERVED.  Nothing else in this module may exit 2, and a caller may rely on
  ``2`` meaning "the protocol blocked this step" and nothing else.

**Model resolution** is ``--model`` > ``$LLM_MODEL`` > ``Defaults.MODEL``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .plan_validator import Issue

__all__ = ["main_cli"]

#: Pass.
EXIT_PASS = 0
#: A negative answer, or no answer: audit ERRORs, a failed run, a bad install.
EXIT_ERROR = 1
#: RESERVED for a HARD ``pre_step_gate`` failure.  See the module docstring.
EXIT_GATE = 2

_PROG = "fsm-llm-harness"
_INSTALL_HINT = "pip install fsm-llm[harness]"

#: Printed instead of a plan step when ``state.md`` has no cursor yet.
_NO_STEP = "-"


# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------


def _import(name: str) -> ModuleType:
    """Import one of this package's modules behind the optional-dep guard.

    Interface contract (one call per subcommand, plus the shared run path):
        - ``name``: a module name relative to :mod:`fsm_llm_harness`.
        - Returns the imported module.
        - Raises ``SystemExit(EXIT_ERROR)`` -- never an ``ImportError`` -- with
          the install hint on stderr, so a partial install produces one legible
          line instead of a traceback.

    ``import_module`` is bound at module scope on purpose: it is stdlib and free
    to import, and binding it here is what lets a test replace it to exercise
    the guard.
    """
    try:
        return import_module(f".{name}", __package__)
    except ImportError as exc:  # pragma: no cover - exercised via _import itself
        print(f"Error: could not import fsm_llm_harness.{name}: {exc}", file=sys.stderr)
        print(f"Make sure deps are installed: {_INSTALL_HINT}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR) from exc


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def resolve_model(explicit: str | None = None) -> str:
    """Resolve the model to dispatch with: flag, then ``$LLM_MODEL``, then default.

    Interface contract (2 call sites: the run path and the ``--model`` report):
        - ``explicit``: the ``--model`` value, or ``None``.
        - A blank or whitespace-only value at either tier is treated as ABSENT
          and falls through.  ``LLM_MODEL=""`` is an unset variable spelled
          badly, never a request for an empty model name.
        - Never raises; performs no I/O.
    """
    constants = _import("constants")
    for candidate in (explicit, os.environ.get(constants.Defaults.ENV_MODEL)):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    model: str = constants.Defaults.MODEL
    return model


def _existing_dir(raw: str) -> Path | None:
    """*raw* as a directory that is ALREADY there, or ``None``.

    Interface contract (4 call sites: resume, status, close and the run path):
        - Returns ``None`` when the path is absent, is not a directory, or is
          unusable.  ``None`` means "nothing to open", never "opened nothing".
        - Never creates anything.  ``PlanMemory`` -- and through it
          ``PlanDirectory`` -- ``mkdir``\\ s its plan directory on construction,
          so a command that opened one to look at it would MANUFACTURE the very
          absence the ``no-plan`` slug exists to report.
    """
    try:
        path = Path(raw).expanduser()
        return path if path.is_dir() else None
    except (OSError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_issues(issues: Sequence[Issue]) -> tuple[int, int, int]:
    """Print every issue and return ``(errors, warnings, infos)``."""
    severity = _import("constants").Severity
    for issue in issues:
        print(f"  {issue.severity:<7} {issue}")
    counts = tuple(
        sum(1 for issue in issues if issue.severity == level)
        for level in (severity.ERROR, severity.WARNING, severity.INFO)
    )
    print(f"{counts[0]} errors, {counts[1]} warnings, {counts[2]} infos")
    return counts  # type: ignore[return-value]


def _no_plan(plan_dir: str, detail: str) -> int:
    """Report the ``no-plan`` slug and hand back the reserved gate code."""
    slug = _import("constants").GateSlug.NO_PLAN
    print(f"GATE:FAIL [{slug}] {plan_dir}: {detail}", file=sys.stderr)
    return EXIT_GATE


def _failures() -> tuple[type[BaseException], ...]:
    """Everything a plan-directory operation can fail with.

    Interface contract (3 call sites: ``new``, ``resume`` and the housekeeping
    reader):
        - ``HarnessError`` covers confinement, ownership and artifact faults --
          everything this package wraps.
        - ``OSError`` covers the filesystem faults it does NOT wrap.  MEASURED:
          ``PlanDirectory.create`` under a ``--plans-dir`` that is a FILE raises
          ``NotADirectoryError`` straight out of ``PlanMemory``'s own ``mkdir``,
          which is not a ``HarnessError`` at all.
        - Catching only the first is the fail-OPEN shape here: an uncaught
          ``OSError`` leaves a traceback and an unspecified exit code where the
          contract promises 1.
    """
    return (_import("exceptions").HarnessError, OSError)


# ---------------------------------------------------------------------------
# Driving a run
# ---------------------------------------------------------------------------


def _drive(directory: Any, goal: str, args: argparse.Namespace) -> int:
    """Run the protocol against an open plan directory and report the outcome.

    Interface contract (2 call sites: ``new`` and ``resume``):
        - ``directory``: an open ``storage.PlanDirectory``.
        - Returns one of the three exit codes; raises nothing of its own.
    """
    constants = _import("constants")
    harness = _import("harness")
    roles = _import("roles")
    tools = _import("tools")

    model = resolve_model(args.model)
    workspace = tools.Workspace(args.workspace)
    print(f"model:     {model}")
    print(f"workspace: {workspace.root}")

    # The driver profile is `HarnessAgent`'s own, with ONE field replaced. Do
    # not restate the other five here: `Defaults.MAX_TURNS`, the two timeouts
    # and the sampling settings are a tuned profile, and a second copy of it in
    # the CLI would drift from the one the tests and the live benches exercise.
    config = harness.HarnessAgent._default_config().model_copy(update={"model": model})
    agent = harness.HarnessAgent(
        worker_factory=roles.build_default_worker_factory(workspace, model=model),
        config=config,
    )
    result = agent.run(
        goal,
        initial_context={
            constants.ContextKeys.PLAN_DIR: str(directory.path),
            constants.ContextKeys.WORKSPACE_ROOT: str(workspace.root),
        },
    )
    return _report_run(agent, result)


def _report_run(agent: Any, result: Any) -> int:
    """Print a finished run and decide its exit code.

    The precedence is deliberate and is the whole of the exit-code contract:
    a HARD gate slug outranks everything, then an ERROR-severity audit finding,
    then the run's own success flag.
    """
    constants = _import("constants")
    for presentation in agent.presentations:
        print(presentation.block)
    for directive in agent.reverts:
        print(f"revert (NOT executed): {directive}")
    print(f"answer: {result.answer}")

    issues = agent.audit_issues
    errors = 0
    if issues is not None:
        print(f"CLOSE audit ({len(issues)} issues):")
        errors = _print_issues(issues)[0]

    slug = result.final_context.get(constants.ContextKeys.LAST_GATE_SLUG)
    if slug in constants.GateSlug.ORDER:
        reason = result.final_context.get(constants.ContextKeys.HALT_REASON, "")
        print(f"GATE:FAIL [{slug}] {reason}".rstrip(), file=sys.stderr)
        return EXIT_GATE
    if errors:
        return EXIT_ERROR
    return EXIT_PASS if result.success else EXIT_ERROR


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_new(args: argparse.Namespace) -> int:
    """Mint a plan directory, seed its ``state.md``, and (unless told not to) run."""
    artifacts = _import("artifacts")
    constants = _import("constants")
    storage = _import("storage")

    try:
        directory = storage.PlanDirectory.create(args.plans_dir)
        seed = artifacts.StateDoc(
            state=constants.HarnessStates.INITIAL,
            iteration=0,
            current_step="0",
            last_transition=f"INIT ({_utc_now()})",
            transition_history=[f"INIT (plan directory created by {_PROG})"],
        )
        directory.write_artifact(constants.ArtifactNames.STATE, seed)
    except _failures() as exc:
        print(f"Error: could not create a plan directory: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"plan-id:   {directory.plan_id}")
    print(f"plan-dir:  {directory.path}")
    if args.create_only:
        return EXIT_PASS
    return _drive(directory, args.goal, args)


def _cmd_resume(args: argparse.Namespace) -> int:
    """Re-open an existing plan directory and continue the protocol from it."""
    storage = _import("storage")

    path = _existing_dir(args.plan_dir)
    if path is None:
        return _no_plan(args.plan_dir, "is not an existing directory")
    directory = storage.PlanDirectory(path)
    try:
        run_state = directory.load_run_state()
    except _failures() as exc:
        return _no_plan(args.plan_dir, f"state.md is unusable: {exc}")
    if run_state is None:
        return _no_plan(args.plan_dir, "has no state.md to resume from")

    goal = args.goal or _recorded_goal(directory)
    if goal is None:
        print(
            f"Error: {args.plan_dir} records no goal (plan.md has no usable "
            "'## Goal' section). Pass --goal to say what this run is for.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    print(f"plan-id:   {directory.plan_id}")
    print(f"plan-dir:  {directory.path}")
    print(
        f"resuming:  {run_state.state} "
        f"(iteration {run_state.iteration}, "
        f"step {run_state.current_step or _NO_STEP}, "
        f"{run_state.fix_attempts} fix attempts)"
    )
    return _drive(directory, goal, args)


def _recorded_goal(directory: Any) -> str | None:
    """The goal ``plan.md`` records, or ``None`` if there is not one to read."""
    artifacts = _import("artifacts")
    constants = _import("constants")
    if not directory.exists(constants.ArtifactNames.PLAN):
        return None
    # A plan.md the artifact schema REJECTS records no goal, full stop. Do NOT
    # "rescue" the `## Goal` section out of a document `PlanDoc` refused: the
    # goal is what the entire run is dispatched against, and lifting it out of
    # a file the validator fails is the CLI trusting text the protocol does
    # not. `None` is the honest answer, and it asks the caller for `--goal`.
    try:
        doc = artifacts.PlanDoc.from_markdown(
            directory.read_text(constants.ArtifactNames.PLAN)
        )
    except _failures():
        return None
    for line in str(doc.body_of("Goal")).splitlines():
        if line.strip():
            return line.strip()
    return None


def _cmd_status(args: argparse.Namespace) -> int:
    """Report where a plan is, and whether its pre-step gate is open."""
    constants = _import("constants")
    storage = _import("storage")
    validator = _import("plan_validator")

    gate = validator.pre_step_gate(args.plan_dir)
    if gate.slug == constants.GateSlug.NO_PLAN:
        return _no_plan(args.plan_dir, gate.detail)

    path = _existing_dir(args.plan_dir)
    if path is None:  # pragma: no cover - unreachable: state.md was just read
        return _no_plan(args.plan_dir, "is not an existing directory")
    directory = storage.PlanDirectory(path)
    try:
        run_state = directory.load_run_state()
    except _failures() as exc:
        return _no_plan(args.plan_dir, f"state.md is unusable: {exc}")
    if run_state is None:  # pragma: no cover - unreachable, as above
        return _no_plan(args.plan_dir, "has no state.md")

    print(f"plan-id:   {directory.plan_id}")
    print(f"plan-dir:  {directory.path}")
    print(f"state:     {run_state.state}")
    print(f"iteration: {run_state.iteration}")
    print(f"step:      {run_state.current_step or _NO_STEP}")
    print(
        f"attempts:  {run_state.fix_attempts} of {constants.Defaults.MAX_FIX_ATTEMPTS}"
    )
    print(f"artifacts: {', '.join(sorted(directory.list_dir())) or '(none)'}")

    # DECISION plan-2026-07-21T191807-bf7ffe24/D-043
    # The gate is re-asked with the plan's OWN state as `expected_state`, not
    # with `pre_step_gate`'s EXECUTE default. Do NOT "simplify" this to the
    # single default-argument call above: `wrong-state` compares `state.md`
    # against the state the CALLER expects, so a healthy plan sitting in EXPLORE
    # would fail that comparison and `status` would exit 2 -- the RESERVED code
    # for "the protocol blocked this step" -- for a plan that is blocking
    # nothing. Asking with the recorded state makes `wrong-state` structurally
    # unreachable here (it is the dispatcher's question, not the reporter's)
    # while leaving `leash-cap` and `iteration-cap` fully live, which is the
    # honest reading of `status`: those two really do mean no further step can
    # be dispatched. The first call is still needed and is not redundant -- it
    # is what answers `no-plan` without constructing a `PlanDirectory`, whose
    # `PlanMemory` would CREATE the missing directory it was asked about.
    # See decisions.md D-043.
    gate = validator.pre_step_gate(args.plan_dir, expected_state=run_state.state)
    print(f"gate:      {gate}")
    exit_code: int = gate.exit_code
    return exit_code


def _cmd_validate(args: argparse.Namespace) -> int:
    """Audit a plan directory and report every issue."""
    validator = _import("plan_validator")
    issues = validator.audit(args.plan_dir, workspace_root=args.workspace)
    print(f"audit {args.plan_dir} ({len(issues)} issues):")
    errors = _print_issues(issues)[0]
    # WARN- and INFO-severity findings are NOT failures. `audit()`'s own
    # contract calls them retrospective advice, and a real plan directory
    # carries a dozen of them while being perfectly runnable; exiting non-zero
    # on those would make the command useless as a gate.
    return EXIT_ERROR if errors else EXIT_PASS


def _cmd_close(args: argparse.Namespace) -> int:
    """Audit, then report (``--apply``: perform) the CLOSE-phase size policies."""
    constants = _import("constants")
    storage = _import("storage")
    validator = _import("plan_validator")

    issues = validator.audit(args.plan_dir, workspace_root=args.workspace)
    print(f"audit {args.plan_dir} ({len(issues)} issues):")
    if _print_issues(issues)[0]:
        print(
            "close REFUSED: the plan directory has ERROR-severity findings. "
            "Protocol memory is not compressed while it is broken -- fix the "
            "errors, or run `validate` to see them again.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    path = _existing_dir(args.plan_dir)
    if path is None:  # pragma: no cover - audit already errors on an absent dir
        return _no_plan(args.plan_dir, "is not an existing directory")

    # DECISION plan-2026-07-21T191807-bf7ffe24/D-044
    # This directory is opened as the ARCHIVIST, and `--apply` is what makes it
    # write anything at all. Both halves are load-bearing.
    # (1) The ROLE: `rules.OWNERSHIP` gives LESSONS.md, SYSTEM.md, FINDINGS.md
    # and DECISIONS.md to `Role.ARCHIVIST` ALONE. Opened with this class's
    # `Role.ORCHESTRATOR` default the dry run would still print (reads are
    # confinement-checked, not ownership-checked) and only `--apply` would fail,
    # with a `HarnessOwnershipError` from inside a size policy -- a refusal at
    # the worst possible moment rather than the right role from the start.
    # (2) The DRY RUN: eviction and the sliding window DELETE content from the
    # cross-plan tier. `evict_lessons` / `apply_sliding_window` are pure
    # `(doc) -> (trimmed, report)` functions and `PlanDirectory`'s methods are
    # the writing ones, so reporting first costs one extra parse and nothing
    # else. Do NOT make `--apply` the default "because CLOSE always compresses":
    # this is the protocol's only lossy operation on memory a human may have
    # written by hand, and it must be asked for.
    # See decisions.md D-044.
    directory = storage.PlanDirectory(path, role=constants.Role.ARCHIVIST)
    ok = _housekeeping(directory, apply=args.apply)
    if not args.apply:
        print("(dry run -- nothing was written; pass --apply to perform these)")
    return EXIT_PASS if ok else EXIT_ERROR


def _housekeeping(directory: Any, *, apply: bool) -> bool:
    """Report -- and with *apply*, perform -- the three CLOSE size policies.

    Interface contract (1 call site today, the ``close`` subcommand; kept
    separate because it is the only part of ``close`` that can WRITE):
        - Prints one line per policy.
        - Returns ``False`` if any policy could not be evaluated, so the caller
          fails closed rather than reporting a clean CLOSE it did not verify.
    """
    constants = _import("constants")
    storage = _import("storage")
    names = constants.ArtifactNames
    ok = True

    def _read(artifact: str) -> Any | None:
        nonlocal ok
        if not directory.exists(artifact):
            print(f"{artifact}: absent")
            return None
        try:
            return directory.read_artifact(artifact)
        except _failures() as exc:
            print(f"{artifact}: could not be read: {exc}", file=sys.stderr)
            ok = False
            return None

    lessons = _read(names.LESSONS)
    if lessons is not None:
        _, report = storage.evict_lessons(lessons)
        print(
            f"{names.LESSONS}: {report.lines_before} -> {report.lines_after} lines "
            f"(cap {report.cap}), {len(report.evicted)} bullets evicted"
        )
        if report.over_cap:
            print(
                f"{names.LESSONS}: STILL over cap after eviction -- protected "
                "[I:5] content alone exceeds it; that needs a human rewrite"
            )
        if apply and report.changed:
            directory.enforce_lessons_cap()

    atlas = _read(names.SYSTEM)
    if atlas is not None:
        report = storage.check_system_cap(atlas)
        state = "OVER CAP" if report.over_cap else "within cap"
        print(
            f"{names.SYSTEM}: {report.lines_before} lines, cap {report.cap} "
            f"({state}; never trimmed automatically)"
        )

    for artifact in storage.PlanDirectory.WINDOWED:
        doc = _read(artifact)
        if doc is None:
            continue
        _, window = storage.apply_sliding_window(doc)
        print(
            f"{artifact}: keep {window.keep}, "
            f"{len(window.kept_plans)} kept, {len(window.trimmed_plans)} trimmed"
            + (f" ({', '.join(window.trimmed_plans)})" if window.changed else "")
        )
        if apply and window.changed:
            directory.apply_sliding_window(artifact)
    return ok


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose usage errors exit 1, not argparse's 2."""

    # DECISION plan-2026-07-21T191807-bf7ffe24/D-042
    # Do NOT delete this class and use a plain `argparse.ArgumentParser`.
    # `ArgumentParser.error()` calls `self.exit(2, ...)`, and 2 is this CLI's
    # RESERVED code for a HARD `pre_step_gate` failure -- the source protocol's
    # own reserved contract, which `ip-orchestrator.md:139` forbids downgrading
    # to advisory. With stock argparse a typo'd flag and "the autonomy leash has
    # stopped this step" would be indistinguishable to any wrapper script, and
    # the wrong one of the two is the safe-looking one: a caller retrying on
    # "usage error" would silently retry past a leash cap.
    # Overriding `error` (rather than catching `SystemExit` at the call site)
    # is what covers every argparse-internal exit path -- unknown flag, missing
    # required argument, bad `type=`, invalid `choices` -- with one rule.
    # `--help` and `--version` are untouched: they exit 0 through `exit()`,
    # not through `error()`.
    # See decisions.md D-042.
    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR)


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    """Add the options every protocol-driving subcommand shares."""
    parser.add_argument(
        "--model",
        default=None,
        help="LLM to dispatch roles with (default: $LLM_MODEL, else the "
        "package default)",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Confined root the code-editing tools act on (default: .)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Interface contract (2 call sites: :func:`main_cli` and the tests):
        - Imports nothing from this package, so ``--help`` costs no model, no
          filesystem access and no optional dependency.
        - Every subparser sets ``func`` to its handler, so dispatch is a lookup
          rather than an if-chain.
    """
    parser = _Parser(
        prog=_PROG,
        description="Drive the iterative-planner protocol over a plan directory.",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    new = subparsers.add_parser(
        "new", help="Mint a plan directory and run the protocol"
    )
    new.add_argument("goal", help="What this run is for")
    new.add_argument(
        "--plans-dir",
        default="plans",
        help="Where plan directories live (default: plans)",
    )
    new.add_argument(
        "--create-only",
        action="store_true",
        help="Mint and seed the plan directory, then stop (no LLM is called)",
    )
    _add_model_options(new)
    new.set_defaults(func=_cmd_new)

    resume = subparsers.add_parser("resume", help="Continue an existing plan directory")
    resume.add_argument("plan_dir", help="The plan directory to resume")
    resume.add_argument(
        "--goal", default=None, help="Override the goal recorded in plan.md"
    )
    _add_model_options(resume)
    resume.set_defaults(func=_cmd_resume)

    status = subparsers.add_parser("status", help="Report a plan directory's position")
    status.add_argument("plan_dir", help="The plan directory to report on")
    status.set_defaults(func=_cmd_status)

    validate = subparsers.add_parser("validate", help="Audit a plan directory")
    validate.add_argument("plan_dir", help="The plan directory to audit")
    validate.add_argument(
        "--workspace",
        default=None,
        help="Source tree to scan for decision anchors (default: no anchor scan)",
    )
    validate.set_defaults(func=_cmd_validate)

    close = subparsers.add_parser(
        "close", help="Audit, then apply the CLOSE size policies"
    )
    close.add_argument("plan_dir", help="The plan directory to close")
    close.add_argument(
        "--workspace",
        default=None,
        help="Source tree to scan for decision anchors (default: no anchor scan)",
    )
    close.add_argument(
        "--apply",
        action="store_true",
        help="Actually evict and trim; without it nothing is written",
    )
    close.set_defaults(func=_cmd_close)
    return parser


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fsm_llm_harness`` and ``fsm-llm-harness``.

    Interface contract:
        - ``argv``: arguments WITHOUT the program name; ``None`` reads
          ``sys.argv[1:]``.
        - Returns the process exit code (``setuptools``' console-script wrapper
          and the ``__main__`` guard below both pass it to ``sys.exit``).
        - Raises ``SystemExit`` only from argparse (``--help``, ``--version``,
          a usage error) and from the optional-dependency guard.
    """
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"{_PROG} {_import('__version__').__version__}")
        return EXIT_PASS
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if handler is None:
        # Fail CLOSED: no subcommand is not a successful no-op.
        build_parser().print_help(sys.stderr)
        return EXIT_ERROR
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main_cli())
