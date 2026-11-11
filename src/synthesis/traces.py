"""
Run traces: what a procedure run left behind, with no value in it.

A ``RunTrace`` holds names, type names, state ids, codes and counts and nothing
else (D-038): never a context value, user text or reply. ``TraceRecorder`` is
the runner's ``trace_sink``; ``trace_from_trial`` builds one from a G6b trial;
the library stores them (``FSMLibrary.add_trace``).

# DECISION plan-2026-11-10T194139-2cefeb61/D-046
# ``RunTrace.__post_init__`` is the ONE place that keeps a trace value-free (the
# recorder and the file reader both build through it). Do NOT add a constructor
# that skips it, put a value or ``type(v).__name__`` in a field, or let the
# recorder keep traces.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from fsm_llm.definitions import TransitionEvaluationResult
from fsm_llm.expressions import is_missing
from fsm_llm.okf import capped_line

from .constants import (
    ABORT_CONTEXT_KEY,
    ABORT_STATE_ID,
    REASON_BUDGET_SPENT,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_OK,
    TRACE_KIND_OTHER,
    TRACE_KINDS,
)

if TYPE_CHECKING:
    from fsm_llm.eval import TrialResult

TRACE_VERSION = 1
_MODES = ("run", "converse", "sim")
_STATUSES = (STATUS_OK, STATUS_ABORTED, STATUS_FAILED)
_OUTCOMES = (None, *(r.value for r in TransitionEvaluationResult))
_KIND_TYPES = (*TRACE_KINDS.values(), TRACE_KIND_OTHER)
_TEXT = 80  # characters of any stored string
_MAX_STATES, _MAX_NAMES = 100, 64
_MAX_OPEN = 256  # runs the recorder follows at once
_EVENTS = ("obtained", "pushed", "stepped", "aborted", "popped", "failed")


def _strs(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(v, str) for v in value
    ):
        raise TypeError("a trace holds names as a list of strings")
    return tuple(capped_line(v, _TEXT) for v in value)


@dataclass(frozen=True)
class RunTrace:
    """One run of one procedure, value-free.

    ``mode``: ``run`` or ``converse`` (the runner's drivers) or ``sim`` (a G6b
    trial of ``scenario`` kind); ``states``: the state after each counted step
    (the last 100); ``keys``: declared names present at the end; ``kinds``:
    ``(key, type name)`` for gate-read keys (``TRACE_KINDS`` names or
    ``other``); ``status``, ``reason``: the runner's result; ``stuck``: the
    state an unfinished run was in; ``outcome``: ``blocked``, ``ambiguous`` or
    ``deterministic`` there.

    Interface contract: the constructor raises ``TypeError`` or ``ValueError``
    for a wrong shape and caps what it keeps (strings 80 characters, keys and
    kinds 64); ``to_row()`` is a JSON-ready dict with ``"v": 1``; ``from_row``
    is its inverse and raises only those two errors, whatever the row holds.
    """

    digest: str
    version: str | None = None
    mode: str = "run"
    scenario: str | None = None
    states: tuple[str, ...] = ()
    steps: int = 0
    keys: tuple[str, ...] = ()
    kinds: tuple[tuple[str, str], ...] = ()
    status: str = STATUS_OK
    reason: str | None = None
    stuck: str | None = None
    outcome: str | None = None

    def __post_init__(self) -> None:
        def put(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        if not isinstance(self.digest, str):
            raise TypeError("a trace digest is a string")
        if self.mode not in _MODES or self.status not in _STATUSES:
            raise ValueError("unknown trace mode or status")
        if self.outcome not in _OUTCOMES:
            raise ValueError("unknown trace outcome")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps is a count")
        for name in ("version", "scenario", "reason", "stuck"):
            text = getattr(self, name)
            if text is not None and not isinstance(text, str):
                raise TypeError(f"{name} is a string or None")
            put(name, text if text is None else capped_line(text, _TEXT))
        put("states", _strs(self.states)[-_MAX_STATES:])
        put("keys", _strs(self.keys)[:_MAX_NAMES])
        if not isinstance(self.kinds, (tuple, list)) or not all(
            isinstance(p, (tuple, list))
            and len(p) == 2
            and isinstance(p[0], str)
            and p[1] in _KIND_TYPES
            for p in self.kinds
        ):
            raise ValueError("kinds are (key, type name) pairs")
        put(
            "kinds",
            tuple((capped_line(k, _TEXT), t) for k, t in self.kinds)[:_MAX_NAMES],
        )

    def to_row(self) -> dict[str, Any]:
        """A JSON-ready dict: the fields (tuples as lists) and ``"v": 1``."""
        return dict(json.loads(json.dumps({"v": TRACE_VERSION, **asdict(self)})))

    @classmethod
    def from_row(cls, row: object) -> RunTrace:
        """The trace of a ``to_row()`` dict; ``ValueError`` or ``TypeError`` for
        any other row (a wrong version, a missing or unknown key, a bad field)."""
        if not isinstance(row, dict) or row.get("v") != TRACE_VERSION:
            raise ValueError("not a version-1 trace row")
        return cls(**{k: v for k, v in row.items() if k != "v"})


class TraceRecorder:
    """The runner's ``trace_sink``: groups events by ``(conv, digest)`` and
    hands each finished run (``popped`` or ``failed``) to ``on_trace``.

    Interface contract: ``TraceRecorder(on_trace)`` (``TypeError`` if not
    callable), called as ``recorder(event, fields)``. It keeps no finished
    trace and follows at most 256 open runs (the oldest is dropped). ``mode``
    is ``converse`` when the ``pushed`` or ``failed`` event has no ``step``. A
    ``failed`` event with no digest yields nothing; an event it does not
    understand is ignored. What ``on_trace`` raises, or an invalid run,
    propagates (the runner swallows a sink's error) after the run is dropped.
    Not thread safe: one recorder per runner.
    """

    def __init__(self, on_trace: Callable[[RunTrace], None]) -> None:
        if not callable(on_trace):
            raise TypeError("on_trace must be callable")
        self._on_trace = on_trace
        self._open: dict[tuple[str, str], dict[str, Any]] = {}

    def __call__(self, event: str, fields: dict[str, Any]) -> None:
        conv, digest = fields.get("conv"), fields.get("digest")
        if (
            event not in _EVENTS
            or not isinstance(conv, str)
            or not isinstance(digest, str)
        ):
            return
        run = self._open.setdefault((conv, digest), {"states": []})
        if len(self._open) > _MAX_OPEN:
            self._open.pop(next(iter(self._open)))
        step, state = fields.get("step"), fields.get("state")
        if event in ("stepped", "aborted"):  # both carry the child's view
            if isinstance(fields.get("keys"), str):
                run["keys"] = tuple(k for k in fields["keys"].split(",") if k)
            if isinstance(fields.get("kinds"), str):
                run["kinds"] = tuple(
                    (k, t)
                    for k, _, t in (
                        p.rpartition(":") for p in fields["kinds"].split(",")
                    )
                    if k and t in _KIND_TYPES
                )
        if event == "obtained" and isinstance(fields.get("version"), str):
            run["version"] = fields["version"]
        elif event == "pushed":
            run["mode"] = "converse" if step is None else "run"
        elif event == "stepped":
            run["steps"] = step if isinstance(step, int) else run.get("steps", 0)
            if isinstance(state, str):
                run["states"].append(state)
        elif event == "aborted":
            run["stuck"], run["outcome"] = state, fields.get("outcome")
        elif event in ("popped", "failed"):
            del self._open[(conv, digest)]
            self._on_trace(
                RunTrace(
                    digest=digest,
                    version=run.get("version"),
                    mode=run.get("mode") or ("converse" if step is None else "run"),
                    states=tuple(run["states"]),
                    steps=run.get("steps", 0),
                    keys=run.get("keys", ()),
                    kinds=run.get("kinds", ()),
                    status=STATUS_FAILED
                    if event == "failed"
                    else str(fields.get("status")),
                    reason=fields.get("reason"),
                    stuck=run.get("stuck"),
                    outcome=run.get("outcome"),
                )
            )


def trace_from_trial(
    trial: TrialResult, digest: str, *, scenario: str, declared: Collection[str]
) -> RunTrace:
    """The ``sim`` trace of one G6b trial of the procedure *digest*.

    Only ``declared`` names (the procedure's inputs and return keys) that the
    trial's context holds are kept: a bulk extraction mints model-chosen names,
    so listing every key would let model text into a trace. ``kinds`` is empty
    (the trial knows no gates), the error text is never copied.
    """
    context = trial.context or {}
    aborted = trial.final_state == ABORT_STATE_ID or bool(
        context.get(ABORT_CONTEXT_KEY)
    )
    status, reason = (
        (STATUS_OK, None)
        if trial.passed
        else (STATUS_FAILED, "error")
        if trial.error
        else (STATUS_ABORTED, REASON_BUDGET_SPENT)
        if aborted
        else (STATUS_FAILED, "check_failed")
    )
    return RunTrace(
        digest=digest,
        mode="sim",
        scenario=scenario,
        states=tuple(trial.visited_states or ()),
        steps=trial.turns_sent,
        keys=tuple(sorted(k for k in declared if not is_missing(context, k))),
        status=status,
        reason=reason,
        stuck=None if trial.ended else trial.final_state,
    )
