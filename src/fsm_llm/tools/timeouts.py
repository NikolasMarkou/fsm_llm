"""The per-call timeout of a tool: the one predicate and the daemon-thread wait."""

from __future__ import annotations

import contextvars
import math
import numbers
import operator
import threading
from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from .constants import Defaults, ErrorMessages, LogMessages
from .errors import ToolExecutionError
from .limits import bounded_repr


class ToolTimedOut(ToolExecutionError):
    """``call_with_timeout`` stopped waiting: the call's outcome is unknown."""


# DECISION plan-2026-10-01T093600-944e2692/D-008: `timeout_seconds` is enforced
# here, in the one execute path, with a daemon worker thread (supersedes D-026
# of plan 65baa765, whose runtime enforcement point no longer exists). Do NOT
# hold `_tools_lock` while waiting, do NOT run a tool without `timeout_seconds` in a
# thread (every call would pay one), and do NOT claim the tool was stopped: a
# Python thread cannot be killed, so a timed-out tool keeps running (one more
# live thread per timed-out call; a tool that keeps hanging accumulates them).
# DECISION plan-2026-10-01T093600-944e2692/D-033: every step that can fail
# runs BEFORE `thread.start()`; do NOT add one after it (a tool that ran must
# never be reported as a plain failure, and a retry-safe one would be re-run),
# and do NOT report a timeout as a failure: its outcome is unknown
# (`ToolResult.timed_out`).
def check_timeout(tool_name: str, timeout_seconds: Any) -> None:
    """Raise ``ToolExecutionError`` (``TOOL_BAD_TIMEOUT``) unless *timeout_seconds* is
    a finite number in ``(0, Defaults.MAX_TOOL_TIMEOUT_S]`` (a bool is not one).

    Interface contract: the one timeout predicate, shared by
    ``call_with_timeout`` (every timed call) and ``HumanInTheLoop.__init__``
    (so a bad ``approval_timeout`` fails at construction). Returns ``None``.
    A number is a ``float`` or a ``numbers.Integral`` (an ``int``, ``np.int64``):
    what ``thread.join(timeout=...)`` took before this check existed.
    ``Decimal``, ``Fraction`` and ``np.float32`` were a ``TypeError`` there and
    stay refused. An integer is compared as an integer, so ``10**400`` is
    refused, not an ``OverflowError``.
    """
    number = (
        operator.index(timeout_seconds)
        if isinstance(timeout_seconds, numbers.Integral)
        and not isinstance(timeout_seconds, bool)
        else timeout_seconds
    )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(number, (int, float))
        or not (isinstance(number, int) or math.isfinite(number))
        or not 0 < number <= Defaults.MAX_TOOL_TIMEOUT_S
    ):
        raise ToolExecutionError(
            ErrorMessages.TOOL_BAD_TIMEOUT.format(
                name=tool_name,
                timeout=bounded_repr(timeout_seconds),
                limit=Defaults.MAX_TOOL_TIMEOUT_S,
            ),
            tool_name=tool_name,
        )


def call_with_timeout(
    call: Callable[[], Any], tool_name: str, timeout_seconds: float
) -> Any:
    """Run *call* in a daemon worker thread and wait at most *timeout_seconds* seconds.

    Interface contract (callers: ``ToolRegistry.execute``, for a tool with
    ``timeout_seconds``, and ``HumanInTheLoop.request_approval``):
        - Raises ``ToolExecutionError`` (``TOOL_BAD_TIMEOUT``) without running
          *call* when *timeout_seconds* is not a finite number in
          ``(0, Defaults.MAX_TOOL_TIMEOUT_S]``.
        - Returns ``call()``'s value, or re-raises its exception, when it ends
          in time. The worker runs in a copy of the caller's context variables.
        - On expiry raises ``ToolTimedOut`` (a ``ToolExecutionError``,
          ``TOOL_TIMED_OUT``: outcome unknown). The worker keeps running; its
          late value or exception is discarded with a WARNING naming the tool
          (``TOOL_LATE_RESULT``).
        - Holds no lock while waiting. Nothing it does after the worker starts
          can raise, except the timeout and the tool's own exception.
        - Also the timed wait of ``HumanInTheLoop.request_approval`` (the
          callback is the "tool"; a ``ToolTimedOut`` means denied).
    """
    check_timeout(tool_name, timeout_seconds)
    timed_out_error = ToolTimedOut(
        ErrorMessages.TOOL_TIMED_OUT.format(name=tool_name, timeout=timeout_seconds),
        tool_name=tool_name,
    )
    late_warning = {
        kind: LogMessages.TOOL_LATE_RESULT.format(
            name=tool_name, timeout=timeout_seconds, outcome=kind
        )
        for kind in ("result", "error")
    }
    finished = threading.Event()
    state_lock = threading.Lock()
    outcome: dict[str, Any] = {}

    def worker() -> None:
        # BaseException: the caller re-raises whatever the tool raised, as an
        # inline call would; nothing is swallowed while the caller still waits.
        try:
            entry: tuple[str, Any] = ("value", call())
        except BaseException as exc:
            entry = ("error", exc)
        with state_lock:
            abandoned = outcome.get("abandoned", False)
            outcome[entry[0]] = entry[1]
        finished.set()
        if abandoned:
            kind = "result" if entry[0] == "value" else "error"
            detail = (
                f": {type(entry[1]).__name__}: {entry[1]}" if kind == "error" else ""
            )
            logger.warning(late_warning[kind] + detail)

    thread = threading.Thread(
        target=contextvars.copy_context().run,
        args=(worker,),
        name=f"fsm-llm-tool-{tool_name}",
        daemon=True,
    )
    thread.start()
    finished.wait(timeout_seconds)
    with state_lock:
        if "value" not in outcome and "error" not in outcome:
            outcome["abandoned"] = True
            raise timed_out_error
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
