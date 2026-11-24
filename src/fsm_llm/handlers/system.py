"""``HandlerSystem``: registration and execution of handlers.

Package ``fsm_llm.handlers`` (plan d454fd62, step 77: split out of the old module
``fsm_llm/handlers.py``). One ``HandlerSystem`` serves every conversation of an ``API``.
"""

from __future__ import annotations

import contextvars
import copy
import math
import numbers
import threading
import traceback
from typing import Any

from fsm_llm.constants import MAX_TIMED_HANDLER_STRAGGLERS
from fsm_llm.logging import logger
from fsm_llm.runtime import (
    EventType,
    current_scope,
    emit,
    visible_entries,
    warn_on_failure,
)

from .errors import HandlerExecutionError
from .protocol import FSMHandler, HandlerErrorMode, HandlerTiming


class _HandlerTimedOut(TimeoutError):
    """The timeout or the straggler cap of a timed handler: the one ``TimeoutError`` the
    ``handler`` event calls ``timeout`` (a handler's own ``TimeoutError`` is an ``error``)."""


# DECISION plan-2026-11-24T193347-3f4530d3/D-008
# The `handler` event is written HERE, by the CALLER's thread after the handler returned, raised
# or was joined, never by the worker of a timed handler: a straggler that returns late must write
# nothing (it could reopen a released writer). It is evidence and best effort (a failed write is a
# WARNING, the handler outcome and the order of handlers are untouched), carries NAMES only (the
# handler's name, its timing, the keys it wrote, never a value) and no fsync. Do NOT write it from
# the worker, add the handler's data, or let a write failure reach the turn.
def _record_handler(
    name: str,
    timing: HandlerTiming,
    error: Exception | None,
    result: dict[str, Any] | None = None,
) -> None:
    """Append ``handler``: ``outcome`` ``ok``, ``error`` or ``timeout`` and the keys written."""
    if current_scope() is None:
        return
    with warn_on_failure(EventType.HANDLER):
        fields: dict[str, Any] = {"name": name, "timing": timing.value, "outcome": "ok"}
        if isinstance(error, _HandlerTimedOut):
            fields["outcome"] = "timeout"
        elif error is not None:
            fields.update(outcome="error", error_type=type(error).__name__)
        elif isinstance(result, dict):
            fields["keys"] = sorted(visible_entries(result))
        emit(EventType.HANDLER, **fields)


class HandlerSystem:
    """
    Central orchestrator for executing custom functions during FSM execution.

    This class manages the registration and execution of handlers, providing error
    handling and context management.

    One ``HandlerSystem`` serves every conversation of an ``API``, so the
    timed-handler straggler cap (``constants.MAX_TIMED_HANDLER_STRAGGLERS``)
    is shared API-wide: stragglers left by one conversation can make a timed
    handler of another fail fast as a timeout.
    """

    def __init__(
        self,
        handler_error_mode: HandlerErrorMode | str = HandlerErrorMode.CONTINUE,
        handler_timeout: float | None = None,
    ):
        """
        Initialize the handler system with specified error handling behavior.

        :param handler_error_mode: How to handle errors during handler execution:
            a ``HandlerErrorMode`` or its value string (``"continue"`` or
            ``"raise"``), coerced once here
        :type handler_error_mode: HandlerErrorMode | str
        :param handler_timeout: Maximum seconds a handler may run before timeout.
            ``None`` disables timeout (default). Reachable through
            ``API(handler_timeout=...)``.
        :type handler_timeout: float | None
        :raises ValueError: If handler_error_mode is not one of: continue, raise
        """
        # Rebound, never mutated in place (copy-on-write, see register_handler).
        self.handlers: list[FSMHandler] = []
        self._registration_lock = threading.Lock()
        self.handler_error_mode = HandlerErrorMode.from_string(handler_error_mode)
        self.handler_timeout = handler_timeout
        # Timed-handler threads still running after their timeout (D-049).
        self._stragglers: list[threading.Thread] = []
        self._stragglers_lock = threading.Lock()

    def register_handler(self, handler: FSMHandler) -> None:
        """
        Register a new handler with the system and maintain priority ordering.

        :param handler: The handler instance to register
        :type handler: FSMHandler
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-019
        # Copy-on-write under a lock: build a NEW sorted list and rebind
        # `self.handlers` in one assignment. Do NOT go back to
        # `self.handlers.append(); self.handlers.sort()`: CPython empties a list
        # for the whole duration of an in-place sort, so a concurrent reader
        # (`handlers_at`, every turn) saw zero handlers and skipped them (C1).
        # Readers take one reference without the lock; do NOT lock them.
        # C6: reject an unorderable priority HERE, naming this handler. Left to
        # `sorted()`, a string priority registered first was accepted and the
        # TypeError surfaced on the NEXT (innocent) registration; a NaN never
        # raised and silently scrambled the order.
        priority = getattr(handler, "priority", 100)
        if isinstance(priority, bool) or not isinstance(priority, numbers.Real):
            raise TypeError(
                f"Handler {getattr(handler, 'name', handler)!r}: priority must be "
                f"a real number, got {type(priority).__name__}"
            )
        if math.isnan(priority):
            raise ValueError(
                f"Handler {getattr(handler, 'name', handler)!r}: priority is NaN"
            )
        with self._registration_lock:
            self.handlers = sorted(
                [*self.handlers, handler], key=lambda h: getattr(h, "priority", 100)
            )

    def handlers_at(self, timing: HandlerTiming) -> list[FSMHandler]:
        """Return the registered handlers that subscribe to ``timing``.

        Contract: pure and cheap (no ``should_execute`` call, no copying);
        returns a new list in priority order. A handler with no ``timings``
        attribute, or ``timings is None``, subscribes to every timing. An empty
        result means ``execute_handlers`` would do nothing at this timing.
        """
        registered = self.handlers  # one snapshot reference (D-019)
        return [
            h
            for h in registered
            if not hasattr(h, "timings") or h.timings is None or timing in h.timings
        ]

    def execute_handlers(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute all qualifying handlers at the specified timing point.

        Properly cascade context updates between handlers, ensuring that each handler
        sees the cumulative changes made by previous handlers.

        :param timing: The lifecycle hook point being executed
        :type timing: HandlerTiming
        :param current_state: Current state identifier of the FSM
        :type current_state: str
        :param target_state: Target state identifier (None if not transitioning)
        :type target_state: str | None
        :param context: Current context data dictionary
        :type context: dict[str, Any]
        :param updated_keys: Set of context keys being updated (for CONTEXT_UPDATE timing)
        :type updated_keys: set[str] | None
        :return: Dictionary containing all context updates from executed handlers
        :rtype: dict[str, Any]
        """
        output_context: dict[str, Any] = {}
        candidates = self.handlers_at(timing)
        if not candidates:
            return output_context

        # DECISION plan-2026-09-20T114608-a8e47b88/D-020
        # `copy.deepcopy(context)` used to run unconditionally, before checking
        # whether ANY handler at this timing would actually pass its own
        # `should_execute()` filter for the current state/target/updated_keys.
        # `should_execute()` never mutates `context` -- confirmed by reading
        # both implementations in this module, not assumed: `BaseHandler`'s
        # default always returns `False` without touching `context`, and
        # `LambdaHandler.should_execute` (the only other implementation here)
        # only ever READS `context` -- directly (`required_keys`/`updated_keys`
        # membership checks) and through user-supplied `condition_lambdas`,
        # which this system's documented contract treats as pure predicates,
        # same as every other condition check in that method. It is therefore
        # safe to probe every candidate's `should_execute()` against the
        # UNCOPIED `context` until the FIRST one that will actually run is
        # found, and pay for the deep copy exactly once, only then. Every
        # handler probed before that point still sees the same original
        # `context` a zero-registered-handler timing already skips entirely
        # via the `candidates` guard above (which complements, and does not
        # replace, the caller-side D-022 optimization in
        # `MessagePipeline.execute_handlers`, `engine/turn/message_pipeline.py`). Do NOT
        # deep-copy per-candidate inside this probe loop -- that would
        # reintroduce the exact cost removed here, just paid earlier and
        # more often.
        updated_context: dict[str, Any] | None = None

        def _report_and_maybe_raise(
            handler_name: str, handler: FSMHandler, exc: Exception
        ) -> None:
            """Log + apply the configured handler_error_mode for a failure at ``handler``.

            Shared by the ``should_execute()`` probe and the actual handler
            execution below so both failure points get identical error-mode
            semantics without duplicating the raise/continue decision twice.
            Raises ``HandlerExecutionError`` (critical or handler_error_mode="raise");
            otherwise returns normally (handler_error_mode="continue").
            """
            # C6: a condition lambda's failure arrives already wrapped by
            # `LambdaHandler.should_execute`; wrapping it again doubled the
            # message and hid the real cause behind `original_error`.
            if isinstance(exc, HandlerExecutionError):
                error = exc
            else:
                error = HandlerExecutionError(handler_name, exc)
            logger.error(f"{error!s}\n{traceback.format_exc()}")
            is_critical = getattr(handler, "critical", False)
            if self.handler_error_mode is HandlerErrorMode.RAISE or is_critical:
                error.partial_context = dict(output_context)
                if error is exc:
                    raise error
                raise error from exc

        # Execute applicable handlers in priority order (lower priority numbers first)
        for handler in candidates:
            handler_name = getattr(handler, "name", handler.__class__.__name__)
            probe_context = context if updated_context is None else updated_context

            try:
                # Check if this handler should execute based on current conditions
                should_run = handler.should_execute(
                    timing, current_state, target_state, probe_context, updated_keys
                )
            except Exception as e:
                _report_and_maybe_raise(handler_name, handler, e)
                continue  # handler_error_mode == "continue": move to the next handler

            if not should_run:
                continue

            # DECISION plan-2026-09-20T114608-a8e47b88/D-025
            # copy.deepcopy(context) now runs OUTSIDE the handler-EXECUTION
            # try/except below (D-020 originally placed it inside that try,
            # alongside should_execute()). should_execute() has ALREADY
            # confirmed this handler will run by this point, so D-020's
            # "defer the copy until the first qualifying handler" benefit is
            # unchanged -- this only moves WHERE the copy is attempted, not
            # WHEN. A non-deep-copyable context value (e.g. a live
            # threading.Lock a caller left in context) is a caller bug, not a
            # handler-execution failure: pre-D-020 it raised a bare
            # TypeError straight out of execute_handlers. With the copy
            # inside the try, the default handler_error_mode="continue" silently
            # swallowed it (empty dict, handler never ran, no exception) and
            # handler_error_mode="raise" misattributed it to a handler that was never
            # actually invoked. Do NOT move this back inside the try below --
            # that reintroduces both regressions. See decisions.md D-025.
            if updated_context is None:
                updated_context = copy.deepcopy(context)

            logger.debug(f"Executing handler {handler_name} at {timing.name}")

            try:
                # Execute the handler with optional timeout
                result = self._execute_single_handler(
                    handler, updated_context, handler_name
                )
                _record_handler(handler_name, timing, None, result)

                # Update context with handler result if valid
                if isinstance(result, dict):
                    updated_context.update(result)
                    output_context.update(result)
                elif result is not None:
                    logger.warning(
                        f"Handler {handler_name} returned a non-dict result "
                        f"({type(result).__name__}); it was ignored"
                    )

                logger.debug(f"Handler {handler_name} completed successfully")

            except Exception as e:
                _record_handler(handler_name, timing, e)
                _report_and_maybe_raise(handler_name, handler, e)
                continue  # handler_error_mode == "continue": move to the next handler

        return output_context

    def _execute_single_handler(
        self, handler: FSMHandler, context: dict[str, Any], handler_name: str
    ) -> dict[str, Any] | None:
        """Execute a single handler, optionally with timeout protection.

        When ``handler_timeout`` is set, the handler runs on a private copy of
        ``context`` in a daemon thread joined with the timeout. A handler that
        finishes in time (returning or raising) has its in-place writes to the
        copy adopted back into ``context``, exactly as if it had run untimed.
        On timeout its result and writes are discarded and ``TimeoutError`` is
        raised; the thread keeps running until the handler returns, but only
        ever touches its own copy. While ``MAX_TIMED_HANDLER_STRAGGLERS`` such
        threads are still alive (counted across every conversation this
        system serves), a new timed call raises ``TimeoutError`` at once
        without starting a thread.
        """
        if self.handler_timeout is None:
            return handler.execute(context)

        # DECISION plan-2026-09-21T203800-8a03483a/D-049
        # Bound the timed-out threads that are still running. Do NOT drop this
        # check to "just start another daemon thread": a handler hung on a
        # network call or a lock leaked one thread per turn for the life of
        # the process (review W3). Do NOT count in-flight threads that have not
        # timed out: one HandlerSystem serves every conversation of an API, so
        # concurrent healthy turns must never be refused.
        with self._stragglers_lock:
            self._stragglers = [t for t in self._stragglers if t.is_alive()]
            straggling = len(self._stragglers)
        if straggling >= MAX_TIMED_HANDLER_STRAGGLERS:
            logger.warning(
                f"Handler '{handler_name}' not started: {straggling} timed-out "
                f"handler threads are still running (cap "
                f"MAX_TIMED_HANDLER_STRAGGLERS={MAX_TIMED_HANDLER_STRAGGLERS})"
            )
            raise _HandlerTimedOut(
                f"Handler '{handler_name}' refused: {straggling} timed-out "
                f"straggler threads still running (cap "
                f"{MAX_TIMED_HANDLER_STRAGGLERS})"
            )

        # DECISION plan-2026-09-21T203800-8a03483a/D-019
        # A timed handler gets its OWN deep copy and runs in a per-call DAEMON
        # thread. Do NOT hand it the shared `context` (a timed-out straggler
        # kept writing into the dict the next handler was reading, C2), and do
        # NOT go back to a ThreadPoolExecutor: its workers are non-daemon, so a
        # straggler blocked interpreter exit. Deepcopy failure falls back to a
        # shallow copy (top-level writes stay isolated) with a WARNING.
        try:
            private = copy.deepcopy(context)
        except Exception as exc:
            logger.warning(
                f"Handler '{handler_name}': context not deep-copyable ({exc!s}); "
                f"running on a shallow copy"
            )
            private = dict(context)

        outcome: dict[str, Any] = {}

        def _run() -> None:
            try:
                outcome["result"] = handler.execute(private)
            except BaseException as exc:  # re-raised on the caller's thread
                outcome["error"] = exc

        # The worker runs in a copy of the caller's context, so a ContextVar the
        # caller set (a monitor handler's states, F-HV-01) is visible to it.
        worker = threading.Thread(
            target=contextvars.copy_context().run,
            args=(_run,),
            name=f"fsm-handler-{handler_name}",
            daemon=True,
        )
        worker.start()
        worker.join(self.handler_timeout)
        if worker.is_alive():
            with self._stragglers_lock:
                self._stragglers.append(worker)
            raise _HandlerTimedOut(
                f"Handler '{handler_name}' timed out after {self.handler_timeout}s"
            )
        # D-049: the thread is done, so `private` is final. Adopt it wholesale
        # (sets, nested edits and deletes), so turning a timeout on does not
        # change what later handlers see. Do NOT adopt before the join says
        # the thread finished: a straggler's writes must stay invisible (C2).
        context.clear()
        context.update(private)
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("result")
