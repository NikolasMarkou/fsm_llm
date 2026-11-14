"""The step driver of ``WorkflowEngine`` (plan d454fd62 step 91; the old ``workflows/engine.py``, same dotted path).

``DriverMixin``: the loop that runs an instance's steps back to back (``_execute_workflow_step``, a loop with
a per-call step budget, not recursion), the run of one step, the handling of a step result and the
transition onto the next step or into a wait. The method bodies are the old ones, moved.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fsm_llm.constants import has_internal_prefix
from fsm_llm.logging import logger

from ..constants import (
    CONTEXT_KEY_CANCELLATION_REASON,
    CONTEXT_KEY_TIMER_INFO,
    CONTEXT_KEY_WAITING_INFO,
    STEP_INTERNAL_WHITELIST,
)
from ..definitions import WorkflowDefinition
from ..exceptions import (
    WorkflowStateError,
    WorkflowStepError,
    WorkflowTimeoutError,
)
from ..models import (
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStepResult,
)
from ..steps.helpers import _STEP_EXECUTOR
from .helpers import _timeout_error
from .surface import _EngineSurface


class DriverMixin(_EngineSurface):
    """The step loop, the run of one step and the transitions of ``WorkflowEngine``."""

    async def _execute_workflow_step(self, instance: WorkflowInstance) -> None:
        """Run ``instance`` from its current step until it pauses, ends or fails.

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-002
        # This is a LOOP with a per-call step budget, not recursion: do NOT
        # go back to `_transition_to_state` -> `_execute_workflow_step`
        # recursion with a depth cap. The old cap (20) failed legitimate
        # acyclic chains longer than 20 steps with an "infinite loop" error,
        # and every exception unwound (and was logged) once per level.

        Step exceptions FAIL the instance and are not raised, except
        ``WorkflowTimeoutError`` (workflow deadline), which is re-raised after
        the instance is FAILED.
        """
        steps_run = 0
        while True:
            # DECISION plan-2026-11-05T134105-5ed71cf2/D-017
            # A cancel is honoured HERE, between steps, not by cancelling the
            # running asyncio task (that would interrupt user code mid-step and
            # leave resources half-released) and not through a second lock in
            # an inner method (089d0ec7/D-003: locks only at outermost entry
            # points). `cancel` cannot get the instance lock until
            # this whole chain returns, so it registers a request first; a
            # step already in flight runs to completion. Do NOT make
            # `cancel` take the lock first or call update_status
            # (CANCELLED) on a terminal instance (089d0ec7/D-015).
            request = self._cancel_requests.get(instance.instance_id)
            if request is not None and not instance.is_terminal():
                instance.context[CONTEXT_KEY_CANCELLATION_REASON] = request.reason
                self._set_status(instance, WorkflowStatus.CANCELLED)
                request.honored = True
                return

            if steps_run >= self.max_steps:
                error = WorkflowStateError(
                    f"Step budget exceeded: {self.max_steps} steps in "
                    f"one engine call for instance {instance.instance_id} "
                    "(a custom step may be routing in a loop)",
                    current_step_id=instance.current_step_id,
                    operation="execute",
                )
                logger.error(str(error))
                self._fail(instance, error)
                return

            next_step_id = await self._run_current_step(instance)
            steps_run += 1
            if next_step_id is None:
                return
            try:
                self._prepare_transition(instance, next_step_id)
            except Exception as e:
                logger.error(f"Error executing step {instance.current_step_id}: {e!s}")
                self._fail(instance, e)
                return

    async def _run_current_step(self, instance: WorkflowInstance) -> str | None:
        """Execute the instance's current step once.

        Returns the state to transition to next, or ``None`` when the run
        stops here (paused, completed, failed).
        """

        # Report the configured max_seconds (not deadline-minus-created_at,
        # which is inflated by the construction-to-start gap). Fall back to the
        # deadline span only if the timeout was not recorded.
        # DECISION plan-2026-09-12T065608-089d0ec7/D-011
        # Do NOT re-introduce `int(...)` truncation here: a sub-second
        # `max_seconds`/deadline span (e.g. 0.5s) would report as `0` in
        # `WorkflowTimeoutError`'s message, which is actively misleading (the
        # `asyncio.wait_for` call below already receives the correct float
        # timeout — only this reporting helper was truncating). See D-011.
        def _step_timeout() -> WorkflowTimeoutError:
            return _timeout_error(
                f"step {instance.current_step_id}",
                self._timeout_seconds(instance),
                instance.instance_id,
            )

        # Check workflow-level timeout (cheap fast-path at the step boundary)
        if instance.deadline is not None and datetime.now(UTC) > instance.deadline:
            error = _step_timeout()
            self._fail(instance, error)
            raise error
        try:
            # Get workflow definition and current step
            workflow_def = self._definition_for(instance)
            current_step = self._get_current_step(
                workflow_def, instance.current_step_id
            )
            step_id = instance.current_step_id

            logger.info(f"Executing step: {step_id} (instance: {instance.instance_id})")
            self._emit("step_started", instance, step_id=step_id)

            # Execute the step. When a workflow-level deadline is set, bound the
            # step itself by the remaining budget so a single long step cannot
            # run unbounded past the deadline.
            token = _STEP_EXECUTOR.set(self.executor)
            try:
                if instance.deadline is not None:
                    remaining = (instance.deadline - datetime.now(UTC)).total_seconds()
                    if remaining <= 0:
                        raise _step_timeout()
                    try:
                        result = await asyncio.wait_for(
                            current_step.execute(instance.context), timeout=remaining
                        )
                    except TimeoutError as e:
                        raise _step_timeout() from e
                else:
                    result = await current_step.execute(instance.context)
            finally:
                _STEP_EXECUTOR.reset(token)

            if not isinstance(result, WorkflowStepResult):
                raise WorkflowStepError(
                    f"execute() returned {type(result).__name__}, "
                    "expected a WorkflowStepResult",
                    step_id=step_id,
                )

            # Update context and history (filter internal keys to prevent overwrites).
            # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
            # TWO layers, and both are load-bearing:
            #   1. `has_internal_prefix` is the canonical predicate. Do NOT
            #      re-inline `k.startswith("_")` -- it is case-sensitive and
            #      blind to `system_`/`internal_`/`__`, which is the F-13 leak.
            #   2. `STEP_INTERNAL_WHITELIST` is a DELIBERATE override on top of
            #      layer 1. Do NOT drop it while "simplifying" the predicate:
            #      _waiting_info and _timer_info must reach the context or
            #      _handle_step_without_transition can no longer detect
            #      waiting/timer steps (invariant I-10). See decisions.md D-003.
            if result.data:
                filtered_data = {
                    k: v
                    for k, v in result.data.items()
                    if not has_internal_prefix(k) or k in STEP_INTERNAL_WHITELIST
                }
                instance.context.update(filtered_data)

            instance.add_history_entry(
                step_id=step_id,
                message=result.message or "",
                data=result.data,
                error=result.error,
            )
            self._emit(
                "step_completed",
                instance,
                step_id=step_id,
                success=result.success,
                next_step_id=result.next_step_id,
                error=result.error,
            )

            # Handle the result
            if result.success:
                return await self._handle_successful_step(instance, result)
            return self._handle_failed_step(instance, result)

        except Exception as e:
            logger.error(f"Error executing step {instance.current_step_id}: {e!s}")
            self._fail(instance, e)
            # Propagate workflow-level timeouts to the caller, matching the
            # behavior of the step-boundary deadline check above.
            if isinstance(e, WorkflowTimeoutError):
                raise
            return None

    @staticmethod
    def _timeout_seconds(instance: WorkflowInstance) -> float:
        if instance.max_seconds is not None:
            return float(instance.max_seconds)
        if instance.deadline is not None:
            return (instance.deadline - instance.created_at).total_seconds()
        return 0.0

    def _get_current_step(self, workflow_def: WorkflowDefinition, step_id: str):
        """Get the current step, raising an error if not found."""
        if step_id not in workflow_def.steps:
            raise WorkflowStateError(
                "Step not found in workflow definition",
                current_step_id=step_id,
                operation="get_step",
            )
        return workflow_def.steps[step_id]

    async def _handle_successful_step(
        self, instance: WorkflowInstance, result: WorkflowStepResult
    ) -> str | None:
        """Handle a successful step execution; return the next state or None."""
        if result.next_step_id:
            return result.next_step_id
        # DECISION plan-2026-09-12T135914-45a654de/D-013
        # A step (e.g. SwitchStep) can signal "this specific route is
        # terminal" by returning next_step_id="" -- distinct from a step
        # that has no next_step_id field at all (also falsy, e.g. a step
        # awaiting an event/timer, whose model default is None). Do NOT
        # re-derive terminality from WorkflowDefinition.get_terminal_states()
        # here: that is a static, WHOLE-STEP analysis (used for
        # reachability warnings) and cannot see that only ONE of a
        # SwitchStep's cases routes to "terminal". This applies to ANY
        # step type whose result carries next_step_id == "" -- not just
        # SwitchStep -- since ConditionStep, APIStep,
        # LLMStep, ParallelStep and RetryStep-wrapped steps all
        # share the same per-route "" == terminal semantics, and
        # definitions.py's own get_terminal_states() already treats ""
        # as a plan-wide terminal-step marker (see
        # `referenced_step_ids.discard("")` in definitions.py).
        #
        # DECISION plan-2026-09-12T135914-45a654de/D-020
        # A prior completion-fix (D-018) narrowed this check to
        # `isinstance(step, SwitchStep)`. A second adversarial review
        # pass found that narrowing REOPENED this exact bug for a
        # RetryStep-wrapped SwitchStep plus 4 other step types. Do NOT
        # reintroduce an isinstance-based narrowing of this predicate --
        # keep the wide `result.next_step_id == ""` check for ANY
        # successful step result. See decisions.md D-020.
        explicitly_terminal = result.next_step_id == ""
        return await self._handle_step_without_transition(
            instance, explicitly_terminal=explicitly_terminal
        )

    def _handle_failed_step(
        self, instance: WorkflowInstance, result: WorkflowStepResult
    ) -> str | None:
        """Handle a failed step result: follow its route or FAIL the instance."""
        logger.warning(f"Step failed: {instance.current_step_id} - {result.message}")

        if result.next_step_id:
            return result.next_step_id
        message = result.message or "Step failed without error message"
        if result.error and result.error not in message:
            message = f"{message} ({result.error})"
        error = WorkflowStepError(message, step_id=instance.current_step_id)
        self._fail(instance, error)
        return None

    def _prepare_transition(
        self, instance: WorkflowInstance, next_step_id: str
    ) -> None:
        """Validate ``next_step_id`` and move the instance onto it (RUNNING)."""
        workflow_def = self._definition_for(instance)

        if next_step_id not in workflow_def.steps:
            raise WorkflowStateError(
                f"Invalid next state: {next_step_id}",
                current_step_id=instance.current_step_id,
                operation="transition",
            )

        logger.info(f"Transitioning from {instance.current_step_id} to {next_step_id}")
        instance.current_step_id = next_step_id
        self._leave_wait(instance, WorkflowStatus.RUNNING)

    def _leave_wait(self, instance: WorkflowInstance, status: WorkflowStatus) -> None:
        """End the instance's wait and move it to ``status``.

        The one exit from a wait, onto the next step (RUNNING) or out of the
        workflow (COMPLETED): drop the instance's listeners and wait timers
        (not its deadline watchdog) so a stale listener cannot wake it later
        at a different step, clear both wait markers, then set the status.
        """
        self._drop_wait_resources(instance.instance_id)
        instance.context.pop(CONTEXT_KEY_WAITING_INFO, None)
        instance.context.pop(CONTEXT_KEY_TIMER_INFO, None)
        self._set_status(instance, status)

    async def _transition_to_state(
        self, instance: WorkflowInstance, next_step_id: str
    ) -> None:
        """Transition to ``next_step_id`` and run from there.

        Raises ``WorkflowStateError`` for an unknown state (the instance is
        unchanged) and ``WorkflowTimeoutError`` for a deadline breach.
        """
        self._prepare_transition(instance, next_step_id)
        await self._execute_workflow_step(instance)

    async def _handle_step_without_transition(
        self, instance: WorkflowInstance, explicitly_terminal: bool = False
    ) -> str | None:
        """Handle a step that doesn't specify a next state.

        Returns a state to continue with (a buffered event satisfied a wait)
        or ``None``.

        Args:
            instance: The workflow instance whose current step just ran.
            explicitly_terminal: True when the step's OWN result signalled
                termination for this invocation (``next_step_id == ""``, e.g. a
                ``SwitchStep`` routed to its terminal branch). When True, the
                instance completes unconditionally, without consulting
                ``WorkflowDefinition.get_terminal_states()``'s static
                whole-step analysis (see D-013 in decisions.md).
        """
        waiting_info = instance.context.get(CONTEXT_KEY_WAITING_INFO) or {}
        timer_info = instance.context.get(CONTEXT_KEY_TIMER_INFO) or {}

        if explicitly_terminal:
            logger.info(
                f"Workflow instance {instance.instance_id} completed successfully "
                "(step signalled explicit terminal route)"
            )
            self._set_status(instance, WorkflowStatus.COMPLETED)
        elif isinstance(waiting_info, dict) and waiting_info.get("waiting_for_event"):
            event_type = waiting_info.get("event_type") or ""
            success_step_id = waiting_info.get("success_step_id") or ""
            correlation_key = waiting_info.get("correlation_key")
            buffered = self._take_buffered_event(instance, event_type, correlation_key)
            if buffered is not None:
                logger.info(
                    f"Workflow instance {instance.instance_id} consumed a buffered "
                    f"'{event_type}' event"
                )
                self._apply_event(instance, waiting_info.get("event_mapping"), buffered)
                if success_step_id:
                    return success_step_id
                self._leave_wait(instance, WorkflowStatus.COMPLETED)
                return None
            logger.info(
                f"Workflow instance {instance.instance_id} is waiting for event"
            )
            self._set_status(instance, WorkflowStatus.WAITING)
            # Auto-register the event listener from step metadata
            await self.register_event_listener(
                instance_id=instance.instance_id,
                event_type=event_type,
                success_step_id=success_step_id,
                timeout_seconds=waiting_info.get("timeout_seconds"),
                timeout_step_id=waiting_info.get("timeout_step_id"),
                event_mapping=waiting_info.get("event_mapping"),
                correlation_key=correlation_key,
            )
        elif isinstance(timer_info, dict) and timer_info.get("waiting_for_timer"):
            logger.info(
                f"Workflow instance {instance.instance_id} is waiting for timer"
            )
            self._set_status(instance, WorkflowStatus.WAITING)
            # Auto-schedule the timer from step metadata
            await self.schedule_timer(
                instance_id=instance.instance_id,
                delay_seconds=timer_info["delay_seconds"],
                next_step_id=timer_info["next_step_id"],
            )
        else:
            # Check if this is a terminal step
            workflow_def = self._definition_for(instance)
            terminal_states = workflow_def.get_terminal_states()

            if instance.current_step_id in terminal_states:
                logger.info(
                    f"Workflow instance {instance.instance_id} completed successfully"
                )
                self._set_status(instance, WorkflowStatus.COMPLETED)
            else:
                logger.warning(
                    f"Step {instance.current_step_id} has no transition and is not "
                    "terminal; the instance stays RUNNING until advance"
                )
        return None
