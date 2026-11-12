"""
Workflow definition and validation for the FSM-LLM Workflow System.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from fsm_llm.definitions import reachable_from
from fsm_llm.logging import logger

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------
from .exceptions import WorkflowDefinitionError, WorkflowValidationError
from .steps import (
    ParallelStep,
    RetryStep,
    TimerStep,
    WaitEventStep,
    WorkflowStep,
)

# --------------------------------------------------------------

# The callable/object fields no step's serialized form can carry (functions
# are not JSON-serializable and have no meaningful round-trip representation).
# Shared by `_serialize_step` and every recursive call site below it so the
# exclusion set is defined exactly once.
_STEP_CALLABLE_EXCLUDE_FIELDS = frozenset(
    {
        "action",
        "api_function",
        "condition",
        "aggregation_function",
        "llm_interface",
    }
)


def _serialize_step(step: WorkflowStep) -> dict[str, Any]:
    """Serialize a single step to a plain dict, recursing into nested steps.

    Interface contract: takes any ``WorkflowStep`` instance; returns a
    JSON-safe ``dict`` with a ``"type"`` key set to the step's class name and
    the 5 known callable fields (`_STEP_CALLABLE_EXCLUDE_FIELDS`) stripped.
    Nested steps are handled recursively so no caller needs its own
    per-step-type dump logic:
    - ``RetryStep.step`` is declared ``Field(exclude=True)`` at the pydantic
      level, so it never appears in ``model_dump()``'s output; this function
      adds it back as ``step_dict["step"]`` via a recursive call.
    - ``ParallelStep.steps`` DOES appear in ``model_dump()``'s output, but
      pydantic's own recursive dump does not apply this exclusion/tagging
      logic to nested steps; this function replaces each entry with its own
      recursive call so nested steps keep their `type` tag and stripped
      callables.

    Mirrors the ``RetryStep`` recursion precedent in
    ``RetryStep.referenced_states``.
    Failure mode: none — every `WorkflowStep` subclass either has no nested
    steps (returned as-is) or is one of the two known nesting shapes above.
    """
    step_dict = step.model_dump(exclude=set(_STEP_CALLABLE_EXCLUDE_FIELDS))
    # Custom steps may carry other callable fields; never emit a function.
    step_dict = {k: v for k, v in step_dict.items() if not callable(v)}
    step_dict["type"] = step.__class__.__name__

    if isinstance(step, RetryStep) and isinstance(step.step, WorkflowStep):
        step_dict["step"] = _serialize_step(step.step)
    elif isinstance(step, ParallelStep):
        step_dict["steps"] = [
            _serialize_step(nested_step)
            if isinstance(nested_step, WorkflowStep)
            else nested_step
            for nested_step in step.steps
        ]

    return step_dict


class WorkflowDefinition(BaseModel):
    """Definition of a workflow."""

    workflow_id: str
    name: str
    description: str = ""
    steps: dict[str, WorkflowStep] = Field(default_factory=dict)
    initial_step_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_step(
        self, step: WorkflowStep, is_initial: bool = False
    ) -> WorkflowDefinition:
        """Add a step to the workflow.

        Raises ``WorkflowDefinitionError`` if a DIFFERENT step with the same
        ``step_id`` is already present (re-adding the same object is a no-op).
        """
        existing = self.steps.get(step.step_id)
        if existing is not None and existing is not step:
            raise WorkflowDefinitionError(
                workflow_id=self.workflow_id,
                message=f"Duplicate step_id '{step.step_id}'",
            )
        self.steps[step.step_id] = step
        if is_initial:
            self.initial_step_id = step.step_id
        return self

    def with_initial_step(self, step: WorkflowStep) -> WorkflowDefinition:
        """Add the initial step to the workflow."""
        return self.with_step(step, is_initial=True)

    @field_validator("initial_step_id")
    @classmethod
    def validate_initial_step(cls, v, info):
        """Validate that the initial step exists in the steps dictionary."""
        if v is not None and "steps" in info.data and v not in info.data["steps"]:
            raise ValueError(f"Initial step '{v}' not found in the steps dictionary")
        return v

    def validate_structure(self) -> None:
        """Validate the workflow definition."""
        errors = []

        # Basic validation
        if not self.workflow_id:
            errors.append("Workflow ID is required")

        if not self.name:
            errors.append("Workflow name is required")

        if not self.steps:
            errors.append("Workflow must have at least one step")

        if not self.initial_step_id:
            errors.append("Initial step ID must be specified")
        elif self.initial_step_id not in self.steps:
            errors.append(f"Initial step '{self.initial_step_id}' not found in steps")

        # Step validation
        errors.extend(self._validate_steps())

        # State transition validation
        errors.extend(self._validate_state_transitions())

        # Reachability validation
        errors.extend(self._validate_reachability())

        # Cycle detection: only a cycle made entirely of steps that run back
        # to back without pausing is an error. A cycle through an event or
        # timer step is a legitimate loop (polling, retry-until, re-approval).
        if self.has_synchronous_cycles():
            errors.append(
                "Workflow contains a synchronous cycle (a loop with no event or "
                "timer step in it would run forever). Put a WaitEventStep or "
                "TimerStep on the loop, or give the chain a terminal state."
            )

        # Warn about states that cannot reach any terminal state
        terminal_states = self.get_terminal_states()
        if terminal_states and self.initial_step_id:
            adjacency = self._adjacency()
            for step_id in self.steps:
                if step_id in terminal_states:
                    continue
                if not reachable_from(adjacency, [step_id]) & terminal_states:
                    logger.warning(
                        f"State '{step_id}' has no path to any terminal state — "
                        "workflow may not terminate from this state"
                    )

        if errors:
            raise WorkflowValidationError(errors)

    def _validate_steps(self) -> list[str]:
        """Validate individual steps."""
        errors = []

        for step_id, step in self.steps.items():
            if step.step_id != step_id:
                errors.append(
                    f"Step ID mismatch: key '{step_id}' != step.step_id '{step.step_id}'"
                )

            if not step.name:
                errors.append(f"Step '{step_id}' must have a name")

            if isinstance(step, ParallelStep):
                seen: set[str] = set()
                for child in step.steps:
                    if child.step_id in seen:
                        errors.append(
                            f"ParallelStep '{step_id}' has duplicate child "
                            f"step_id '{child.step_id}'"
                        )
                    seen.add(child.step_id)

        return errors

    def _validate_state_transitions(self) -> list[str]:
        """Validate that all referenced states exist."""
        errors = []
        all_states = set(self.steps.keys())

        for step_id, step in self.steps.items():
            for state in step.referenced_states():
                if state not in all_states:
                    errors.append(
                        f"Step '{step_id}' references unknown state: '{state}'"
                    )

        return errors

    def _adjacency(self) -> dict[str, set[str]]:
        """Step id -> the states it can route to (``referenced_states``)."""
        return {
            step_id: step.referenced_states() for step_id, step in self.steps.items()
        }

    def _validate_reachability(self) -> list[str]:
        """Validate that all states are reachable from the initial state."""
        if not self.initial_step_id:
            return ["Cannot validate reachability without initial step"]

        reachable_states = reachable_from(self._adjacency(), [self.initial_step_id])
        all_states = set(self.steps.keys())
        unreachable_states = all_states - reachable_states

        errors = []
        for state in unreachable_states:
            errors.append(
                f"State '{state}' is not reachable from initial state '{self.initial_step_id}'"
            )

        return errors

    def get_terminal_states(self) -> set[str]:
        """Get all terminal states (states with no outgoing transitions)."""
        return {
            step_id
            for step_id, step in self.steps.items()
            if not step.referenced_states()
        }

    def has_cycles(self) -> bool:
        """Check if the workflow graph has any cycle (including loops that
        pass through event or timer steps; see ``has_synchronous_cycles``).

        Uses an iterative DFS with WHITE/GRAY/BLACK coloring to avoid
        RecursionError on large workflow graphs (W-ISSUE-003). Consistent
        with the iterative walk of ``reachable_from``.
        """
        return self._has_cycle(pausing_breaks_cycle=False)

    def has_synchronous_cycles(self) -> bool:
        """Check for a cycle whose steps all run back to back without pausing.

        Edges leaving a pausing step (``WaitEventStep``/``TimerStep``, or a
        ``RetryStep`` wrapping one) are ignored: the engine returns control at
        such a step, so a loop through it cannot spin inside one engine call.

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-002
        # validate() rejects ONLY synchronous cycles. Do NOT go back to
        # rejecting every cycle: polling / retry-until loops through a timer
        # or event step are the classic workflow shape and were impossible to
        # register. The per-call step budget (engine `max_steps`)
        # still bounds custom steps that route dynamically.
        """
        return self._has_cycle(pausing_breaks_cycle=True)

    @staticmethod
    def _is_pausing_step(step: WorkflowStep) -> bool:
        inner: object = step
        while isinstance(inner, RetryStep):
            inner = inner.step
        return isinstance(inner, (WaitEventStep, TimerStep))

    def _has_cycle(self, pausing_breaks_cycle: bool) -> bool:
        if not self.initial_step_id:
            return False

        # DECISION plan_2026-05-29_d9092060/D-005 [STALE]
        # NOTE: do NOT revert to the recursive has_cycle() inner function --
        # Python's default recursion limit (~1000) causes RecursionError on
        # linear workflows with >1000 steps. See decisions.md D-005.
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in self.steps}

        def successors(node: str) -> set[str]:
            step = self.steps.get(node)
            if step is None:
                return set()
            if pausing_breaks_cycle and self._is_pausing_step(step):
                return set()
            return step.referenced_states()

        for start in list(self.steps):
            if color.get(start, WHITE) != WHITE:
                continue
            stack: list[tuple[str, bool]] = [(start, False)]
            while stack:
                node, returning = stack.pop()
                if returning:
                    color[node] = BLACK
                    continue
                node_color = color.get(node, WHITE)
                if node_color == GRAY:
                    return True
                if node_color == BLACK:
                    continue
                color[node] = GRAY
                stack.append((node, True))  # mark for post-visit (back-edge close)
                for succ in successors(node):
                    succ_color = color.get(succ, WHITE)
                    if succ_color == GRAY:
                        return True
                    if succ_color == WHITE:
                        stack.append((succ, False))
        return False

    def serialize(self) -> dict[str, Any]:
        """Serialize the workflow definition to a dictionary."""
        # Convert to dict, but handle special cases like callables
        workflow_dict = self.model_dump(exclude={"steps"})

        # Handle steps separately, recursing into any nested steps
        # (ParallelStep.steps, RetryStep.step) via _serialize_step so they
        # round-trip with their `type` tag and stripped callables too.
        steps_dict = {
            step_id: _serialize_step(step) for step_id, step in self.steps.items()
        }

        workflow_dict["steps"] = steps_dict
        return workflow_dict
