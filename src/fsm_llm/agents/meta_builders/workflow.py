"""``WorkflowArtifactBuilder``: incrementally builds a workflow definition (plan d454fd62 step 89; the
class body is the old ``agents/meta_builders.py`` one)."""

from __future__ import annotations

from typing import Any, ClassVar

from fsm_llm.agents.constants import MetaDefaults
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.exceptions import BuilderError

from .base import ArtifactBuilder


class WorkflowArtifactBuilder(ArtifactBuilder):
    """Incrementally builds a workflow definition."""

    VALID_STEP_TYPES: ClassVar[set[str]] = {
        "auto_transition",
        "api_call",
        "condition",
        "llm_processing",
        "wait_for_event",
        "timer",
        "parallel",
        "conversation",
    }

    def __init__(self) -> None:
        super().__init__()
        self.workflow_id: str | None = None
        self.name: str | None = None
        self.description: str | None = None
        self.initial_step_id: str | None = None
        self.steps: dict[str, dict[str, Any]] = {}

    @property
    def artifact_type(self) -> ArtifactType:
        return ArtifactType.WORKFLOW

    # -- Mutation methods ---------------------------------------------------

    def set_overview(
        self,
        workflow_id: str,
        name: str,
        description: str,
    ) -> WorkflowArtifactBuilder:
        """Set basic workflow metadata."""
        self.workflow_id = workflow_id.strip()
        self.name = name.strip()
        self.description = description.strip()
        return self

    def add_step(
        self,
        step_id: str,
        step_type: str,
        name: str,
        description: str = "",
        config: dict[str, Any] | None = None,
    ) -> WorkflowArtifactBuilder:
        """Add a workflow step."""
        warnings: list[str] = []
        step_id = step_id.strip()

        if not step_id:
            raise BuilderError("Step ID cannot be empty", action="add_step")

        if step_type not in self.VALID_STEP_TYPES:
            warnings.append(
                f"Unknown step type '{step_type}'. "
                f"Valid: {', '.join(sorted(self.VALID_STEP_TYPES))}"
            )

        if step_id in self.steps:
            warnings.append(f"Step '{step_id}' already exists; overwriting")

        self.steps[step_id] = {
            "step_id": step_id,
            "step_type": step_type,
            "name": name.strip(),
            "description": description.strip(),
            "transitions": [],
        }
        if config:
            self.steps[step_id]["config"] = config

        if len(self.steps) == 1 and self.initial_step_id is None:
            self.initial_step_id = step_id
            warnings.append(f"Auto-set initial step to '{step_id}'")

        self._warnings.extend(warnings)
        return self

    def remove_step(self, step_id: str) -> WorkflowArtifactBuilder:
        """Remove a step. A missing item is a no-op; returns self."""
        if step_id not in self.steps:
            return self
        del self.steps[step_id]

        # Clean up transitions
        for s in self.steps.values():
            s["transitions"] = [t for t in s["transitions"] if t["target"] != step_id]

        if self.initial_step_id == step_id:
            self.initial_step_id = None
        return self

    def set_step_transition(
        self,
        from_step: str,
        to_step: str,
        condition: str | None = None,
    ) -> WorkflowArtifactBuilder:
        """Add a transition between steps."""
        if from_step not in self.steps:
            raise BuilderError(
                f"Source step '{from_step}' not found",
                action="set_step_transition",
            )
        if to_step not in self.steps:
            raise BuilderError(
                f"Target step '{to_step}' not found",
                action="set_step_transition",
            )

        transition: dict[str, Any] = {"target": to_step}
        if condition:
            transition["condition"] = condition

        self.steps[from_step]["transitions"].append(transition)
        return self

    def set_initial_step(self, step_id: str) -> WorkflowArtifactBuilder:
        """Set the initial step."""
        if step_id not in self.steps:
            raise BuilderError(f"Step '{step_id}' not found", action="set_initial_step")
        self.initial_step_id = step_id
        return self

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Produce workflow definition dict."""
        return {
            "workflow_id": self.workflow_id or "unnamed_workflow",
            "name": self.name or "Unnamed Workflow",
            "description": self.description or "",
            "initial_step_id": self.initial_step_id or "",
            "steps": dict(self.steps),
        }

    # -- Validation ---------------------------------------------------------

    def validate_partial(self) -> list[str]:
        warnings: list[str] = []
        if not self.name:
            warnings.append("Workflow name not set")
        if not self.steps:
            warnings.append("No steps defined yet")
        for sid, step in self.steps.items():
            if not step["transitions"]:
                warnings.append(f"Step '{sid}' has no outgoing transitions (terminal?)")
        return warnings

    def validate_complete(self) -> list[str]:
        """Return structural errors; an empty list means a complete *spec*.

        A workflow artifact is not directly loadable by ``fsm_llm.workflows``:
        its steps need Python callables (``APIStep.api_function``,
        ``ConditionStep.condition``, ...) that a JSON spec cannot carry, so a
        caller still wires them by hand. This checks required metadata, the
        initial step, transition targets and that every ``step_type`` is in
        ``VALID_STEP_TYPES``.
        """
        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-004: do NOT add a
        # WorkflowDefinition(**self.to_dict()) load check here without an
        # adapter that supplies step callables; it can never pass.
        errors: list[str] = []
        if not self.workflow_id:
            errors.append("Workflow ID is required")
        if not self.name:
            errors.append("Workflow name is required")
        if not self.steps:
            errors.append("At least one step is required")
        if not self.initial_step_id:
            errors.append("Initial step is required")
        elif self.initial_step_id not in self.steps:
            errors.append(f"Initial step '{self.initial_step_id}' not found")

        # Check step types are known and transition targets exist
        for sid, step in self.steps.items():
            if step["step_type"] not in self.VALID_STEP_TYPES:
                errors.append(
                    f"Step '{sid}' has unknown step type '{step['step_type']}'"
                )
            for t in step["transitions"]:
                if t["target"] not in self.steps:
                    errors.append(
                        f"Step '{sid}' has transition to unknown step '{t['target']}'"
                    )

        return errors

    def get_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.workflow_id:
            missing.append("Workflow ID")
        if not self.name:
            missing.append("Workflow name")
        if not self.steps:
            missing.append("At least one step")
        elif not self.initial_step_id:
            missing.append("Initial step")
        return missing

    _TOTAL_REQUIRED: ClassVar[int] = 4  # id, name, a step, initial step

    def _get_minimal_summary(self) -> str:
        return "\n".join(
            [
                "=== Workflow Builder Status ===",
                f"Name: {self.name or '(not set)'} | Initial: {self.initial_step_id or '(not set)'}",
                *self._graph_minimal_lines(
                    self.steps, self.initial_step_id, "Steps", "Connections"
                ),
                *self._missing_line(),
            ]
        )

    def _get_standard_summary(self) -> str:
        parts: list[str] = [
            "=== Workflow Builder Status ===",
            f"Name: {self.name or '(not set)'}",
            f"Initial step: {self.initial_step_id or '(not set)'}",
            f"Steps ({len(self.steps)}):",
        ]
        for sid, step in self.steps.items():
            marker = self._node_markers(sid, step, self.initial_step_id)
            name = step["name"][: MetaDefaults.SUMMARY_LINE_WIDTH]
            parts.append(f"  - {sid}{marker} ({step['step_type']}): {name}")
            for t in step["transitions"]:
                parts.append(f"    -> {t['target']}")
        return "\n".join([*parts, *self._missing_line()])

    def _get_full_summary(self) -> str:
        parts: list[str] = []
        parts.append("=== Workflow Builder Status ===")
        parts.append(f"ID: {self.workflow_id or '(not set)'}")
        parts.append(f"Name: {self.name or '(not set)'}")
        parts.append(f"Description: {self.description or '(not set)'}")
        parts.append(f"Initial step: {self.initial_step_id or '(not set)'}")
        parts.append(f"Steps ({len(self.steps)}):")

        for sid, step in self.steps.items():
            marker = self._node_markers(sid, step, self.initial_step_id)
            parts.append(f"  - {sid}{marker} ({step['step_type']}): {step['name']}")
            for t in step["transitions"]:
                cond_str = f" [if: {t['condition']}]" if t.get("condition") else ""
                parts.append(f"    -> {t['target']}{cond_str}")

        return "\n".join([*parts, *self._still_missing_lines()])
