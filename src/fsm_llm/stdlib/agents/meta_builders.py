from __future__ import annotations

"""
Artifact builders for incrementally constructing FSMs, Workflows, and Agents.

Each builder maintains partial state and provides:
- Mutation methods (add/remove/update)
- Incremental validation
- Progress tracking
- Summary generation for LLM context injection
- Final serialization to validated definitions
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import ValidationError

from fsm_llm.dialog.definitions import FSMDefinition
from fsm_llm.logging import logger

from .constants import MetaDefaults
from .definitions import ArtifactType, BuildProgress
from .exceptions import BuilderError


class ArtifactBuilder(ABC):
    """Base class for incrementally building artifacts."""

    @property
    @abstractmethod
    def artifact_type(self) -> ArtifactType: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def validate_partial(self) -> list[str]: ...

    @abstractmethod
    def validate_complete(self) -> list[str]: ...

    @abstractmethod
    def get_progress(self) -> BuildProgress: ...

    @abstractmethod
    def get_missing_fields(self) -> list[str]: ...

    @abstractmethod
    def get_summary(self, detail_level: str = "full") -> str:
        """Human-readable summary for LLM context injection.

        Args:
            detail_level: One of ``"minimal"`` (names + counts),
                ``"standard"`` (names + short descriptions + targets),
                or ``"full"`` (everything including instructions).
        """
        ...


# ---------------------------------------------------------------------------
# FSM Builder
# ---------------------------------------------------------------------------


class FSMBuilder(ArtifactBuilder):
    """Incrementally builds an FSMDefinition."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.description: str | None = None
        self.persona: str | None = None
        self.initial_state: str | None = None
        self.states: dict[str, dict[str, Any]] = {}

    @property
    def artifact_type(self) -> ArtifactType:
        return ArtifactType.FSM

    # -- Mutation methods ---------------------------------------------------

    def set_overview(
        self,
        name: str,
        description: str,
        persona: str | None = None,
    ) -> list[str]:
        """Set basic FSM metadata. Returns list of warnings."""
        warnings: list[str] = []
        self.name = name.strip()
        self.description = description.strip()
        if persona:
            self.persona = persona.strip()
        if not self.name:
            warnings.append("FSM name cannot be empty")
        if not self.description:
            warnings.append("FSM description cannot be empty")
        return warnings

    def add_state(
        self,
        state_id: str,
        description: str,
        purpose: str,
        extraction_instructions: str | None = None,
        response_instructions: str | None = None,
    ) -> list[str]:
        """Add a state. Returns list of warnings."""
        warnings: list[str] = []
        state_id = state_id.strip()

        if not state_id:
            raise BuilderError("State ID cannot be empty", action="add_state")
        if state_id in self.states:
            warnings.append(f"State '{state_id}' already exists; overwriting")

        self.states[state_id] = {
            "id": state_id,
            "description": description.strip(),
            "purpose": purpose.strip(),
            "transitions": [],
        }
        if extraction_instructions:
            self.states[state_id]["extraction_instructions"] = (
                extraction_instructions.strip()
            )
        if response_instructions:
            self.states[state_id]["response_instructions"] = (
                response_instructions.strip()
            )

        # Auto-set initial state if this is the first state
        if len(self.states) == 1 and self.initial_state is None:
            self.initial_state = state_id
            warnings.append(
                f"Auto-set initial state to '{state_id}' (first state added)"
            )

        logger.debug(f"Added state '{state_id}' to FSM builder")
        return warnings

    def remove_state(self, state_id: str) -> bool:
        """Remove a state. Returns True if removed."""
        if state_id not in self.states:
            return False

        del self.states[state_id]

        # Clean up transitions pointing to this state
        for _sid, state in self.states.items():
            state["transitions"] = [
                t for t in state["transitions"] if t["target_state"] != state_id
            ]

        # Clear initial state if it was removed
        if self.initial_state == state_id:
            self.initial_state = None

        logger.debug(f"Removed state '{state_id}' from FSM builder")
        return True

    def update_state(self, state_id: str, **fields: Any) -> list[str]:
        """Update fields on an existing state. Returns list of warnings."""
        if state_id not in self.states:
            raise BuilderError(
                f"State '{state_id}' not found",
                action="update_state",
            )

        warnings: list[str] = []
        allowed = {
            "description",
            "purpose",
            "extraction_instructions",
            "response_instructions",
        }

        for key, value in fields.items():
            if key not in allowed:
                warnings.append(f"Ignoring unknown field '{key}'")
                continue
            if value is None:
                warnings.append(f"Ignoring None value for '{key}'")
                continue
            if not isinstance(value, str):
                warnings.append(
                    f"Field '{key}' should be a string, "
                    f"got {type(value).__name__}; converting"
                )
                value = str(value)
            value = value.strip()
            self.states[state_id][key] = value

        return warnings

    def add_transition(
        self,
        from_state: str,
        target_state: str,
        description: str,
        priority: int = MetaDefaults.DEFAULT_PRIORITY,
        conditions: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add a transition between states. Returns list of warnings."""
        warnings: list[str] = []

        if from_state not in self.states:
            raise BuilderError(
                f"Source state '{from_state}' not found",
                action="add_transition",
            )
        if target_state not in self.states:
            raise BuilderError(
                f"Target state '{target_state}' not found",
                action="add_transition",
            )

        transition: dict[str, Any] = {
            "target_state": target_state,
            "description": description.strip(),
            "priority": priority,
        }

        if conditions:
            transition["conditions"] = conditions

        self.states[from_state]["transitions"].append(transition)
        logger.debug(f"Added transition '{from_state}' -> '{target_state}'")
        return warnings

    def remove_transition(self, from_state: str, target_state: str) -> bool:
        """Remove a transition. Returns True if removed."""
        if from_state not in self.states:
            return False

        original = self.states[from_state]["transitions"]
        filtered = [t for t in original if t["target_state"] != target_state]
        if len(filtered) == len(original):
            return False

        self.states[from_state]["transitions"] = filtered
        return True

    def set_initial_state(self, state_id: str) -> list[str]:
        """Set the initial state. Returns list of warnings."""
        warnings: list[str] = []
        if state_id not in self.states:
            raise BuilderError(
                f"State '{state_id}' not found",
                action="set_initial_state",
            )
        self.initial_state = state_id
        return warnings

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Produce dict compatible with FSMDefinition."""
        result: dict[str, Any] = {
            "name": self.name or "unnamed_fsm",
            "description": self.description or "No description",
            "initial_state": self.initial_state or "",
            "version": "4.1",
            "states": dict(self.states),
        }
        if self.persona:
            result["persona"] = self.persona
        return result

    # -- Validation ---------------------------------------------------------

    def validate_partial(self) -> list[str]:
        """Return warnings about the current partial state."""
        warnings: list[str] = []

        if not self.name:
            warnings.append("FSM name not set")
        if not self.description:
            warnings.append("FSM description not set")
        if not self.states:
            warnings.append("No states defined yet")
        if self.initial_state and self.initial_state not in self.states:
            warnings.append(f"Initial state '{self.initial_state}' does not exist")

        # Check for states without transitions (potential dead ends)
        for sid, state in self.states.items():
            if not state["transitions"]:
                # Terminal states are ok, just note them
                warnings.append(
                    f"State '{sid}' has no outgoing transitions (terminal?)"
                )

        return warnings

    def validate_complete(self) -> list[str]:
        """Attempt to create FSMDefinition and check reachability."""
        errors: list[str] = []

        if not self.name:
            errors.append("FSM name is required")
        if not self.description:
            errors.append("FSM description is required")
        if not self.states:
            errors.append("At least one state is required")
        if not self.initial_state:
            errors.append("Initial state is required")

        if errors:
            return errors

        # Structural validation via FSMDefinition
        try:
            FSMDefinition(**self.to_dict())
        except ValidationError as e:
            for err in e.errors():
                loc = " -> ".join(str(part) for part in err["loc"])
                errors.append(f"{loc}: {err['msg']}")
        except (TypeError, KeyError, ValueError) as e:
            errors.append(str(e))

        # Reachability: BFS from initial state
        if self.initial_state and self.initial_state in self.states:
            reachable: set[str] = set()
            queue = [self.initial_state]
            while queue:
                current = queue.pop(0)
                if current in reachable:
                    continue
                reachable.add(current)
                for t in self.states.get(current, {}).get("transitions", []):
                    target = t.get("target_state", "")
                    if target and target not in reachable:
                        queue.append(target)

            unreachable = set(self.states.keys()) - reachable
            if unreachable:
                errors.append(
                    f"Unreachable states (not connected from initial state): "
                    f"{', '.join(sorted(unreachable))}"
                )

            # Check transitions point to existing states
            for sid in self.states:
                for t in self.states[sid]["transitions"]:
                    target = t.get("target_state", "")
                    if target and target not in self.states:
                        errors.append(
                            f"State '{sid}' has transition to non-existent "
                            f"state '{target}'"
                        )

        return errors

    # -- Progress -----------------------------------------------------------

    def get_missing_fields(self) -> list[str]:
        """Return list of required things still missing."""
        missing: list[str] = []
        if not self.name:
            missing.append("FSM name")
        if not self.description:
            missing.append("FSM description")
        if not self.states:
            missing.append("At least one state")
        elif not self.initial_state:
            missing.append("Initial state")
        else:
            # Check if we have at least one terminal state
            has_terminal = any(not s["transitions"] for s in self.states.values())
            if not has_terminal:
                missing.append(
                    "At least one terminal state (state with no outgoing transitions)"
                )

            # Check states have basic fields
            for sid, s in self.states.items():
                if not s.get("description"):
                    missing.append(f"Description for state '{sid}'")
                if not s.get("purpose"):
                    missing.append(f"Purpose for state '{sid}'")

        return missing

    def get_progress(self) -> BuildProgress:
        """Return current build progress."""
        missing = self.get_missing_fields()
        warnings = self.validate_partial()

        # Count required items: name, description, initial_state, >=1 state, >=1 terminal
        total = 5
        completed = total - min(len(missing), total)

        return BuildProgress(
            total_required=total,
            completed=completed,
            missing=missing,
            warnings=warnings,
        )

    def get_summary(self, detail_level: str = "full") -> str:
        """Human-readable summary for LLM context injection."""
        if detail_level == "minimal":
            return self._get_minimal_summary()
        if detail_level == "standard":
            return self._get_standard_summary()
        return self._get_full_summary()

    def _get_minimal_summary(self) -> str:
        parts: list[str] = [
            "=== FSM Builder Status ===",
            f"Name: {self.name or '(not set)'} | Initial: {self.initial_state or '(not set)'}",
        ]
        if self.states:
            state_labels = []
            for sid in self.states:
                label = sid
                if sid == self.initial_state:
                    label += " [INITIAL]"
                if not self.states[sid]["transitions"]:
                    label += " [TERMINAL]"
                state_labels.append(label)
            parts.append(f"States ({len(self.states)}): {', '.join(state_labels)}")
            total_transitions = sum(len(s["transitions"]) for s in self.states.values())
            parts.append(f"Transitions: {total_transitions} defined")
        else:
            parts.append("States: none yet")

        missing = self.get_missing_fields()
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        return "\n".join(parts)

    def _get_standard_summary(self) -> str:
        parts: list[str] = [
            "=== FSM Builder Status ===",
            f"Name: {self.name or '(not set)'}",
            f"Initial state: {self.initial_state or '(not set)'}",
            f"States ({len(self.states)}):",
        ]
        for sid, state in self.states.items():
            marker = " [INITIAL]" if sid == self.initial_state else ""
            if not state["transitions"]:
                marker += " [TERMINAL]"
            desc = state.get("description", "")[:40]
            parts.append(f"  - {sid}{marker}: {desc}")
            for t in state["transitions"]:
                parts.append(f"    -> {t['target_state']}")

        missing = self.get_missing_fields()
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        return "\n".join(parts)

    def _get_full_summary(self) -> str:
        parts: list[str] = []

        parts.append("=== FSM Builder Status ===")
        parts.append(f"Name: {self.name or '(not set)'}")
        parts.append(f"Description: {self.description or '(not set)'}")
        if self.persona:
            parts.append(f"Persona: {self.persona}")
        parts.append(f"Initial state: {self.initial_state or '(not set)'}")
        parts.append(f"States ({len(self.states)}):")

        for sid, state in self.states.items():
            marker = " [INITIAL]" if sid == self.initial_state else ""
            is_terminal = not state["transitions"]
            terminal_marker = " [TERMINAL]" if is_terminal else ""
            parts.append(
                f"  - {sid}{marker}{terminal_marker}: {state.get('description', '(no description)')}"
            )

            if state.get("extraction_instructions"):
                parts.append(
                    f"    extraction: {state['extraction_instructions'][: MetaDefaults.SUMMARY_TRUNCATE_WIDTH]}..."
                )
            if state.get("response_instructions"):
                parts.append(
                    f"    response: {state['response_instructions'][: MetaDefaults.SUMMARY_TRUNCATE_WIDTH]}..."
                )

            for t in state["transitions"]:
                cond_count = len(t.get("conditions", []))
                cond_str = f" ({cond_count} conditions)" if cond_count else ""
                parts.append(
                    f"    -> {t['target_state']}: {t['description']}{cond_str}"
                )

        missing = self.get_missing_fields()
        if missing:
            parts.append("\nStill missing:")
            for m in missing:
                parts.append(f"  - {m}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Workflow Builder
# ---------------------------------------------------------------------------


class WorkflowBuilder(ArtifactBuilder):
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
    ) -> list[str]:
        """Set basic workflow metadata."""
        warnings: list[str] = []
        self.workflow_id = workflow_id.strip()
        self.name = name.strip()
        self.description = description.strip()
        return warnings

    def add_step(
        self,
        step_id: str,
        step_type: str,
        name: str,
        description: str = "",
        config: dict[str, Any] | None = None,
    ) -> list[str]:
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

        return warnings

    def remove_step(self, step_id: str) -> bool:
        """Remove a step. Returns True if removed."""
        if step_id not in self.steps:
            return False
        del self.steps[step_id]

        # Clean up transitions
        for s in self.steps.values():
            s["transitions"] = [t for t in s["transitions"] if t["target"] != step_id]

        if self.initial_step_id == step_id:
            self.initial_step_id = None
        return True

    def set_step_transition(
        self,
        from_step: str,
        to_step: str,
        condition: str | None = None,
    ) -> list[str]:
        """Add a transition between steps."""
        warnings: list[str] = []

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
        return warnings

    def set_initial_step(self, step_id: str) -> list[str]:
        """Set the initial step."""
        if step_id not in self.steps:
            raise BuilderError(f"Step '{step_id}' not found", action="set_initial_step")
        self.initial_step_id = step_id
        return []

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

        # Check transition targets exist
        for sid, step in self.steps.items():
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

    def get_progress(self) -> BuildProgress:
        missing = self.get_missing_fields()
        warnings = self.validate_partial()
        total = 4  # id, name, >=1 step, initial step
        completed = total - min(len(missing), total)
        return BuildProgress(
            total_required=total,
            completed=completed,
            missing=missing,
            warnings=warnings,
        )

    def get_summary(self, detail_level: str = "full") -> str:
        if detail_level == "minimal":
            return self._get_minimal_summary()
        if detail_level == "standard":
            return self._get_standard_summary()
        return self._get_full_summary()

    def _get_minimal_summary(self) -> str:
        parts: list[str] = [
            "=== Workflow Builder Status ===",
            f"Name: {self.name or '(not set)'} | Initial: {self.initial_step_id or '(not set)'}",
        ]
        if self.steps:
            step_labels = []
            for sid in self.steps:
                label = sid
                if sid == self.initial_step_id:
                    label += " [INITIAL]"
                if not self.steps[sid]["transitions"]:
                    label += " [TERMINAL]"
                step_labels.append(label)
            parts.append(f"Steps ({len(self.steps)}): {', '.join(step_labels)}")
            total_transitions = sum(len(s["transitions"]) for s in self.steps.values())
            parts.append(f"Connections: {total_transitions} defined")
        else:
            parts.append("Steps: none yet")

        missing = self.get_missing_fields()
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        return "\n".join(parts)

    def _get_standard_summary(self) -> str:
        parts: list[str] = [
            "=== Workflow Builder Status ===",
            f"Name: {self.name or '(not set)'}",
            f"Initial step: {self.initial_step_id or '(not set)'}",
            f"Steps ({len(self.steps)}):",
        ]
        for sid, step in self.steps.items():
            marker = " [INITIAL]" if sid == self.initial_step_id else ""
            if not step["transitions"]:
                marker += " [TERMINAL]"
            parts.append(
                f"  - {sid}{marker} ({step['step_type']}): {step['name'][:40]}"
            )
            for t in step["transitions"]:
                parts.append(f"    -> {t['target']}")

        missing = self.get_missing_fields()
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        return "\n".join(parts)

    def _get_full_summary(self) -> str:
        parts: list[str] = []
        parts.append("=== Workflow Builder Status ===")
        parts.append(f"ID: {self.workflow_id or '(not set)'}")
        parts.append(f"Name: {self.name or '(not set)'}")
        parts.append(f"Description: {self.description or '(not set)'}")
        parts.append(f"Initial step: {self.initial_step_id or '(not set)'}")
        parts.append(f"Steps ({len(self.steps)}):")

        for sid, step in self.steps.items():
            marker = " [INITIAL]" if sid == self.initial_step_id else ""
            is_terminal = not step["transitions"]
            terminal_marker = " [TERMINAL]" if is_terminal else ""
            parts.append(
                f"  - {sid}{marker}{terminal_marker} ({step['step_type']}): {step['name']}"
            )
            for t in step["transitions"]:
                cond_str = f" [if: {t['condition']}]" if t.get("condition") else ""
                parts.append(f"    -> {t['target']}{cond_str}")

        missing = self.get_missing_fields()
        if missing:
            parts.append("\nStill missing:")
            for m in missing:
                parts.append(f"  - {m}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent Builder
# ---------------------------------------------------------------------------


class AgentBuilder(ArtifactBuilder):
    """Incrementally builds an agent configuration."""

    VALID_AGENT_TYPES: ClassVar[set[str]] = {
        "react",
        "plan_execute",
        "reflexion",
        "rewoo",
        "evaluator_optimizer",
        "maker_checker",
        "prompt_chain",
        "self_consistency",
        "debate",
        "orchestrator",
        "adapt",
    }

    def __init__(self) -> None:
        self.agent_type: str | None = None
        self.name: str | None = None
        self.description: str | None = None
        self.config: dict[str, Any] = {
            "model": MetaDefaults.AGENT_MODEL,
            "max_iterations": MetaDefaults.AGENT_MAX_ITERATIONS,
            "timeout_seconds": MetaDefaults.AGENT_TIMEOUT_SECONDS,
            "temperature": MetaDefaults.AGENT_TEMPERATURE,
            "max_tokens": MetaDefaults.AGENT_MAX_TOKENS,
        }
        self.tools: list[dict[str, Any]] = []

    @property
    def artifact_type(self) -> ArtifactType:
        return ArtifactType.AGENT

    # -- Mutation methods ---------------------------------------------------

    # Common LLM misspellings/variations → canonical type
    _AGENT_TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "react_agent": "react",
        "reactagent": "react",
        "plan_and_execute": "plan_execute",
        "plan_execute_agent": "plan_execute",
        "planexecute": "plan_execute",
        "self_consistent": "self_consistency",
        "eval_opt": "evaluator_optimizer",
        "eval_optimize": "evaluator_optimizer",
        "maker_check": "maker_checker",
        "reflexion_agent": "reflexion",
        "debate_agent": "debate",
        "orchestrate": "orchestrator",
        "adapt_agent": "adapt",
        "prompt_chaining": "prompt_chain",
        "rewoo_agent": "rewoo",
    }

    def set_agent_type(self, agent_type: str) -> list[str]:
        """Set the agent pattern type."""
        warnings: list[str] = []
        agent_type = agent_type.strip().lower()
        # Resolve common LLM variations
        agent_type = self._AGENT_TYPE_ALIASES.get(agent_type, agent_type)
        if agent_type not in self.VALID_AGENT_TYPES:
            raise BuilderError(
                f"Unknown agent type '{agent_type}'. "
                f"Valid: {', '.join(sorted(self.VALID_AGENT_TYPES))}",
                action="set_agent_type",
            )
        self.agent_type = agent_type
        return warnings

    def set_overview(
        self,
        name: str,
        description: str,
    ) -> list[str]:
        """Set basic agent metadata."""
        self.name = name.strip()
        self.description = description.strip()
        return []

    # Allowed config fields and their expected types
    _CONFIG_ALLOWED: ClassVar[set[str]] = {
        "model",
        "max_iterations",
        "timeout_seconds",
        "temperature",
        "max_tokens",
    }
    _CONFIG_NUMERIC: ClassVar[set[str]] = {"timeout_seconds", "temperature"}
    _CONFIG_INT: ClassVar[set[str]] = {"max_iterations", "max_tokens"}

    def set_config(self, **kwargs: Any) -> list[str]:
        """Update agent config fields."""
        warnings: list[str] = []
        for key, value in kwargs.items():
            if key not in self._CONFIG_ALLOWED:
                warnings.append(f"Ignoring unknown config field '{key}'")
                continue
            if key == "model":
                if not isinstance(value, str):
                    warnings.append(
                        f"Config 'model' expects str, "
                        f"got {type(value).__name__}; skipping"
                    )
                    continue
            elif key in self._CONFIG_INT:
                if not isinstance(value, int):
                    warnings.append(
                        f"Config '{key}' expects int, "
                        f"got {type(value).__name__}; skipping"
                    )
                    continue
            elif key in self._CONFIG_NUMERIC:
                if not isinstance(value, int | float):
                    warnings.append(
                        f"Config '{key}' expects number, "
                        f"got {type(value).__name__}; skipping"
                    )
                    continue
            self.config[key] = value
        return warnings

    def add_tool(
        self,
        name: str,
        description: str,
        parameter_schema: dict[str, Any] | None = None,
    ) -> list[str]:
        """Add a tool definition."""
        warnings: list[str] = []
        name = name.strip()
        if not name:
            raise BuilderError("Tool name cannot be empty", action="add_tool")

        # Check for duplicates
        for t in self.tools:
            if t["name"] == name:
                warnings.append(f"Tool '{name}' already exists; overwriting")
                self.tools = [t for t in self.tools if t["name"] != name]
                break

        tool_def: dict[str, Any] = {
            "name": name,
            "description": description.strip(),
        }
        if parameter_schema:
            tool_def["parameter_schema"] = parameter_schema
        self.tools.append(tool_def)
        return warnings

    def remove_tool(self, name: str) -> bool:
        """Remove a tool. Returns True if removed."""
        original_len = len(self.tools)
        self.tools = [t for t in self.tools if t["name"] != name]
        return len(self.tools) < original_len

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Produce agent configuration dict."""
        return {
            "name": self.name or "unnamed_agent",
            "description": self.description or "",
            "agent_type": self.agent_type or "",
            "config": dict(self.config),
            "tools": list(self.tools),
        }

    # -- Validation ---------------------------------------------------------

    def validate_partial(self) -> list[str]:
        warnings: list[str] = []
        if not self.agent_type:
            warnings.append("Agent type not set")
        if not self.tools:
            warnings.append("No tools defined yet")
        return warnings

    def validate_complete(self) -> list[str]:
        errors: list[str] = []
        if not self.agent_type:
            errors.append("Agent type is required")
        if not self.name:
            errors.append("Agent name is required")
        if not self.tools:
            errors.append("At least one tool is required")
        for t in self.tools:
            if not t.get("name"):
                errors.append("Tool name is required")
            if not t.get("description"):
                errors.append(f"Tool '{t.get('name', '?')}' needs a description")
        return errors

    def get_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.agent_type:
            missing.append("Agent type")
        if not self.name:
            missing.append("Agent name")
        if not self.tools:
            missing.append("At least one tool")
        return missing

    def get_progress(self) -> BuildProgress:
        missing = self.get_missing_fields()
        warnings = self.validate_partial()
        total = 3
        completed = total - min(len(missing), total)
        return BuildProgress(
            total_required=total,
            completed=completed,
            missing=missing,
            warnings=warnings,
        )

    def get_summary(self, detail_level: str = "full") -> str:
        if detail_level == "minimal":
            return self._get_minimal_summary()
        if detail_level == "standard":
            return self._get_standard_summary()
        return self._get_full_summary()

    def _get_minimal_summary(self) -> str:
        parts: list[str] = [
            "=== Agent Builder Status ===",
            f"Name: {self.name or '(not set)'} | Type: {self.agent_type or '(not set)'}",
        ]
        if self.tools:
            tool_names = [t["name"] for t in self.tools]
            parts.append(f"Tools ({len(self.tools)}): {', '.join(tool_names)}")
        else:
            parts.append("Tools: none yet")

        missing = self.get_missing_fields()
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        return "\n".join(parts)

    def _get_standard_summary(self) -> str:
        parts: list[str] = [
            "=== Agent Builder Status ===",
            f"Name: {self.name or '(not set)'}",
            f"Agent type: {self.agent_type or '(not set)'}",
            f"Tools ({len(self.tools)}):",
        ]
        for t in self.tools:
            parts.append(f"  - {t['name']}: {t['description'][:40]}")

        missing = self.get_missing_fields()
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        return "\n".join(parts)

    def _get_full_summary(self) -> str:
        parts: list[str] = []
        parts.append("=== Agent Builder Status ===")
        parts.append(f"Name: {self.name or '(not set)'}")
        parts.append(f"Description: {self.description or '(not set)'}")
        parts.append(f"Agent type: {self.agent_type or '(not set)'}")
        parts.append(
            f"Config: model={self.config.get('model')}, "
            f"max_iter={self.config.get('max_iterations')}, "
            f"temp={self.config.get('temperature')}"
        )
        parts.append(f"Tools ({len(self.tools)}):")
        for t in self.tools:
            schema_str = ""
            if t.get("parameter_schema"):
                params = t["parameter_schema"].get("properties", {})
                schema_str = f" (params: {', '.join(params.keys())})" if params else ""
            parts.append(f"  - {t['name']}: {t['description']}{schema_str}")

        missing = self.get_missing_fields()
        if missing:
            parts.append("\nStill missing:")
            for m in missing:
                parts.append(f"  - {m}")

        return "\n".join(parts)
