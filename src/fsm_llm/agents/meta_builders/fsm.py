"""``FSMArtifactBuilder``: incrementally builds an ``FSMDefinition`` (plan d454fd62 step 89; the
class body is the old ``agents/meta_builders.py`` one)."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import ValidationError

from fsm_llm.agents.constants import MetaDefaults
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.exceptions import BuilderError
from fsm_llm.core.definitions import FSMDefinition, reachable_from
from fsm_llm.logging import logger

from .base import ArtifactBuilder


class FSMArtifactBuilder(ArtifactBuilder):
    """Incrementally builds an FSMDefinition."""

    def __init__(self) -> None:
        super().__init__()
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
    ) -> FSMArtifactBuilder:
        """Set basic FSM metadata. Warnings go to ``take_warnings()``; returns self."""
        warnings: list[str] = []
        self.name = name.strip()
        self.description = description.strip()
        if persona:
            self.persona = persona.strip()
        if not self.name:
            warnings.append("FSM name cannot be empty")
        if not self.description:
            warnings.append("FSM description cannot be empty")
        self._warnings.extend(warnings)
        return self

    def add_state(
        self,
        state_id: str,
        description: str,
        purpose: str,
        extraction_instructions: str | None = None,
        response_instructions: str | None = None,
    ) -> FSMArtifactBuilder:
        """Add a state. Warnings go to ``take_warnings()``; returns self."""
        warnings: list[str] = []
        state_id = state_id.strip()

        if not state_id:
            # DECISION plan-2026-10-02T052921-89b03f61/D-005
            # Call-time BuilderError refusals (here and in every mutator below)
            # are the one carve-out from "validate only in build()". Do NOT
            # move them into build(): the LLM tools and _assemble_* need
            # per-call feedback, and a bad model reply must skip one item, not
            # abort the whole artifact.
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
        self._warnings.extend(warnings)
        return self

    def remove_state(self, state_id: str) -> FSMArtifactBuilder:
        """Remove a state. A missing item is a no-op; returns self."""
        if state_id not in self.states:
            return self

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
        return self

    def update_state(self, state_id: str, **fields: Any) -> FSMArtifactBuilder:
        """Update fields on an existing state. Warnings go to ``take_warnings()``; returns self."""
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

        self._warnings.extend(warnings)
        return self

    def add_transition(
        self,
        from_state: str,
        target_state: str,
        description: str,
        priority: int = MetaDefaults.DEFAULT_PRIORITY,
        conditions: list[dict[str, Any]] | None = None,
    ) -> FSMArtifactBuilder:
        """Add a transition between states; returns self."""
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
        return self

    def remove_transition(
        self, from_state: str, target_state: str
    ) -> FSMArtifactBuilder:
        """Remove a transition. A missing item is a no-op; returns self."""
        if from_state not in self.states:
            return self

        original = self.states[from_state]["transitions"]
        filtered = [t for t in original if t["target_state"] != target_state]
        if len(filtered) == len(original):
            return self

        self.states[from_state]["transitions"] = filtered
        return self

    def set_initial_state(self, state_id: str) -> FSMArtifactBuilder:
        """Set the initial state; returns self."""
        if state_id not in self.states:
            raise BuilderError(
                f"State '{state_id}' not found",
                action="set_initial_state",
            )
        self.initial_state = state_id
        return self

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Produce dict compatible with FSMDefinition."""
        result: dict[str, Any] = {
            "name": self.name or "unnamed_fsm",
            "description": self.description or "No description",
            "initial_state": self.initial_state or "",
            "version": FSMDefinition.model_fields["version"].default,
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

        # Structural validation via FSMDefinition. Its own structure validator
        # (dangling targets, orphans) runs only when no field failed, and it
        # then reports the defect itself (empty ``loc``): do not say it again.
        structure_checked = True
        try:
            FSMDefinition(**self.to_dict())
        except ValidationError as e:
            for err in e.errors():
                loc = " -> ".join(str(part) for part in err["loc"])
                errors.append(f"{loc}: {err['msg']}")
                structure_checked = structure_checked and not err["loc"]
        except (TypeError, KeyError, ValueError) as e:
            errors.append(str(e))
            structure_checked = False

        # A field error stopped that check: reachability and targets here.
        if (
            not structure_checked
            and self.initial_state
            and self.initial_state in self.states
        ):
            reachable = reachable_from(
                {
                    sid: [
                        target
                        for t in state["transitions"]
                        if (target := t.get("target_state", ""))
                    ]
                    for sid, state in self.states.items()
                },
                [self.initial_state],
            )

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

    _TOTAL_REQUIRED: ClassVar[int] = (
        5  # name, description, initial state, a state, a terminal
    )

    def _get_minimal_summary(self) -> str:
        return "\n".join(
            [
                "=== FSM Builder Status ===",
                f"Name: {self.name or '(not set)'} | Initial: {self.initial_state or '(not set)'}",
                *self._graph_minimal_lines(
                    self.states, self.initial_state, "States", "Transitions"
                ),
                *self._missing_line(),
            ]
        )

    def _get_standard_summary(self) -> str:
        parts: list[str] = [
            "=== FSM Builder Status ===",
            f"Name: {self.name or '(not set)'}",
            f"Initial state: {self.initial_state or '(not set)'}",
            f"States ({len(self.states)}):",
        ]
        for sid, state in self.states.items():
            marker = self._node_markers(sid, state, self.initial_state)
            desc = state.get("description", "")[: MetaDefaults.SUMMARY_LINE_WIDTH]
            parts.append(f"  - {sid}{marker}: {desc}")
            for t in state["transitions"]:
                parts.append(f"    -> {t['target_state']}")
        return "\n".join([*parts, *self._missing_line()])

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
            marker = self._node_markers(sid, state, self.initial_state)
            parts.append(
                f"  - {sid}{marker}: {state.get('description', '(no description)')}"
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
                cond_count = len(t.get("conditions") or [])
                cond_str = f" ({cond_count} conditions)" if cond_count else ""
                parts.append(
                    f"    -> {t['target_state']}: {t['description']}{cond_str}"
                )

        return "\n".join([*parts, *self._still_missing_lines()])
