"""
Workflow definition and validation for the LLM-FSM Workflow System.
"""

from typing import Dict, List, Any, Optional, Set
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .exceptions import WorkflowDefinitionError, WorkflowValidationError
from .steps import WorkflowStep, AutoTransitionStep, APICallStep, ConditionStep

# --------------------------------------------------------------


class WorkflowDefinition(BaseModel):
    """Definition of a workflow."""
    workflow_id: str
    name: str
    description: str = ""
    steps: Dict[str, WorkflowStep] = Field(default_factory=dict)
    initial_step_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def with_step(self, step: WorkflowStep, is_initial: bool = False) -> 'WorkflowDefinition':
        """Add a step to the workflow."""
        self.steps[step.step_id] = step
        if is_initial:
            self.initial_step_id = step.step_id
        return self

    def with_initial_step(self, step: WorkflowStep) -> 'WorkflowDefinition':
        """Add the initial step to the workflow."""
        return self.with_step(step, is_initial=True)

    @field_validator('initial_step_id')
    @classmethod
    def validate_initial_step(cls, v, info):
        """Validate that the initial step exists in the steps dictionary."""
        if v is not None and 'steps' in info.data and v not in info.data['steps']:
            raise ValueError(f"Initial step '{v}' not found in the steps dictionary")
        return v

    def validate(self) -> None:
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

        if errors:
            raise WorkflowValidationError(errors)

    def _validate_steps(self) -> List[str]:
        """Validate individual steps."""
        errors = []

        for step_id, step in self.steps.items():
            if step.step_id != step_id:
                errors.append(f"Step ID mismatch: key '{step_id}' != step.step_id '{step.step_id}'")

            if not step.name:
                errors.append(f"Step '{step_id}' must have a name")

        return errors

    def _validate_state_transitions(self) -> List[str]:
        """Validate that all referenced states exist."""
        errors = []
        all_states = set(self.steps.keys())

        for step_id, step in self.steps.items():
            # Check different step types for state references
            referenced_states = self._get_referenced_states(step)

            for state in referenced_states:
                if state not in all_states:
                    errors.append(f"Step '{step_id}' references unknown state: '{state}'")

        return errors

    def _get_referenced_states(self, step: WorkflowStep) -> Set[str]:
        """Get all states referenced by a step."""
        referenced_states = set()

        if isinstance(step, AutoTransitionStep):
            referenced_states.add(step.next_state)
        elif isinstance(step, APICallStep):
            referenced_states.add(step.success_state)
            referenced_states.add(step.failure_state)
        elif isinstance(step, ConditionStep):
            referenced_states.add(step.true_state)
            referenced_states.add(step.false_state)
        # Add more step types as needed...

        return referenced_states

    def _validate_reachability(self) -> List[str]:
        """Validate that all states are reachable from the initial state."""
        if not self.initial_step_id:
            return ["Cannot validate reachability without initial step"]

        reachable_states = self._find_reachable_states(self.initial_step_id)
        all_states = set(self.steps.keys())
        unreachable_states = all_states - reachable_states

        errors = []
        for state in unreachable_states:
            errors.append(f"State '{state}' is not reachable from initial state '{self.initial_step_id}'")

        return errors

    def _find_reachable_states(self, start_state: str) -> Set[str]:
        """Find all states reachable from the start state using DFS."""
        visited = set()
        stack = [start_state]

        while stack:
            current_state = stack.pop()
            if current_state in visited:
                continue

            visited.add(current_state)

            if current_state in self.steps:
                step = self.steps[current_state]
                next_states = self._get_referenced_states(step)

                for next_state in next_states:
                    if next_state not in visited:
                        stack.append(next_state)

        return visited

    def get_terminal_states(self) -> Set[str]:
        """Get all terminal states (states with no outgoing transitions)."""
        terminal_states = set()

        for step_id, step in self.steps.items():
            referenced_states = self._get_referenced_states(step)
            if not referenced_states:
                terminal_states.add(step_id)

        return terminal_states

    def has_cycles(self) -> bool:
        """Check if the workflow has cycles."""
        if not self.initial_step_id:
            return False

        visited = set()
        rec_stack = set()

        def has_cycle(state: str) -> bool:
            visited.add(state)
            rec_stack.add(state)

            if state in self.steps:
                step = self.steps[state]
                next_states = self._get_referenced_states(step)

                for next_state in next_states:
                    if next_state not in visited:
                        if has_cycle(next_state):
                            return True
                    elif next_state in rec_stack:
                        return True

            rec_stack.remove(state)
            return False

        return has_cycle(self.initial_step_id)

    def serialize(self) -> Dict[str, Any]:
        """Serialize the workflow definition to a dictionary."""
        # Convert to dict, but handle special cases like callables
        workflow_dict = self.model_dump(exclude={"steps"})

        # Handle steps separately
        steps_dict = {}
        for step_id, step in self.steps.items():
            step_type = step.__class__.__name__
            step_dict = step.model_dump(
                exclude={"action", "api_function", "condition", "aggregation_function", "llm_interface"}
            )
            step_dict["type"] = step_type
            steps_dict[step_id] = step_dict

        workflow_dict["steps"] = steps_dict
        return workflow_dict

# --------------------------------------------------------------


class WorkflowValidator:
    """Utility class for validating workflows."""

    @staticmethod
    def validate_workflow(workflow: WorkflowDefinition) -> List[str]:
        """Validate a workflow definition and return a list of errors."""
        try:
            workflow.validate()
            return []
        except WorkflowValidationError as e:
            return e.validation_errors

    @staticmethod
    def validate_workflow_collection(workflows: Dict[str, WorkflowDefinition]) -> Dict[str, List[str]]:
        """Validate a collection of workflows."""
        validation_results = {}

        for workflow_id, workflow in workflows.items():
            errors = WorkflowValidator.validate_workflow(workflow)
            if errors:
                validation_results[workflow_id] = errors

        return validation_results

    @staticmethod
    def check_workflow_dependencies(workflows: Dict[str, WorkflowDefinition]) -> List[str]:
        """Check for dependencies between workflows (if supported in the future)."""
        # Placeholder for future dependency checking
        return []

# --------------------------------------------------------------
