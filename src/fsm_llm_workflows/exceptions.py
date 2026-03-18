"""
Custom exceptions for the LLM-FSM Workflow System.
"""

from typing import Optional, Any, Dict


class WorkflowError(Exception):
    """Base exception for all workflow-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class WorkflowDefinitionError(WorkflowError):
    """Error in workflow definition structure or content."""

    def __init__(self, workflow_id: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.workflow_id = workflow_id
        super().__init__(f"Workflow '{workflow_id}': {message}", details)


class WorkflowStepError(WorkflowError):
    """Error during workflow step execution."""

    def __init__(self, step_id: str, message: str, cause: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.step_id = step_id
        self.cause = cause

        error_details = details or {}
        if cause:
            error_details['cause'] = str(cause)
            error_details['cause_type'] = type(cause).__name__

        super().__init__(f"Step '{step_id}': {message}", error_details)


class WorkflowInstanceError(WorkflowError):
    """Error related to workflow instance management."""

    def __init__(self, instance_id: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.instance_id = instance_id
        super().__init__(f"Instance '{instance_id}': {message}", details)


class WorkflowTimeoutError(WorkflowError):
    """Workflow operation timed out."""

    def __init__(self, operation: str, timeout_seconds: int, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.timeout_seconds = timeout_seconds

        error_details = details or {}
        error_details['timeout_seconds'] = timeout_seconds

        super().__init__(f"Operation '{operation}' timed out after {timeout_seconds} seconds", error_details)


class WorkflowValidationError(WorkflowError):
    """Validation error in workflow configuration."""

    def __init__(self, validation_errors: list, details: Optional[Dict[str, Any]] = None):
        self.validation_errors = validation_errors

        error_summary = f"Validation failed with {len(validation_errors)} error(s):\n"
        error_summary += "\n".join(f"  - {error}" for error in validation_errors)

        error_details = details or {}
        error_details['validation_errors'] = validation_errors

        super().__init__(error_summary, error_details)


class WorkflowStateError(WorkflowError):
    """Error related to workflow state transitions or management."""

    def __init__(self, current_state: str, operation: str, message: str,
                 details: Optional[Dict[str, Any]] = None):
        self.current_state = current_state
        self.operation = operation

        error_details = details or {}
        error_details['current_state'] = current_state
        error_details['operation'] = operation

        super().__init__(f"State '{current_state}' - {operation}: {message}", error_details)


class WorkflowEventError(WorkflowError):
    """Error in event processing or event listener management."""

    def __init__(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.event_type = event_type

        error_details = details or {}
        error_details['event_type'] = event_type

        super().__init__(f"Event '{event_type}': {message}", error_details)


class WorkflowResourceError(WorkflowError):
    """Error related to workflow resource management (timers, listeners, etc.)."""

    def __init__(self, resource_type: str, resource_id: str, message: str,
                 details: Optional[Dict[str, Any]] = None):
        self.resource_type = resource_type
        self.resource_id = resource_id

        error_details = details or {}
        error_details['resource_type'] = resource_type
        error_details['resource_id'] = resource_id

        super().__init__(f"{resource_type} '{resource_id}': {message}", error_details)