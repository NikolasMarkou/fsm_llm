"""
Custom exceptions for the FSM-LLM Workflow System.

Every subclass of ``WorkflowError`` is built like the core exceptions:
``Cls(message, *, <id>=None, details=None)``. The message comes first, the
identifying data are keyword-only and are kept as attributes of the same name.
When an id is given, the text carries it as a prefix (``"Step 'a': ..."``);
an id left out adds no prefix.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.definitions import FSMError


class WorkflowError(FSMError):
    """Base exception for all workflow-related errors."""


class WorkflowDefinitionError(WorkflowError):
    """Error in workflow definition structure or content."""

    def __init__(
        self,
        message: str,
        *,
        workflow_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.workflow_id = workflow_id
        text = (
            message if workflow_id is None else f"Workflow '{workflow_id}': {message}"
        )
        super().__init__(text, details)


class WorkflowStepError(WorkflowError):
    """Error during workflow step execution."""

    def __init__(
        self,
        message: str,
        *,
        step_id: str | None = None,
        cause: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.step_id = step_id
        self.cause = cause

        error_details = dict(details or {})
        if cause:
            error_details["cause"] = str(cause)
            error_details["cause_type"] = type(cause).__name__

        text = message if step_id is None else f"Step '{step_id}': {message}"
        super().__init__(text, error_details)


class WorkflowInstanceError(WorkflowError):
    """Error related to workflow instance management."""

    def __init__(
        self,
        message: str,
        *,
        instance_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.instance_id = instance_id
        text = (
            message if instance_id is None else f"Instance '{instance_id}': {message}"
        )
        super().__init__(text, details)


class WorkflowTimeoutError(WorkflowError):
    """Workflow operation timed out."""

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        instance_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        #: The workflow instance that timed out, when known. Lets a caller of
        #: ``start`` find the (now FAILED) instance even though the
        #: call raised before returning its id.
        self.instance_id = instance_id

        error_details = dict(details or {})
        if timeout_seconds is not None:
            error_details["timeout_seconds"] = timeout_seconds
        if instance_id is not None:
            error_details["instance_id"] = instance_id

        super().__init__(message, error_details)


class WorkflowValidationError(WorkflowError):
    """Validation error in workflow configuration.

    ``validation_errors`` are appended to ``message``, one per line.
    """

    def __init__(
        self,
        message: str,
        *,
        validation_errors: list | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.validation_errors = list(validation_errors or [])

        text = message
        if self.validation_errors:
            text += ":\n" + "\n".join(
                f"  - {error}" for error in self.validation_errors
            )

        error_details = dict(details or {})
        error_details["validation_errors"] = self.validation_errors

        super().__init__(text, error_details)


class WorkflowStateError(WorkflowError):
    """Error related to workflow state transitions or management."""

    def __init__(
        self,
        message: str,
        *,
        current_step_id: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.current_step_id = current_step_id
        self.operation = operation

        error_details = dict(details or {})
        if current_step_id is not None:
            error_details["current_step_id"] = current_step_id
        if operation is not None:
            error_details["operation"] = operation

        prefix = "" if current_step_id is None else f"State '{current_step_id}'"
        if operation is not None:
            prefix = f"{prefix} - {operation}" if prefix else operation
        super().__init__(f"{prefix}: {message}" if prefix else message, error_details)


class WorkflowEventError(WorkflowError):
    """Error in event processing or event listener management."""

    def __init__(
        self,
        message: str,
        *,
        event_type: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.event_type = event_type

        error_details = dict(details or {})
        if event_type is not None:
            error_details["event_type"] = event_type

        text = message if event_type is None else f"Event '{event_type}': {message}"
        super().__init__(text, error_details)


class WorkflowResourceError(WorkflowError):
    """Error related to workflow resource management (timers, listeners, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id

        error_details = dict(details or {})
        if resource_type is not None:
            error_details["resource_type"] = resource_type
        if resource_id is not None:
            error_details["resource_id"] = resource_id

        prefix = " ".join(
            part
            for part in (
                resource_type,
                None if resource_id is None else f"'{resource_id}'",
            )
            if part
        )
        super().__init__(f"{prefix}: {message}" if prefix else message, error_details)
