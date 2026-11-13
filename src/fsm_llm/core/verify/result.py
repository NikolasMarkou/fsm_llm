"""The result container of one validation run (errors, warnings, info).

Layer ``core`` (plan d454fd62, step 75): split out of the old ``fsm_llm.validator``.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.logging import logger


class FSMValidationResult:
    """
    Container for FSM validation results.

    This class stores and categorizes validation findings into errors, warnings,
    and informational messages for comprehensive FSM analysis.

    Attributes:
        fsm_name (str): Name of the validated FSM
        is_valid (bool): Overall validity status (True if no errors)
        errors (list[str]): List of errors that make the FSM invalid
        warnings (list[str]): List of warnings about potential issues
        info (list[str]): List of informational insights about the FSM
    """

    def __init__(self, fsm_name: str):
        """
        Initialize a validation result container.

        Args:
            fsm_name: Name of the FSM being validated
        """
        self.fsm_name = fsm_name
        self.is_valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def add_error(self, message: str):
        """
        Add an error message and mark the FSM as invalid.

        Errors represent critical issues that prevent the FSM from functioning correctly.

        Args:
            message: The error message to add
        """
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"Validation error in {self.fsm_name}: {message}")

    def add_warning(self, message: str):
        """
        Add a warning message.

        Warnings represent potential issues that don't make the FSM invalid
        but might cause problems or indicate design flaws.

        Args:
            message: The warning message to add
        """
        self.warnings.append(message)
        logger.warning(f"Validation warning in {self.fsm_name}: {message}")

    def add_info(self, message: str):
        """
        Add an informational message.

        Info messages provide insights about the FSM structure and characteristics.

        Args:
            message: The informational message to add
        """
        self.info.append(message)
        logger.info(message)

    def as_dict(self) -> dict[str, Any]:
        """
        Convert the result to a dictionary.

        Returns:
            Dict containing validation results in a structured format
        """
        return {
            "fsm_name": self.fsm_name,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }

    def __str__(self) -> str:
        """
        Convert the result to a human-readable string for display.

        Returns:
            Formatted string representation of validation results
        """
        lines = [f"=== Validation Results for {self.fsm_name} ==="]

        # Overall status
        if self.is_valid:
            lines.append("[VALID] FSM is VALID")
        else:
            lines.append("[INVALID] FSM is INVALID")

        # Group and format different message types
        if self.errors:
            lines.append("\n[ERROR] ERRORS:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")

        if self.warnings:
            lines.append("\n[WARN] WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")

        if self.info:
            lines.append("\n[INFO] INFO:")
            for i, info in enumerate(self.info, 1):
                lines.append(f"  {i}. {info}")

        return "\n".join(lines)
