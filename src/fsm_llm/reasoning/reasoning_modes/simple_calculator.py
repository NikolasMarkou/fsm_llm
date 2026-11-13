"""SIMPLE CALCULATOR FSM - Basic arithmetic operations.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

simple_calculator_fsm = {
    "name": "simple_calculator",
    "description": "Performs simple arithmetic calculations with error handling",
    "initial_state": "extract_elements",
    "persona": "You are a precise calculator that performs arithmetic operations accurately.",
    "states": {
        "extract_elements": {
            "description": "Extract operands and operator from problem",
            "purpose": f"Extract '{ContextKeys.OPERAND1}', '{ContextKeys.OPERAND2}', and '{ContextKeys.OPERATOR}'",
            "required_context_keys": [
                ContextKeys.OPERAND1,
                ContextKeys.OPERAND2,
                ContextKeys.OPERATOR,
            ],
            "extraction_instructions": f"""
            Extract the mathematical elements from the problem:

            - Identify the first number (operand1)
            - Identify the second number (operand2)
            - Identify the operation (+, -, *, /, ^, etc.)
            - Handle decimal numbers and negative numbers
            - Extract from problem_components if available, otherwise parse from problem_statement

            For expressions like "2 + 3", set operand1=2, operand2=3, operator="+"

            {_IMPORTANT_NO_QUESTIONS} Extract elements from the available problem information.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "perform_calculation",
                    "description": "Elements extracted successfully",
                }
            ],
        },
        "perform_calculation": {
            "description": "Calculate the arithmetic result",
            "purpose": f"Calculate and store result in '{ContextKeys.CALCULATION_RESULT}'",
            "required_context_keys": [ContextKeys.CALCULATION_RESULT],
            "extraction_instructions": f"""
            Perform the arithmetic calculation:

            - Execute the operation: operand1 operator operand2
            - Handle basic operations: +, -, *, /, ^(power)
            - Manage edge cases: division by zero, overflow, invalid operations
            - Store the numerical result in calculation_result
            - If error occurs, store error description in calculation_error

            Examples:
            - 2 + 3 = 5
            - 10 / 2 = 5
            - 5 * 4 = 20
            - 2^3 = 8

            {_IMPORTANT_NO_QUESTIONS} Perform calculation with extracted elements.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
