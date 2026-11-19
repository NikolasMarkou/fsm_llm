"""SIMPLE CALCULATOR FSM - Basic arithmetic operations.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
The model writes ONE expression; the CalculatorExecutor handler evaluates it on
entry to `perform_calculation` (`ReasoningHandlers.execute_calculation`).
"""

from fsm_llm.reasoning.constants import (
    CALCULATION_STATE,
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS

_EXPRESSION_INSTRUCTIONS = f"""
            Write the problem as ONE Python arithmetic expression:

            - Use only numbers, + - * / // % ** and parentheses
            - Use / for division, // only for whole-number division
            - "15% of 240" is "0.15 * 240"; "2^3" is "2 ** 3"; "(3 + 4) x 5" is "(3 + 4) * 5"
            - Write numbers without thousands separators ("1000", not "1,000")
            - Do NOT compute the answer yourself: a program evaluates the expression
            - Use problem_components if available, otherwise problem_statement

            {_IMPORTANT_NO_QUESTIONS} Write the expression from the available problem information.
            """

simple_calculator_fsm = {
    "name": "simple_calculator",
    "description": "Performs simple arithmetic calculations with error handling",
    "initial_state": "extract_elements",
    "persona": "You are a precise calculator that performs arithmetic operations accurately.",
    "handler_only_keys": [
        ContextKeys.CALCULATION_RESULT,
        ContextKeys.CALCULATION_ERROR,
    ],
    "states": {
        "extract_elements": {
            "description": "Write the problem as one arithmetic expression",
            "purpose": f"Extract '{ContextKeys.CALCULATION_EXPRESSION}'",
            "field_extractions": [
                {
                    "field_name": ContextKeys.CALCULATION_EXPRESSION,
                    "field_type": "str",
                    "extraction_instructions": _EXPRESSION_INSTRUCTIONS,
                }
            ],
            "extraction_instructions": _EXPRESSION_INSTRUCTIONS,
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": CALCULATION_STATE,
                    "description": "Expression extracted",
                }
            ],
        },
        CALCULATION_STATE: {
            "description": "The handler evaluates the expression (no model call)",
            "purpose": f"Hold '{ContextKeys.CALCULATION_RESULT}' or '{ContextKeys.CALCULATION_ERROR}'",
            "response_instructions": "",
            "transitions": [],
        },
    },
}
