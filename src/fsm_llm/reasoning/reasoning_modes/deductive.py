"""DEDUCTIVE REASONING FSM - Logical reasoning from general to specific.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

deductive_fsm = {
    "name": "deductive_reasoning",
    "description": "Apply general principles and rules to reach specific conclusions through valid logical reasoning",
    "initial_state": "identify_premises",
    "persona": "You are a logical thinker who applies established principles and rules to reach certain conclusions through valid reasoning.",
    "states": {
        "identify_premises": {
            "description": "Identify the general rules, principles, and assumptions",
            "purpose": f"Establish '{ContextKeys.PREMISES}' and identify '{ContextKeys.ASSUMPTIONS}'",
            "required_context_keys": [ContextKeys.PREMISES, ContextKeys.ASSUMPTIONS],
            "extraction_instructions": f"""
            Identify the starting points for deductive reasoning:

            - What general rules, laws, or principles apply to this problem?
            - What facts or givens can we assume as true?
            - What established knowledge is relevant?
            - What premises does any argument rely on?
            - What assumptions are being made (both stated and unstated)?

            Be explicit about both obvious and hidden assumptions that underlie the reasoning.

            {_IMPORTANT_NO_QUESTIONS} Work with the premises and information already provided in the problem.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "apply_logic",
                    "description": "Premises and assumptions identified",
                }
            ],
        },
        "apply_logic": {
            "description": "Apply logical rules to derive conclusions step by step",
            "purpose": f"Document '{ContextKeys.LOGICAL_STEPS}' and '{ContextKeys.INTERMEDIATE_CONCLUSIONS}'",
            "required_context_keys": [
                ContextKeys.LOGICAL_STEPS,
                ContextKeys.INTERMEDIATE_CONCLUSIONS,
            ],
            "extraction_instructions": f"""
            Apply logical reasoning systematically:

            - What follows logically from the established premises?
            - What intermediate conclusions can be drawn at each step?
            - What logical rules or forms are being applied (modus ponens, syllogism, etc.)?
            - How does each step follow necessarily from the previous ones?
            - What can be concluded with certainty at each stage?

            Show your logical work clearly, step by step, ensuring valid inferences.

            {_IMPORTANT_NO_QUESTIONS} Apply logical reasoning to the premises you've identified.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "derive_conclusion",
                    "description": "Logical steps applied",
                }
            ],
        },
        "derive_conclusion": {
            "description": "Reach final conclusions and assess logical validity",
            "purpose": f"State final '{ContextKeys.CONCLUSION}' and assess '{ContextKeys.LOGICAL_VALIDITY}'",
            "required_context_keys": [
                ContextKeys.CONCLUSION,
                ContextKeys.LOGICAL_VALIDITY,
            ],
            "extraction_instructions": f"""
            Derive the final conclusion and validate the reasoning:

            - What specific conclusion follows from the complete logical chain?
            - Is the reasoning logically valid (do conclusions follow necessarily)?
            - Does the conclusion adequately address the original problem?
            - Are there any logical gaps, errors, or invalid inferences?
            - How certain can we be about the final conclusion?

            State your conclusion clearly and provide honest assessment of the logical validity.

            {_IMPORTANT_NO_QUESTIONS} Derive your conclusion from the logical steps you've taken.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
