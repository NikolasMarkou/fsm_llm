"""ABDUCTIVE REASONING FSM - Finding best explanations for observations.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

abductive_fsm = {
    "name": "abductive_reasoning",
    "description": "Find the best explanation for puzzling observations through systematic inference to the best explanation",
    "initial_state": "identify_observations",
    "persona": "You are a detective and investigator who excels at finding the most plausible explanations for puzzling observations and unexplained phenomena.",
    "states": {
        "identify_observations": {
            "description": "Identify key observations that require explanation",
            "purpose": f"Catalog '{ContextKeys.OBSERVATIONS}' and identify '{ContextKeys.SURPRISING_ELEMENTS}' that require explanation",
            "required_context_keys": [
                ContextKeys.OBSERVATIONS,
                ContextKeys.SURPRISING_ELEMENTS,
            ],
            "extraction_instructions": f"""
            Systematically identify what needs to be explained:

            - What specific facts, phenomena, or observations are we trying to explain?
            - What seems surprising, unexpected, anomalous, or puzzling?
            - What patterns, behaviors, or outcomes need to be accounted for?
            - What data points or evidence require explanation?
            - What aspects of the situation seem to call for further understanding?

            Focus on concrete, specific observations rather than interpretations or preliminary explanations.

            {_IMPORTANT_NO_QUESTIONS} Work with the observations already provided in the problem context.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "generate_hypotheses",
                    "description": "Key observations identified",
                }
            ],
        },
        "generate_hypotheses": {
            "description": "Generate multiple potential explanations",
            "purpose": f"Create '{ContextKeys.POTENTIAL_HYPOTHESES}' with '{ContextKeys.HYPOTHESIS_RATIONALES}' for each explanation",
            "required_context_keys": [
                ContextKeys.POTENTIAL_HYPOTHESES,
                ContextKeys.HYPOTHESIS_RATIONALES,
            ],
            "extraction_instructions": f"""
            Generate multiple plausible explanations for the observations:

            - What could reasonably account for what we observe?
            - Consider different types of causes (direct, indirect, systemic, multiple contributing factors)
            - Think about both obvious and less obvious potential explanations
            - Include competing or alternative hypotheses that might explain the same phenomena
            - Consider explanations at different levels (individual, systemic, environmental, etc.)

            For each hypothesis, provide a clear rationale explaining why it could account for the observations.

            {_IMPORTANT_NO_QUESTIONS} Generate comprehensive hypotheses based on the observations you've identified.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "evaluate_hypotheses",
                    "description": "Hypotheses generated",
                }
            ],
        },
        "evaluate_hypotheses": {
            "description": "Systematically evaluate each hypothesis against standard criteria",
            "purpose": f"Create '{ContextKeys.HYPOTHESIS_EVALUATIONS}' using '{ContextKeys.EVALUATION_CRITERIA}' for systematic assessment",
            "required_context_keys": [
                ContextKeys.HYPOTHESIS_EVALUATIONS,
                ContextKeys.EVALUATION_CRITERIA,
            ],
            "extraction_instructions": f"""
            Systematically evaluate each hypothesis using standard criteria for explanatory adequacy:

            - Explanatory scope: How comprehensively does it explain the observations?
            - Simplicity/parsimony: Is it unnecessarily complex or does it invoke minimal assumptions?
            - Plausibility: How likely is it given our background knowledge and experience?
            - Testability: Can it be verified, falsified, or further investigated?
            - Consistency: Does it fit coherently with other established knowledge?
            - Predictive power: Does it suggest new predictions or help anticipate future observations?

            Rate each hypothesis systematically on these dimensions with clear justifications.

            {_IMPORTANT_NO_QUESTIONS} Evaluate based on the hypotheses you've systematically generated.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "select_best_explanation",
                    "description": "Hypotheses evaluated",
                }
            ],
        },
        "select_best_explanation": {
            "description": "Select most plausible explanation with clear justification",
            "purpose": f"Choose '{ContextKeys.BEST_HYPOTHESIS}' with '{ContextKeys.SELECTION_JUSTIFICATION}', '{ContextKeys.CONFIDENCE_IN_EXPLANATION}', and '{ContextKeys.NEXT_STEPS_FOR_VALIDATION}'",
            "required_context_keys": [
                ContextKeys.BEST_HYPOTHESIS,
                ContextKeys.SELECTION_JUSTIFICATION,
                ContextKeys.CONFIDENCE_IN_EXPLANATION,
                ContextKeys.NEXT_STEPS_FOR_VALIDATION,
            ],
            "extraction_instructions": f"""
            Select the best explanation through careful comparative analysis:

            - Which hypothesis best balances all the evaluative criteria?
            - Why is this the most plausible and compelling explanation overall?
            - What is your confidence level in this explanation (1-10 scale) and why?
            - What would you need to do to further test, validate, or investigate this explanation?
            - What are the key strengths of this explanation and what limitations or uncertainties remain?

            Provide clear, detailed justification for your selection and acknowledge appropriate levels of uncertainty.

            {_IMPORTANT_NO_QUESTIONS} Make your selection based on your systematic evaluation process.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
