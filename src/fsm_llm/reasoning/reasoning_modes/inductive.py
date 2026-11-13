"""INDUCTIVE REASONING FSM - Reasoning from specific to general patterns.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

inductive_fsm = {
    "name": "inductive_reasoning",
    "description": "Reason from specific observations to discover general patterns and principles",
    "initial_state": "gather_observations",
    "persona": "You are an empirical thinker who discovers patterns by carefully examining specific examples and building general understanding from evidence.",
    "states": {
        "gather_observations": {
            "description": "Collect and organize specific observations and data points",
            "purpose": f"Identify '{ContextKeys.OBSERVATIONS}' and '{ContextKeys.DATA_POINTS}' relevant to the problem",
            "required_context_keys": [
                ContextKeys.OBSERVATIONS,
                ContextKeys.DATA_POINTS,
            ],
            "extraction_instructions": f"""
            Systematically gather specific, concrete observations:

            - What specific examples, cases, or instances are available?
            - What data points, measurements, or facts are relevant?
            - What have we observed in similar situations or contexts?
            - What specific behaviors, outcomes, or phenomena are documented?
            - What concrete evidence is available for analysis?

            Focus on collecting concrete, specific observations rather than general statements or theories.

            {_IMPORTANT_NO_QUESTIONS} Work with the observations and data available in the problem context.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "identify_commonalities",
                    "description": "Observations gathered",
                }
            ],
        },
        "identify_commonalities": {
            "description": "Find patterns and commonalities across observations",
            "purpose": f"Identify '{ContextKeys.COMMONALITIES}' and '{ContextKeys.TRENDS}' in the data",
            "required_context_keys": [ContextKeys.COMMONALITIES, ContextKeys.TRENDS],
            "extraction_instructions": f"""
            Systematically look for patterns across your observations:

            - What do multiple cases, examples, or instances have in common?
            - What trends, regularities, or consistencies emerge from the data?
            - What relationships appear consistently across different observations?
            - What factors or variables seem to be associated or correlated?
            - What sequences, progressions, or developmental patterns do you notice?

            Identify both obvious and subtle patterns that might not be immediately apparent.

            {_IMPORTANT_NO_QUESTIONS} Find patterns in the observations you've systematically gathered.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "form_hypothesis",
                    "description": "Commonalities identified",
                }
            ],
        },
        "form_hypothesis": {
            "description": "Form general hypothesis based on observed patterns",
            "purpose": f"Create '{ContextKeys.HYPOTHESIS}' supported by '{ContextKeys.SUPPORTING_EVIDENCE}'",
            "required_context_keys": [
                ContextKeys.HYPOTHESIS,
                ContextKeys.SUPPORTING_EVIDENCE,
            ],
            "extraction_instructions": f"""
            Form a well-grounded general hypothesis from the identified patterns:

            - What general rule, principle, or law might explain the observed patterns?
            - What predictions can you make about future or unobserved cases?
            - What broader principle or generalization seems to apply?
            - How would you clearly state this generalization?
            - What specific evidence best supports this hypothesis?

            Make your hypothesis specific enough to be testable while being general enough to be useful for prediction.

            {_IMPORTANT_NO_QUESTIONS} Form your hypothesis based on the patterns you've systematically identified.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "test_generalization",
                    "description": "Hypothesis formed",
                }
            ],
        },
        "test_generalization": {
            "description": "Test the strength and limits of the generalization",
            "purpose": f"Evaluate with '{ContextKeys.TEST_RESULTS}', '{ContextKeys.COUNTER_EXAMPLES}', and '{ContextKeys.GENERALIZATION_STRENGTH}'",
            "required_context_keys": [
                ContextKeys.TEST_RESULTS,
                ContextKeys.COUNTER_EXAMPLES,
                ContextKeys.GENERALIZATION_STRENGTH,
            ],
            "extraction_instructions": f"""
            Rigorously test your generalization against available evidence:

            - How well does it predict or explain other cases not used in forming it?
            - What counter-examples, exceptions, or contradictory evidence exist?
            - Under what conditions does the generalization hold or fail?
            - How strong is the inductive support based on sample size and quality?
            - What would strengthen or weaken confidence in this generalization?

            Rate the strength of your generalization from 1-10 based on the quality and quantity of supporting evidence.

            {_IMPORTANT_NO_QUESTIONS} Test your generalization rigorously with all available information.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
