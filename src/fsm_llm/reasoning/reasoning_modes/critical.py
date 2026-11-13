"""CRITICAL REASONING FSM - Rigorous evaluation of arguments and evidence.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

critical_fsm = {
    "name": "critical_reasoning",
    "description": "Systematic critical evaluation of arguments, claims, evidence, and reasoning to distinguish sound from unsound conclusions",
    "initial_state": "identify_claims",
    "persona": "You are a rigorous critical thinker who carefully evaluates arguments, evidence, and reasoning to separate truth from error and strong arguments from weak ones.",
    "states": {
        "identify_claims": {
            "description": "Identify and categorize the main claims and arguments",
            "purpose": f"Extract '{ContextKeys.CLAIMS}' and '{ContextKeys.ARGUMENTS}' from the problem or text",
            "required_context_keys": [ContextKeys.CLAIMS, ContextKeys.ARGUMENTS],
            "extraction_instructions": f"""
            Systematically identify what is being claimed or argued:

            - What are the main conclusions, assertions, or claims being made?
            - What specific arguments are presented to support these claims?
            - What is the overall thesis, position, or central argument?
            - Are there implicit or unstated assumptions underlying the arguments?
            - How are factual claims distinguished from opinions, interpretations, or value judgments?

            Clearly separate main arguments from supporting points and distinguish between different types of claims.

            {_IMPORTANT_NO_QUESTIONS} Work with the claims and arguments present in the given material.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "examine_evidence",
                    "description": "Claims and arguments identified",
                }
            ],
        },
        "examine_evidence": {
            "description": "Critically examine the quality and sufficiency of supporting evidence",
            "purpose": f"Assess '{ContextKeys.EVIDENCE_QUALITY}' and identify '{ContextKeys.EVIDENCE_GAPS}'",
            "required_context_keys": [
                ContextKeys.EVIDENCE_QUALITY,
                ContextKeys.EVIDENCE_GAPS,
            ],
            "extraction_instructions": f"""
            Rigorously examine the evidence supporting the identified claims:

            - What evidence is actually provided? Is it relevant, sufficient, and appropriate?
            - How reliable, credible, and authoritative are the sources?
            - Is the evidence current, representative, and methodologically sound?
            - What important evidence is missing that would strengthen or weaken the argument?
            - Are there potential biases, conflicts of interest, or limitations in how evidence was collected or presented?

            Be specific about both strengths and critical weaknesses of the evidence base.

            {_IMPORTANT_NO_QUESTIONS} Evaluate the evidence that is already available in the material.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {"target_state": "analyze_logic", "description": "Evidence examined"}
            ],
        },
        "analyze_logic": {
            "description": "Analyze logical structure and identify reasoning flaws",
            "purpose": f"Conduct '{ContextKeys.LOGICAL_ANALYSIS}', identify '{ContextKeys.ASSUMPTIONS}' and '{ContextKeys.FALLACIES}'",
            "required_context_keys": [
                ContextKeys.LOGICAL_ANALYSIS,
                ContextKeys.ASSUMPTIONS,
                ContextKeys.FALLACIES,
            ],
            "extraction_instructions": f"""
            Systematically analyze the logical structure and identify flaws:

            - Do the conclusions actually follow logically from the stated premises?
            - What key assumptions are being made (both stated and unstated)?
            - Are there identifiable logical fallacies (ad hominem, straw man, false dilemma, appeal to authority, etc.)?
            - Are there gaps in reasoning, unsupported logical leaps, or non sequiturs?
            - Is the reasoning internally consistent throughout the argument?

            Identify specific logical strengths and weaknesses with clear examples.

            {_IMPORTANT_NO_QUESTIONS} Analyze the logical structure of what has been provided.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "consider_alternatives",
                    "description": "Logic analyzed",
                }
            ],
        },
        "consider_alternatives": {
            "description": "Consider alternative explanations and strong counter-arguments",
            "purpose": f"Identify '{ContextKeys.ALTERNATIVE_EXPLANATIONS}' and '{ContextKeys.COUNTER_ARGUMENTS}'",
            "required_context_keys": [
                ContextKeys.ALTERNATIVE_EXPLANATIONS,
                ContextKeys.COUNTER_ARGUMENTS,
            ],
            "extraction_instructions": f"""
            Systematically consider alternatives and opposing viewpoints:

            - What alternative explanations, interpretations, or conclusions are plausible?
            - What are the strongest possible counter-arguments to the main claims?
            - What would informed opponents or skeptics of this position argue?
            - Are there other reasonable ways to interpret the same evidence?
            - What additional considerations or factors might change the evaluation?

            Present fair and intellectually honest alternatives rather than weak straw man arguments.

            {_IMPORTANT_NO_QUESTIONS} Generate thoughtful alternatives based on your systematic analysis.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "form_judgment",
                    "description": "Alternatives considered",
                }
            ],
        },
        "form_judgment": {
            "description": "Form comprehensive critical assessment with justified confidence level",
            "purpose": f"Provide '{ContextKeys.CRITICAL_ASSESSMENT}' and '{ContextKeys.CONFIDENCE_RATING}'",
            "required_context_keys": [
                ContextKeys.CRITICAL_ASSESSMENT,
                ContextKeys.CONFIDENCE_RATING,
            ],
            "extraction_instructions": f"""
            Form a well-reasoned overall critical judgment:

            - How strong and convincing are the arguments when all factors are considered?
            - What are the most significant strengths and critical weaknesses?
            - How much confidence should we reasonably have in the main claims?
            - What would most significantly strengthen or weaken these arguments?
            - What is your final, balanced assessment of the overall reasoning quality?

            Provide a fair, balanced evaluation with a confidence rating from 1-10 and clear justification.

            {_IMPORTANT_NO_QUESTIONS} Form your critical judgment based on your complete systematic analysis.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
