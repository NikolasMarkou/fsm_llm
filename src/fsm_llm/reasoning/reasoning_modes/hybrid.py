"""HYBRID REASONING FSM - Integrated multi-approach reasoning with loop prevention.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    HYBRID_EVALUATION_STATE,
    HYBRID_FIRST_STATE,
    ContextKeys,
    Defaults,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

hybrid_fsm = {
    "name": "hybrid_reasoning",
    "description": "Systematically combines multiple reasoning approaches for comprehensive problem solving with loop prevention mechanisms",
    "initial_state": HYBRID_FIRST_STATE,
    "persona": "You are a master strategist who skillfully combines different reasoning approaches to tackle complex problems from multiple complementary angles.",
    # The loop counter is written only by the HybridLoopCounter handler on
    # critical_evaluation exit (D-054); the model never sets it.
    "handler_only_keys": [ContextKeys.HYBRID_LOOP_COUNT],
    "states": {
        HYBRID_FIRST_STATE: {
            "description": "Break problem into components requiring different reasoning approaches",
            "purpose": f"Map '{ContextKeys.PROBLEM_ASPECTS}' to reasoning types in '{ContextKeys.REASONING_MAP}'",
            "required_context_keys": [
                ContextKeys.PROBLEM_ASPECTS,
                ContextKeys.REASONING_MAP,
            ],
            "extraction_instructions": f"""
            Systematically analyze the problem to identify aspects requiring different reasoning approaches:

            - What parts need systematic analytical breakdown and decomposition?
            - What components require logical deduction from established principles?
            - Where might creative, innovative thinking provide valuable insights?
            - What aspects need critical evaluation of arguments or evidence?
            - What patterns might inductive reasoning help reveal from available data?

            Create a comprehensive map of problem aspects to appropriate reasoning approaches.

            {_IMPORTANT_NO_QUESTIONS} Analyze the problem as presented to identify reasoning needs.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "apply_analytical",
                    "description": "Components identified and mapped to reasoning approaches",
                }
            ],
        },
        "apply_analytical": {
            "description": "Apply systematic analytical reasoning to understand problem structure",
            "purpose": f"Create '{ContextKeys.ANALYTICAL_BREAKDOWN}' and '{ContextKeys.COMPONENT_RELATIONSHIPS}'",
            "required_context_keys": [
                ContextKeys.ANALYTICAL_BREAKDOWN,
                ContextKeys.COMPONENT_RELATIONSHIPS,
            ],
            "extraction_instructions": f"""
            Apply systematic analytical thinking to the problem:

            - Break down complex aspects into simpler, more manageable components
            - Identify relationships, dependencies, and interactions between components
            - Understand the underlying structure and organizational principles
            - Analyze how different elements contribute to the overall problem
            - Document systematic findings from the analytical decomposition

            Provide thorough analytical breakdown that will inform other reasoning approaches.

            {_IMPORTANT_NO_QUESTIONS} Proceed with systematic analytical breakdown based on your component identification.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "apply_logical",
                    "description": "Analytical reasoning systematically applied",
                }
            ],
        },
        "apply_logical": {
            "description": "Apply logical reasoning to derive sound conclusions",
            "purpose": f"Establish '{ContextKeys.LOGICAL_CONCLUSIONS}' and '{ContextKeys.REASONING_CHAIN}'",
            "required_context_keys": [
                ContextKeys.LOGICAL_CONCLUSIONS,
                ContextKeys.REASONING_CHAIN,
            ],
            "extraction_instructions": f"""
            Apply systematic logical reasoning to the analytical findings:

            - What can be logically deduced from the analytical breakdown and identified relationships?
            - What logical steps follow necessarily from the evidence and established facts?
            - What conclusions can be drawn with high confidence based on logical inference?
            - How do the logical pieces fit together to form coherent understanding?
            - What clear chain of reasoning emerges from the logical analysis?

            Build a clear, valid chain of logical reasoning that builds on the analytical foundation.

            {_IMPORTANT_NO_QUESTIONS} Apply logical reasoning systematically to your analytical findings.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "apply_creative",
                    "description": "Logical reasoning systematically applied",
                }
            ],
        },
        "apply_creative": {
            "description": "Apply creative thinking to generate novel approaches and insights",
            "purpose": f"Generate '{ContextKeys.CREATIVE_INSIGHTS}' and '{ContextKeys.NOVEL_APPROACHES}'",
            "required_context_keys": [
                ContextKeys.CREATIVE_INSIGHTS,
                ContextKeys.NOVEL_APPROACHES,
            ],
            "extraction_instructions": f"""
            Apply creative thinking to complement and enhance the analytical and logical work:

            - What new perspectives or insights emerge from viewing the analysis creatively?
            - How might innovative approaches complement or extend the logical conclusions?
            - What novel solutions become possible when we think beyond conventional boundaries?
            - What creative connections or unexpected relationships can enhance understanding?
            - How can creative thinking add value to the systematic analysis already completed?

            Generate creative insights that meaningfully enhance rather than contradict the systematic reasoning.

            {_IMPORTANT_NO_QUESTIONS} Generate creative insights building on your comprehensive analytical and logical work.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": HYBRID_EVALUATION_STATE,
                    "description": "Creative reasoning systematically applied",
                }
            ],
        },
        HYBRID_EVALUATION_STATE: {
            "description": "Critically evaluate all findings with systematic loop prevention",
            "purpose": f"Create '{ContextKeys.EVALUATION_RESULTS}' and determine if refinement needed (maximum {Defaults.MAX_HYBRID_LOOPS} loops)",
            "required_context_keys": [ContextKeys.EVALUATION_RESULTS],
            "extraction_instructions": f"""
            Critically evaluate the integration of all reasoning approaches:

            - How effectively do the analytical, logical, and creative approaches complement each other?
            - Are there significant contradictions, gaps, or inconsistencies that need resolution?
            - What are the key strengths and potential weaknesses of this combined approach?
            - Are there critical flaws or missing elements that would justify returning for refinement?
            - How robust and comprehensive is the overall reasoning when considered together?

            Set needs_refinement=True only for serious, fundamental issues.

            {_IMPORTANT_NO_QUESTIONS} Evaluate comprehensively based on your multi-faceted analysis.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "integrate_solution",
                    "description": "Ready to integrate (no critical issues or loop limit reached)",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "No critical issues found or maximum loops reached",
                            "logic": {
                                "or": [
                                    {
                                        "!=": [
                                            {"var": ContextKeys.NEEDS_REFINEMENT},
                                            True,
                                        ]
                                    },
                                    {
                                        ">=": [
                                            {"var": [ContextKeys.HYBRID_LOOP_COUNT, 0]},
                                            Defaults.MAX_HYBRID_LOOPS,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": HYBRID_FIRST_STATE,
                    "description": "Refinement needed (limited loops remaining)",
                    "priority": 2,
                    "conditions": [
                        {
                            "description": "Critical issues found and loops available",
                            "logic": {
                                "and": [
                                    {
                                        "==": [
                                            {"var": ContextKeys.NEEDS_REFINEMENT},
                                            True,
                                        ]
                                    },
                                    {
                                        "<": [
                                            {"var": [ContextKeys.HYBRID_LOOP_COUNT, 0]},
                                            Defaults.MAX_HYBRID_LOOPS,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
            ],
        },
        "integrate_solution": {
            "description": "Integrate insights from all reasoning approaches into comprehensive solution",
            "purpose": f"Create '{ContextKeys.INTEGRATED_SOLUTION}' with '{ContextKeys.REASONING_SYNTHESIS_NOTES}'",
            "required_context_keys": [
                ContextKeys.INTEGRATED_SOLUTION,
                ContextKeys.REASONING_SYNTHESIS_NOTES,
            ],
            "extraction_instructions": f"""
            Systematically integrate all reasoning approaches into a comprehensive, unified solution:

            - How do analytical insights, logical conclusions, and creative innovations combine synergistically?
            - What is the most complete and nuanced understanding of the problem that emerges?
            - How do different reasoning types reinforce, complement, or constructively challenge each other?
            - What is the best integrated approach to solving or addressing the problem?
            - How does the combination create understanding that is greater than the sum of individual parts?

            Synthesize rather than merely summarize—create genuine integration that leverages the strengths of each approach.

            {_IMPORTANT_NO_QUESTIONS} Integrate systematically based on your comprehensive multi-approach analysis.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "finalize_hybrid",
                    "description": "Solution comprehensively integrated",
                }
            ],
        },
        "finalize_hybrid": {
            "description": "Present final hybrid solution with complete reasoning synthesis",
            "purpose": f"Finalize '{ContextKeys.FINAL_HYBRID_SOLUTION}' and '{ContextKeys.REASONING_SYNTHESIS}'",
            "required_context_keys": [
                ContextKeys.FINAL_HYBRID_SOLUTION,
                ContextKeys.REASONING_SYNTHESIS,
            ],
            "extraction_instructions": f"""
            Present the final comprehensive hybrid solution with complete synthesis:

            - What is your final, most comprehensive and well-reasoned solution?
            - How did each reasoning type (analytical, logical, creative, critical) contribute uniquely?
            - What is the particular strength and advantage of this multi-faceted reasoning approach?
            - How confident are you in this hybrid solution and why?
            - What would be the most logical next steps for implementation, testing, or further development?

            Provide a complete synthesis that demonstrates and showcases the power of systematically combined reasoning approaches.

            {_IMPORTANT_NO_QUESTIONS} Present your final integrated solution based on the complete hybrid reasoning process.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
