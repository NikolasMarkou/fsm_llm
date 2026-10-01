"""
Complete FSM definitions for the reasoning engine.

Contains all finite state machine definitions for reasoning strategies:
orchestrator, classifier, and 9 specialized reasoning FSMs.
All FSMs use standardized context keys from constants.py.

Every key the model writes is one core typed field (``field_extractions``)
with its own instructions and a narrowed prompt context; no state has
state-level ``extraction_instructions`` (no bulk extraction call), and only
the states whose reply is kept (``final_answer`` and each strategy FSM's
terminal state) have ``response_instructions``: every other state is silent
and makes no Pass-2 call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm import typed_field_extraction
from fsm_llm.definitions import TypedFieldType

from .constants import (
    ANSWER_RESPONSE_INSTRUCTIONS,
    COMPOSE_INSTRUCTION,
    HYBRID_EVALUATION_STATE,
    MERGED_RESULT_KEYS,
    ORCHESTRATOR_HANDLER_ONLY_KEYS,
    ClassifierStates,
    ContextKeys,
    Defaults,
    OrchestratorStates,
)

_C = COMPOSE_INSTRUCTION
_K = ContextKeys

# Keys the first state of every strategy FSM reads besides the problem
# statement: what the engine passes into a pushed strategy FSM.
_PROBLEM_READS: tuple[str, ...] = (
    _K.PROBLEM_TYPE,
    _K.PROBLEM_COMPONENTS,
    _K.CONSTRAINTS,
)


def _field(
    name: str,
    field_type: TypedFieldType,
    instructions: str,
    *,
    reads: Sequence[str] = (),
    required: bool = True,
) -> dict[str, Any]:
    """One typed field of a reasoning state.

    Contract: core ``typed_field_extraction`` with ``context_keys`` set to
    ``problem_statement`` followed by ``reads`` (the keys the value is worked
    out from; a key absent from context is simply not shown). Raises what the
    core builder raises (an envelope-key name, an internal-prefixed key).
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-055: every model-written
    # reasoning key is a typed field on a narrowed context. Do NOT put back
    # state-level extraction_instructions or required_context_keys for these
    # keys: the first buys a bulk call per step, the second mints an untyped
    # `any` field that reads the whole context (reasoning_trace included).
    # Do NOT add a field for a handler-only key (ORCHESTRATOR_HANDLER_ONLY_KEYS,
    # hybrid_loop_count): the model must never write the verdict or a counter.
    return typed_field_extraction(
        name,
        field_type,
        instructions,
        context_keys=(_K.PROBLEM_STATEMENT, *reads),
        required=required,
    )


def _state(
    state_id: str,
    description: str,
    purpose: str,
    fields: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    *,
    response_instructions: str = "",
) -> dict[str, Any]:
    """One reasoning state: typed fields only, silent unless given
    ``response_instructions`` (only answer states are, see the module
    docstring). Returns the raw state dict ``FSMDefinition`` accepts."""
    return {
        "id": state_id,
        "description": description,
        "purpose": purpose,
        "field_extractions": list(fields),
        "response_instructions": response_instructions,
        "transitions": list(transitions),
    }


def _then(target_state: str, description: str) -> list[dict[str, Any]]:
    """The single unconditional transition of a linear chain state."""
    return [{"target_state": target_state, "description": description}]


# ============================================================================
# ORCHESTRATOR FSM - Main control flow with retry management
# ============================================================================

_STRATEGY_NAMES = (
    "simple_calculator (arithmetic), analytical (systematic breakdown), "
    "deductive (logic from premises), inductive (patterns from cases), "
    "creative (new ideas), critical (judging an argument or evidence), "
    "abductive (best explanation of a phenomenon), analogical (reasoning by "
    "similarity), hybrid (several approaches combined)"
)

orchestrator_fsm = {
    "name": "reasoning_orchestrator",
    "description": "Orchestrates various reasoning strategies with retry limits and loop prevention.",
    "initial_state": OrchestratorStates.PROBLEM_ANALYSIS,
    "persona": "You are a reasoning guide helping to solve problems step by step. Be clear, logical, and thorough.",
    # DECISION plan-2026-10-01T093600-944e2692/D-054: the validation verdict,
    # retry counters, confidence and strategy choice are handler-owned. Do NOT
    # take a key out of this list to let the model "help" (a bulk reply with
    # validation_result: true would open the gate) and do NOT list them in a
    # state's required_context_keys (never extracted, the validator warns).
    "handler_only_keys": list(ORCHESTRATOR_HANDLER_ONLY_KEYS),
    "states": {
        OrchestratorStates.PROBLEM_ANALYSIS: _state(
            OrchestratorStates.PROBLEM_ANALYSIS,
            "Initial analysis of the problem",
            f"Analyze the '{_K.PROBLEM_STATEMENT}' to identify '{_K.PROBLEM_TYPE}' and '{_K.PROBLEM_COMPONENTS}'",
            [
                _field(
                    _K.PROBLEM_TYPE,
                    "str",
                    f"{_C}A short label for the kind of problem in "
                    "problem_statement: arithmetic, logic, explanation, "
                    "creative, evaluation, analysis or similar. Use "
                    "'arithmetic' for a calculation.",
                ),
                _field(
                    _K.PROBLEM_COMPONENTS,
                    "list",
                    f"{_C}A JSON list of the key parts of the problem: the "
                    "given facts, quantities or premises, and what is asked. "
                    "For a calculation, the numbers and the operation.",
                ),
            ],
            [
                {
                    "target_state": OrchestratorStates.STRATEGY_SELECTION,
                    "description": "Problem analyzed successfully",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Problem type and components identified",
                            "requires_context_keys": [
                                _K.PROBLEM_TYPE,
                                _K.PROBLEM_COMPONENTS,
                            ],
                        }
                    ],
                }
            ],
        ),
        OrchestratorStates.STRATEGY_SELECTION: _state(
            OrchestratorStates.STRATEGY_SELECTION,
            "Select appropriate reasoning strategy",
            f"Choose '{_K.REASONING_STRATEGY}' based on problem analysis. For arithmetic, choose 'simple_calculator'.",
            [
                _field(
                    _K.REASONING_STRATEGY,
                    "str",
                    f"{_C}The reasoning strategy for the problem, exactly one "
                    f"of: {_STRATEGY_NAMES}. Use the classified_problem_type "
                    "value when it is set; use simple_calculator when "
                    "problem_type is arithmetic.",
                    reads=(
                        _K.PROBLEM_TYPE,
                        _K.PROBLEM_COMPONENTS,
                        _K.CLASSIFIED_PROBLEM_TYPE,
                    ),
                ),
                _field(
                    _K.STRATEGY_RATIONALE,
                    "str",
                    f"{_C}One sentence on why the reasoning_strategy value "
                    "fits the problem.",
                    reads=(_K.PROBLEM_TYPE, _K.REASONING_STRATEGY),
                ),
            ],
            _then(OrchestratorStates.EXECUTE_REASONING, "Strategy selected"),
        ),
        # The strategy FSM runs here, pushed and popped by the engine's
        # before_step hook: nothing to extract, nothing to say.
        OrchestratorStates.EXECUTE_REASONING: _state(
            OrchestratorStates.EXECUTE_REASONING,
            "Execute selected reasoning strategy",
            "Apply the chosen reasoning approach through specialized FSM execution",
            [],
            _then(OrchestratorStates.SYNTHESIZE_SOLUTION, "Reasoning completed"),
        ),
        OrchestratorStates.SYNTHESIZE_SOLUTION: _state(
            OrchestratorStates.SYNTHESIZE_SOLUTION,
            "Synthesize solution from reasoning results",
            f"Create '{_K.PROPOSED_SOLUTION}' and '{_K.KEY_INSIGHTS}' from reasoning results",
            [
                _field(
                    _K.PROPOSED_SOLUTION,
                    "any",
                    f"{_C}Your complete answer to problem_statement, built "
                    "from the strategy results given (for a calculation, the "
                    "calculation_result value). State the answer itself "
                    "first, then the reasoning that supports it.",
                    reads=MERGED_RESULT_KEYS,
                ),
                _field(
                    _K.KEY_INSIGHTS,
                    "list",
                    f"{_C}A JSON list of 2 to 4 short insights that support "
                    "the answer.",
                    reads=(
                        _K.PROPOSED_SOLUTION,
                        *(k for k in MERGED_RESULT_KEYS if k != _K.KEY_INSIGHTS),
                    ),
                ),
            ],
            _then(OrchestratorStates.VALIDATE_REFINE, "Solution synthesized"),
        ),
        # The verdict and the retry counters are written by handlers only
        # (D-054): this state extracts nothing.
        OrchestratorStates.VALIDATE_REFINE: _state(
            OrchestratorStates.VALIDATE_REFINE,
            "Validate solution with retry limit protection",
            f"Check '{_K.VALIDATION_RESULT}' and retry if needed (max {Defaults.MAX_RETRIES} times)",
            [],
            [
                {
                    "target_state": OrchestratorStates.FINAL_ANSWER,
                    "description": "Solution valid or max retries reached",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Valid solution or retry limit hit",
                            "logic": {
                                "or": [
                                    {"==": [{"var": _K.VALIDATION_RESULT}, True]},
                                    {"==": [{"var": _K.MAX_RETRIES_REACHED}, True]},
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": OrchestratorStates.EXECUTE_REASONING,
                    "description": "Retry reasoning (if under limit)",
                    "priority": 2,
                    "conditions": [
                        {
                            "description": "Invalid and can retry",
                            "logic": {
                                "and": [
                                    {"==": [{"var": _K.VALIDATION_RESULT}, False]},
                                    {"!=": [{"var": _K.MAX_RETRIES_REACHED}, True]},
                                ]
                            },
                        }
                    ],
                },
            ],
        ),
        OrchestratorStates.FINAL_ANSWER: _state(
            OrchestratorStates.FINAL_ANSWER,
            "Present final answer with complete reasoning trace",
            f"Set '{_K.FINAL_SOLUTION}' and final metadata",
            [
                _field(
                    _K.FINAL_SOLUTION,
                    "any",
                    f"{_C}The final answer to problem_statement as the user "
                    "will read it: the proposed_solution value restated "
                    "clearly and completely (without a proposed_solution, "
                    "your own best answer to the problem). When "
                    "max_retries_reached is true, add one sentence saying the "
                    "answer could not be fully validated.",
                    reads=(
                        _K.PROPOSED_SOLUTION,
                        _K.KEY_INSIGHTS,
                        _K.MAX_RETRIES_REACHED,
                    ),
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# CLASSIFIER FSM - Problem classification and strategy recommendation
# ============================================================================

# The classifier's reply is never read (the engine reads its context only), so
# even its terminal state is silent.
_CLASSIFIER_READS: tuple[str, ...] = (_K.PROBLEM_TYPE, _K.PROBLEM_COMPONENTS)

classifier_fsm = {
    "name": "problem_classifier",
    "description": "Classifies problems to determine the most appropriate reasoning strategy",
    "initial_state": ClassifierStates.ANALYZE_DOMAIN,
    "persona": "You are an expert problem analyst who identifies the best reasoning approach for any given problem.",
    "states": {
        ClassifierStates.ANALYZE_DOMAIN: _state(
            ClassifierStates.ANALYZE_DOMAIN,
            "Identify problem domain and context",
            f"Determine '{_K.PROBLEM_DOMAIN}' and '{_K.DOMAIN_INDICATORS}'",
            [
                _field(
                    _K.PROBLEM_DOMAIN,
                    "str",
                    f"{_C}The primary domain of the problem, one of: "
                    "mathematics, logic, creativity, analysis, evaluation, "
                    "empirical, explanatory, comparative.",
                    reads=_CLASSIFIER_READS,
                ),
                _field(
                    _K.DOMAIN_INDICATORS,
                    "list",
                    f"{_C}A JSON list of the features of the problem that "
                    "point to the problem_domain value (numbers and "
                    "operations, premises, a request for ideas, an argument "
                    "to judge, observations to explain, a comparison).",
                    reads=(*_CLASSIFIER_READS, _K.PROBLEM_DOMAIN),
                ),
            ],
            _then(ClassifierStates.ANALYZE_STRUCTURE, "Domain identified"),
        ),
        ClassifierStates.ANALYZE_STRUCTURE: _state(
            ClassifierStates.ANALYZE_STRUCTURE,
            "Analyze problem structure and complexity",
            f"Identify '{_K.PROBLEM_STRUCTURE}' and '{_K.STRUCTURAL_ELEMENTS}'",
            [
                _field(
                    _K.PROBLEM_STRUCTURE,
                    "str",
                    f"{_C}The structure of the problem, one of: simple (one "
                    "step), sequential, hierarchical, network, complex.",
                    reads=(*_CLASSIFIER_READS, _K.PROBLEM_DOMAIN),
                ),
                _field(
                    _K.STRUCTURAL_ELEMENTS,
                    "list",
                    f"{_C}A JSON list of the problem's structural elements: "
                    "its components, relationships, dependencies and "
                    "constraints.",
                    reads=(*_CLASSIFIER_READS, _K.PROBLEM_STRUCTURE),
                ),
            ],
            _then(ClassifierStates.IDENTIFY_REASONING_NEEDS, "Structure analyzed"),
        ),
        ClassifierStates.IDENTIFY_REASONING_NEEDS: _state(
            ClassifierStates.IDENTIFY_REASONING_NEEDS,
            "Identify specific reasoning requirements",
            f"Determine '{_K.REASONING_REQUIREMENTS}' and '{_K.KEY_CHALLENGES}'",
            [
                _field(
                    _K.REASONING_REQUIREMENTS,
                    "str",
                    f"{_C}The kind of reasoning the problem needs, in a few "
                    "words: decomposition, deduction from premises, pattern "
                    "generalization, idea generation, critical evaluation, "
                    "best explanation, analogy, or several combined. Use "
                    "'direct computation' for a plain calculation.",
                    reads=(
                        *_CLASSIFIER_READS,
                        _K.PROBLEM_DOMAIN,
                        _K.PROBLEM_STRUCTURE,
                    ),
                ),
                _field(
                    _K.KEY_CHALLENGES,
                    "list",
                    f"{_C}A JSON list of the main challenges the reasoning "
                    "must handle.",
                    reads=(*_CLASSIFIER_READS, _K.REASONING_REQUIREMENTS),
                ),
            ],
            _then(ClassifierStates.RECOMMEND_STRATEGY, "Needs identified"),
        ),
        ClassifierStates.RECOMMEND_STRATEGY: _state(
            ClassifierStates.RECOMMEND_STRATEGY,
            "Recommend optimal reasoning strategy",
            f"Set '{_K.RECOMMENDED_REASONING_TYPE}', '{_K.STRATEGY_JUSTIFICATION}', and '{_K.ALTERNATIVE_APPROACHES}'",
            [
                _field(
                    _K.RECOMMENDED_REASONING_TYPE,
                    "str",
                    f"{_C}The best reasoning strategy for the problem, exactly "
                    f"one of: {_STRATEGY_NAMES}.",
                    reads=(
                        *_CLASSIFIER_READS,
                        _K.PROBLEM_DOMAIN,
                        _K.PROBLEM_STRUCTURE,
                        _K.REASONING_REQUIREMENTS,
                    ),
                ),
                _field(
                    _K.STRATEGY_JUSTIFICATION,
                    "str",
                    f"{_C}One sentence on why the recommended_reasoning_type "
                    "value fits the problem.",
                    reads=(_K.REASONING_REQUIREMENTS, _K.RECOMMENDED_REASONING_TYPE),
                ),
                _field(
                    _K.ALTERNATIVE_APPROACHES,
                    "list",
                    f"{_C}A JSON list of 1 or 2 other strategy names from the "
                    "same set that could also work.",
                    reads=(_K.RECOMMENDED_REASONING_TYPE,),
                    required=False,
                ),
            ],
            [],
        ),
    },
}


# ============================================================================
# SIMPLE CALCULATOR FSM - Basic arithmetic operations
# ============================================================================

simple_calculator_fsm = {
    "name": "simple_calculator",
    "description": "Performs simple arithmetic calculations with error handling",
    "initial_state": "extract_elements",
    "persona": "You are a precise calculator that performs arithmetic operations accurately.",
    "states": {
        "extract_elements": _state(
            "extract_elements",
            "Extract operands and operator from problem",
            f"Extract '{_K.OPERAND1}', '{_K.OPERAND2}', and '{_K.OPERATOR}'",
            [
                _field(
                    _K.OPERAND1,
                    "any",
                    "The first number of the calculation the problem asks "
                    "for (decimals and negative numbers allowed).",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.OPERAND2,
                    "any",
                    "The second number of the calculation the problem asks "
                    "for (decimals and negative numbers allowed).",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.OPERATOR,
                    "str",
                    "The operation between the two numbers: +, -, *, /, ^ "
                    "(power) or %.",
                    reads=_PROBLEM_READS,
                ),
            ],
            _then("perform_calculation", "Elements extracted successfully"),
        ),
        "perform_calculation": _state(
            "perform_calculation",
            "Calculate the arithmetic result",
            f"Calculate and store result in '{_K.CALCULATION_RESULT}'",
            [
                _field(
                    _K.CALCULATION_RESULT,
                    "any",
                    f"{_C}The result of the calculation problem_statement asks "
                    "for: work it out step by step (operand1 operator "
                    "operand2 for a single operation) and give the final "
                    "number, with its unit when the problem has one.",
                    reads=(_K.OPERAND1, _K.OPERAND2, _K.OPERATOR),
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# ANALYTICAL REASONING FSM - Systematic decomposition and analysis
# ============================================================================

_ANALYTICAL_DECOMPOSED = (_K.COMPONENTS, _K.ATTRIBUTES, _K.RELATIONSHIPS)

analytical_fsm = {
    "name": "analytical_reasoning",
    "description": "Analytical reasoning through systematic decomposition and component analysis",
    "initial_state": "decompose",
    "persona": "You are a methodical analytical thinker who breaks down complex problems systematically.",
    "states": {
        "decompose": _state(
            "decompose",
            "Break down the problem into component parts",
            f"Identify '{_K.COMPONENTS}', '{_K.ATTRIBUTES}', and '{_K.RELATIONSHIPS}'",
            [
                _field(
                    _K.COMPONENTS,
                    "list",
                    f"{_C}A JSON list of the smaller, manageable parts the "
                    "problem breaks into.",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.ATTRIBUTES,
                    "any",
                    f"{_C}The key attributes of each component in the "
                    "components value.",
                    reads=(_K.COMPONENTS,),
                ),
                _field(
                    _K.RELATIONSHIPS,
                    "list",
                    f"{_C}A JSON list of short statements of how the "
                    "components relate to and depend on each other.",
                    reads=(_K.COMPONENTS,),
                ),
            ],
            _then("analyze_components", "Decomposition complete"),
        ),
        "analyze_components": _state(
            "analyze_components",
            "Analyze each component in detail",
            f"Create '{_K.COMPONENT_ANALYSIS}' and identify '{_K.DATA_REQUIREMENTS}'",
            [
                _field(
                    _K.COMPONENT_ANALYSIS,
                    "any",
                    f"{_C}Your analysis of each component: its role, its "
                    "properties, how it contributes to the whole and how "
                    "much it matters.",
                    reads=_ANALYTICAL_DECOMPOSED,
                ),
                _field(
                    _K.DATA_REQUIREMENTS,
                    "list",
                    f"{_C}A JSON list of extra information a complete "
                    "analysis would need (an empty list when nothing is "
                    "missing).",
                    reads=(_K.COMPONENTS, _K.COMPONENT_ANALYSIS),
                ),
            ],
            _then("identify_patterns", "Component analysis complete"),
        ),
        "identify_patterns": _state(
            "identify_patterns",
            "Find patterns and dependencies between components",
            f"Identify '{_K.PATTERNS}', '{_K.CAUSAL_LINKS}', and '{_K.DEPENDENCIES}'",
            [
                _field(
                    _K.PATTERNS,
                    "list",
                    f"{_C}A JSON list of the recurring patterns and "
                    "regularities across the components.",
                    reads=(*_ANALYTICAL_DECOMPOSED, _K.COMPONENT_ANALYSIS),
                ),
                _field(
                    _K.CAUSAL_LINKS,
                    "list",
                    f"{_C}A JSON list of cause-effect links between the "
                    "components, each as 'cause -> effect'.",
                    reads=(*_ANALYTICAL_DECOMPOSED, _K.COMPONENT_ANALYSIS),
                ),
                _field(
                    _K.DEPENDENCIES,
                    "list",
                    f"{_C}A JSON list of what depends on what, including any "
                    "feedback loops.",
                    reads=(*_ANALYTICAL_DECOMPOSED, _K.COMPONENT_ANALYSIS),
                ),
            ],
            _then("integrate_findings", "Patterns identified"),
        ),
        "integrate_findings": _state(
            "integrate_findings",
            "Synthesize understanding from all analytical work",
            f"Create '{_K.INTEGRATED_ANALYSIS}' and '{_K.KEY_INSIGHTS}'",
            [
                _field(
                    _K.INTEGRATED_ANALYSIS,
                    "any",
                    f"{_C}Your integrated answer to problem_statement drawn "
                    "from the analysis: what the components, patterns and "
                    "causal links mean together, ending with the conclusion "
                    "that answers the problem.",
                    reads=(
                        _K.COMPONENTS,
                        _K.COMPONENT_ANALYSIS,
                        _K.PATTERNS,
                        _K.CAUSAL_LINKS,
                        _K.DEPENDENCIES,
                    ),
                ),
                _field(
                    _K.KEY_INSIGHTS,
                    "list",
                    f"{_C}A JSON list of the 2 to 4 most important insights "
                    "of the analysis.",
                    reads=(_K.INTEGRATED_ANALYSIS, _K.PATTERNS, _K.CAUSAL_LINKS),
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# DEDUCTIVE REASONING FSM - Logical reasoning from general to specific
# ============================================================================

deductive_fsm = {
    "name": "deductive_reasoning",
    "description": "Apply general principles and rules to reach specific conclusions through valid logical reasoning",
    "initial_state": "identify_premises",
    "persona": "You are a logical thinker who applies established principles and rules to reach certain conclusions through valid reasoning.",
    "states": {
        "identify_premises": _state(
            "identify_premises",
            "Identify the general rules, principles, and assumptions",
            f"Establish '{_K.PREMISES}' and identify '{_K.ASSUMPTIONS}'",
            [
                _field(
                    _K.PREMISES,
                    "list",
                    f"{_C}A JSON list of the premises: the rules, principles "
                    "and facts the problem gives or relies on, each as a "
                    "short statement.",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.ASSUMPTIONS,
                    "list",
                    f"{_C}A JSON list of the stated and unstated assumptions "
                    "the reasoning rests on.",
                    reads=(_K.PREMISES,),
                ),
            ],
            _then("apply_logic", "Premises and assumptions identified"),
        ),
        "apply_logic": _state(
            "apply_logic",
            "Apply logical rules to derive conclusions step by step",
            f"Document '{_K.LOGICAL_STEPS}' and '{_K.INTERMEDIATE_CONCLUSIONS}'",
            [
                _field(
                    _K.LOGICAL_STEPS,
                    "list",
                    f"{_C}A JSON list of the inference steps from the "
                    "premises, in order, each naming the rule it uses "
                    "(modus ponens, modus tollens, syllogism, ...).",
                    reads=(_K.PREMISES, _K.ASSUMPTIONS),
                ),
                _field(
                    _K.INTERMEDIATE_CONCLUSIONS,
                    "list",
                    f"{_C}A JSON list of what each step establishes with certainty.",
                    reads=(_K.PREMISES, _K.LOGICAL_STEPS),
                ),
            ],
            _then("derive_conclusion", "Logical steps applied"),
        ),
        "derive_conclusion": _state(
            "derive_conclusion",
            "Reach final conclusions and assess logical validity",
            f"State final '{_K.CONCLUSION}' and assess '{_K.LOGICAL_VALIDITY}'",
            [
                _field(
                    _K.CONCLUSION,
                    "any",
                    f"{_C}The conclusion that answers problem_statement, "
                    "stated plainly (a yes or no first when the problem asks "
                    "a yes/no question), followed by the one-line reason.",
                    reads=(
                        _K.PREMISES,
                        _K.LOGICAL_STEPS,
                        _K.INTERMEDIATE_CONCLUSIONS,
                    ),
                ),
                _field(
                    _K.LOGICAL_VALIDITY,
                    "bool",
                    "Your judgment: true when the conclusion follows "
                    "necessarily from the premises, false when the chain has "
                    "a gap or an invalid step.",
                    reads=(_K.PREMISES, _K.LOGICAL_STEPS, _K.CONCLUSION),
                    required=False,
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# INDUCTIVE REASONING FSM - Reasoning from specific to general patterns
# ============================================================================

inductive_fsm = {
    "name": "inductive_reasoning",
    "description": "Reason from specific observations to discover general patterns and principles",
    "initial_state": "gather_observations",
    "persona": "You are an empirical thinker who discovers patterns by carefully examining specific examples and building general understanding from evidence.",
    "states": {
        "gather_observations": _state(
            "gather_observations",
            "Collect and organize specific observations and data points",
            f"Identify '{_K.OBSERVATIONS}' and '{_K.DATA_POINTS}' relevant to the problem",
            [
                _field(
                    _K.OBSERVATIONS,
                    "list",
                    f"{_C}A JSON list of concrete, specific observations, "
                    "examples or cases relevant to the problem.",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.DATA_POINTS,
                    "list",
                    f"{_C}A JSON list of the facts, measurements or data "
                    "points relevant to the problem.",
                    reads=(*_PROBLEM_READS, _K.OBSERVATIONS),
                ),
            ],
            _then("identify_commonalities", "Observations gathered"),
        ),
        "identify_commonalities": _state(
            "identify_commonalities",
            "Find patterns and commonalities across observations",
            f"Identify '{_K.COMMONALITIES}' and '{_K.TRENDS}' in the data",
            [
                _field(
                    _K.COMMONALITIES,
                    "list",
                    f"{_C}A JSON list of what several observations have in common.",
                    reads=(_K.OBSERVATIONS, _K.DATA_POINTS),
                ),
                _field(
                    _K.TRENDS,
                    "list",
                    f"{_C}A JSON list of the trends, regularities and "
                    "correlations across the observations.",
                    reads=(_K.OBSERVATIONS, _K.DATA_POINTS),
                ),
            ],
            _then("form_hypothesis", "Commonalities identified"),
        ),
        "form_hypothesis": _state(
            "form_hypothesis",
            "Form general hypothesis based on observed patterns",
            f"Create '{_K.HYPOTHESIS}' supported by '{_K.SUPPORTING_EVIDENCE}'",
            [
                _field(
                    _K.HYPOTHESIS,
                    "any",
                    f"{_C}The general rule or principle that explains the "
                    "observed patterns and answers problem_statement, stated "
                    "specifically enough to test.",
                    reads=(_K.OBSERVATIONS, _K.COMMONALITIES, _K.TRENDS),
                ),
                _field(
                    _K.SUPPORTING_EVIDENCE,
                    "list",
                    f"{_C}A JSON list of the observations that best support "
                    "the hypothesis.",
                    reads=(_K.OBSERVATIONS, _K.HYPOTHESIS),
                ),
            ],
            _then("test_generalization", "Hypothesis formed"),
        ),
        "test_generalization": _state(
            "test_generalization",
            "Test the strength and limits of the generalization",
            f"Evaluate with '{_K.TEST_RESULTS}', '{_K.COUNTER_EXAMPLES}', and '{_K.GENERALIZATION_STRENGTH}'",
            [
                _field(
                    _K.TEST_RESULTS,
                    "any",
                    f"{_C}How well the hypothesis predicts or explains cases "
                    "not used to form it, and when it holds or fails.",
                    reads=(_K.HYPOTHESIS, _K.SUPPORTING_EVIDENCE),
                ),
                _field(
                    _K.COUNTER_EXAMPLES,
                    "list",
                    f"{_C}A JSON list of counter-examples or exceptions to the "
                    "hypothesis (an empty list when none is known).",
                    reads=(_K.HYPOTHESIS,),
                ),
                _field(
                    _K.GENERALIZATION_STRENGTH,
                    "float",
                    "Your rating from 1 to 10 of how strongly the evidence "
                    "supports the hypothesis.",
                    reads=(_K.HYPOTHESIS, _K.SUPPORTING_EVIDENCE, _K.TEST_RESULTS),
                    required=False,
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# CREATIVE REASONING FSM - Novel solution generation through creative thinking
# ============================================================================

creative_fsm = {
    "name": "creative_reasoning",
    "description": "Generate novel and innovative solutions through divergent and convergent creative thinking processes",
    "initial_state": "explore_perspectives",
    "persona": "You are an innovative creative thinker who generates novel solutions by seeing problems from fresh perspectives and making unexpected connections.",
    "states": {
        "explore_perspectives": _state(
            "explore_perspectives",
            "Explore the problem from multiple creative perspectives",
            f"Generate '{_K.PERSPECTIVES}' and '{_K.REFRAMINGS}' of the problem",
            [
                _field(
                    _K.PERSPECTIVES,
                    "list",
                    f"{_C}A JSON list of different angles on the problem (how "
                    "a child, an artist, an engineer or someone from another "
                    "field would see it).",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.REFRAMINGS,
                    "list",
                    f"{_C}A JSON list of fresh ways to frame the problem "
                    "(flipped assumptions, removed constraints, metaphors).",
                    reads=(*_PROBLEM_READS, _K.PERSPECTIVES),
                ),
            ],
            _then("generate_ideas", "Multiple perspectives explored"),
        ),
        "generate_ideas": _state(
            "generate_ideas",
            "Brainstorm creative and unconventional ideas without judgment",
            f"Create '{_K.CREATIVE_IDEAS}' and '{_K.UNCONVENTIONAL_APPROACHES}'",
            [
                _field(
                    _K.CREATIVE_IDEAS,
                    "list",
                    f"{_C}A JSON list of many creative ideas for the problem, "
                    "favouring novelty over practicality.",
                    reads=(_K.PERSPECTIVES, _K.REFRAMINGS),
                ),
                _field(
                    _K.UNCONVENTIONAL_APPROACHES,
                    "list",
                    f"{_C}A JSON list of approaches that break conventional "
                    "thinking about the problem.",
                    reads=(_K.PERSPECTIVES, _K.REFRAMINGS),
                ),
            ],
            _then("combine_concepts", "Ideas generated"),
        ),
        "combine_concepts": _state(
            "combine_concepts",
            "Combine and synthesize ideas in novel ways",
            f"Create '{_K.COMBINATIONS}' and develop '{_K.NOVEL_SOLUTIONS}'",
            [
                _field(
                    _K.COMBINATIONS,
                    "list",
                    f"{_C}A JSON list of new combinations of the ideas, each "
                    "merging elements of two or more.",
                    reads=(_K.CREATIVE_IDEAS, _K.UNCONVENTIONAL_APPROACHES),
                ),
                _field(
                    _K.NOVEL_SOLUTIONS,
                    "list",
                    f"{_C}A JSON list of complete, novel solutions to the "
                    "problem built from those combinations.",
                    reads=(
                        _K.CREATIVE_IDEAS,
                        _K.UNCONVENTIONAL_APPROACHES,
                        _K.COMBINATIONS,
                    ),
                ),
            ],
            _then("evaluate_novelty", "Concepts combined"),
        ),
        "evaluate_novelty": _state(
            "evaluate_novelty",
            "Evaluate creative solutions for novelty, feasibility, and impact",
            f"Select '{_K.BEST_CREATIVE_SOLUTION}' and rate '{_K.INNOVATION_RATING}'",
            [
                _field(
                    _K.BEST_CREATIVE_SOLUTION,
                    "any",
                    f"{_C}Your final creative answer to problem_statement, "
                    "complete as asked (when the problem asks for several "
                    "items, give all of them), with a short reason for the "
                    "choice.",
                    reads=(
                        _K.CREATIVE_IDEAS,
                        _K.COMBINATIONS,
                        _K.NOVEL_SOLUTIONS,
                    ),
                ),
                _field(
                    _K.INNOVATION_RATING,
                    "float",
                    "Your rating from 1 to 10 of how novel and original the "
                    "chosen answer is.",
                    reads=(_K.BEST_CREATIVE_SOLUTION,),
                    required=False,
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# CRITICAL REASONING FSM - Rigorous evaluation of arguments and evidence
# ============================================================================

critical_fsm = {
    "name": "critical_reasoning",
    "description": "Systematic critical evaluation of arguments, claims, evidence, and reasoning to distinguish sound from unsound conclusions",
    "initial_state": "identify_claims",
    "persona": "You are a rigorous critical thinker who carefully evaluates arguments, evidence, and reasoning to separate truth from error and strong arguments from weak ones.",
    "states": {
        "identify_claims": _state(
            "identify_claims",
            "Identify and categorize the main claims and arguments",
            f"Extract '{_K.CLAIMS}' and '{_K.ARGUMENTS}' from the problem or text",
            [
                _field(
                    _K.CLAIMS,
                    "list",
                    f"{_C}A JSON list of the main claims or conclusions made "
                    "in the problem, each marked as fact, opinion or value "
                    "judgment.",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.ARGUMENTS,
                    "list",
                    f"{_C}A JSON list of the arguments offered for those "
                    "claims, including unstated ones.",
                    reads=(*_PROBLEM_READS, _K.CLAIMS),
                ),
            ],
            _then("examine_evidence", "Claims and arguments identified"),
        ),
        "examine_evidence": _state(
            "examine_evidence",
            "Critically examine the quality and sufficiency of supporting evidence",
            f"Assess '{_K.EVIDENCE_QUALITY}' and identify '{_K.EVIDENCE_GAPS}'",
            [
                _field(
                    _K.EVIDENCE_QUALITY,
                    "any",
                    f"{_C}Your assessment of the evidence behind the claims: "
                    "is it relevant, sufficient, representative and free of "
                    "bias?",
                    reads=(_K.CLAIMS, _K.ARGUMENTS),
                ),
                _field(
                    _K.EVIDENCE_GAPS,
                    "list",
                    f"{_C}A JSON list of the evidence that is missing and "
                    "would strengthen or weaken the argument.",
                    reads=(_K.CLAIMS, _K.ARGUMENTS, _K.EVIDENCE_QUALITY),
                ),
            ],
            _then("analyze_logic", "Evidence examined"),
        ),
        "analyze_logic": _state(
            "analyze_logic",
            "Analyze logical structure and identify reasoning flaws",
            f"Conduct '{_K.LOGICAL_ANALYSIS}', identify '{_K.ASSUMPTIONS}' and '{_K.FALLACIES}'",
            [
                _field(
                    _K.LOGICAL_ANALYSIS,
                    "any",
                    f"{_C}Your analysis of whether the conclusions follow "
                    "from the premises: gaps, leaps and inconsistencies.",
                    reads=(_K.CLAIMS, _K.ARGUMENTS, _K.EVIDENCE_QUALITY),
                ),
                _field(
                    _K.ASSUMPTIONS,
                    "list",
                    f"{_C}A JSON list of the stated and unstated assumptions "
                    "the argument makes.",
                    reads=(_K.CLAIMS, _K.ARGUMENTS),
                ),
                _field(
                    _K.FALLACIES,
                    "list",
                    f"{_C}A JSON list of the logical fallacies in the "
                    "argument, each named (ad hominem, straw man, false "
                    "dilemma, appeal to authority, ...) with a few words on "
                    "where it occurs; an empty list when there is none.",
                    reads=(_K.CLAIMS, _K.ARGUMENTS, _K.LOGICAL_ANALYSIS),
                ),
            ],
            _then("consider_alternatives", "Logic analyzed"),
        ),
        "consider_alternatives": _state(
            "consider_alternatives",
            "Consider alternative explanations and strong counter-arguments",
            f"Identify '{_K.ALTERNATIVE_EXPLANATIONS}' and '{_K.COUNTER_ARGUMENTS}'",
            [
                _field(
                    _K.ALTERNATIVE_EXPLANATIONS,
                    "list",
                    f"{_C}A JSON list of other plausible interpretations or "
                    "conclusions from the same evidence.",
                    reads=(_K.CLAIMS, _K.LOGICAL_ANALYSIS),
                ),
                _field(
                    _K.COUNTER_ARGUMENTS,
                    "list",
                    f"{_C}A JSON list of the strongest counter-arguments to "
                    "the main claims.",
                    reads=(_K.CLAIMS, _K.LOGICAL_ANALYSIS, _K.FALLACIES),
                ),
            ],
            _then("form_judgment", "Alternatives considered"),
        ),
        "form_judgment": _state(
            "form_judgment",
            "Form comprehensive critical assessment with justified confidence level",
            f"Provide '{_K.CRITICAL_ASSESSMENT}' and '{_K.CONFIDENCE_RATING}'",
            [
                _field(
                    _K.CRITICAL_ASSESSMENT,
                    "any",
                    f"{_C}Your overall judgment of the argument in "
                    "problem_statement: whether it holds, its main flaws "
                    "named, and why.",
                    reads=(
                        _K.CLAIMS,
                        _K.EVIDENCE_GAPS,
                        _K.LOGICAL_ANALYSIS,
                        _K.FALLACIES,
                        _K.COUNTER_ARGUMENTS,
                    ),
                ),
                _field(
                    _K.CONFIDENCE_RATING,
                    "float",
                    "Your rating from 1 to 10 of how confident you are in "
                    "that judgment.",
                    reads=(_K.CRITICAL_ASSESSMENT,),
                    required=False,
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# ABDUCTIVE REASONING FSM - Finding best explanations for observations
# ============================================================================

abductive_fsm = {
    "name": "abductive_reasoning",
    "description": "Find the best explanation for puzzling observations through systematic inference to the best explanation",
    "initial_state": "identify_observations",
    "persona": "You are a detective and investigator who excels at finding the most plausible explanations for puzzling observations and unexplained phenomena.",
    "states": {
        "identify_observations": _state(
            "identify_observations",
            "Identify key observations that require explanation",
            f"Catalog '{_K.OBSERVATIONS}' and identify '{_K.SURPRISING_ELEMENTS}' that require explanation",
            [
                _field(
                    _K.OBSERVATIONS,
                    "list",
                    f"{_C}A JSON list of the concrete facts or phenomena the "
                    "problem asks to explain.",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.SURPRISING_ELEMENTS,
                    "list",
                    f"{_C}A JSON list of what is surprising or puzzling in "
                    "those observations.",
                    reads=(_K.OBSERVATIONS,),
                ),
            ],
            _then("generate_hypotheses", "Key observations identified"),
        ),
        "generate_hypotheses": _state(
            "generate_hypotheses",
            "Generate multiple potential explanations",
            f"Create '{_K.POTENTIAL_HYPOTHESES}' with '{_K.HYPOTHESIS_RATIONALES}' for each explanation",
            [
                _field(
                    _K.POTENTIAL_HYPOTHESES,
                    "list",
                    f"{_C}A JSON list of 2 to 4 competing explanations that "
                    "could account for the observations.",
                    reads=(_K.OBSERVATIONS, _K.SURPRISING_ELEMENTS),
                ),
                _field(
                    _K.HYPOTHESIS_RATIONALES,
                    "any",
                    f"{_C}For each potential hypothesis, why it could account "
                    "for the observations.",
                    reads=(_K.OBSERVATIONS, _K.POTENTIAL_HYPOTHESES),
                ),
            ],
            _then("evaluate_hypotheses", "Hypotheses generated"),
        ),
        "evaluate_hypotheses": _state(
            "evaluate_hypotheses",
            "Systematically evaluate each hypothesis against standard criteria",
            f"Create '{_K.HYPOTHESIS_EVALUATIONS}' using '{_K.EVALUATION_CRITERIA}' for systematic assessment",
            [
                _field(
                    _K.HYPOTHESIS_EVALUATIONS,
                    "any",
                    f"{_C}Your evaluation of each potential hypothesis for "
                    "explanatory scope, simplicity, plausibility, "
                    "testability and consistency with known facts.",
                    reads=(
                        _K.OBSERVATIONS,
                        _K.POTENTIAL_HYPOTHESES,
                        _K.HYPOTHESIS_RATIONALES,
                    ),
                ),
                _field(
                    _K.EVALUATION_CRITERIA,
                    "list",
                    f"{_C}A JSON list of the criteria the evaluation used.",
                    reads=(_K.HYPOTHESIS_EVALUATIONS,),
                ),
            ],
            _then("select_best_explanation", "Hypotheses evaluated"),
        ),
        "select_best_explanation": _state(
            "select_best_explanation",
            "Select most plausible explanation with clear justification",
            f"Choose '{_K.BEST_HYPOTHESIS}' with '{_K.SELECTION_JUSTIFICATION}', '{_K.CONFIDENCE_IN_EXPLANATION}', and '{_K.NEXT_STEPS_FOR_VALIDATION}'",
            [
                _field(
                    _K.BEST_HYPOTHESIS,
                    "any",
                    f"{_C}The most plausible explanation, the one that "
                    "answers problem_statement, stated plainly with the "
                    "mechanism behind it.",
                    reads=(
                        _K.OBSERVATIONS,
                        _K.POTENTIAL_HYPOTHESES,
                        _K.HYPOTHESIS_EVALUATIONS,
                    ),
                ),
                _field(
                    _K.SELECTION_JUSTIFICATION,
                    "any",
                    f"{_C}Why that explanation beats the others.",
                    reads=(_K.BEST_HYPOTHESIS, _K.HYPOTHESIS_EVALUATIONS),
                ),
                _field(
                    _K.CONFIDENCE_IN_EXPLANATION,
                    "float",
                    "Your rating from 1 to 10 of how confident you are in "
                    "the best_hypothesis value.",
                    reads=(_K.BEST_HYPOTHESIS, _K.HYPOTHESIS_EVALUATIONS),
                    required=False,
                ),
                _field(
                    _K.NEXT_STEPS_FOR_VALIDATION,
                    "list",
                    f"{_C}A JSON list of ways to test or confirm the explanation.",
                    reads=(_K.BEST_HYPOTHESIS,),
                    required=False,
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# ANALOGICAL REASONING FSM - Transfer insights through analogical thinking
# ============================================================================

analogical_fsm = {
    "name": "analogical_reasoning",
    "description": "Transfer insights and solutions via systematic analogical reasoning and pattern matching across domains",
    "initial_state": "define_target_problem",
    "persona": "You are an expert at finding meaningful connections and analogies. You help solve problems by identifying similar situations and transferring insights across domains.",
    "states": {
        "define_target_problem": _state(
            "define_target_problem",
            "Clearly define and characterize the target problem",
            f"Analyze the problem to identify '{_K.TARGET_PROBLEM_DESCRIPTION}' and '{_K.KEY_FEATURES_OF_TARGET}'",
            [
                _field(
                    _K.TARGET_PROBLEM_DESCRIPTION,
                    "any",
                    f"{_C}The core challenge of the problem and the kind of "
                    "answer it seeks, in one or two sentences.",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.KEY_FEATURES_OF_TARGET,
                    "list",
                    f"{_C}A JSON list of the structural and functional "
                    "features a good analogy must share with the problem.",
                    reads=(*_PROBLEM_READS, _K.TARGET_PROBLEM_DESCRIPTION),
                ),
            ],
            _then("find_source_analogs", "Target problem clearly defined"),
        ),
        "find_source_analogs": _state(
            "find_source_analogs",
            "Identify potential analogous situations across various domains",
            f"Find '{_K.POTENTIAL_ANALOGS}' with '{_K.RATIONALE_FOR_CHOICE}' and '{_K.SIMILARITY_CRITERIA_USED}'",
            [
                _field(
                    _K.POTENTIAL_ANALOGS,
                    "list",
                    f"{_C}A JSON list of 2 to 4 situations from other domains "
                    "(nature, technology, history, ...) that share the "
                    "problem's structure.",
                    reads=(_K.TARGET_PROBLEM_DESCRIPTION, _K.KEY_FEATURES_OF_TARGET),
                ),
                _field(
                    _K.RATIONALE_FOR_CHOICE,
                    "any",
                    f"{_C}Why each potential analog could offer insight into "
                    "the problem.",
                    reads=(_K.KEY_FEATURES_OF_TARGET, _K.POTENTIAL_ANALOGS),
                ),
                _field(
                    _K.SIMILARITY_CRITERIA_USED,
                    "list",
                    f"{_C}A JSON list of the similarity criteria used to pick "
                    "the analogs.",
                    reads=(_K.KEY_FEATURES_OF_TARGET, _K.POTENTIAL_ANALOGS),
                ),
            ],
            _then("map_correspondences", "Source analogs identified"),
        ),
        "map_correspondences": _state(
            "map_correspondences",
            "Create systematic mapping between source analog and target problem",
            f"Select best analog and create detailed mapping with '{_K.SELECTED_ANALOG}', '{_K.ANALOGICAL_MAPPING}', '{_K.IDENTIFIED_SIMILARITIES}', '{_K.IDENTIFIED_DIFFERENCES}'",
            [
                _field(
                    _K.SELECTED_ANALOG,
                    "any",
                    f"{_C}The potential analog with the strongest structural "
                    "similarity to the problem.",
                    reads=(
                        _K.KEY_FEATURES_OF_TARGET,
                        _K.POTENTIAL_ANALOGS,
                        _K.RATIONALE_FOR_CHOICE,
                    ),
                ),
                _field(
                    _K.ANALOGICAL_MAPPING,
                    "any",
                    f"{_C}What in the selected analog corresponds to what in "
                    "the problem (A corresponds to X, B to Y).",
                    reads=(_K.KEY_FEATURES_OF_TARGET, _K.SELECTED_ANALOG),
                ),
                _field(
                    _K.IDENTIFIED_SIMILARITIES,
                    "list",
                    f"{_C}A JSON list of the strongest similarities between "
                    "the selected analog and the problem.",
                    reads=(_K.SELECTED_ANALOG, _K.ANALOGICAL_MAPPING),
                ),
                _field(
                    _K.IDENTIFIED_DIFFERENCES,
                    "list",
                    f"{_C}A JSON list of the important differences that limit "
                    "the analogy.",
                    reads=(_K.SELECTED_ANALOG, _K.ANALOGICAL_MAPPING),
                ),
            ],
            _then("transfer_insights", "Correspondences systematically mapped"),
        ),
        "transfer_insights": _state(
            "transfer_insights",
            "Transfer knowledge and solutions from analog to target domain",
            f"Generate '{_K.TRANSFERRED_INSIGHTS_OR_SOLUTIONS}' and '{_K.POTENTIAL_INFERENCES}'",
            [
                _field(
                    _K.TRANSFERRED_INSIGHTS_OR_SOLUTIONS,
                    "any",
                    f"{_C}The solutions, principles or mechanisms from the "
                    "selected analog that apply to the problem, and how they "
                    "apply.",
                    reads=(
                        _K.SELECTED_ANALOG,
                        _K.ANALOGICAL_MAPPING,
                        _K.IDENTIFIED_SIMILARITIES,
                        _K.IDENTIFIED_DIFFERENCES,
                    ),
                ),
                _field(
                    _K.POTENTIAL_INFERENCES,
                    "list",
                    f"{_C}A JSON list of predictions or inferences about the "
                    "problem that the analogy suggests.",
                    reads=(_K.ANALOGICAL_MAPPING, _K.TRANSFERRED_INSIGHTS_OR_SOLUTIONS),
                ),
            ],
            _then("evaluate_analogy_fit", "Insights transferred"),
        ),
        "evaluate_analogy_fit": _state(
            "evaluate_analogy_fit",
            "Critically evaluate the analogy's validity and practical utility",
            f"Assess analogy with '{_K.ANALOGY_STRENGTHS}', '{_K.ANALOGY_WEAKNESSES_OR_LIMITATIONS}', '{_K.ADAPTED_SOLUTION_OR_UNDERSTANDING}', '{_K.ANALOGY_CONFIDENCE_RATING}'",
            [
                _field(
                    _K.ANALOGY_STRENGTHS,
                    "list",
                    f"{_C}A JSON list of the most compelling aspects of the analogy.",
                    reads=(_K.SELECTED_ANALOG, _K.IDENTIFIED_SIMILARITIES),
                ),
                _field(
                    _K.ANALOGY_WEAKNESSES_OR_LIMITATIONS,
                    "list",
                    f"{_C}A JSON list of where the analogy breaks down or misleads.",
                    reads=(_K.SELECTED_ANALOG, _K.IDENTIFIED_DIFFERENCES),
                ),
                _field(
                    _K.ADAPTED_SOLUTION_OR_UNDERSTANDING,
                    "any",
                    f"{_C}Your answer to problem_statement adapted from the "
                    "analogy: the transferred insight fitted to the problem, "
                    "stated plainly.",
                    reads=(
                        _K.SELECTED_ANALOG,
                        _K.TRANSFERRED_INSIGHTS_OR_SOLUTIONS,
                        _K.POTENTIAL_INFERENCES,
                        _K.IDENTIFIED_DIFFERENCES,
                    ),
                ),
                _field(
                    _K.ANALOGY_CONFIDENCE_RATING,
                    "float",
                    "Your rating from 1 to 10 of how confident you are in "
                    "the adapted answer.",
                    reads=(_K.ADAPTED_SOLUTION_OR_UNDERSTANDING,),
                    required=False,
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# HYBRID REASONING FSM - Integrated multi-approach reasoning with loop prevention
# ============================================================================

_HYBRID_WORK = (
    _K.ANALYTICAL_BREAKDOWN,
    _K.LOGICAL_CONCLUSIONS,
    _K.CREATIVE_INSIGHTS,
)

hybrid_fsm = {
    "name": "hybrid_reasoning",
    "description": "Systematically combines multiple reasoning approaches for comprehensive problem solving with loop prevention mechanisms",
    "initial_state": "identify_components",
    "persona": "You are a master strategist who skillfully combines different reasoning approaches to tackle complex problems from multiple complementary angles.",
    # The loop counter is written only by the HybridLoopCounter handler on
    # critical_evaluation exit (D-054); the model never sets it.
    "handler_only_keys": [_K.HYBRID_LOOP_COUNT],
    "states": {
        "identify_components": _state(
            "identify_components",
            "Break problem into components requiring different reasoning approaches",
            f"Map '{_K.PROBLEM_ASPECTS}' to reasoning types in '{_K.REASONING_MAP}'",
            [
                _field(
                    _K.PROBLEM_ASPECTS,
                    "list",
                    f"{_C}A JSON list of the aspects of the problem that need "
                    "different kinds of reasoning.",
                    reads=_PROBLEM_READS,
                ),
                _field(
                    _K.REASONING_MAP,
                    "any",
                    f"{_C}For each problem aspect, the reasoning approach it "
                    "needs (analytical, logical, creative, critical or "
                    "inductive).",
                    reads=(_K.PROBLEM_ASPECTS,),
                ),
            ],
            _then(
                "apply_analytical",
                "Components identified and mapped to reasoning approaches",
            ),
        ),
        "apply_analytical": _state(
            "apply_analytical",
            "Apply systematic analytical reasoning to understand problem structure",
            f"Create '{_K.ANALYTICAL_BREAKDOWN}' and '{_K.COMPONENT_RELATIONSHIPS}'",
            [
                _field(
                    _K.ANALYTICAL_BREAKDOWN,
                    "any",
                    f"{_C}Your analytical breakdown of the problem into "
                    "simpler parts and how each contributes to the whole.",
                    reads=(_K.PROBLEM_ASPECTS, _K.REASONING_MAP),
                ),
                _field(
                    _K.COMPONENT_RELATIONSHIPS,
                    "any",
                    f"{_C}The relationships, dependencies and interactions "
                    "between those parts.",
                    reads=(_K.PROBLEM_ASPECTS, _K.ANALYTICAL_BREAKDOWN),
                ),
            ],
            _then("apply_logical", "Analytical reasoning systematically applied"),
        ),
        "apply_logical": _state(
            "apply_logical",
            "Apply logical reasoning to derive sound conclusions",
            f"Establish '{_K.LOGICAL_CONCLUSIONS}' and '{_K.REASONING_CHAIN}'",
            [
                _field(
                    _K.LOGICAL_CONCLUSIONS,
                    "list",
                    f"{_C}A JSON list of what follows logically from the "
                    "analytical breakdown.",
                    reads=(_K.ANALYTICAL_BREAKDOWN, _K.COMPONENT_RELATIONSHIPS),
                ),
                _field(
                    _K.REASONING_CHAIN,
                    "list",
                    f"{_C}A JSON list of the logical steps, in order, that "
                    "lead to those conclusions.",
                    reads=(_K.ANALYTICAL_BREAKDOWN, _K.LOGICAL_CONCLUSIONS),
                ),
            ],
            _then("apply_creative", "Logical reasoning systematically applied"),
        ),
        "apply_creative": _state(
            "apply_creative",
            "Apply creative thinking to generate novel approaches and insights",
            f"Generate '{_K.CREATIVE_INSIGHTS}' and '{_K.NOVEL_APPROACHES}'",
            [
                _field(
                    _K.CREATIVE_INSIGHTS,
                    "list",
                    f"{_C}A JSON list of new perspectives or connections that "
                    "extend the analytical and logical work.",
                    reads=(_K.ANALYTICAL_BREAKDOWN, _K.LOGICAL_CONCLUSIONS),
                ),
                _field(
                    _K.NOVEL_APPROACHES,
                    "list",
                    f"{_C}A JSON list of unconventional approaches to the problem.",
                    reads=(_K.LOGICAL_CONCLUSIONS, _K.CREATIVE_INSIGHTS),
                ),
            ],
            _then(HYBRID_EVALUATION_STATE, "Creative reasoning systematically applied"),
        ),
        HYBRID_EVALUATION_STATE: _state(
            HYBRID_EVALUATION_STATE,
            "Critically evaluate all findings with systematic loop prevention",
            f"Create '{_K.EVALUATION_RESULTS}' and determine if refinement needed (maximum {Defaults.MAX_HYBRID_LOOPS} loops)",
            [
                _field(
                    _K.EVALUATION_RESULTS,
                    "any",
                    f"{_C}Your critical evaluation of the combined work: how "
                    "the approaches complement each other, contradictions or "
                    "gaps, strengths and weaknesses.",
                    reads=_HYBRID_WORK,
                ),
                # DECISION plan-2026-10-01T093600-944e2692/D-055: the back
                # edge reads an explicit bool field. Do NOT drop it back to a
                # sentence in state-level instructions: with no bulk call
                # nothing would ever write needs_refinement.
                _field(
                    _K.NEEDS_REFINEMENT,
                    "bool",
                    "Your judgment: true only when the evaluation_results "
                    "value names a serious, fundamental flaw that another "
                    "pass would fix; false otherwise.",
                    reads=(*_HYBRID_WORK, _K.EVALUATION_RESULTS),
                    required=False,
                ),
            ],
            [
                {
                    "target_state": "integrate_solution",
                    "description": "Ready to integrate (no critical issues or loop limit reached)",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "No critical issues found or maximum loops reached",
                            "logic": {
                                "or": [
                                    {"!=": [{"var": _K.NEEDS_REFINEMENT}, True]},
                                    {
                                        ">=": [
                                            {"var": [_K.HYBRID_LOOP_COUNT, 0]},
                                            Defaults.MAX_HYBRID_LOOPS,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": "identify_components",
                    "description": "Refinement needed (limited loops remaining)",
                    "priority": 2,
                    "conditions": [
                        {
                            "description": "Critical issues found and loops available",
                            "logic": {
                                "and": [
                                    {"==": [{"var": _K.NEEDS_REFINEMENT}, True]},
                                    {
                                        "<": [
                                            {"var": [_K.HYBRID_LOOP_COUNT, 0]},
                                            Defaults.MAX_HYBRID_LOOPS,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
            ],
        ),
        "integrate_solution": _state(
            "integrate_solution",
            "Integrate insights from all reasoning approaches into comprehensive solution",
            f"Create '{_K.INTEGRATED_SOLUTION}' with '{_K.REASONING_SYNTHESIS_NOTES}'",
            [
                _field(
                    _K.INTEGRATED_SOLUTION,
                    "any",
                    f"{_C}Your integrated answer to problem_statement that "
                    "combines the analytical, logical and creative work and "
                    "addresses the evaluation.",
                    reads=(*_HYBRID_WORK, _K.EVALUATION_RESULTS),
                ),
                _field(
                    _K.REASONING_SYNTHESIS_NOTES,
                    "any",
                    f"{_C}How the reasoning approaches reinforced or "
                    "challenged each other in that answer.",
                    reads=(*_HYBRID_WORK, _K.INTEGRATED_SOLUTION),
                ),
            ],
            _then("finalize_hybrid", "Solution comprehensively integrated"),
        ),
        "finalize_hybrid": _state(
            "finalize_hybrid",
            "Present final hybrid solution with complete reasoning synthesis",
            f"Finalize '{_K.FINAL_HYBRID_SOLUTION}' and '{_K.REASONING_SYNTHESIS}'",
            [
                _field(
                    _K.FINAL_HYBRID_SOLUTION,
                    "any",
                    f"{_C}Your final, complete answer to problem_statement, "
                    "stated plainly first, then the reasoning behind it.",
                    reads=(
                        _K.INTEGRATED_SOLUTION,
                        _K.REASONING_SYNTHESIS_NOTES,
                        _K.EVALUATION_RESULTS,
                    ),
                ),
                _field(
                    _K.REASONING_SYNTHESIS,
                    "any",
                    f"{_C}How each reasoning type contributed to the final answer.",
                    reads=(_K.FINAL_HYBRID_SOLUTION, _K.REASONING_SYNTHESIS_NOTES),
                ),
            ],
            [],
            response_instructions=ANSWER_RESPONSE_INSTRUCTIONS,
        ),
    },
}


# ============================================================================
# FSM REGISTRY - Complete collection for easy access and management
# ============================================================================

ALL_REASONING_FSMS = {
    "orchestrator": orchestrator_fsm,
    "classifier": classifier_fsm,
    "simple_calculator": simple_calculator_fsm,
    "analytical": analytical_fsm,
    "deductive": deductive_fsm,
    "inductive": inductive_fsm,
    "creative": creative_fsm,
    "critical": critical_fsm,
    "abductive": abductive_fsm,
    "analogical": analogical_fsm,
    "hybrid": hybrid_fsm,
}
