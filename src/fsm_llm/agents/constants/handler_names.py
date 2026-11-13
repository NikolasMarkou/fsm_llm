"""Handler priorities and registration names of the agent handlers."""

from __future__ import annotations


class HandlerPriorities:
    """Explicit priorities for agent handler execution order.

    Lower numbers execute first.  When multiple handlers share the same
    timing hook (e.g. PRE_TRANSITION), priority determines their order.
    """

    HITL_GATE = 10  # Must run first to block unapproved tools
    ITERATION_LIMITER = 50  # Budget check before transitions
    TOOL_EXECUTOR = 100  # Execute tool after approval + budget checks
    END_CONVERSATION = 200  # Finalize trace on conversation end
    ERROR = 200  # Log errors


class HandlerNames:
    """Handler names for registration."""

    TOOL_EXECUTOR = "AgentToolExecutor"
    ITERATION_LIMITER = "AgentIterationLimiter"
    THINK_FRESH_KEYS = "AgentThinkFreshKeys"
    FEEDBACK_CONSUMED = "AgentFeedbackConsumed"
    APPROVAL_FRESH_KEYS = "AgentApprovalFreshKeys"
    HITL_GATE = "AgentHITLGate"
    ERROR = "AgentError"

    # Pattern-specific handlers
    REFLEXION_EVALUATOR = "ReflexionEvaluator"
    REFLEXION_REFLECTOR = "ReflexionReflector"
    REFLEXION_FRESH_KEYS = "ReflexionFreshKeys"
    PLAN_STEP_EXECUTOR = "PlanStepExecutor"
    PLAN_STEP_CHECKER = "PlanStepChecker"
    PLAN_STEP_FRESH_KEYS = "PlanStepFreshKeys"
    PLAN_REPLANNER = "PlanReplanCounter"
    REWOO_EXECUTOR = "REWOOExecutor"
    EVAL_OPT_EVALUATOR = "EvalOptEvaluator"
    EVAL_OPT_REFINE_ENTRY = "EvalOptRefineEntry"
    EVAL_OPT_REFINE_EXIT = "EvalOptRefineExit"
    MAKER_CHECKER_CHECKER = "MakerCheckerChecker"
    MAKER_CHECKER_REVISE_ENTRY = "MakerCheckerReviseEntry"
    MAKER_CHECKER_REVISE_EXIT = "MakerCheckerReviseExit"
    MAKER_CHECKER_FORCE_PASS = "MakerCheckerForcePass"
    MAKER_CHECKER_REVISION_COUNT = "MakerCheckerRevisionCount"
    CHAIN_GATE_CHECKER = "ChainGateChecker"
    ORCHESTRATOR_DELEGATOR = "OrchestratorDelegator"
    ORCHESTRATOR_DECISION_RESET = "OrchestratorDecisionReset"
    DEBATE_JUDGE = "DebateJudge"
    DEBATE_CONSENSUS_RESET = "DebateConsensusReset"
    ADAPT_DECOMPOSE_TRACKER = "ADaPTAssessor"
    ADAPT_SUBTASK_EXECUTOR = "subtask_executor"
    CONTEXT_COMPACTOR = "AgentContextCompactor"
    OBSERVATION_SUMMARIZER = "AgentObservationSummarizer"
