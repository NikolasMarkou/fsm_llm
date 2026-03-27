from __future__ import annotations

"""
Constants for the agents package.
"""

from fsm_llm.constants import DEFAULT_LLM_MODEL

# ---------------------------------------------------------------------------
# ReAct states (original)
# ---------------------------------------------------------------------------


class AgentStates:
    """States in the ReAct agent FSM."""

    THINK = "think"
    ACT = "act"
    CONCLUDE = "conclude"
    AWAIT_APPROVAL = "await_approval"


# ---------------------------------------------------------------------------
# Pattern-specific states
# ---------------------------------------------------------------------------


class ReflexionStates:
    """States in the Reflexion agent FSM."""

    THINK = "think"
    ACT = "act"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    CONCLUDE = "conclude"


class PlanExecuteStates:
    """States in the Plan-and-Execute agent FSM."""

    PLAN = "plan"
    EXECUTE_STEP = "execute_step"
    CHECK_RESULT = "check_result"
    REPLAN = "replan"
    SYNTHESIZE = "synthesize"


class REWOOStates:
    """States in the REWOO agent FSM."""

    PLAN_ALL = "plan_all"
    EXECUTE_PLANS = "execute_plans"
    SOLVE = "solve"


class EvalOptStates:
    """States in the Evaluator-Optimizer agent FSM."""

    GENERATE = "generate"
    EVALUATE = "evaluate"
    REFINE = "refine"
    OUTPUT = "output"


class MakerCheckerStates:
    """States in the Maker-Checker agent FSM."""

    MAKE = "make"
    CHECK = "check"
    REVISE = "revise"
    OUTPUT = "output"


class PromptChainStates:
    """States in the Prompt Chaining agent FSM (dynamic)."""

    OUTPUT = "output"
    STEP_PREFIX = "step_"
    GATE_PREFIX = "gate_"


class SelfConsistencyStates:
    """States in the Self-Consistency agent FSM."""

    GENERATE = "generate"
    AGGREGATE = "aggregate"


class OrchestratorStates:
    """States in the Orchestrator-Workers agent FSM."""

    ORCHESTRATE = "orchestrate"
    DELEGATE = "delegate"
    COLLECT = "collect"
    SYNTHESIZE = "synthesize"


class DebateStates:
    """States in the Debate agent FSM."""

    PROPOSE = "propose"
    CRITIQUE = "critique"
    COUNTER = "counter"
    JUDGE = "judge"
    CONCLUDE = "conclude"


class ADaPTStates:
    """States in the ADaPT agent FSM."""

    ATTEMPT = "attempt"
    ASSESS = "assess"
    DECOMPOSE = "decompose"
    COMBINE = "combine"


# ---------------------------------------------------------------------------
# Context keys
# ---------------------------------------------------------------------------


class ContextKeys:
    """Standard context keys used across the agents package."""

    # Task
    TASK = "task"

    # Tool selection (extracted by LLM in think state)
    TOOL_NAME = "tool_name"
    TOOL_INPUT = "tool_input"
    REASONING = "reasoning"
    SHOULD_TERMINATE = "should_terminate"

    # Tool execution results
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    TOOL_STATUS = "tool_status"

    # Observations accumulated across iterations
    OBSERVATIONS = "observations"
    OBSERVATION_COUNT = "observation_count"

    # Final answer
    FINAL_ANSWER = "final_answer"
    CONFIDENCE = "confidence"

    # Budget tracking
    ITERATION_COUNT = "iteration_count"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

    # HITL
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"

    # Agent trace
    AGENT_TRACE = "agent_trace"

    # Reflexion
    EVALUATION_PASSED = "evaluation_passed"
    EVALUATION_SCORE = "evaluation_score"
    EVALUATION_FEEDBACK = "evaluation_feedback"
    EPISODIC_MEMORY = "episodic_memory"
    REFLECTION_COUNT = "reflection_count"

    # Plan-and-Execute
    PLAN_STEPS = "plan_steps"
    CURRENT_STEP_INDEX = "current_step_index"
    STEP_RESULTS = "step_results"
    ALL_STEPS_COMPLETE = "all_steps_complete"
    STEP_FAILED = "step_failed"

    # REWOO
    EVIDENCE = "evidence"
    PLAN_BLUEPRINT = "plan_blueprint"

    # Evaluator-Optimizer
    GENERATED_OUTPUT = "generated_output"
    EVALUATION_RESULT = "evaluation_result"
    REFINEMENT_FEEDBACK = "refinement_feedback"
    REFINEMENT_COUNT = "refinement_count"

    # Maker-Checker
    DRAFT_OUTPUT = "draft_output"
    CHECKER_FEEDBACK = "checker_feedback"
    CHECKER_PASSED = "checker_passed"
    REVISION_COUNT = "revision_count"

    # Prompt Chaining
    CHAIN_STEP_INDEX = "chain_step_index"
    CHAIN_STEP_RESULT = "chain_step_result"
    CHAIN_RESULTS = "chain_results"
    GATE_PASSED = "gate_passed"

    # Self-Consistency
    SAMPLES = "samples"
    AGGREGATED_ANSWER = "aggregated_answer"

    # Orchestrator-Workers
    SUBTASKS = "subtasks"
    WORKER_RESULTS = "worker_results"
    DELEGATION_PLAN = "delegation_plan"
    ALL_COLLECTED = "all_collected"

    # Debate
    PROPOSITION = "proposition"
    CRITIQUE = "critique"
    COUNTER_ARGUMENT = "counter_argument"
    JUDGE_VERDICT = "judge_verdict"
    DEBATE_ROUNDS = "debate_rounds"
    CURRENT_ROUND = "current_round"
    CONSENSUS_REACHED = "consensus_reached"

    # ADaPT
    ATTEMPT_RESULT = "attempt_result"
    ATTEMPT_SUCCEEDED = "attempt_succeeded"
    SUBTASK_RESULTS = "subtask_results"
    CURRENT_DEPTH = "current_depth"


# ---------------------------------------------------------------------------
# Handler names
# ---------------------------------------------------------------------------


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
    OBSERVATION_TRACKER = "AgentObservationTracker"
    HITL_GATE = "AgentHITLGate"
    END_CONVERSATION = "AgentEndConversation"
    ERROR = "AgentError"

    # Pattern-specific handlers
    REFLEXION_EVALUATOR = "ReflexionEvaluator"
    REFLEXION_REFLECTOR = "ReflexionReflector"
    PLAN_STEP_EXECUTOR = "PlanStepExecutor"
    PLAN_STEP_CHECKER = "PlanStepChecker"
    REWOO_EXECUTOR = "REWOOExecutor"
    EVAL_OPT_EVALUATOR = "EvalOptEvaluator"
    MAKER_CHECKER_CHECKER = "MakerCheckerChecker"
    CHAIN_GATE_CHECKER = "ChainGateChecker"
    ORCHESTRATOR_DELEGATOR = "OrchestratorDelegator"
    DEBATE_JUDGE = "DebateJudge"
    ADAPT_ASSESSOR = "ADaPTAssessor"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class Defaults:
    """Default configuration values."""

    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.5
    MAX_TOKENS = 1000
    MAX_ITERATIONS = 10
    TIMEOUT_SECONDS = 300.0
    MAX_OBSERVATION_LENGTH = 2000
    MAX_OBSERVATIONS = 20
    CONFIDENCE_THRESHOLD = 0.3

    # Message sent to advance the FSM conversation loop
    CONTINUE_MESSAGE = "Continue."

    # Multiplier for computing hard iteration ceiling from max_iterations.
    # Each agent cycle uses multiple FSM transitions; this factor provides
    # headroom so the FSM can finish its current cycle before the budget
    # check fires.
    FSM_BUDGET_MULTIPLIER = 3

    # Reflexion
    MAX_REFLECTIONS = 3
    EVALUATION_THRESHOLD = 0.7

    # Plan-and-Execute
    MAX_PLAN_STEPS = 10
    MAX_REPLANS = 2

    # Evaluator-Optimizer
    MAX_REFINEMENTS = 3

    # Maker-Checker
    MAX_REVISIONS = 3
    QUALITY_THRESHOLD = 0.7

    # Self-Consistency
    NUM_SAMPLES = 5
    SAMPLE_TEMPERATURE_RANGE = (0.5, 1.0)

    # Orchestrator
    MAX_WORKERS = 5

    # Debate
    MAX_DEBATE_ROUNDS = 3

    # ADaPT
    MAX_DECOMPOSITION_DEPTH = 3


# ---------------------------------------------------------------------------
# Error and log messages
# ---------------------------------------------------------------------------


class ReasoningIntegrationKeys:
    """Context keys for reasoning-agent integration (namespaced to avoid collision)."""

    REASONING_RESULT = "reasoning_integration_result"
    REASONING_TYPE_USED = "reasoning_integration_type_used"
    REASONING_CONFIDENCE = "reasoning_integration_confidence"
    REASONING_TOOL_NAME = "reason"


# ---------------------------------------------------------------------------
# Meta-builder states and constants
# ---------------------------------------------------------------------------


class MetaBuilderStates:
    """States in the MetaBuilderAgent FSM."""

    INTAKE = "intake"
    PLAN = "plan"
    BUILD = "build"
    REVIEW = "review"
    REVISE = "revise"
    OUTPUT = "output"


class MetaDefaults:
    """Default configuration values for the meta-builder."""

    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    MAX_TURNS = 50

    # Builder defaults
    DEFAULT_PRIORITY = 100
    SUMMARY_TRUNCATE_WIDTH = 80

    # ReactAgent configuration for the build phase
    BUILD_MAX_ITERATIONS = 25
    BUILD_TIMEOUT_SECONDS = 120.0
    BUILD_TEMPERATURE = 0.3
    BUILD_MAX_TOKENS = 1000

    # Agent builder defaults (for the agent artifact being built)
    AGENT_MODEL = "gpt-4o-mini"
    AGENT_MAX_ITERATIONS = 10
    AGENT_TIMEOUT_SECONDS = 300.0
    AGENT_TEMPERATURE = 0.5
    AGENT_MAX_TOKENS = 1000


class DecisionWords:
    """Word sets for detecting user approval or revision intent."""

    APPROVE: frozenset[str] = frozenset(
        {
            "approve",
            "approved",
            "yes",
            "y",
            "ok",
            "okay",
            "looks good",
            "lgtm",
            "accept",
            "accepted",
            "confirm",
            "confirmed",
            "good",
            "great",
            "perfect",
            "ship it",
            "go ahead",
            "sounds good",
            "fine",
            "done",
            "correct",
            "right",
            "build it",
            "just build it",
            "do it",
            "that works",
            "sure",
            "yep",
            "yeah",
            "nice",
            "awesome",
            "keep it",
            "give it to me",
            "give me",
            "show me",
            "let me see",
            "send it",
            "save it",
            "export",
            "output",
            "generate",
            "finish",
            "complete",
            "wrap up",
            "all good",
            "no changes",
        }
    )

    REVISE: frozenset[str] = frozenset(
        {
            "revise",
            "revision",
            "change",
            "changes",
            "modify",
            "edit",
            "update",
            "fix",
            "no",
            "nope",
            "redo",
            "wrong",
            "incorrect",
            "not right",
            "needs work",
            "not quite",
            "try again",
            "add",
            "remove",
            "delete",
            "rename",
            "replace",
            "move",
            "instead",
            "different",
            "but",
        }
    )


class MetaLogMessages:
    """Standard log message templates for meta-builder."""

    META_STARTED = "Meta-agent started with model={model}"
    ARTIFACT_CLASSIFIED = "Artifact type classified as: {artifact_type}"
    BUILD_STARTED = "Build phase started for {artifact_type}"
    BUILD_COMPLETE = "Build complete: {artifact_type} '{name}'"
    REVIEW_STARTED = "Review phase: {error_count} errors, {warning_count} warnings"
    REVISION_STARTED = "Revision requested: {revision}"


class MetaErrorMessages:
    """Standard error messages for meta-builder."""

    BUILDER_NOT_INITIALIZED = "Builder has not been initialized yet"
    CONVERSATION_NOT_STARTED = "Conversation has not been started"
    CONVERSATION_ALREADY_STARTED = "Conversation has already been started"
    INVALID_ARTIFACT_TYPE = "Invalid artifact type: '{artifact_type}'"


class ErrorMessages:
    """Standard error messages."""

    BUDGET_EXHAUSTED = "Agent exceeded maximum iterations ({limit})"
    TOOL_NOT_FOUND = "Tool '{name}' not found in registry"
    TOOL_EXECUTION_FAILED = "Tool '{name}' execution failed: {error}"
    APPROVAL_DENIED = "Human denied approval for: {action}"
    TIMEOUT = "Agent timed out after {seconds:.1f}s"
    NO_TOOLS = "Cannot create agent with empty tool registry"
    MAX_REFLECTIONS = "Maximum reflections ({limit}) reached"
    MAX_REFINEMENTS = "Maximum refinements ({limit}) reached"
    MAX_REVISIONS = "Maximum revisions ({limit}) reached"
    MAX_DEPTH = "Maximum decomposition depth ({limit}) reached"
    EMPTY_CHAIN = "Cannot create prompt chain agent with empty chain"
    NO_SAMPLES = "num_samples must be at least 1"


class LogMessages:
    """Standard log message templates."""

    AGENT_STARTED = "Agent started with {tool_count} tools, model={model}"
    TOOL_SELECTED = "Selected tool: {name} with input: {input}"
    TOOL_EXECUTED = "Tool '{name}' executed successfully"
    TOOL_FAILED = "Tool '{name}' failed: {error}"
    ITERATION = "Iteration {current}/{max}"
    AGENT_COMPLETE = "Agent completed in {iterations} iterations"
    APPROVAL_REQUESTED = "Requesting approval for: {action}"
    APPROVAL_RESULT = "Approval {result} for: {action}"
    ESCALATION = "Escalating to human: {reason}"
    REFLECTION = "Reflection {current}/{max}: {summary}"
    PLAN_STEP = "Executing plan step {current}/{total}: {description}"
    EVALUATION = "Evaluation {result}: score={score}"
    DEBATE_ROUND = "Debate round {current}/{max}"
    DECOMPOSITION = "Decomposing task at depth {depth}"
