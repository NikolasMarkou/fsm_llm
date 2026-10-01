"""
Constants for the agents package.
"""

from __future__ import annotations

from fsm_llm.constants import DEFAULT_LLM_MODEL

# ---------------------------------------------------------------------------
# Agent FSM state names
# ---------------------------------------------------------------------------
# DECISION plan-2026-09-30T062855-07ad3f8c/D-047: each class names every state
# of its pattern's FSM, and the builders (`fsm_definitions.py`,
# `parallel_react.py`), the per-state prompt maps and the handler
# registrations all read these names. Do NOT write a state id as a string
# literal in a builder or a handler registration, and do NOT drop a member
# because only the builder reads it: the same literal in a builder and in a
# handler registration is a lockstep invariant nothing checks (a rename in one
# place leaves a handler that never fires). See decisions.md D-047.


class AgentStates:
    """States in the ReAct agent FSM (also ParallelReact, ReasoningReact,
    VerifiedReact, AutoMemory) and the shared ``await_approval`` state."""

    THINK = "think"
    ACT = "act"
    AWAIT_APPROVAL = "await_approval"
    CONCLUDE = "conclude"


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
    # With HITL, the shared approval state (`_await_approval_state`).
    AWAIT_APPROVAL = AgentStates.AWAIT_APPROVAL


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


class SelfConsistencyStates:
    """States in the Self-Consistency agent FSM."""

    GENERATE = "generate"


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

    # Sentinel value for "no tool selected"
    NO_TOOL = "none"

    # Tool execution results
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    TOOL_STATUS = "tool_status"

    # Observations accumulated across iterations
    OBSERVATIONS = "observations"
    OBSERVATION_COUNT = "observation_count"

    # No pattern writes or reads it and no state extracts it (D-046): the
    # answer is the speaking state's reply or a pattern answer key. Kept in
    # RUN_OUTPUT_KEYS, so a caller cannot put a ready-made answer into the
    # context the final state's prompt shows.
    FINAL_ANSWER = "final_answer"
    CONFIDENCE = "confidence"

    # Budget tracking
    ITERATION_COUNT = "iteration_count"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    # Why a run was forced to stop (a ``StopReason`` value), written only by
    # the handler that forced it (stall, forced pass); unset on a forced stop
    # means ``StopReason.MAX_ITERATIONS``.
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-027: public, NOT
    # internal-prefixed. Do NOT rename it to `_forced_stop_reason`:
    # `API.get_data` drops internal keys, so the result seam never saw it and a
    # forced pass read as success=True. Only a `StopReason.FORCED` value counts
    # and it can only lower `success`; caller context never supplies it
    # (`RUN_OUTPUT_KEYS`).
    FORCED_STOP_REASON = "forced_stop_reason"
    # Why the last loop turn did nothing (an executor warning, a HITL denial),
    # for the next think turn's prompt. Not a transient key: the compactor
    # would delete it before think reads it (LOOP-06, D-021 of plan 06a5ec0a).
    AGENT_FEEDBACK = "agent_feedback"
    # Gated calls a human approver refused in this run and that did not run: a
    # list of sentences (tool, redacted parameters, "was refused ... and was
    # not performed", `handlers.refusal_record`), appended only by the HITL
    # driver and removed only when the same call is approved later and runs
    # (D-045). Kept to the end of the run and read by the conclude prompt.
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-034: do NOT report a denial to conclude through
    # `agent_feedback` (cleared on think exit, so the answer then claimed the
    # refused action happened) and do NOT record it as an observation (it
    # would count as tool evidence, LOOP-06). Each entry states a final fact;
    # do NOT word it as a skipped or pending item, the model reads those as
    # work to retry (06a5ec0a/D-049). See decisions.md D-034.
    REFUSED_ACTIONS = "refused_actions"
    # The state a transition leaves, written by core on every transition
    # (absent before the first one). Read by the ReAct-family limiter.
    CURRENT_STATE = "_current_state"

    # HITL
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    # Driver-only grant: the exact approved call. Internal prefix, so core drops
    # it from every model extraction; only the approval driver writes it.
    DRIVER_APPROVAL = "_approval_granted"
    # Count of driver grants the executor spent, written by its delta. A value
    # behind the executor's own count means a spending delta was discarded.
    APPROVALS_SPENT = "_approvals_spent"

    # Agent trace
    AGENT_TRACE = "agent_trace"

    # Reflexion
    EVALUATION_PASSED = "evaluation_passed"
    EVALUATION_SCORE = "evaluation_score"
    EVALUATION_FEEDBACK = "evaluation_feedback"
    EPISODIC_MEMORY = "episodic_memory"
    REFLECTION_COUNT = "reflection_count"
    REFLECTION = "reflection"
    LESSONS = "lessons"

    # Plan-and-Execute
    PLAN_STEPS = "plan_steps"
    CURRENT_STEP_INDEX = "current_step_index"
    STEP_RESULTS = "step_results"
    ALL_STEPS_COMPLETE = "all_steps_complete"
    STEP_FAILED = "step_failed"
    STEP_RESULT = "step_result"
    CURRENT_STEP_DESCRIPTION = "current_step_description"
    # The plan a replan revises (stash, see RESULT_DROPPED_CONTEXT_KEYS).
    PREVIOUS_PLAN_STEPS = "previous_plan_steps"

    # REWOO
    EVIDENCE = "evidence"
    # One entry per executed plan step: {"id", "tool_name", "success"}. The
    # success rule reads it (at least one True), not the evidence mapping.
    EVIDENCE_STATUS = "evidence_status"
    PLAN_BLUEPRINT = "plan_blueprint"

    # Evaluator-Optimizer
    GENERATED_OUTPUT = "generated_output"
    EVALUATION_RESULT = "evaluation_result"
    REFINEMENT_FEEDBACK = "refinement_feedback"
    REFINEMENT_COUNT = "refinement_count"
    PREVIOUS_OUTPUT = "previous_output"

    # Maker-Checker
    DRAFT_OUTPUT = "draft_output"
    CHECKER_FEEDBACK = "checker_feedback"
    CHECKER_PASSED = "checker_passed"
    REVISION_COUNT = "revision_count"
    PREVIOUS_DRAFT = "previous_draft"

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
    # Subtasks over max_workers that never ran; kept out of worker_results
    # (D-049 of plan 06a5ec0a).
    SKIPPED_SUBTASKS = "skipped_subtasks"
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
    # How a decomposition's subtasks combine: "AND" (all) or "OR" (any).
    OPERATOR = "operator"
    CURRENT_DEPTH = "current_depth"


class StopReason:
    """Why an agent run ended: the value of ``AgentResult.stop_reason``.

    ``ANSWERED`` and ``EVIDENCE`` go with ``success=True``; every other
    value goes with ``success=False``.
    """

    # Concluded with a real answer: an answer key or an executed tool call.
    ANSWERED = "answered"
    # Succeeded on executed work: planner evidence, subtasks, a harness role.
    EVIDENCE = "evidence"
    # The iteration budget forced the stop; the last output still ships.
    MAX_ITERATIONS = "max_iterations"
    # A judge overrode a failing (or missing) verdict at a revision or budget
    # limit (EvalOpt, MakerChecker, Debate at num_rounds), or Reflexion hit
    # max_reflections without a passing evaluation. A genuine pass on the
    # limit's round is ANSWERED.
    FORCED_PASS = "forced_pass"
    # Consecutive turns with no tool selected forced the stop.
    STALLED = "stalled"
    # The answer was rejected by a verifier.
    VERIFICATION_FAILED = "verification_failed"
    # The run ended without a real result (prose fallback only, no evidence,
    # a malformed or failed turn, no successful subtask or sample).
    NO_RESULT = "no_result"
    # A prompt-chain validation gate failed.
    GATE_FAILED = "gate_failed"
    # The conversation was ended from outside the run (a hook, another
    # thread, a monitor) before it reached a final state; recorded by
    # ``BaseAgent._run_conversation_loop``, the last output still ships.
    ENDED = "ended"

    # Reasons a forcing handler may record in ``ContextKeys.FORCED_STOP_REASON``
    # (the prompt-chain gate handler records ``GATE_FAILED``; the run loop
    # records ``ENDED``).
    FORCED: frozenset[str] = frozenset(
        {MAX_ITERATIONS, FORCED_PASS, STALLED, GATE_FAILED, ENDED}
    )


# ---------------------------------------------------------------------------
# Handler names
# ---------------------------------------------------------------------------


# DECISION plan-2026-09-24T091842-c1d5bfbc/D-012: the redo stash (set by
# ``make_redraft_handlers`` on entry to a redo state) is run-time prompt input,
# not a result. BaseAgent drops these keys from ``AgentResult.final_context``
# only. Do NOT give them an internal prefix instead: internal keys are also
# filtered out of prompts, which hides the previous draft from the checker and
# refiner that need it. Add any new ``previous_*`` stash key here.
RESULT_DROPPED_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        ContextKeys.PREVIOUS_DRAFT,
        ContextKeys.PREVIOUS_OUTPUT,
        ContextKeys.PREVIOUS_PLAN_STEPS,
    }
)


# Loop values a ReAct think turn produces (react, reasoning_react,
# parallel_react): cleared on think entry so each turn extracts them afresh.
# The executors clear the tool selection themselves.
REACT_THINK_FRESH_KEYS: tuple[str, ...] = (
    ContextKeys.REASONING,
    ContextKeys.SHOULD_TERMINATE,
)


# DECISION plan-2026-09-29T103145-06a5ec0a/D-002: keys a run writes for itself
# (tool selection and results, evidence counters, termination, approval state and
# the driver-only grant). Caller context never supplies them; they are removed
# only by ``base.strip_caller_context`` (see the anchor there). Do NOT add the
# whole internal-prefix family here: ``_sensitive``-style policy inputs and the
# harness roots legitimately arrive through ``initial_context``.
RUN_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        ContextKeys.OBSERVATION_COUNT,
        ContextKeys.OBSERVATIONS,
        ContextKeys.SHOULD_TERMINATE,
        ContextKeys.FINAL_ANSWER,
        ContextKeys.TOOL_NAME,
        ContextKeys.TOOL_INPUT,
        ContextKeys.TOOL_RESULT,
        ContextKeys.TOOL_STATUS,
        ContextKeys.TOOL_ERROR,
        ContextKeys.APPROVAL_REQUIRED,
        ContextKeys.APPROVAL_GRANTED,
        ContextKeys.REASONING,
        ContextKeys.DRIVER_APPROVAL,
        ContextKeys.APPROVALS_SPENT,
        ContextKeys.FORCED_STOP_REASON,
        ContextKeys.AGENT_FEEDBACK,
        ContextKeys.REFUSED_ACTIONS,
    }
)


# DECISION plan-2026-09-29T103145-06a5ec0a/D-051: keys only framework handlers
# write (limiters, stall detector, forcing handlers, tool executors). Every
# agent FSM lists them in core ``handler_only_keys``, so no extraction channel
# (bulk, per-field, post-transition) can plant them: a model writing
# ``max_iterations_reached`` or ``forced_stop_reason`` flipped ``success``,
# and ``observation_count`` is the conclude evidence guard. Do NOT drop a key
# here to let a state extract it; internal-prefixed keys
# (``_approvals_spent``, ``_approval_granted``) need no entry, core never
# extracts them.
FRAMEWORK_ONLY_KEYS: tuple[str, ...] = (
    ContextKeys.MAX_ITERATIONS_REACHED,
    ContextKeys.FORCED_STOP_REASON,
    ContextKeys.ITERATION_COUNT,
    ContextKeys.OBSERVATION_COUNT,
    ContextKeys.REFUSED_ACTIONS,
)


# DECISION plan-2026-09-29T103145-06a5ec0a/D-003: constructor kwargs that reach
# ``BaseAgent.__init__(**api_kwargs)`` only by mistake. The agent-constructor
# names mean the pattern cannot use them (HITL or tools silently ignored and
# forwarded to litellm); the config-owned names collide with the explicit
# ``model``/``temperature``/``max_tokens`` that ``_create_api`` passes. Do NOT
# turn this into a whitelist of allowed kwargs: litellm passthrough (``seed``,
# ``timeout``, ``caching``, ``api_base``, ...) and API kwargs (``handlers``,
# ``llm_interface``, ...) are open-ended and must keep flowing.
MISPLACED_AGENT_KWARGS: frozenset[str] = frozenset(
    {"hitl", "tools", "evaluation_fn", "approval_callback"}
)
CONFIG_OWNED_KWARGS: frozenset[str] = frozenset({"model", "temperature", "max_tokens"})


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
    END_CONVERSATION = "AgentEndConversation"
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
    CHAIN_GATE_CHECKER = "ChainGateChecker"
    ORCHESTRATOR_DELEGATOR = "OrchestratorDelegator"
    ORCHESTRATOR_DECISION_RESET = "OrchestratorDecisionReset"
    DEBATE_JUDGE = "DebateJudge"
    DEBATE_CONSENSUS_RESET = "DebateConsensusReset"
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
    MCP_TIMEOUT_SECONDS = 30.0
    MAX_OBSERVATION_LENGTH = 2000
    MAX_OBSERVATIONS = 20
    CONFIDENCE_THRESHOLD = 0.3
    MIN_ANSWER_LENGTH = 5
    MAX_TASK_PREVIEW_LENGTH = 200
    # Cap on AgentConfig.instructions. They are prefixed to prompt slots that
    # core caps at 5000 chars; a ReAct think slot grows with the tool list, so
    # long instructions plus a large registry can still fail the FSM load.
    MAX_INSTRUCTIONS_LENGTH = 2000

    # Multiplier for computing hard iteration ceiling from max_iterations.
    # Each agent cycle uses multiple FSM transitions; this factor provides
    # headroom so the FSM can finish its current cycle before the budget
    # check fires.
    FSM_BUDGET_MULTIPLIER = 3
    # Loop turns a ReAct-family run may still need after a forced stop to
    # reach its conclude state (Reflexion: think -> act -> evaluate ->
    # conclude). ``AgentHandlers.check_iteration_limit`` forces the stop this
    # many transitions before the hard ceiling (capped at ``max_iterations``).
    FORCED_STOP_MARGIN = 3

    # Reflexion
    MAX_REFLECTIONS = 3

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
    # Heuristic budget factor for sizing the debate FSM iteration cap
    # (num_rounds * FSM_BUDGET_MULTIPLIER * DEBATE_STATES_PER_ROUND). NOTE: a
    # debate round actually traverses 4 states (propose/critique/counter/judge);
    # this 2 is intentionally an under-count that the FSM_BUDGET_MULTIPLIER (3)
    # headroom compensates for at typical round counts. Do not read it as the
    # literal states-per-round (AP3-004).
    DEBATE_STATES_PER_ROUND = 2

    # ADaPT
    MAX_DECOMPOSITION_DEPTH = 3
    # Subtasks run per decomposition; extra ones are dropped with a WARNING.
    ADAPT_MAX_SUBTASKS = 8

    # AgentServer: agent runs in flight at once; one more request gets 503.
    SERVER_MAX_CONCURRENT = 8


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
# Meta-builder constants
# ---------------------------------------------------------------------------


class MetaDefaults:
    """Default configuration values for the meta-builder."""

    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    MAX_TURNS = 50

    # Builder defaults
    DEFAULT_PRIORITY = 100
    SUMMARY_TRUNCATE_WIDTH = 80

    # ReactAgent configuration for the build phase
    BUILD_MAX_ITERATIONS = 25
    BUILD_TIMEOUT_SECONDS = 120.0
    BUILD_TEMPERATURE = 0.3

    # Agent builder defaults (for the agent artifact being built)
    AGENT_MODEL = "gpt-4o-mini"
    AGENT_MAX_ITERATIONS = 10
    AGENT_TIMEOUT_SECONDS = 300.0
    AGENT_TEMPERATURE = 0.5
    AGENT_MAX_TOKENS = 1000


class MetaLogMessages:
    """Standard log message templates for meta-builder."""

    META_STARTED = "Meta-agent started with model={model}"
    BUILD_STARTED = "Build phase started for {artifact_type}"


class MetaErrorMessages:
    """Standard error messages for meta-builder."""

    CONVERSATION_NOT_STARTED = "Conversation has not been started"
    CONVERSATION_ALREADY_STARTED = "Conversation has already been started"


class ErrorMessages:
    """Standard error messages."""

    TOOL_NOT_FOUND = "Tool '{name}' not found in registry"
    TOOL_EXECUTION_FAILED = "Tool '{name}' execution failed: {error}"
    EMPTY_CHAIN = "Cannot create prompt chain agent with empty chain"
    NO_SAMPLES = "num_samples must be at least 1"
    PROMPT_SLOT_OVERFLOW = (
        "Agent prompt too long: {slot} exceeds core's {limit}-character "
        "instruction limit. It holds AgentConfig.instructions ({instructions} "
        "characters) and, in tool-using patterns, the tool catalogue ({tools} "
        "tools). Shorten AgentConfig.instructions (create_agent system_prompt=), "
        "shorten tool descriptions or register fewer tools."
    )


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
